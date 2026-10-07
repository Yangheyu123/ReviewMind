"""AACR-Bench 黑盒外部评测（评测体系 v3 §外部基准）。

协议：
- 不修改其 Ground Truth：正/负样本按语言分层子集（每语言取 change_line_count
  最小的 4 正 / 3 负，确定性排序，不随机）；
- 当前 Agent（完整 ReviewMind：分组路由 + 工具检索 + 锚点 + filter + 置信度地板）
  黑盒跑每个 PR，findings 转为其评论格式（note/path/from_line/to_line/side）；
- 判卷复用其官方 judge.py（path → side → 行号重叠(k=1) → 语义匹配），
  语义阶段用其官方 Mock 模式（本地相似度，确定性、零 LLM-judge）；
- 指标：其官方 Precision/Recall/Line 系 + F1/NoiseRate + tokens/耗时；
  负样本无 GT：所有产出均按误报计，观察"强行找问题"行为。

用法：py -3 scripts/run_aacr.py [--pos-per-lang 4] [--neg-per-lang 3]
"""

from __future__ import annotations

import argparse
import asyncio
import copy
import json
import os
import sys
import time
from collections import defaultdict
from pathlib import Path

BACKEND = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BACKEND))
os.environ.setdefault("LLM_MOCK_MODE", "false")

AACR_ROOT = BACKEND.parent.parent / "aacr-bench"
sys.path.insert(0, str(AACR_ROOT / "evaluation"))
os.environ["JUDGE_USE_MOCK"] = "true"  # 必须在 import judge 前设置：确定性语义匹配

from app.core.config import settings  # noqa: E402
from app.models.review_job import ReviewJob  # noqa: E402
from app.services.review_job_store import ReviewJobStore  # noqa: E402
from app.services.github_client import GitHubClient  # noqa: E402
from app.agent_loop.engine import run_engine  # noqa: E402
from judge import evaluate_comments, compute_cr_statistics  # noqa: E402  AACR 官方判卷器


def stratified(samples: list[dict], per_lang: int) -> list[dict]:
    """按语言分层，各取 change_line_count 最小的 per_lang 个（确定性排序）。"""
    by_lang: dict[str, list[dict]] = defaultdict(list)
    for s in samples:
        by_lang[s.get("project_main_language") or "?"].append(s)
    picked: list[dict] = []
    for lang in sorted(by_lang):
        bucket = sorted(by_lang[lang], key=lambda s: (s.get("change_line_count") or 0,
                                                      s.get("githubPrUrl") or ""))
        picked.extend(bucket[:per_lang])
    return picked


def finding_to_comment(f: dict) -> dict:
    """ReviewMind finding → AACR 生成评论格式（side 固定 RIGHT=新代码侧）。"""
    line = f.get("anchored_line") or f.get("line")
    note = str(f.get("description") or "")
    if f.get("suggestion"):
        note += "\nSuggestion: " + str(f["suggestion"])
    # side 与 GT 同用小写（judge 阶段 2 大小写敏感）
    return {"note": note, "path": str(f.get("file") or ""), "side": "right",
            "from_line": int(line) if line else None, "to_line": int(line) if line else None}


async def run_reviewmind(pr_url: str, tag: str) -> dict:
    store = ReviewJobStore()
    client = GitHubClient(token=settings.github_token or os.environ.get("GITHUB_TOKEN"))
    job = ReviewJob(job_id=f"aacr_{tag}_{int(time.time()*1000)}", pr_url=pr_url)
    await store.create(job)
    result = await run_engine(store, client, job)
    meta = result.get("engine_meta", {})
    return {"findings": meta.get("findings", []),
            "tokens_used": meta.get("tokens_used", 0)}


async def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pos-per-lang", type=int, default=4)
    ap.add_argument("--neg-per-lang", type=int, default=3)
    ap.add_argument("--out", default="dataset/aacr_results.json")
    args = ap.parse_args()

    if settings.github_token:
        os.environ["GITHUB_TOKEN"] = settings.github_token
    # 基准适配（报告披露）：GT 为英文，agent 输出切英文以适配其确定性语义匹配
    settings.review_output_language = "en"

    positives = stratified(json.loads((AACR_ROOT / "dataset/positive_samples.json")
                                      .read_text(encoding="utf-8")), args.pos_per_lang)
    negatives = stratified(json.loads((AACR_ROOT / "dataset/negative_samples.json")
                                      .read_text(encoding="utf-8")), args.neg_per_lang)
    print(f"AACR 分层子集：正 {len(positives)} PR / 负 {len(negatives)} PR", file=sys.stderr)

    # ---- 正样本：跑 agent → 官方 judge 判卷 ----
    pos_out: list[dict] = []
    tot = {"expected": 0, "generated": 0, "line_match": 0, "semantic_match": 0,
           "tokens": 0, "elapsed": 0}
    for i, inst in enumerate(positives):
        url = inst["githubPrUrl"]
        t0 = time.time()
        try:
            run = await run_reviewmind(url, f"p{i}")
            comments = [finding_to_comment(f) for f in run["findings"]]
            refs = copy.deepcopy(inst.get("comments", []))  # 不改动原 GT
            await evaluate_comments(refs, comments, k=1)
            stats = compute_cr_statistics(refs, len(comments))
        except Exception as exc:
            print(f"[POS] {url} FAILED: {exc}", file=sys.stderr)
            stats = {"expected_notes": 0, "generated_notes": 0, "line_match_count": 0,
                     "semantic_match_count": 0}
            run = {"findings": [], "tokens_used": 0}
        elapsed = round(time.time() - t0, 1)
        tot["expected"] += stats["expected_notes"]; tot["generated"] += stats["generated_notes"]
        tot["line_match"] += stats["line_match_count"]; tot["semantic_match"] += stats["semantic_match_count"]
        tot["tokens"] += run.get("tokens_used", 0); tot["elapsed"] += elapsed
        pos_out.append({"url": url, "lang": inst.get("project_main_language"),
                        "stats": stats, "elapsed_s": elapsed,
                        "findings": run["findings"]})
        print(f"[POS] {i+1}/{len(positives)} {inst.get('project_main_language')} "
              f"gen={stats['generated_notes']} sem={stats['semantic_match_count']} "
              f"line={stats['line_match_count']} {elapsed}s", file=sys.stderr)
        await asyncio.sleep(3)

    g, e = tot["generated"], tot["expected"]
    p = tot["semantic_match"] / g if g else 0.0
    r = tot["semantic_match"] / e if e else 0.0
    pos_summary = {
        "prs": len(positives), "expected_comments": e, "generated_comments": g,
        "precision": round(p, 4), "recall": round(r, 4),
        "f1": round(2*p*r/(p+r), 4) if (p+r) else 0.0,
        "noise_rate": round(1-p, 4),
        "line_precision": round(tot["line_match"]/g, 4) if g else 0.0,
        "line_recall": round(tot["line_match"]/e, 4) if e else 0.0,
        "tokens_per_pr": round(tot["tokens"]/max(len(positives),1), 0),
        "seconds_per_pr": round(tot["elapsed"]/max(len(positives),1), 1),
        "judge": "AACR judge.py, semantic=mock(deterministic), k=1",
    }

    # ---- 负样本：无 GT，所有产出按误报计（强行找问题观察） ----
    neg_out: list[dict] = []
    neg_findings = neg_tokens = neg_elapsed = neg_with_findings = 0
    for i, inst in enumerate(negatives):
        url = inst["githubPrUrl"]
        t0 = time.time()
        try:
            run = await run_reviewmind(url, f"n{i}")
        except Exception as exc:
            print(f"[NEG] {url} FAILED: {exc}", file=sys.stderr)
            run = {"findings": [], "tokens_used": 0}
        elapsed = round(time.time() - t0, 1)
        n_f = len(run["findings"])
        neg_findings += n_f; neg_tokens += run.get("tokens_used", 0); neg_elapsed += elapsed
        neg_with_findings += 1 if n_f else 0
        neg_out.append({"url": url, "lang": inst.get("project_main_language"),
                        "findings": n_f, "elapsed_s": elapsed})
        print(f"[NEG] {i+1}/{len(negatives)} {inst.get('project_main_language')} "
              f"findings={n_f} {elapsed}s", file=sys.stderr)
        await asyncio.sleep(3)
    neg_summary = {
        "prs": len(negatives), "findings_per_pr": round(neg_findings/max(len(negatives),1), 2),
        "false_positive_findings": neg_findings,
        "prs_with_findings(强行找问题率)": round(neg_with_findings/max(len(negatives),1), 4),
        "tokens_per_pr": round(neg_tokens/max(len(negatives),1), 0),
        "seconds_per_pr": round(neg_elapsed/max(len(negatives),1), 1),
    }

    Path(args.out).write_text(json.dumps({
        "config": {"pos_per_lang": args.pos_per_lang, "neg_per_lang": args.neg_per_lang,
                   "model": settings.llm_model_review,
                   "review_min_confidence": settings.review_min_confidence,
                   "grouping": "auto", "filter": True,
                   "judge": "AACR official judge.py, semantic=mock, k=1",
                   "sampling": "per-language stratified, smallest change_line_count first"},
        "positive_summary": pos_summary, "negative_summary": neg_summary,
        "positive_results": pos_out, "negative_results": neg_out,
    }, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"positive": pos_summary, "negative": neg_summary},
                     ensure_ascii=False, indent=2))


if __name__ == "__main__":
    asyncio.run(main())
