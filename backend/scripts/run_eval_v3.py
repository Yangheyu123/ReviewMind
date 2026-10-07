"""评测体系 v3：自建回归集 + 四级消融（评测报告 §消融）。

组定义（消融阶梯，同一模型/数据集/配置锁定下）：
  A  Baseline      —— LLM 直读 diff（单次调用）
  RC +RepoContext  —— 单 reviewer + 仓内检索工具（快照/code_search/find_symbol），
                      无分组路由（review_grouping_mode=single）
  C  +Routing      —— §16.1 分层分组 Send() fan-out 多 reviewer（filter 关）
  D  +Arbitration  —— C + filter 反思（完整 ReviewMind，含置信度地板）

指标（命中判定复用 run_eval.label_hit，纯规则）：
  Precision(下界) = 命中 findings / 总 findings；Recall = 命中标签 / 总标签；
  F1；NoiseRate = 1 - Precision；findings/PR；tokens/PR；秒/PR。

可复现性：输出 config 锁定 模型/prompt 指纹/温度/数据集指纹/各引擎开关/rg 可用性。

用法：py -3 scripts/run_eval_v3.py --groups A,RC,C,D --out dataset/eval_v3.json
"""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import os
import shutil
import sys
import time
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.environ.setdefault("LLM_MOCK_MODE", "false")

from app.core.config import settings  # noqa: E402
from app.core.llm import LLMClient  # noqa: E402
from run_eval import (  # noqa: E402
    REVIEW_DIRECTIVE, label_hit, load_pr_material, run_group_a,
)
from analyze_misses import purify  # noqa: E402


async def run_engine_group_v3(item: dict, material: dict, tag: str,
                              enable_filter: bool, single_group: bool) -> dict:
    """引擎组（RC/C/D 共用）：single_group=True 为消融 RC 档。"""
    from app.models.review_job import ReviewJob
    from app.services.review_job_store import ReviewJobStore
    from app.services.github_client import GitHubClient
    from app.agent_loop.engine import run_engine

    settings.review_enable_filter = enable_filter
    settings.review_grouping_mode = "single" if single_group else "auto"
    store = ReviewJobStore()
    client = GitHubClient(token=settings.github_token or os.environ.get("GITHUB_TOKEN"))
    job = ReviewJob(job_id=f"ev3_{tag}_{item['repo'].replace('/','_')}_{item['pr_number']}_{int(time.time())}",
                    pr_url=item["pr_url"])
    await store.create(job)
    result = await run_engine(store, client, job)
    meta = result.get("engine_meta", {})
    return {
        "findings": meta.get("findings", []),
        "tokens_used": meta.get("tokens_used", 0),
        "llm_request_count": meta.get("llm_request_count", 0),
    }


def score(results: list[dict], items: list[dict]) -> dict:
    by_key = {f"{r['repo']}#{r['pr_number']}": r for r in results}
    total_labels = sum(len(i["labels"]) for i in items)
    hit_labels = hit_findings = total_findings = matched = 0
    tokens = elapsed = 0
    for it in items:
        r = by_key.get(f"{it['repo']}#{it['pr_number']}")
        if r is None:
            continue
        matched += 1
        fs = r["findings"]
        total_findings += len(fs)
        tokens += r.get("tokens_used", 0) or 0
        elapsed += r.get("elapsed_s", 0) or 0
        hit_labels += sum(1 for lb in it["labels"] if any(label_hit(f, lb) for f in fs))
        hit_findings += sum(1 for f in fs if any(label_hit(f, lb) for lb in it["labels"]))
    p = hit_findings / total_findings if total_findings else 0.0
    r_ = hit_labels / total_labels if total_labels else 0.0
    f1 = 2 * p * r_ / (p + r_) if (p + r_) else 0.0
    n = matched or 1
    return {
        "matched_prs": matched, "labels": total_labels,
        "recall": round(r_, 4), "precision_lb": round(p, 4), "f1": round(f1, 4),
        "noise_rate": round(1 - p, 4),
        "findings_per_pr": round(total_findings / n, 2),
        "tokens_per_pr": round(tokens / n, 0), "seconds_per_pr": round(elapsed / n, 1),
    }


async def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="dataset/labels.json")
    ap.add_argument("--groups", default="A,RC,C,D")
    ap.add_argument("--limit", type=int, default=45)
    ap.add_argument("--out", default="dataset/eval_v3.json")
    args = ap.parse_args()

    data = json.loads(Path(args.dataset).read_text(encoding="utf-8"))
    items = purify(data["items"][: args.limit])
    print(f"dataset: {len(items)} PRs / {sum(len(i['labels']) for i in items)} labels", file=sys.stderr)

    from app.services.github_client import GitHubClient
    if settings.github_token:
        os.environ["GITHUB_TOKEN"] = settings.github_token
    client = GitHubClient(token=settings.github_token or os.environ.get("GITHUB_TOKEN"))
    llm = LLMClient()
    materials = {}
    for item in items:
        materials[item["pr_url"]] = await load_pr_material(item, client)

    from app.agent_loop.engine import GROUP_SYSTEM_PROMPT
    prompt_hash = hashlib.sha256(
        (GROUP_SYSTEM_PROMPT + REVIEW_DIRECTIVE).encode("utf-8")).hexdigest()[:12]
    dataset_hash = hashlib.sha256(Path(args.dataset).read_bytes()).hexdigest()[:12]

    group_specs = {
        "A": dict(kind="direct"),
        "RC": dict(kind="engine", enable_filter=False, single_group=True),
        "C": dict(kind="engine", enable_filter=False, single_group=False),
        "D": dict(kind="engine", enable_filter=True, single_group=False),
    }

    all_results: dict[str, list[dict]] = {}
    for group in [g.strip().upper() for g in args.groups.split(",")]:
        spec = group_specs.get(group)
        if spec is None:
            continue
        settings.review_grouping_mode = "auto"
        results: list[dict] = []
        for i, item in enumerate(items):
            t0 = time.time()
            material = materials[item["pr_url"]]
            try:
                if spec["kind"] == "direct":
                    res = await run_group_a(item, material, llm)
                    res["tokens_used"] = sum(r.get("total_tokens") or 0
                                             for r in res.get("llm_records", []))
                else:
                    res = await run_engine_group_v3(item, material, group,
                                                    spec["enable_filter"], spec["single_group"])
            except Exception as exc:
                print(f"[{group}] {item['repo']}#{item['pr_number']} FAILED: {exc}", file=sys.stderr)
                res = {"findings": [], "tokens_used": 0}
            res["elapsed_s"] = round(time.time() - t0, 1)
            res["repo"], res["pr_number"] = item["repo"], item["pr_number"]
            results.append(res)
            print(f"[{group}] {i+1}/{len(items)} {item['repo']}#{item['pr_number']} "
                  f"findings={len(res['findings'])} {res['elapsed_s']}s "
                  f"tokens={res.get('tokens_used', 0)}", file=sys.stderr)
            await asyncio.sleep(3)
        all_results[group] = results
        settings.review_grouping_mode = "auto"

    summary = {g: score(all_results[g], items) for g in all_results}
    config_lock = {
        "model": settings.llm_model_review, "prompt_hash": prompt_hash,
        "temperature_direct": 0.1, "temperature_tool_loop": 0.0,
        "dataset_hash": dataset_hash, "dataset": args.dataset,
        "hit_rule": "same file AND (line dist<=15 OR word overlap>=0.5)",
        "review_min_confidence": settings.review_min_confidence,
        "rg_available": bool(shutil.which("rg")),
        "groups": {g: group_specs[g] for g in all_results},
    }
    Path(args.out).write_text(json.dumps({
        "config": config_lock, "summary": summary, "results": all_results,
    }, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    asyncio.run(main())
