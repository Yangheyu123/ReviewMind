"""四组对照评测（Phase 5，改进方案 §9）。

组定义：
  A 直读 baseline —— diff 塞单 prompt，无工具无检索，一次调用出 findings
  B 向量基线     —— A + 检索增强（本地 TF-IDF 于 base 快照，top-3 相关片段注入 prompt；
                     本地向量替代 API embedding——账户无 embedding 额度，结论谨慎外推）
  C 主链路       —— run_engine（快照+grep+AST 工具循环），filter 关闭
  D 完整架构     —— C + filter 反思

指标口径（§9.2）：
  hit_rate = 命中标签数 / 标签总数；命中 = finding.file == label.file 且
             (行号距离 ≤15 或 描述词重叠 ≥30%)；
  findings_per_pr、hit_findings_rate = 命中产出率（命中 findings / 全部 findings，
             Precision 的下界代理——标签之外且为真的发现被低估，口径已知）。

用法：
  GITHUB_TOKEN=... python scripts/run_eval.py --groups A,B,C,D --limit 10
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import re
import sys
import time
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.environ.setdefault("LLM_MOCK_MODE", "false")

from app.core.config import settings  # noqa: E402
from app.core.llm import LLMClient  # noqa: E402
from app.core.llm_usage import llm_context  # noqa: E402

REVIEW_DIRECTIVE = """你是资深代码审查员。审查下面的 diff，找出真实缺陷（安全/正确性/性能/测试）。
只报告有把握的问题；每条给出 file（新文件路径）、line（新文件行号，尽力）、level（CRITICAL/HIGH/MEDIUM/LOW/INFO）、
type、description（简体中文）、existing_code（从 diff 逐字摘出的代码段）。
严格只输出 JSON：{"findings": [...]}，没有问题输出 {"findings": []}。"""

_HIT_WINDOW = 15
_STOPWORDS = set("the a an to of in for is are be with on at this that it and or if else "
                 "not you we i def class return function var let const import from as "
                 "的 了 在 是 和 与 有 对 不".split())


# ---------------------------------------------------------------------------
# 公共：拉取并缓存 PR 材料
# ---------------------------------------------------------------------------

async def load_pr_material(item: dict, client) -> dict:
    """拉取 PR 信息 + 变更文件（磁盘缓存，评测可重跑）。"""
    cache_dir = Path("dataset/cache")
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache = cache_dir / f"{item['repo'].replace('/', '_')}#{item['pr_number']}.json"
    if cache.exists():
        return json.loads(cache.read_text(encoding="utf-8"))

    from app.schemas.github import GitHubPullRequestRef
    from app.services.github_url_parser import parse_github_pr_url
    pr_ref = parse_github_pr_url(item["pr_url"])
    info = await client.fetch_pull_request(pr_ref)
    files = await client.fetch_pull_request_files(pr_ref)
    material = {
        "pr_ref": {"owner": pr_ref.owner, "repo": pr_ref.repo, "pull_number": pr_ref.pull_number,
                   "html_url": str(pr_ref.html_url)},
        "head_sha": info.head.sha, "base_sha": info.base.sha,
        "title": info.title,
        "files": [f.model_dump(mode="json") for f in files],
    }
    cache.write_text(json.dumps(material, ensure_ascii=False), encoding="utf-8")
    return material


def build_diff_text(material: dict, max_chars: int = 24000) -> str:
    parts = []
    total = 0
    for f in material["files"]:
        patch = f.get("patch") or ""
        if len(patch) > 6000:
            patch = patch[:6000] + "\n... (truncated)"
        seg = f"## {f['filename']}\n```diff\n{patch}\n```"
        total += len(seg)
        if total > max_chars:
            parts.append("... (diff 已截断)")
            break
        parts.append(seg)
    return "\n".join(parts)


# ---------------------------------------------------------------------------
# 组 A：直读 baseline
# ---------------------------------------------------------------------------

async def run_group_a(item: dict, material: dict, llm: LLMClient) -> dict:
    prompt = f"PR: {material['title']}\n\n{build_diff_text(material)}"
    with llm_context("eval", group="A", phase="direct") as ctx:
        try:
            content = await llm.chat(
                [{"role": "system", "content": REVIEW_DIRECTIVE},
                 {"role": "user", "content": prompt}],
            )
            findings = _parse_json_findings(content)
        except Exception as exc:
            findings = []
            print(f"[A] LLM failed: {exc}", file=sys.stderr)
    return {"findings": findings, "llm_records": ctx["records"]}


# ---------------------------------------------------------------------------
# 组 B：向量（TF-IDF 本地替身）检索增强
# ---------------------------------------------------------------------------

def _tokenize(text: str) -> list[str]:
    return [t for t in re.findall(r"[a-zA-Z_]\w{1,}", text.lower()) if t not in _STOPWORDS]


def tfidf_top_related(material: dict, snapshot_root, top_k: int = 3) -> list[dict]:
    """用 diff 新增文本作 query，在快照文件上做 TF-IDF 余弦检索。"""
    from app.services.source_snapshot import iter_snapshot_files, read_snapshot_file

    query_tokens = _tokenize("\n".join(
        line[1:] for f in material["files"] for line in (f.get("patch") or "").splitlines()
        if line.startswith("+")))
    if not query_tokens or snapshot_root is None:
        return []
    q_tf: dict[str, int] = {}
    for t in query_tokens:
        q_tf[t] = q_tf.get(t, 0) + 1

    scored: list[tuple[float, str, str]] = []
    files = [p for p in iter_snapshot_files(snapshot_root) if p.endswith((".py", ".js", ".ts", ".java"))][:400]
    doc_stats: list[dict] = []
    df: dict[str, int] = {}
    for path in files:
        src = read_snapshot_file(snapshot_root, path) or ""
        tf = {}
        for t in _tokenize(src):
            tf[t] = tf.get(t, 0) + 1
        doc_stats.append({"path": path, "tf": tf, "src": src})
        for t in tf:
            df[t] = df.get(t, 0) + 1
    n_docs = max(len(doc_stats), 1)
    import math
    for d in doc_stats:
        score = 0.0
        for t, q in q_tf.items():
            if t in d["tf"]:
                idf = math.log(1 + n_docs / df[t])
                score += q * (1 + math.log(d["tf"][t])) * idf
        if score > 0:
            lines = d["src"].splitlines()
            best_line = 0
            best_hits = -1
            for i, ln in enumerate(lines[: len(lines)]):
                hits = sum(1 for t in q_tf if t in ln.lower())
                if hits > best_hits:
                    best_hits, best_line = hits, i
            snippet = "\n".join(lines[max(0, best_line - 8): best_line + 12])[:1500]
            scored.append((score, d["path"], snippet))
    scored.sort(key=lambda x: -x[0])
    return [{"file": p, "snippet": s} for _, p, s in scored[:top_k]]


async def run_group_b(item: dict, material: dict, llm: LLMClient, snapshot_root) -> dict:
    related = tfidf_top_related(material, snapshot_root)
    ctx_parts = ["以下是与本 diff 最相关的仓库既有代码（TF-IDF 检索），供参考："]
    for r in related:
        ctx_parts.append(f"### {r['file']}\n```\n{r['snippet']}\n```")
    related_text = "\n".join(ctx_parts) if related else "（无检索结果）"
    prompt = f"PR: {material['title']}\n\n{build_diff_text(material)}\n\n{related_text}"
    with llm_context("eval", group="B", phase="direct+retrieval") as ctx:
        try:
            content = await llm.chat(
                [{"role": "system", "content": REVIEW_DIRECTIVE},
                 {"role": "user", "content": prompt}],
            )
            findings = _parse_json_findings(content)
        except Exception as exc:
            findings = []
            print(f"[B] LLM failed: {exc}", file=sys.stderr)
    return {"findings": findings, "llm_records": ctx["records"]}


# ---------------------------------------------------------------------------
# 组 C/D：run_engine（filter 开关区分）
# ---------------------------------------------------------------------------

async def run_engine_group(item: dict, material: dict, tag: str, enable_filter: bool) -> dict:
    from app.models.review_job import ReviewJob
    from app.services.review_job_store import ReviewJobStore
    from app.services.github_client import GitHubClient
    from app.agent_loop.engine import run_engine

    settings.review_enable_filter = enable_filter
    store = ReviewJobStore()
    client = GitHubClient(token=os.environ.get("GITHUB_TOKEN") or None)
    job = ReviewJob(job_id=f"eval_{tag}_{item['repo'].replace('/','_')}_{item['pr_number']}_{int(time.time())}",
                    pr_url=item["pr_url"])
    await store.create(job)
    result = await run_engine(store, client, job)
    findings = result.get("engine_meta", {}).get("findings", [])
    return {"findings": findings, "job_id": job.job_id}


# ---------------------------------------------------------------------------
# 指标
# ---------------------------------------------------------------------------

def _tokens(text: str) -> set[str]:
    return set(_tokenize(text)) - _STOPWORDS


def label_hit(finding: dict, label: dict) -> bool:
    f_file = str(finding.get("file") or "").strip()
    l_file = str(label.get("file") or "").strip()
    if not f_file or not l_file or (f_file not in l_file and l_file not in f_file):
        return False
    try:
        dist = abs(int(finding.get("line") or 0) - int(label.get("line") or 0))
    except (TypeError, ValueError):
        dist = 999
    if dist <= _HIT_WINDOW:
        return True
    overlap = _tokens(str(finding.get("description") or "")) & _tokens(str(label.get("body") or ""))
    denom = min(len(_tokens(str(label.get("body") or ""))) or 1, 12)
    return len(overlap) / denom >= 0.5


def score_group(results: list[dict], labels_items: list[dict]) -> dict:
    """按 repo+PR 号连接标签与结果——禁止按位置配对（数据集评测后被增删会静默错位）。"""
    by_key = {f"{r['repo']}#{r['pr_number']}": r for r in results}
    total_labels = sum(len(i["labels"]) for i in labels_items)
    hit_labels = 0
    hit_findings = 0
    total_findings = 0
    matched = 0
    for item in labels_items:
        res = by_key.get(f"{item['repo']}#{item['pr_number']}")
        if res is None:
            continue  # 数据集快照与结果失配：按 key 丢弃（口径可审计），不猜位置
        matched += 1
        findings = res["findings"]
        total_findings += len(findings)
        for label in item["labels"]:
            if any(label_hit(f, label) for f in findings):
                hit_labels += 1
        for f in findings:
            if any(label_hit(f, label) for label in item["labels"]):
                hit_findings += 1
    n = matched or 1
    return {
        "prs": n,
        "matched_prs": matched,
        "dataset_prs": len(labels_items),
        "total_labels": total_labels,
        "hit_labels": hit_labels,
        "hit_rate": round(hit_labels / total_labels, 4) if total_labels else None,
        "total_findings": total_findings,
        "findings_per_pr": round(total_findings / n, 2),
        "hit_findings_rate": round(hit_findings / total_findings, 4) if total_findings else None,
    }


# ---------------------------------------------------------------------------

def _parse_json_findings(content: str) -> list[dict]:
    try:
        m = re.search(r"\{.*\}", content, re.S)
        data = json.loads(m.group(0) if m else content)
        return data.get("findings", []) if isinstance(data, dict) else []
    except (json.JSONDecodeError, AttributeError):
        return []


async def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="dataset/labels.json")
    ap.add_argument("--groups", default="A,B,C,D")
    ap.add_argument("--limit", type=int, default=10)
    ap.add_argument("--out", default="dataset/eval_results.json")
    args = ap.parse_args()

    data = json.loads(Path(args.dataset).read_text(encoding="utf-8"))
    items = data["items"][: args.limit]
    # 标签净化（弱标签噪声控制，口径记录于 evaluation.md）：
    # 剔除问句（含疑问句式开头）、引用块、suggestion 代码块、changelog 类文件
    import re as _re
    _question_start = _re.compile(r"^(is|are|can|does|do|should|would|could|will|has|have|was|were)\b", _re.I)
    _changelog = _re.compile(r"(versionhistory|changelog|changes\.rst|news|history)", _re.I)

    def _keep_label(label: dict) -> bool:
        body = (label.get("body") or "").strip()
        if not body:
            return False
        if body.endswith("?") or "?" in body[:60]:
            return False
        if body.startswith(">") or body.startswith("```suggestion"):
            return False
        if _question_start.search(body):
            return False
        if _changelog.search(label.get("file") or ""):
            return False
        return True
    for it in items:
        it["labels"] = [l for l in it["labels"] if _keep_label(l)]
    # 只评有标签的 PR（无标签 PR 不消耗 LLM 额度）
    items = [it for it in items if it["labels"]]
    print(f"purified: {len(items)} PRs / {sum(len(i['labels']) for i in items)} labels", file=sys.stderr)
    print(f"dataset: {len(items)} PRs / {sum(len(i['labels']) for i in items)} labels", file=sys.stderr)

    from app.services.github_client import GitHubClient
    # token 以 .env（settings）为准——引擎侧 client 读 os.environ，这里统一注入，
    # 否则 shell 未导出 GITHUB_TOKEN 时引擎会以未认证身份跑（60 次/小时配额必炸）
    if settings.github_token:
        os.environ["GITHUB_TOKEN"] = settings.github_token
    client = GitHubClient(token=settings.github_token or os.environ.get("GITHUB_TOKEN"))
    llm = LLMClient()
    materials = {}
    for item in items:
        materials[item["pr_url"]] = await load_pr_material(item, client)

    all_results: dict[str, list[dict]] = {}
    for group in args.groups.split(","):
        group = group.strip().upper()
        results: list[dict] = []
        for i, item in enumerate(items):
            t0 = time.time()
            material = materials[item["pr_url"]]
            try:
                if group == "A":
                    res = await run_group_a(item, material, llm)
                elif group == "B":
                    from app.services.source_snapshot import ensure_snapshot
                    from app.schemas.github import GitHubPullRequestRef
                    pr = material["pr_ref"]
                    root = await ensure_snapshot(client, GitHubPullRequestRef(**pr), material["base_sha"])
                    res = await run_group_b(item, material, llm, root)
                elif group == "C":
                    res = await run_engine_group(item, material, tag="C", enable_filter=False)
                elif group == "D":
                    res = await run_engine_group(item, material, tag="D", enable_filter=True)
                else:
                    continue
            except Exception as exc:
                print(f"[{group}] {item['repo']}#{item['pr_number']} FAILED: {exc}", file=sys.stderr)
                res = {"findings": [], "error": str(exc)}
            res["repo"] = item["repo"]
            res["pr_number"] = item["pr_number"]
            results.append(res)
            print(f"[{group}] {i+1}/{len(items)} {item['repo']}#{item['pr_number']} "
                  f"findings={len(res['findings'])} {time.time()-t0:.0f}s", file=sys.stderr)
            await asyncio.sleep(3)  # PR 间隔：降低突发密度，规避 GitHub 二级限流
        all_results[group] = results

    summary = {g: score_group(all_results[g], items) for g in all_results}
    import hashlib
    dataset_hash = hashlib.sha256(Path(args.dataset).read_bytes()).hexdigest()[:12]
    mismatch = [f"{it['repo']}#{it['pr_number']}" for it in items
                if not any(f"{r['repo']}#{r['pr_number']}" == f"{it['repo']}#{it['pr_number']}"
                           for r in all_results.get("C", []))]
    if mismatch:
        print(f"WARNING: {len(mismatch)} 个标签 PR 在结果中缺失（按 key 丢弃）: {mismatch[:5]}", file=sys.stderr)
    Path(args.out).write_text(json.dumps({
        "config": {"limit": args.limit, "model": settings.llm_model_review,
                   "hit_window": _HIT_WINDOW, "dataset_hash": dataset_hash,
                   "dataset": args.dataset},
        "summary": summary,
        "results": all_results,
    }, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    asyncio.run(main())
