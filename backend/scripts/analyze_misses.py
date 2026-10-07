"""召回短板归因诊断（只读分析，不改系统）。

对 eval_full.json 里 D 组漏掉的每条标签逐条归因：
  1. filter_deleted  —— C 组命中而 D 组未命中（反思环节删掉的）
  2. no_finding_file —— D 组在该文件零产出（生产侧缺口）
       2a. lang_gap  —— 文件扩展名不在 工具索引/分组解析 覆盖内（.rs/.rst/.md/go...）
       2b. discipline—— 核心语言文件仍零产出（agent 纪律 / 上下文 / 分组视野）
  3. miss_criteria   —— D 组同文件有产出但未达命中线（行距>15 且 词重叠<0.5）
同时统计"A 命中而 D 漏掉"的回归集（评测报告的主短板）。

按 repo+PR 号连接 labels 与 eval 结果（labels.json 在评测后有增删，禁止按位置配对）。

用法：py -3 scripts/analyze_misses.py
"""

from __future__ import annotations

import json
import os
import re
import sys
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.environ.setdefault("LLM_MOCK_MODE", "true")

from run_eval import label_hit, _tokens  # noqa: E402  口径与评测完全一致

TOOL_LANG_EXTS = (".py", ".js", ".jsx", ".ts", ".tsx", ".mjs", ".cjs", ".java")


def purify(items: list[dict]) -> list[dict]:
    """与 run_eval.main 完全一致的标签净化。"""
    question_start = re.compile(r"^(is|are|can|does|do|should|would|could|will|has|have|was|were)", re.I)
    changelog = re.compile(r"(versionhistory|changelog|changes\.rst|news|history)", re.I)

    def keep(label: dict) -> bool:
        body = (label.get("body") or "").strip()
        if not body:
            return False
        if body.endswith("?") or "?" in body[:60]:
            return False
        if body.startswith(">") or body.startswith("```suggestion"):
            return False
        if question_start.search(body):
            return False
        if changelog.search(label.get("file") or ""):
            return False
        return True

    for it in items:
        it["labels"] = [l for l in it["labels"] if keep(l)]
    return [it for it in items if it["labels"]]


def main() -> None:
    root = Path(__file__).resolve().parent.parent
    data = json.loads((root / "dataset/labels.json").read_text(encoding="utf-8"))
    items = purify(data["items"][:45])
    eval_full = json.loads((root / "dataset/eval_full.json").read_text(encoding="utf-8"))
    res = eval_full["results"]

    # 按 repo+PR 号建索引（评测跑的是当时的 34 PR；labels.json 现存 33）
    def key(r: dict) -> str:
        return f"{r['repo']}#{r['pr_number']}"

    idx = {g: {key(r): r for r in res[g]} for g in ("A", "C", "D")}
    dropped_prs = [key(it) for it in items if key(it) not in idx["A"]]
    if dropped_prs:
        print(f"注意：labels.json 中 {len(dropped_prs)} 个 PR 不在评测结果里（评测后数据集有增删）: {dropped_prs}\n")

    buckets: dict[str, list[dict]] = {"filter_deleted": [], "lang_gap": [], "discipline": [], "miss_criteria": []}
    near_miss: list[tuple[int, float]] = []

    for item in items:
        k = key(item)
        if k not in idx["D"]:
            continue
        ra, rc, rd = idx["A"][k], idx["C"][k], idx["D"][k]
        for label in item["labels"]:
            if any(label_hit(f, label) for f in rd["findings"]):
                continue
            hit_a = any(label_hit(f, label) for f in ra["findings"])
            hit_c = any(label_hit(f, label) for f in rc["findings"])
            l_file = str(label.get("file") or "")
            same_file = [f for f in rd["findings"]
                         if str(f.get("file") or "") and (str(f["file"]) in l_file or l_file in str(f["file"]))]
            info = {"pr": k, "file": l_file, "body": (label.get("body") or "").replace("\n", " ")[:70],
                    "hit_a": hit_a}
            if hit_c:
                buckets["filter_deleted"].append(info)
            elif not same_file:
                ext = Path(l_file).suffix.lower()
                buckets["lang_gap" if ext not in TOOL_LANG_EXTS else "discipline"].append(info)
            else:
                buckets["miss_criteria"].append(info)
                best = None
                for f in same_file:
                    try:
                        dist = abs(int(f.get("line") or 0) - int(label.get("line") or 0))
                    except (TypeError, ValueError):
                        dist = 999
                    lt = _tokens(str(label.get("body") or ""))
                    ft = _tokens(str(f.get("description") or ""))
                    ov = len(lt & ft) / max(len(lt) or 1, 1)
                    if best is None or (dist, -ov) < (best[0], -best[1]):
                        best = (dist, round(ov, 2))
                near_miss.append(best)

    total = sum(len(i["labels"]) for i in items if key(i) in idx["D"])
    print(f"现存可对照 {sum(1 for i in items if key(i) in idx['D'])} PR / {total} 标签；"
          f"D 组漏掉 {sum(len(v) for v in buckets.values())} 条\n")
    for k, v in buckets.items():
        a_n = sum(1 for x in v if x["hit_a"])
        print(f"[{k}] {len(v)} 条（A 命中过的回归损失 {a_n} 条）")
        for x in v[:8]:
            print(f"    {'A✓' if x['hit_a'] else 'A✗'} {x['pr']}  {x['file']}")
            print(f"       {x['body']}")
    if near_miss:
        print(f"\n近失样本（同文件最近 finding 的行距/重叠度）: {near_miss}")


if __name__ == "__main__":
    main()
