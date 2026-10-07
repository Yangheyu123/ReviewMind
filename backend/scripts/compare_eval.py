"""评测对比：v1（改造前 eval_full.json）vs v2（改造后 eval_cd_v2.json）。

A 组沿用 v1 结果作对照（同模型同端点）；C/D 用 v2。
输出四组口径一致的指标表 + 回填简历所需的结论数字。

用法：py -3 scripts/compare_eval.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from run_eval import label_hit  # noqa: E402
from analyze_misses import purify  # noqa: E402


def main() -> None:
    root = Path(__file__).resolve().parent.parent
    v1 = json.loads((root / "dataset/eval_full.json").read_text(encoding="utf-8"))
    v2_path = root / "dataset/eval_cd_v2.json"
    if not v2_path.exists():
        print("eval_cd_v2.json 尚未生成")
        return
    v2 = json.loads(v2_path.read_text(encoding="utf-8"))

    items = purify(json.loads((root / "dataset/labels.json").read_text(encoding="utf-8"))["items"][:45])
    by_key_v1 = {g: {f"{r['repo']}#{r['pr_number']}": r for r in v1["results"][g]} for g in ("A", "C", "D")}
    by_key_v2 = {g: {f"{r['repo']}#{r['pr_number']}": r for r in v2["results"][g]} for g in v2["results"]}

    total_labels = sum(len(i["labels"]) for i in items)

    def score(res_index, group):
        hit_labels = hit_findings = total_findings = matched = 0
        for it in items:
            res = res_index.get(group, {}).get(f"{it['repo']}#{it['pr_number']}")
            if res is None:
                continue
            matched += 1
            fs = res["findings"]
            total_findings += len(fs)
            hit_labels += sum(1 for lb in it["labels"] if any(label_hit(f, lb) for f in fs))
            hit_findings += sum(1 for f in fs if any(label_hit(f, lb) for lb in it["labels"]))
        return {
            "matched": matched,
            "hit_labels": hit_labels,
            "recall": round(hit_labels / total_labels, 4),
            "findings": total_findings,
            "per_pr": round(total_findings / max(matched, 1), 2),
            "efficiency": round(hit_findings / total_findings, 4) if total_findings else None,
        }

    rows = [
        ("A 直读(v1)", score(by_key_v1, "A")),
        ("C v1(改造前)", score(by_key_v1, "C")),
        ("C v2(改造后)", score(by_key_v2, "C") if "C" in by_key_v2 else None),
        ("D v1(改造前)", score(by_key_v1, "D")),
        ("D v2(改造后)", score(by_key_v2, "D") if "D" in by_key_v2 else None),
    ]
    print(f"标签总数 {total_labels}（净化口径 {len(items)} PR）\n")
    print(f"{'组':<14}{'命中':>5}{'召回':>9}{'findings':>10}{'条/PR':>8}{'单条效率':>9}{'matched':>9}")
    for name, s in rows:
        if s is None:
            print(f"{name:<14}  (无)")
            continue
        print(f"{name:<14}{s['hit_labels']:>5}{s['recall']:>9.1%}{s['findings']:>10}"
              f"{s['per_pr']:>8}{(s['efficiency'] or 0):>9.1%}{s['matched']:>9}")


if __name__ == "__main__":
    main()
