"""评测集构造脚本（Phase 5，改进方案 §9.1）。

机械规则（先定死再抓，不按结果挑）：
- 仓库与时间窗：REPOS 列表 × 最近 MERGE_WINDOW_DAYS 天的已合并 PR；
- 标签 = 双证据弱标签：
  证据 1：PR review comment 命中"缺陷类意见"关键词（排除提问/纯风格）；
  证据 2：该 comment 之后 PR 有新 push（作者用代码变更回应了意见）；
- 每条标签存 PR 链接 + 评论链接 + 文件/行号，全程可溯源。

输出：dataset/labels.json
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
import urllib.request
from datetime import datetime, timedelta, timezone

TOKEN = os.environ.get("GITHUB_TOKEN", "")
API = "https://api.github.com"

REPOS = ["sqlalchemy/sqlalchemy", "pydantic/pydantic", "agronholm/anyio", "langchain-ai/langchain", "encode/starlette", "encode/httpx", "pallets/click", "pallets/flask", "tiangolo/fastapi"]
MERGE_WINDOW_DAYS = 400          # 机械规则：近 400 天内合并的 PR
MAX_PRS_PER_REPO = 150           # 每仓库最多扫描的 merged PR 数
TARGET_LABELS = 40               # 收满即停（30+ 目标，留余量）

# 缺陷类意见信号（英文为主，含少量中文）；排除纯风格/提问
DEFECT_SIGNALS = re.compile(
    r"\b(bug|error|wrong|incorrect|broken|break(s|ing)?|missing|forgot|"
    r"n\+1|inject(ion)?|null|none check|leak|race|deadlock|overflow|"
    r"vulnerab|secur(e|ity)|fail(s|ing|ed)? (to|when)|unhandled|edge case|"
    r"off[- ]by[- ]one|regress)\b", re.IGNORECASE)
EXCLUDE_SIGNALS = re.compile(
    r"^(nic|cool|thanks|thank you|lgtm|good|awesome|nice)|\b(typo|naming|style|format|docstring)\b",
    re.IGNORECASE)


def api_get(path: str, params: str = "") -> dict | list:
    url = f"{API}{path}{'?' + params if params else ''}"
    req = urllib.request.Request(url, headers={
        "Authorization": f"Bearer {TOKEN}",
        "Accept": "application/vnd.github+json",
        "User-Agent": "reviewmind-eval",
    })
    for attempt in range(5):
        try:
            with urllib.request.urlopen(req, timeout=30) as resp:
                return json.load(resp)
        except urllib.error.HTTPError as e:
            if e.code in (403, 429):
                time.sleep(20)
                continue
            raise
        except (urllib.error.URLError, OSError, TimeoutError) as e:
            # 网络抖动（SSL EOF 等）：退避重试
            print(f"[retry {attempt+1}] {type(e).__name__}: {e}", file=sys.stderr)
            time.sleep(5 + attempt * 5)
    return {}


def parse_dt(s: str) -> datetime:
    return datetime.fromisoformat(s.replace("Z", "+00:00"))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="dataset/labels.json")
    ap.add_argument("--max-labels", type=int, default=TARGET_LABELS)
    args = ap.parse_args()

    cutoff = datetime.now(timezone.utc) - timedelta(days=MERGE_WINDOW_DAYS)
    labels: list[dict] = []
    prs_scanned = 0

    for repo in REPOS:
        if len(labels) >= args.max_labels:
            break
        page = 1
        repo_prs = 0
        while repo_prs < MAX_PRS_PER_REPO and len(labels) < args.max_labels:
            prs = api_get(f"/repos/{repo}/pulls",
                          f"state=closed&sort=updated&direction=desc&per_page=50&page={page}")
            if not prs:
                break
            page += 1
            for pr in prs:
                if repo_prs >= MAX_PRS_PER_REPO or len(labels) >= args.max_labels:
                    break
                if not pr.get("merged_at"):
                    continue
                merged_at = parse_dt(pr["merged_at"])
                if merged_at < cutoff:
                    continue
                repo_prs += 1
                prs_scanned += 1
                number = pr["number"]
                # 机械规则：拿全部 review comments（行级评论质量最高）
                comments = api_get(f"/repos/{repo}/pulls/{number}/comments", "per_page=100")
                if not comments:
                    continue
                # 证据 2 的素材：commits 时间线 + 作者修复确认（issue comments）
                commits = api_get(f"/repos/{repo}/pulls/{number}/commits", "per_page=100")
                commit_times = [parse_dt(c["commit"]["committer"]["date"]) for c in commits
                                if c.get("commit", {}).get("committer", {}).get("date")]
                author_login = (pr.get("user") or {}).get("login", "")
                issue_comments = api_get(f"/repos/{repo}/issues/{number}/comments", "per_page=100")
                fix_ack = any(
                    (c.get("user") or {}).get("login") == author_login
                    and re.search(r"(fixed|done|updated|addressed|applied|corrected|good catch)",
                                  (c.get("body") or ""), re.IGNORECASE)
                    for c in issue_comments
                )
                # 证据 2c（更弱兜底）：缺陷意见来自他人（非作者自评）且作者有任意回复
                author_replied = any(
                    (c.get("user") or {}).get("login") == author_login for c in issue_comments
                )

                pr_labels: list[dict] = []
                for c in comments:
                    body = c.get("body") or ""
                    if len(body) < 15:
                        continue
                    # 排除 bot 评论（AI reviewer 产物不能当 ground truth）
                    comment_login = (c.get("user") or {}).get("login", "")
                    if "[bot]" in comment_login or "bot" in comment_login.lower() or body.lstrip().startswith("<!--"):
                        continue
                    if not DEFECT_SIGNALS.search(body) or EXCLUDE_SIGNALS.search(body):
                        continue
                    created = parse_dt(c["created_at"])
                    # 双证据（任一）：意见后新 push（代码回应）或作者修复确认（语言回应）
                    pushed_after = any(t > created for t in commit_times)
                    comment_author = (c.get("user") or {}).get("login", "")
                    by_reviewer = comment_author and comment_author != author_login
                    if not (pushed_after or fix_ack or (by_reviewer and author_replied)):
                        continue
                    pr_labels.append({
                        "file": c.get("path"),
                        "line": c.get("line") or c.get("original_line") or 0,
                        "body": body[:400],
                        "comment_url": c.get("html_url"),
                        "commented_at": c["created_at"],
                    })
                if pr_labels:
                    labels.append({
                        "repo": repo,
                        "pr_number": number,
                        "pr_url": pr["html_url"],
                        "title": pr["title"][:120],
                        "changed_files": pr.get("changed_files"),
                        "merged_at": pr["merged_at"],
                        "labels": pr_labels[:3],  # 每 PR 最多 3 条，避免单 PR 主导
                    })
                    print(f"[+] {repo}#{number}: {len(pr_labels)} labels", file=sys.stderr)
                time.sleep(0.2)

    out = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "rules": {
            "repos": REPOS, "merge_window_days": MERGE_WINDOW_DAYS,
            "max_prs_per_repo": MAX_PRS_PER_REPO,
            "evidence": "review comment (defect-signal regex, non-style) + push-after-comment",
        },
        "stats": {
            "prs_scanned": prs_scanned,
            "prs_with_labels": len(labels),
            "total_labels": sum(len(p["labels"]) for p in labels),
        },
        "items": labels,
    }
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)
    print(json.dumps(out["stats"], ensure_ascii=False))


if __name__ == "__main__":
    main()
