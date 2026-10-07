"""审查记忆（Phase 2.5）：跨任务的历史 findings 积累与预注入。

机制（对齐改进方案 §6 / Phase 2.5 定稿）：
- 写入：每次审查完成后，findings 连同锚点/分类存入 review_memory 表；
- 检索：新 PR 的 PRE 阶段，按「repo + 变更文件路径」精确匹配 + 符号词重叠
  排序，召回本仓库历史发现，作为"审查记忆"段注入组 prompt；
- 价值场景：捕获复制粘贴式回归（上月修过的 bug 被抄进新代码）与仓库特有
  反复出现的审查主题——静态语言清单（rules/）覆盖不了的仓库私有知识；
- 冷启动：空库零命中、单次 SQL 查询的零成本，不阻塞管线；
- 升级点：当前为关键词级检索（无 embedding 依赖）；账户具备 embedding 额度后，
  将 description/existing_code 向量化入 pgvector，检索升级为语义级——接口不变。
"""

from __future__ import annotations

import logging
import re
from typing import Any

logger = logging.getLogger(__name__)

_MAX_MEMORY_HINTS = 8          # 注入 prompt 的历史发现上限
_MIN_DESCRIPTION_CHARS = 10    # 过短的描述不入库（占位/空发现无记忆价值）
_MEMORY_WINDOW_DAYS = 180      # 只召回近 180 天（过旧的记忆可能已过时）


def _store():
    from app.services.review_job_store import review_job_store
    return review_job_store


async def save_findings_to_memory(repo: str, findings: list[dict[str, Any]]) -> int:
    """审查完成后写入记忆。失败降级为日志（不阻塞主流程）。"""
    if not repo or not findings:
        return 0
    try:
        return await _store().save_review_memory(repo, findings, _MIN_DESCRIPTION_CHARS)
    except Exception as exc:
        logger.warning("[MEMORY] save failed (repo=%s): %s", repo, exc)
        return 0


async def recall_memory(repo: str, changed_files: list[str]) -> list[dict[str, Any]]:
    """召回本仓库在相同/相近文件上的历史发现（关键词级：repo+路径匹配+词重叠排序）。"""
    if not repo or not changed_files:
        return []
    try:
        rows = await _store().recall_review_memory(repo, changed_files, _MEMORY_WINDOW_DAYS)
    except Exception as exc:
        logger.warning("[MEMORY] recall failed (repo=%s): %s", repo, exc)
        return []

    # 词重叠排序：历史发现与本次变更文件名/目录的词面相关度
    def score(row: dict[str, Any]) -> int:
        text = f"{row.get('file', '')} {row.get('description', '')}".lower()
        return sum(1 for t in _tokens(" ".join(changed_files)) if t in text)

    ranked = sorted(rows, key=score, reverse=True)[:_MAX_MEMORY_HINTS]
    return ranked


def render_memory_hints(memories: list[dict[str, Any]]) -> str:
    """渲染注入组 prompt 的"审查记忆"段。"""
    if not memories:
        return ""
    lines = [
        "## 本仓库历史审查记忆（过往审查在相近代码上的发现，重点核查是否复现）",
        "",
    ]
    for m in memories:
        when = str(m.get("created_at", ""))[:10]
        lines.append(
            f"- [{when}] `{m.get('file')}:{m.get('line')}` "
            f"[{str(m.get('category') or m.get('type') or 'other')}] "
            f"{str(m.get('description') or '')[:160]}"
        )
    lines.append("")
    lines.append(
        "注意：以上是历史发现而非本次结论——若本次 diff 未触及同类代码请忽略；"
        "若同类模式复现（如复制粘贴回归），按正常取证流程确认后提交。"
    )
    return "\n".join(lines)


def _tokens(text: str) -> set[str]:
    stop = set("the a an to of in for is are with on at py ts js java test tests src".split())
    return {t for t in re.findall(r"[a-z_]\w{2,}", text.lower()) if t not in stop}
