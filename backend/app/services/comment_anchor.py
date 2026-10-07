"""评论锚点定位三级流水（Phase 3）。

消费 submit_finding 收集的 existing_code（模型从 diff 逐字摘出的最小代码段），
产出精确行号——行号是匹配出来的，不是模型编的。

三级流水（对齐改进方案 §7.1 / 阿里 OCR diff/resolver 范式）：
  L1 hunk 匹配：锚点在 diff 新增行（+ 行）中逐字出现 → 新文件行号，status=hunk
  L2 全文匹配：锚点在 head 版本文件全文唯一/首处出现 → status=fulltext
  L3 跨文件唯一迁移：锚点在快照其它文件全文唯一出现 → 迁移 file/line，status=relocated
  兜底：unanchored——保留模型给的行号但显式降权（confidence × 0.5），不静默。
"""

from __future__ import annotations

import logging
from typing import Any

logger = logging.getLogger(__name__)


def _strip_line(text: str) -> str:
    """去掉 diff 行前缀（+/-/空格）后比较。"""
    return text[1:] if text[:1] in ("+", "-", " ") else text


def _match_in_patch(patch: str, anchor: str) -> tuple[int, bool] | None:
    """在 patch 的 hunk 里找锚点（按行序列逐字匹配，忽略行首 diff 标记）。

    Returns:
        (新文件行号, 是否在新增行上) 或 None（未命中 / 仅命中删除行）。
    """
    anchor_lines = [ln.strip() for ln in anchor.strip().splitlines() if ln.strip()]
    if not anchor_lines:
        return None

    new_line = 0
    lines = patch.splitlines()
    i = 0
    while i < len(lines):
        raw = lines[i]
        if raw.startswith("@@"):
            # @@ -old,count +new,count @@
            try:
                head = raw.split("+", 1)[1].split(",", 1)[0]
                new_line = int(head) - 1
            except (IndexError, ValueError):
                new_line = 0
            i += 1
            continue
        prefix = raw[:1]
        body = _strip_line(raw).strip()
        if prefix == "+":
            new_line += 1
            if body == anchor_lines[0]:
                # 尝试整段匹配
                j = i
                matched = True
                for k, al in enumerate(anchor_lines):
                    if j + k >= len(lines):
                        matched = False
                        break
                    cand = _strip_line(lines[j + k]).strip()
                    if cand != al:
                        matched = False
                        break
                if matched:
                    return (new_line, True)
            i += 1
            continue
        if prefix == "-":
            # 删除行不产生新行号；锚点首行若命在删除行上不算新增代码问题
            if body == anchor_lines[0]:
                return None
            i += 1
            continue
        # context 行
        new_line += 1
        i += 1
    return None


def _match_in_source(source: str, anchor: str) -> int | None:
    """锚点在全文中的首处行号（按行序列匹配，容许缩进差异）。"""
    anchor_lines = [ln.strip() for ln in anchor.strip().splitlines() if ln.strip()]
    if not anchor_lines:
        return None
    src_lines = [ln.strip() for ln in source.splitlines()]
    for start in range(len(src_lines) - len(anchor_lines) + 1):
        if src_lines[start : start + len(anchor_lines)] == anchor_lines:
            return start + 1
    return None


def _count_in_source(source: str, anchor: str) -> int:
    """锚点在全文出现次数（判断唯一性）。"""
    count = 0
    anchor_lines = [ln.strip() for ln in anchor.strip().splitlines() if ln.strip()]
    src_lines = [ln.strip() for ln in source.splitlines()]
    for start in range(len(src_lines) - len(anchor_lines) + 1):
        if src_lines[start : start + len(anchor_lines)] == anchor_lines:
            count += 1
    return count


async def anchor_finding(finding: dict[str, Any], ctx: Any) -> dict[str, Any]:
    """对单条 finding 做三级锚定，回写 anchored_line / anchor_status / confidence。

    ctx: ToolContext（提供变更文件 patch 与快照/全文读取）。
    """
    file = str(finding.get("file") or "")
    anchor = str(finding.get("existing_code") or "")
    finding["anchor_status"] = "unanchored"
    finding["anchored_line"] = finding.get("line", 0)

    if not file or not anchor.strip():
        finding["confidence"] = round(float(finding.get("confidence", 0.5)) * 0.5, 3)
        return finding

    # L1 hunk 匹配（变更文件 patch）
    patch = ctx.patch_of(file)
    if patch:
        hit = _match_in_patch(patch, anchor)
        if hit is not None:
            line, _on_added = hit
            finding["anchor_status"] = "hunk"
            finding["anchored_line"] = line
            finding["line"] = line
            return finding

    # L2 全文匹配（head 内容，懒取）
    source = await ctx.get_source(file)
    if source:
        occurrences = _count_in_source(source, anchor)
        if occurrences == 1:
            line = _match_in_source(source, anchor)
            if line:
                finding["anchor_status"] = "fulltext"
                finding["anchored_line"] = line
                finding["line"] = line
                return finding

    # L3 跨文件唯一迁移（快照面：找锚点唯一出现的其它文件）
    if ctx.snapshot_root is not None:
        from app.agent_loop.tool_context import search_in_sources

        result = search_in_sources(ctx, anchor.strip().splitlines()[0].strip(), is_regex=False, file_glob=None)
        if not result.get("truncated"):
            hits = [h for h in result.get("hits", []) if h["file"] != file]
            if len(hits) == 1:
                other = hits[0]
                other_source = await ctx.get_source(other["file"])
                if other_source and _count_in_source(other_source, anchor) == 1:
                    finding["anchor_status"] = "relocated"
                    finding["file"] = other["file"]
                    finding["anchored_line"] = other["line"]
                    finding["line"] = other["line"]
                    return finding

    # 兜底：未锚定，保留模型行号但显式降权（不静默）
    finding["confidence"] = round(float(finding.get("confidence", 0.5)) * 0.5, 3)
    logger.info("[ANCHOR] unanchored finding file=%s model_line=%s", file, finding.get("line"))
    return finding
