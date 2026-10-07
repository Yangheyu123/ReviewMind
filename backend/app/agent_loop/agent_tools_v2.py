"""ReviewAgent 的六个只读/提交工具（Phase 1 工具面）。

只读取证类：file_read / code_search / find_symbol / file_read_diff
提交终止类：submit_finding / task_done

设计约束（对齐改进方案 §4）：
- 工具面全部只读 + 提交两类，无 shell/写文件能力（第一道沙箱）；
- 路径 confinement 到本组变更文件（tool_context._confine）；
- file_read 500 行截断 + IS_TRUNCATED 标记；
- code_search 纯 Python re（无子进程，免疫注入），100 条命中封顶；
- submit_finding 强制带 existing_code 锚点字段（Phase 3 锚点定位的输入）。
"""

from __future__ import annotations

import logging
from typing import Any

from pydantic import BaseModel, Field

from app.agent_loop.tool_context import (
    ToolContext,
    find_symbol_in_sources,
    read_file_lines,
    search_in_sources,
)
from app.agent_loop.tools import Tool, ToolRegistry

logger = logging.getLogger(__name__)


class FileReadArgs(BaseModel):
    path: str = Field(description="变更文件路径")
    start_line: int | None = Field(default=None, ge=1, description="起始行（1-based，含）")
    end_line: int | None = Field(default=None, ge=1, description="结束行（含）")


class FileReadTool(Tool[FileReadArgs]):
    name = "file_read"
    description = (
        "读取变更文件 head 版本的内容（带行号）。单次最多 500 行，超出会截断并标记 "
        "is_truncated。可用 hunk 头 @@-x,y +m,n@@ 推导要读的行区间。"
    )
    args_model = FileReadArgs

    def __init__(self, ctx: ToolContext) -> None:
        self._ctx = ctx

    async def _run(self, args: FileReadArgs) -> Any:
        if self._ctx._confine(args.path) is None:
            return {"error": f"path not accessible: {args.path}（仅限本组变更文件）"}
        source = await self._ctx.get_source(args.path)
        if source is None:
            return {"error": f"failed to fetch file content: {args.path}"}
        return read_file_lines(self._ctx, args.path, args.start_line, args.end_line)


class CodeSearchArgs(BaseModel):
    pattern: str = Field(description="搜索模式：字面量或正则（is_regex=true 时）")
    is_regex: bool = Field(default=False, description="是否按正则解释 pattern")
    file_glob: str | None = Field(default=None, description="可选文件过滤，如 *.py / src/**")


class CodeSearchTool(Tool[CodeSearchArgs]):
    name = "code_search"
    description = (
        "在已读取（缓存）的变更文件源码中搜索字面量/正则，返回文件:行号:内容，"
        "最多 100 条命中。先用 file_read 预热文件可扩大搜索面。"
    )
    args_model = CodeSearchArgs

    def __init__(self, ctx: ToolContext) -> None:
        self._ctx = ctx

    async def _run(self, args: CodeSearchArgs) -> Any:
        if not args.pattern or args.pattern.startswith("-"):
            return {"error": "invalid pattern（不允许以 - 开头）"}
        return search_in_sources(self._ctx, args.pattern, args.is_regex, args.file_glob)


class FindSymbolArgs(BaseModel):
    symbol: str = Field(description="函数/方法/类名")


class FindSymbolTool(Tool[FindSymbolArgs]):
    name = "find_symbol"
    description = (
        "在已缓存的源码 AST 符号表中查定义位置（文件、行区间、签名片段）。"
        "注意：引用反查请用 code_search 搜符号名（名字级匹配，非语义级）。"
    )
    args_model = FindSymbolArgs

    def __init__(self, ctx: ToolContext) -> None:
        self._ctx = ctx

    async def _run(self, args: FindSymbolArgs) -> Any:
        return find_symbol_in_sources(self._ctx, args.symbol)


class FileReadDiffArgs(BaseModel):
    path: str = Field(description="变更文件路径")


class FileReadDiffTool(Tool[FileReadDiffArgs]):
    name = "file_read_diff"
    description = "查看其它变更文件的 diff patch（跨文件上下文，不消耗行数上限）。"
    args_model = FileReadDiffArgs

    def __init__(self, ctx: ToolContext) -> None:
        self._ctx = ctx

    async def _run(self, args: FileReadDiffArgs) -> Any:
        patch = self._ctx.patch_of(args.path)
        if patch is None:
            return {"error": f"path not accessible or no patch: {args.path}"}
        return {"file": args.path, "patch": patch[:8000]}


# 粗分类枚举（受控词表）：空间稳定、边界清晰，供去重/统计/筛选消费；
# 细分判断放 type_detail 自由文本，保证新颖问题不被硬塞错桶
FINDING_CATEGORIES = (
    "correctness", "performance", "security", "concurrency",
    "resource-leak", "error-handling", "api-misuse", "test-gap", "other",
)


class FindingComment(BaseModel):
    file: str = Field(description="问题所在文件")
    line: int = Field(default=0, ge=0, description="新文件行号（1-based）；无法定位填 0")
    existing_code: str = Field(description="从 diff 中逐字摘出的最小代码段（锚点，用于精确定位）")
    level: str = Field(description="CRITICAL/HIGH/MEDIUM/LOW/INFO")
    category: str = Field(
        description=(
            "粗分类，必须取以下之一：correctness(逻辑错误/边界/空值) / performance(n+1、"
            "重复计算、阻塞IO) / security(注入/越权/泄密) / concurrency(竞态/死锁/共享可变状态) / "
            "resource-leak(未关闭/泄漏) / error-handling(吞错/异常边界) / api-misuse(用错接口契约) / "
            "test-gap(缺失关键测试) / other(以上皆不适用)"
        ))
    type_detail: str = Field(default="", description="细分判断（自由文本），如 'n+1 in loop' / 'clock skew'")
    confidence: float = Field(default=0.5, ge=0, le=1)
    description: str = Field(description="问题描述（简体中文）")
    suggestion: str = Field(default="", description="修复建议（简体中文）")


class SubmitFindingArgs(BaseModel):
    comments: list[FindingComment] = Field(description="批量提交的 findings（可空）")
    summary: str = Field(default="", description="本组审查结论摘要（简体中文）")


class SubmitFindingTool(Tool[SubmitFindingArgs]):
    name = "submit_finding"
    description = (
        "批量提交本组审查发现。每条必须包含 existing_code（从 diff 逐字摘出的锚点代码段）。"
        "没有发现可提交空列表 + summary。提交后仍可继续取证。"
    )
    args_model = SubmitFindingArgs

    def __init__(self, ctx: ToolContext, group_label: str = "") -> None:
        self._ctx = ctx
        self._group = group_label
        self.last_summary: str = ""

    async def _run(self, args: SubmitFindingArgs) -> Any:
        from app.services.comment_anchor import anchor_finding

        accepted = 0
        anchored = 0
        for c in args.comments:
            if not c.existing_code.strip():
                continue  # 锚点缺失的 finding 直接拒收
            category = str(c.category).strip().lower()
            if category not in FINDING_CATEGORIES:
                continue  # 非法粗分类直接拒收（受控词表硬约束）
            finding = {
                "file": c.file, "line": c.line, "existing_code": c.existing_code,
                "level": c.level.upper(), "category": category,
                "type_detail": c.type_detail, "confidence": c.confidence,
                "description": c.description, "suggestion": c.suggestion,
                "group": self._group,
                "source": "llm",
            }
            # Phase 3：三级锚定流水（hunk→全文→跨文件迁移），行号由匹配产生
            finding = await anchor_finding(finding, self._ctx)
            if finding["anchor_status"] != "unanchored":
                anchored += 1
            self._ctx.findings.append(finding)
            accepted += 1
        if args.summary:
            self.last_summary = args.summary
        return {
            "accepted": accepted,
            "anchored": anchored,
            "unanchored": accepted - anchored,
            "rejected_no_anchor": len(args.comments) - accepted,
            "total_findings_so_far": len(self._ctx.findings),
            "note": "anchored=hunk/fulltext/relocated 三级流水命中；unanchored 的行号来自模型、已降权",
        }


class TaskDoneArgs(BaseModel):
    state: str = Field(default="DONE", description="DONE 或 FAILED")
    summary: str = Field(default="", description="收工摘要")


class TaskDoneTool(Tool[TaskDoneArgs]):
    name = "task_done"
    description = "结束本组审查任务。证据收集完毕、findings 已提交后调用。state=DONE 正常结束。"
    args_model = TaskDoneArgs

    def __init__(self) -> None:
        self.requested_state: str | None = None
        self.summary: str = ""

    async def _run(self, args: TaskDoneArgs) -> Any:
        self.requested_state = args.state.upper() if args.state else "DONE"
        self.summary = args.summary
        return {"task_state": self.requested_state}


def build_tool_registry(ctx: ToolContext, group_label: str = "") -> tuple[ToolRegistry, dict[str, Tool]]:
    """装配 Phase 1 工具面。返回 (registry, 工具实例表)——循环需要拿到 task_done/submit 实例的状态。"""
    tools: list[Tool] = [
        FileReadTool(ctx),
        CodeSearchTool(ctx),
        FindSymbolTool(ctx),
        FileReadDiffTool(ctx),
        SubmitFindingTool(ctx, group_label),
        TaskDoneTool(),
    ]
    registry = ToolRegistry()
    instances = {}
    for t in tools:
        registry.register(t)
        instances[t.name] = t
    return registry, instances
