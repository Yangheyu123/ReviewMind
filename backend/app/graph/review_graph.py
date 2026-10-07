"""LangGraph Review Workflow：将 pipeline + agents 编排为有序图节点。"""

from __future__ import annotations

import asyncio
import json
import logging
from dataclasses import dataclass
from typing import Any

from app.graph.nodes import (
    node_ast_context,
    node_diff_filter,
    node_finding_validator,
    node_fetch_files_async,
    node_fetch_pr_async,
    node_parse_diff,
    node_performance_agent,
    node_rag_context,
    node_report_agent,
    node_risk_judge,
    node_security_agent,
    node_summary_agent,
    node_tech_stack_analysis,
    node_test_agent,
)
from app.graph.state import ReviewGraphState
from app.models.review_job import ReviewJob
from app.schemas.review import (
    ChangedFile,
    ChangedSymbol,
    ReviewJobStatus,
    ReviewReport,
    ReviewReportStats,
)
from app.services.github_client import GitHubClient
from app.services.review_job_store import ReviewJobStore


logger = logging.getLogger(__name__)


@dataclass
class ReviewGraphResult:
    pr_info: dict[str, Any]
    filtered_files: dict[str, Any]
    parsed_diff: list[dict[str, Any]]
    warnings: list[str]
    # Phase 4：引擎观测元数据（findings/tokens/llm 请求数），外层合入 pipeline_result
    engine_meta: dict[str, Any] | None = None


_CRITICAL_NODES = {"fetch_pr", "fetch_files"}

_NODE_ORDER = [
    "fetch_pr", "fetch_files", "diff_filter", "parse_diff", "ast_context", "rag_context",
    "tech_stack_analysis",
    "summary_agent", "security_agent", "performance_agent", "test_agent",
    "finding_validator",
    "risk_judge", "report_agent",
]

_PRE_NODES = ["fetch_pr", "fetch_files", "diff_filter", "parse_diff", "ast_context", "rag_context", "tech_stack_analysis"]
_AGENT_NODES = ["summary_agent", "security_agent", "performance_agent", "test_agent"]
_POST_NODES = ["finding_validator", "risk_judge", "report_agent"]

_NODE_DISPLAY: dict[str, tuple[str, str]] = {
    "fetch_pr": ("FETCH_PR", "正在拉取 GitHub PR 信息"),
    "fetch_files": ("FETCH_FILES", "正在拉取 PR 文件列表"),
    "diff_filter": ("DIFF_FILTER", "正在过滤无意义 Diff"),
    "parse_diff": ("DIFF_PARSE", "正在解析变更行"),
    "ast_context": ("AST_CONTEXT", "正在提取 AST 上下文"),
    "rag_context": ("RAG_CONTEXT", "正在检索项目知识库相似代码"),
    "tech_stack_analysis": ("TECH_STACK", "正在分析项目技术栈"),
    "summary_agent": ("SUMMARY_AGENT", "Summary Agent 运行中"),
    "security_agent": ("SECURITY_AGENT", "Security Agent 运行中"),
    "performance_agent": ("PERFORMANCE_AGENT", "Performance Agent 运行中"),
    "test_agent": ("TEST_AGENT", "Test Agent 运行中"),
    "finding_validator": ("FINDING_VALIDATOR", "正在过滤框架误报"),
    "risk_judge": ("RISK_JUDGE", "风险聚合中"),
    "report_agent": ("REPORT_AGENT", "生成报告中"),
}

_NODE_PERCENT: dict[str, int] = {
    "fetch_pr": 15, "fetch_files": 30, "diff_filter": 45, "parse_diff": 55,
    "ast_context": 60, "rag_context": 64, "tech_stack_analysis": 68,
    "summary_agent": 70, "security_agent": 75,
    "performance_agent": 80, "test_agent": 85,
    "finding_validator": 90, "risk_judge": 92, "report_agent": 98,
}


class ReviewGraph:
    def __init__(self, store: ReviewJobStore, github_client: GitHubClient | None = None) -> None:
        self._store = store
        self._github_client = github_client or GitHubClient()

    async def run(self, job: ReviewJob, config: dict[str, Any] | None = None) -> ReviewGraphResult:
        """执行完整的 review graph 工作流。

        当 ``settings.review_use_agent_loop`` 开启时，委托给 ReviewOrchestrator
        （Agent Loop：Planner → Executor → Finalizer）；否则走原有编排路径。
        两种路径返回结构兼容的 ReviewGraphResult，对外接口不变。
        """
        from app.core.config import settings

        if settings.review_use_agent_loop:
            from app.agent_loop.engine import run_engine
            result = await run_engine(self._store, self._github_client, job)
            return ReviewGraphResult(
                pr_info=result.get("pr_info", {}),
                filtered_files=result.get("filtered_files", {}),
                parsed_diff=result.get("parsed_diff", []),
                warnings=result.get("warnings", []),
                engine_meta=result.get("engine_meta"),
            )

        state = ReviewGraphState(job_id=job.job_id, pr_url=job.pr_url, config=config or {})
        await self._store.update_status(job.job_id, ReviewJobStatus.running)
        logger.info("[GRAPH] Starting pipeline | job=%s pr_url=%s nodes=%d", job.job_id, job.pr_url, len(_NODE_ORDER))

        # --- pre 段：串行 ---
        for node_name in _PRE_NODES:
            if state.error:
                break
            display_step, display_msg = _NODE_DISPLAY.get(node_name, (node_name.upper(), node_name))
            percent = _NODE_PERCENT.get(node_name, 50)
            await self._add_progress(job.job_id, display_step, percent, display_msg)
            state = await self._run_node(node_name, state)

            if node_name == "parse_diff" and not state.error:
                included = state.filtered_files.get("included_files", [])
                file_names = [f.get("filename", "") for f in included if isinstance(f, dict)]
                if file_names:
                    await self._store.add_progress_event(
                        job.job_id,
                        {"type": "chunk", "target": "files", "content": json.dumps(file_names, ensure_ascii=False)},
                    )

            if state.error and node_name in _CRITICAL_NODES:
                await self._store.update_status(job.job_id, ReviewJobStatus.failed, error_message=state.error)
                await self._add_progress(job.job_id, f"{display_step}_FAILED", percent, state.error, event_type="warning")
                return ReviewGraphResult(pr_info=state.pr_info, filtered_files=state.filtered_files, parsed_diff=state.parsed_diff, warnings=state.warnings)

        # --- agents 段：并行 ---
        if not state.error:
            await self._add_progress(job.job_id, "AGENTS", 70, "多 Agent 并行分析中")
            results = await asyncio.gather(
                *[self._run_node(n, state) for n in _AGENT_NODES],
                return_exceptions=True,
            )
            for node_name, res in zip(_AGENT_NODES, results):
                if isinstance(res, Exception):
                    logger.error("[GRAPH] Agent %s raised: %s", node_name, res)
                    state.warnings.append(f"{node_name.upper()}: {res}")

        # --- post 段：串行 ---
        if not state.error:
            for node_name in _POST_NODES:
                display_step, display_msg = _NODE_DISPLAY.get(node_name, (node_name.upper(), node_name))
                percent = _NODE_PERCENT.get(node_name, 50)
                await self._add_progress(job.job_id, display_step, percent, display_msg)
                state = await self._run_node(node_name, state)

        await self._finish(job.job_id, state)
        return ReviewGraphResult(pr_info=state.pr_info, filtered_files=state.filtered_files, parsed_diff=state.parsed_diff, warnings=state.warnings)

    async def _run_via_orchestrator(
        self, job: ReviewJob, config: dict[str, Any] | None
    ) -> ReviewGraphResult:
        """委托给 Agent Loop（ReviewOrchestrator）执行。"""
        from app.agent_loop.orchestrator import ReviewOrchestrator

        orchestrator = ReviewOrchestrator(self._store, self._github_client)
        result = await orchestrator.run(job, config=config)
        return ReviewGraphResult(
            pr_info=result.pr_info,
            filtered_files=result.filtered_files,
            parsed_diff=result.parsed_diff,
            warnings=result.warnings,
        )

    async def _run_node(self, node_name: str, state: ReviewGraphState) -> ReviewGraphState:
        gc = self._github_client
        st = self._store
        async_dispatch: dict[str, Any] = {
            "fetch_pr": lambda s: node_fetch_pr_async(s, gc, st),
            "fetch_files": lambda s: node_fetch_files_async(s, gc, st),
            "ast_context": lambda s: node_ast_context(s, gc, st),
            "rag_context": lambda s: node_rag_context(s, gc, st),
            "summary_agent": lambda s: node_summary_agent(s, gc, st),
            "security_agent": lambda s: node_security_agent(s, gc, st),
            "performance_agent": lambda s: node_performance_agent(s, gc, st),
            "test_agent": lambda s: node_test_agent(s, gc, st),
            "risk_judge": lambda s: node_risk_judge(s, gc, st),
        }
        sync_dispatch: dict[str, Any] = {
            "diff_filter": lambda s: _sync(node_diff_filter, s, gc, st),
            "parse_diff": lambda s: _sync(node_parse_diff, s, gc, st),
            "tech_stack_analysis": lambda s: _sync(node_tech_stack_analysis, s, gc, st),
            "finding_validator": lambda s: _sync(node_finding_validator, s, gc, st),
            "report_agent": lambda s: _sync(node_report_agent, s, gc, st),
        }
        if node_name in async_dispatch:
            return await async_dispatch[node_name](state)
        handler = sync_dispatch.get(node_name)
        if handler is not None:
            return await handler(state)
        return state

    async def _finish(self, job_id: str, state: ReviewGraphState) -> None:
        """将 graph 输出组装为 ReviewReport 并保存到 store。"""
        changed_files = [
            ChangedFile(
                filename=f["filename"], status=f.get("status", "unknown"),
                additions=f.get("additions", 0), deletions=f.get("deletions", 0),
                changes=f.get("changes", f.get("additions", 0) + f.get("deletions", 0)),
                patch=f.get("patch"), risk_count=0,
            )
            for f in state.filtered_files.get("included_files", [])
        ]
        # 变更符号来自 AST 提取结果（降级的文件级上下文没有 symbol，跳过）
        changed_symbols = [
            ChangedSymbol(
                file=ctx["file"], symbol=ctx["symbol"], language=ctx.get("language", "unknown"),
                start_line=ctx.get("start_line", 0), end_line=ctx.get("end_line", 0),
                changed_lines=ctx.get("changed_lines", []), code=ctx.get("code"),
            )
            for ctx in state.ast_contexts
            if ctx.get("symbol")
        ]

        if state.report_output:
            output = state.report_output
            report = ReviewReport(
                summary=output.summary, risk_level=output.risk_level,
                stats=ReviewReportStats(), changed_files=changed_files,
                changed_symbols=changed_symbols, findings=output.findings,
                review_comment=output.review_comment,
            )
        else:
            report = ReviewReport(
                summary=state.summary_text or f"PR 分析完成，共 {len(state.parsed_diff)} 个文件。",
                risk_level="LOW", stats=ReviewReportStats(),
                changed_files=changed_files, changed_symbols=changed_symbols, findings=[],
                review_comment="## AI Review Summary\n\n基础分析完成。",
            )

        if state.aggregated_risk:
            for f in state.aggregated_risk.findings:
                level = f.level.upper()
                if level == "CRITICAL": report.stats.critical += 1
                elif level == "HIGH": report.stats.high += 1
                elif level in ("MEDIUM", "WARNING"): report.stats.medium += 1
                elif level == "LOW": report.stats.low += 1
                else: report.stats.suggestion += 1

        await self._store.update_status(job_id, ReviewJobStatus.completed, report=report)
        await self._add_progress(job_id, "DONE", 100, "Review 工作流已完成")

    async def _add_progress(self, job_id: str, step: str, percent: int, message: str, event_type: str = "progress") -> None:
        await self._store.add_progress_event(
            job_id, {"type": event_type, "step": step, "percent": percent, "message": message},
        )


async def _sync(fn, state: ReviewGraphState, github_client, store) -> ReviewGraphState:
    """将同步节点包装为协程。"""
    return fn(state, github_client, store)