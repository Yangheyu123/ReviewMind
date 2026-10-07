"""ReviewOrchestrator：串联 Agent Loop 三层，对接现有 graph 入口。

这是 Agent Loop 的总装层，职责：
1. 跑预处理节点（复用 graph/nodes.py 的确定性节点）生成 ContextSnapshot；
2. Planner → Executor → Finalizer 三层串联；
3. 把 Finalizer 产出组装为与现有 ReviewGraph 一致的 ReviewReport，推送 SSE 进度；
4. 返回与 ReviewGraphResult 兼容的结果，供 ReviewGraph 无缝委托。

设计原则：
- 进度事件协议与现有 _NODE_DISPLAY / _add_progress 完全一致，前端无感知；
- 复用现有预处理节点的实现，不重写 fetch/diff_filter/ast/rag/tech_stack；
- 任何一层失败都走降级（Planner 有 fallback_plan，Executor 隔离故障，Finalizer 确定性），
  确保主流程永不因 Agent Loop 故障而崩溃。
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any

from app.agent_loop.executor import Executor
from app.agent_loop.finalizer import Finalizer
from app.agent_loop.planner import PlannerAgent
from app.agent_loop.schemas import ContextSnapshot
from app.graph.nodes import (
    node_ast_context,
    node_diff_filter,
    node_fetch_files_async,
    node_fetch_pr_async,
    node_parse_diff,
    node_rag_context,
    node_tech_stack_analysis,
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
class OrchestratorResult:
    """与 ReviewGraphResult 兼容的产出。"""

    pr_info: dict[str, Any]
    filtered_files: dict[str, Any]
    parsed_diff: list[dict[str, Any]]
    warnings: list[str]


class ReviewOrchestrator:
    """Agent Loop 总装：预处理 → Planner → Executor → Finalizer。"""

    def __init__(self, store: ReviewJobStore, github_client: GitHubClient) -> None:
        self._store = store
        self._github_client = github_client

    async def run(self, job: ReviewJob, config: dict[str, Any] | None = None) -> OrchestratorResult:
        """执行完整的 Agent Loop Review 工作流。"""
        warnings: list[str] = []
        await self._store.update_status(job.job_id, ReviewJobStatus.running)
        logger.info("[ORCHESTRATOR] start | job=%s pr_url=%s", job.job_id, job.pr_url)

        # ---- 第一阶段：预处理（复用现有确定性节点，串行）----
        state = ReviewGraphState(job_id=job.job_id, pr_url=job.pr_url, config=config or {})
        state = await self._run_preprocessing(state, job.job_id, warnings)

        # 关键节点失败 → 直接终止（与现有 _CRITICAL_NODES 行为一致）
        if state.error:
            await self._store.update_status(
                job.job_id, ReviewJobStatus.failed, error_message=state.error
            )
            await self._progress(job.job_id, "PREPROCESS_FAILED", 30, state.error, "warning")
            return OrchestratorResult(
                pr_info=state.pr_info,
                filtered_files=state.filtered_files,
                parsed_diff=state.parsed_diff,
                warnings=warnings,
            )

        # ---- 第二阶段：构建 ContextSnapshot ----
        snapshot = ContextSnapshot(
            job_id=job.job_id,
            pr_url=job.pr_url,
            pr_info=state.pr_info,
            filtered_files=state.filtered_files,
            parsed_diff=state.parsed_diff,
            ast_contexts=state.ast_contexts,
            rag_contexts=state.rag_contexts,
            tech_stack_prompt=state.tech_stack_prompt,
            config=config or {},
        )

        # ---- 第三阶段：Planner 决策 ----
        await self._progress(job.job_id, "PLANNER", 70, "Planner 正在制定审查计划")

        # 通过 on_step 回调把 Planner 的工具调用过程实时推给前端
        async def on_planner_step(step_no, step, tool_result):
            tc = step.tool_call
            if tc:
                tool_args = ", ".join(f"{k}={v}" for k, v in tc.arguments.items()) if tc.arguments else ""
                await self._progress(
                    job.job_id, "PLANNER", 70,
                    f"调用工具 {tc.name}({tool_args}) — 第 {step_no + 1} 步",
                )

        planner = PlannerAgent(snapshot, on_step=on_planner_step)
        plan = await planner.plan()
        if plan.reasoning:
            logger.info("[ORCHESTRATOR] planner decision: %s", plan.reasoning)
        await self._progress(
            job.job_id, "PLANNER_DONE", 72,
            f"审查计划：{', '.join(plan.dimension_names())}（{plan.overall_risk_hint or '未知风险'}）",
        )

        # ---- 第四阶段：Executor 并行执行 ----
        await self._progress(job.job_id, "EXECUTOR", 75, f"并行执行 {len(plan.dimensions)} 个维度")
        executor = Executor(snapshot)
        dimension_results = await executor.execute(plan)
        for dr in dimension_results:
            if not dr.success:
                warnings.append(f"{dr.dimension.value.upper()}_EXEC: {dr.error}")

        # 推送 finding 事件（与现有 node_*_agent 的 SSE 行为对齐）
        await self._push_findings(job.job_id, dimension_results)

        # ---- 第五阶段：Finalizer 聚合 + 报告 ----
        await self._progress(job.job_id, "FINALIZER", 90, "正在聚合结果与生成报告")
        finalizer = Finalizer(snapshot)
        finalized = finalizer.finalize(dimension_results, job_id=job.job_id)

        # ---- 第六阶段：组装 ReviewReport 并保存（与 _finish 对齐）----
        await self._finish(job.job_id, snapshot, finalized, warnings)

        return OrchestratorResult(
            pr_info=state.pr_info,
            filtered_files=state.filtered_files,
            parsed_diff=state.parsed_diff,
            warnings=warnings,
        )

    # ------------------------------------------------------------------
    # 预处理：复用现有节点
    # ------------------------------------------------------------------

    async def _run_preprocessing(
        self, state: ReviewGraphState, job_id: str, warnings: list[str]
    ) -> ReviewGraphState:
        """串行执行预处理节点，复用 graph/nodes.py 的实现。"""
        gc = self._github_client
        st = self._store

        # 顺序依赖：fetch_pr → fetch_files → diff_filter → parse_diff → ast → rag → tech_stack
        await self._progress(job_id, "FETCH_PR", 15, "正在拉取 GitHub PR 信息")
        state = await node_fetch_pr_async(state, gc, st)
        if state.error:
            return state

        await self._progress(job_id, "FETCH_FILES", 30, "正在拉取 PR 文件列表")
        state = await node_fetch_files_async(state, gc, st)
        if state.error:
            return state

        await self._progress(job_id, "DIFF_FILTER", 45, "正在过滤无意义 Diff")
        state = node_diff_filter(state, gc, st)
        await self._progress(job_id, "DIFF_PARSE", 55, "正在解析变更行")
        state = node_parse_diff(state, gc, st)
        await self._progress(job_id, "AST_CONTEXT", 60, "正在提取 AST 上下文")
        state = await node_ast_context(state, gc, st)
        if state.warnings:
            warnings.extend(state.warnings)

        await self._progress(job_id, "RAG_CONTEXT", 64, "正在检索项目知识库相似代码")
        state = await node_rag_context(state, gc, st)
        await self._progress(job_id, "TECH_STACK", 68, "正在分析项目技术栈")
        state = node_tech_stack_analysis(state, gc, st)
        return state

    # ------------------------------------------------------------------
    # SSE 进度推送（与现有协议一致）
    # ------------------------------------------------------------------

    async def _progress(
        self, job_id: str, step: str, percent: int, message: str, event_type: str = "progress"
    ) -> None:
        await self._store.add_progress_event(
            job_id, {"type": event_type, "step": step, "percent": percent, "message": message}
        )

    async def _push_findings(self, job_id: str, dimension_results: list) -> None:
        """推送 finding 事件，与现有 node_*_agent 的 SSE 行为对齐。"""
        for dr in dimension_results:
            if dr.result is None:
                continue
            for f in dr.result.findings:
                await self._store.add_progress_event(
                    job_id,
                    {
                        "id": f.id, "agent": f.agent, "file": f.file, "line": f.line,
                        "symbol": getattr(f, "symbol", "") or "", "level": f.level,
                        "finding_type": f.type, "confidence": f.confidence,
                        "description": f.description, "suggestion": f.suggestion,
                        "type": "finding",
                    },
                )

    # ------------------------------------------------------------------
    # 报告组装（与 ReviewGraph._finish 对齐）
    # ------------------------------------------------------------------

    async def _finish(self, job_id: str, snapshot: ContextSnapshot, finalized, warnings: list[str]) -> None:
        included = snapshot.filtered_files.get("included_files", [])
        changed_files = [
            ChangedFile(
                filename=f["filename"], status=f.get("status", "unknown"),
                additions=f.get("additions", 0), deletions=f.get("deletions", 0),
                changes=f.get("changes", f.get("additions", 0) + f.get("deletions", 0)),
                patch=f.get("patch"), risk_count=0,
            )
            for f in included
        ]

        # 变更符号来自 AST 提取结果（降级的文件级上下文没有 symbol，跳过）
        changed_symbols = [
            ChangedSymbol(
                file=ctx["file"], symbol=ctx["symbol"], language=ctx.get("language", "unknown"),
                start_line=ctx.get("start_line", 0), end_line=ctx.get("end_line", 0),
                changed_lines=ctx.get("changed_lines", []), code=ctx.get("code"),
            )
            for ctx in snapshot.ast_contexts
            if ctx.get("symbol")
        ]

        report = ReviewReport(
            summary=finalized.report_output.summary,
            risk_level=finalized.report_output.risk_level,
            stats=ReviewReportStats(),
            changed_files=changed_files,
            changed_symbols=changed_symbols,
            findings=finalized.report_output.findings,
            review_comment=finalized.report_output.review_comment,
        )

        # 统计风险等级分布（与 _finish 一致）
        for f in finalized.aggregated_risk.findings:
            level = f.level.upper()
            if level == "CRITICAL":
                report.stats.critical += 1
            elif level == "HIGH":
                report.stats.high += 1
            elif level in ("MEDIUM", "WARNING"):
                report.stats.medium += 1
            elif level == "LOW":
                report.stats.low += 1
            else:
                report.stats.suggestion += 1

        await self._store.update_status(job_id, ReviewJobStatus.completed, report=report)
        await self._progress(job_id, "DONE", 100, "Review Agent Loop 已完成")
