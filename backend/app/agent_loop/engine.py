"""LangGraph 审查引擎（Phase 1：agent_loop 转正后的主引擎）。

架构（对齐改进方案 §4）：
- StateGraph 骨架：pre（确定性预处理，复用既有节点函数）→ Send() fan-out
  到 review_group（每组一个 ReviewAgent 工具循环，map-reduce）→ 条件边
  route_after_review（预算耗尽 → grace_round；PRE 失败 → finish_failed）→ aggregate → END；
- checkpointer：AsyncPostgresSaver（thread_id = job_id），任务状态可恢复；
- 进度事件：沿用 store.add_progress_event（SSE 管线不变）。

LangGraph 承重特性使用清单（采用门槛 ≥3，见改进方案 §0）：
1. Send() 分组并行 fan-out ✅  2. 条件边（预算闸路由/失败短路）✅  3. AsyncPostgresSaver ✅
"""

from __future__ import annotations

import logging
import operator
from typing import Annotated, Any, TypedDict

from langgraph.graph import END, START, StateGraph
from langgraph.types import Send
from langchain_core.runnables import RunnableConfig

from app.core.config import settings
from app.core.llm import LLMClient
from app.models.review_job import ReviewJob
from app.schemas.review import (
    ChangedFile,
    ChangedSymbol,
    ReviewFinding,
    ReviewJobStatus,
    ReviewReport,
    ReviewReportStats,
)
from app.services.github_client import GitHubClient
from app.services.github_url_parser import parse_github_pr_url
from app.services.review_job_store import ReviewJobStore

logger = logging.getLogger(__name__)

# 分组：每 ≤GROUP_MAX_FILES 个文件一组；组数硬上限防实例数爆炸（改进方案 §14 墙 2）
GROUP_MAX_FILES = 10
GROUPS_HARD_CAP = 12


class EngineState(TypedDict, total=False):
    job_id: str
    pr_url: str
    head_sha: str
    # PRE 产物
    pr_info: dict[str, Any]
    included_files: list[dict[str, Any]]
    excluded_files: list[dict[str, Any]]
    snapshot_root: str          # base tarball 快照根（空串 = 无快照）
    memory_hints: str           # 审查记忆渲染段（空串 = 无历史发现）
    snapshot_files: list[str]
    parsed_diff: list[dict[str, Any]]
    ast_contexts: list[dict[str, Any]]
    groups: list[list[str]]
    # fan-out 汇聚（reducer 合并多组写入）
    findings: Annotated[list[dict[str, Any]], operator.add]
    group_summaries: Annotated[list[str], operator.add]
    tokens_used: Annotated[list[int], operator.add]
    llm_records: Annotated[list[dict[str, Any]], operator.add]
    budget_exceeded: Annotated[list[bool], operator.add]
    warnings: Annotated[list[str], operator.add]
    # 控制
    error: str | None
    group_files: list[str]          # Send 注入的分组上下文
    group_index: int


# ---------------------------------------------------------------------------
# 依赖注入：store/client/job 等非序列化对象走 RunnableConfig.configurable，
# 不进 EngineState（checkpointer 会序列化 state）。
# ---------------------------------------------------------------------------

def _deps(config: dict[str, Any]) -> dict[str, Any]:
    conf = config.get("configurable", config) if isinstance(config, dict) else {}
    return {
        "store": conf["store"],
        "github_client": conf["github_client"],
        "job": conf["job"],
    }


# ---------------------------------------------------------------------------
# PRE：确定性预处理（复用既有节点函数，载体是旧 ReviewGraphState）
# ---------------------------------------------------------------------------

async def _run_pre(store: ReviewJobStore, github_client: GitHubClient, job: ReviewJob) -> dict[str, Any]:
    from app.graph.nodes import (
        node_ast_context,
        node_diff_filter,
        node_fetch_files_async,
        node_fetch_pr_async,
        node_parse_diff,
    )
    from app.graph.state import ReviewGraphState

    async def step(name: str, percent: int, message: str) -> None:
        await store.add_progress_event(job.job_id, {
            "type": "progress", "step": name, "percent": percent, "message": message,
        })

    carrier = ReviewGraphState(job_id=job.job_id, pr_url=job.pr_url, config={})
    await step("FETCH_PR", 15, "正在拉取 GitHub PR 信息")
    carrier = await node_fetch_pr_async(carrier, github_client, store)
    if carrier.error:
        return {"error": carrier.error}
    await step("FETCH_FILES", 30, "正在拉取 PR 文件列表")
    carrier = await node_fetch_files_async(carrier, github_client, store)
    if carrier.error:
        return {"error": carrier.error}
    await step("DIFF_FILTER", 45, "正在过滤无意义 Diff")
    carrier = node_diff_filter(carrier, github_client, store)
    await step("DIFF_PARSE", 55, "正在解析变更行")
    carrier = node_parse_diff(carrier, github_client, store)
    if carrier.error:
        return {"error": carrier.error}
    await step("AST_CONTEXT", 60, "正在提取 AST 上下文")
    carrier = await node_ast_context(carrier, github_client, store)
    await step("SNAPSHOT", 66, "正在准备源码快照（base tarball）")
    snapshot_root, snapshot_files = await _prepare_snapshot(github_client, carrier)
    if snapshot_root is None:
        warnings_extra = "源码快照不可用，检索面退化为变更文件集"
    else:
        warnings_extra = ""
    await step("AGENTS", 70, "多 Agent 并行分析中（每组一个工具循环实例）")

    included = carrier.filtered_files.get("included_files", [])
    filenames = [f["filename"] for f in included]
    # §16.1 分层分组：import 依赖图 → 目录亲和 → 顺序兜底（快照已就绪，可读 head 源码）
    from app.services.grouping import build_groups_v2
    from app.services.source_snapshot import read_snapshot_file

    def _read_src(path: str) -> str | None:
        if snapshot_root is None:
            return None
        return read_snapshot_file(snapshot_root, path)

    if settings.review_grouping_mode == "single":
        # 消融 RC 档：单 reviewer + 仓内上下文工具，无分组路由（评测体系 v3）
        groups, layer = [filenames], "single-reviewer-ablation"
    else:
        v2_groups, layer = build_groups_v2(included, _read_src)
        groups = v2_groups[:GROUPS_HARD_CAP]
    dropped = sum(len(g) for g in groups[GROUPS_HARD_CAP:]) if settings.review_grouping_mode != "single" else 0
    await step("GROUPING", 67, f"分组完成（{layer}，{len(groups)} 组）")

    # Phase 2.5：审查记忆预注入（查询词确定，无需 agent 自主检索；空库零命中零成本）
    memory_hints = ""
    if settings.review_enable_memory:
        try:
            from app.services.review_memory import recall_memory, render_memory_hints
            repo_full = f"{carrier.pr_info.get('owner', '')}/{carrier.pr_info.get('repo', '')}"
            memories = await recall_memory(repo_full, [f["filename"] for f in included])
            memory_hints = render_memory_hints(memories)
            if memories:
                await step("MEMORY", 68, f"召回 {len(memories)} 条本仓库历史审查发现")
        except Exception as exc:
            logger.warning("[ENGINE] memory recall failed: %s", exc)
    warnings = list(carrier.warnings)
    if dropped:
        warnings.append(
            f"文件数超出组数硬上限（{GROUPS_HARD_CAP}×{GROUP_MAX_FILES}），{dropped} 个文件本轮跳过（coverage 受限）"
        )
    return {
        "pr_info": carrier.pr_info,
        "included_files": included,
        "excluded_files": carrier.filtered_files.get("excluded_files", []),
        "snapshot_root": str(snapshot_root) if snapshot_root else "",
        "memory_hints": memory_hints,
        "snapshot_files": snapshot_files,
        "parsed_diff": carrier.parsed_diff,
        "ast_contexts": carrier.ast_contexts,
        "groups": groups,
        "warnings": warnings + ([warnings_extra] if warnings_extra else []),
        "error": None,
    }


async def _prepare_snapshot(github_client, carrier) -> tuple[Any, list[str]]:
    """下载 base tarball 快照（同步阻塞调用放线程池不必要——tarball 下载在
    download_tarball 内是异步的；此处保持同步签名以适配既有节点调用风格，
    失败降级不阻塞管线）。"""
    from app.services.source_snapshot import ensure_snapshot, iter_snapshot_files

    base_sha = (carrier.pr_info.get("base") or {}).get("sha", "")
    pr_ref = None
    try:
        from app.schemas.github import GitHubPullRequestRef
        pr_ref = GitHubPullRequestRef(
            owner=carrier.pr_info.get("owner", ""),
            repo=carrier.pr_info.get("repo", ""),
            pull_number=carrier.pr_info.get("pull_number", 0),
            html_url=carrier.pr_info.get("html_url", ""),
        )
        root = await ensure_snapshot(github_client, pr_ref, base_sha)
        if root is None:
            return None, []
        files = iter_snapshot_files(root)
        return root, files
    except Exception as exc:
        logger.warning("[ENGINE] snapshot prepare failed: %s", exc)
        return None, []


def _build_groups(filenames: list[str]) -> tuple[list[list[str]], int]:
    groups = [filenames[i : i + GROUP_MAX_FILES] for i in range(0, len(filenames), GROUP_MAX_FILES)]
    if len(groups) > GROUPS_HARD_CAP:
        dropped = sum(len(g) for g in groups[GROUPS_HARD_CAP:])
        return groups[:GROUPS_HARD_CAP], dropped
    return groups, 0


# ---------------------------------------------------------------------------
# 路由（条件边）
# ---------------------------------------------------------------------------

def route_pre(state: EngineState) -> Any:
    """PRE 出口：失败短路到 finish_failed；成功则 Send() fan-out 每组一个 review_group。"""
    if state.get("error"):
        return "finish_failed"
    base = {k: v for k, v in state.items()
            if k not in ("findings", "group_summaries", "tokens_used", "budget_exceeded", "warnings")}
    return [
        Send("review_group", {**base, "group_files": g, "group_index": i})
        for i, g in enumerate(state.get("groups", []))
    ]


async def post_review(state: EngineState, config) -> dict[str, Any]:
    """fan-out 汇聚节点（空操作）。

    多个 Send 分支的出口边会被逐分支评估，条件路由若直接挂在 review_group 上，
    aggregate 会被调度多次（实测 completed->completed 非法转移）。汇聚后再路由。
    """
    return {}


def route_after_review(state: EngineState) -> str:
    """post_review 出口：预算耗尽 → grace_round；否则聚合。"""
    if state.get("error"):
        return "finish_failed"
    if any(state.get("budget_exceeded", [])):
        return "grace_round"
    return "aggregate"


# ---------------------------------------------------------------------------
# 节点
# ---------------------------------------------------------------------------

async def pre_node(state: EngineState, config: RunnableConfig) -> dict[str, Any]:
    deps = _deps(config)
    store: ReviewJobStore = deps["store"]
    client: GitHubClient = deps["github_client"]
    job: ReviewJob = deps["job"]
    await store.update_status(job.job_id, ReviewJobStatus.running)
    pre = await _run_pre(store, client, job)
    if pre.get("error"):
        await store.update_status(job.job_id, ReviewJobStatus.failed, error_message=pre["error"])
    return pre


GROUP_SYSTEM_PROMPT = """你是资深代码审查员。对分配给你的变更文件组做全维度审查（安全、性能、测试、正确性）。

工作方式（ReAct 取证循环）：
1. 阅读 diff，形成怀疑点；
2. 不许凭空下结论——用工具取证：
   - file_read(path, start_line, end_line)：读变更文件 head 版本全文（hunk 头 @@-x,y +m,n@@ 可推算行区间）
   - code_search(pattern, is_regex?, file_glob?)：全仓检索（含 base 快照；支持字面量/正则与文件过滤）
   - find_symbol(symbol)：查函数定义位置与行区间
   - file_read_diff(path)：看其它变更文件的 diff
3. 证据闭合后 submit_finding 批量提交（每条必须带 existing_code：从 diff 中逐字摘出的最小代码段）；
4. 全部完成后调用 task_done(state="DONE")。

纪律：
- 证据与置信分离：结论必须有代码证据，但把握程度用 confidence（0-1）表达——真实可疑但
  尚未完全确证的问题以低 confidence 提交（管线会按置信度排序降权），不要因为不确定就不提交；
  纯风格偏好与臆测不要提交；
- 测试文件是一等审查对象：重点看断言缺口、期望值错误、缺失的边界用例、flaky 模式
  （sleep 等待/共享状态/顺序依赖）；
- existing_code 必须与 diff 中内容逐字一致（用于锚点定位）；
- level ∈ CRITICAL/HIGH/MEDIUM/LOW/INFO；category 必须从给定枚举选（细分判断写 type_detail）；description/suggestion 用简体中文；
- 单条 finding 只描述一个问题。"""


def _group_system_prompt() -> str:
    """按配置切换输出语言：英文 GT 基准（AACR-Bench）跑分时用英文描述。"""
    if settings.review_output_language == "en":
        return GROUP_SYSTEM_PROMPT.replace("description/suggestion 用简体中文",
                                           "description/suggestion in English")
    return GROUP_SYSTEM_PROMPT


def _group_user_prompt(state: EngineState, group_files: list[str]) -> str:
    parts = ["你负责审查以下变更文件组（全维度：安全/性能/测试/正确性）：\n"]
    # Phase 3：按组内语言注入高信号审查清单（高信号 + 反例约束，宁缺毋滥）
    from app.rules import load_rules_for_group
    rule_text = load_rules_for_group(group_files)
    if rule_text:
        parts.append("## 语言审查清单（只报告有证据的高信号问题）")
        parts.append(rule_text)
        parts.append("")
    memory = state.get("memory_hints") or ""
    if memory:
        parts.append(memory)
        parts.append("")
    parsed_map = {d.get("file"): d for d in state.get("parsed_diff", [])}
    for name in group_files:
        d = parsed_map.get(name, {})
        parts.append(f"## {name} (+{d.get('additions', 0)}/-{d.get('deletions', 0)})")
        patch = d.get("patch") or ""
        if len(patch) > 8000:
            patch = patch[:8000] + "\n... (truncated)"
        parts.append(f"```diff\n{patch}\n```" if patch else "（无 patch 内容）")
        parts.append("")
    other = [d for d in state.get("parsed_diff", []) if d.get("file") not in group_files]
    if other:
        lines = [f"{d.get('file')} (+{d.get('additions', 0)}/-{d.get('deletions', 0)})"
                 for d in other[:40]]
        suffix = f" 等 {len(other)} 个" if len(other) > 40 else ""
        parts.append("## 全 PR 变更文件清单（跨组视野；其它文件 diff 可用 file_read_diff 查看）\n"
                     + "、".join(lines) + suffix)
    return "\n".join(parts)


async def review_group(state: EngineState, config: RunnableConfig) -> dict[str, Any]:
    """组内 ReviewAgent 工具循环节点（fan-out 的每个实例）。"""
    store: ReviewJobStore = _deps(config)["store"]
    client: GitHubClient = _deps(config)["github_client"]
    group_files: list[str] = state.get("group_files", [])
    group_index: int = state.get("group_index", 0)
    total_groups = len(state.get("groups", []))
    label = f"g{group_index}"

    await store.add_progress_event(state["job_id"], {
        "type": "progress", "step": f"AGENT_{label}", "percent": 70,
        "message": f"Agent 审查组 {group_index + 1}/{total_groups}：{', '.join(group_files[:3])}{'…' if len(group_files) > 3 else ''}",
    })

    llm = LLMClient()
    if llm._mock_mode or not llm.is_configured:
        return _fallback_group_findings(state, group_files)

    from app.agent_loop.review_agent import ReviewAgent
    from app.agent_loop.tool_context import ToolContext

    head_sha = (state.get("pr_info", {}).get("head") or {}).get("sha", "")
    pr_ref = parse_github_pr_url(state["pr_url"])
    from pathlib import Path
    snapshot_root_str = state.get("snapshot_root") or ""
    ctx = ToolContext(
        github_client=client, pr_ref=pr_ref, head_sha=head_sha,
        changed_files=state.get("included_files", []), group_files=group_files,
        snapshot_root=Path(snapshot_root_str) if snapshot_root_str else None,
        snapshot_files=state.get("snapshot_files", []),
    )
    from app.core.llm_usage import llm_context

    agent = ReviewAgent(llm, ctx, model=settings.llm_model_review, group_label=label)
    with llm_context(state["job_id"], group=label, phase="review") as usage_ctx:
        result = await agent.run(_group_system_prompt(), _group_user_prompt(state, group_files))

    # Phase 3：filter 反思（工作流步骤，非 agent）——只删 diff 可证明错误的 findings；
    # 受保护主题一票豁免；LLM 不可用则全保留（保守方向），不阻塞管线
    if result.findings and settings.review_enable_filter:
        from app.agents.filter_agent import filter_findings
        # 核查 prompt 只喂组内补丁（控制上下文）；删除校验的证据源是全量变更文件
        # （锚点迁移后的 finding 可能指向组外文件，按组内清单判 Ground A 会误删）
        patches = {
            f["filename"]: (f.get("patch") or "")
            for f in state.get("included_files", [])
            if f["filename"] in group_files
        }
        all_patches = {
            f["filename"]: (f.get("patch") or "")
            for f in state.get("included_files", [])
        }
        with llm_context(state["job_id"], group=label, phase="filter") as filter_ctx:
            result.findings, filtered_meta = await filter_findings(
                result.findings, patches, llm, all_patches=all_patches,
            )
        usage_ctx["records"].extend(filter_ctx["records"])
        if filtered_meta.get("status") == "applied" and (
            filtered_meta.get("removed") or filtered_meta.get("unverified_kept")
        ):
            await store.add_progress_event(state["job_id"], {
                "type": "warning", "code": "FILTER_REMOVED",
                "message": (
                    f"反思节点剔除 {len(filtered_meta.get('removed', []))} 条可证明错误的 findings"
                    f"（受保护豁免 {filtered_meta.get('protected_kept', 0)} 条，"
                    f"证据校验拦截 {filtered_meta.get('unverified_kept', 0)} 条）"
                ),
            })

    # LLM 全程不可用（如余额耗尽/密钥失效）→ 降级规则版并显式可观测，
    # 不允许以 "completed + 0 findings" 伪装成功（静默降级反模式）
    if result.stop_reason.startswith("llm_error"):
        await store.add_progress_event(state["job_id"], {
            "type": "warning", "code": "LLM_UNAVAILABLE",
            "message": f"LLM 调用失败，本组降级为规则审查：{result.stop_reason[:200]}",
        })
        fallback = _fallback_group_findings(state, group_files)
        for i, f in enumerate(fallback["findings"]):
            await store.add_progress_event(state["job_id"], {
                "type": "finding",
                "id": f"{state['job_id']}_{label}_fb{i}",
                "agent": f.get("agent", "review_agent"),
                "file": f.get("file", ""), "line": f.get("line", 0),
                "level": f.get("level", "INFO"), "type": f.get("category") or f.get("type", "review"),
                "confidence": f.get("confidence", 0.3),
                "description": f.get("description", ""), "suggestion": f.get("suggestion", ""),
            })
        fallback["warnings"] = [result.stop_reason[:300]]
        return fallback

    for i, f in enumerate(result.findings):
        await store.add_progress_event(state["job_id"], {
            "type": "finding",
            "id": f"{state['job_id']}_{label}_{i}",
            "agent": "review_agent",
            "file": f["file"], "line": f.get("line", 0),
            "level": f.get("level", "INFO"), "type": f.get("category") or f.get("type", "review"),
            "confidence": f.get("confidence", 0.5),
            "description": f.get("description", ""),
            "suggestion": f.get("suggestion", ""),
        })

    return {
        "findings": result.findings,
        "group_summaries": [result.summary] if result.summary else [],
        "tokens_used": [result.tokens_used],
        "budget_exceeded": [result.stop_reason == "budget_exceeded"],
        "llm_records": usage_ctx["records"],
    }


_CATEGORY_KEYWORDS = {
    "security": "security", "injection": "security", "performance": "performance",
    "n-plus-1": "performance", "test": "test-gap", "concurrency": "concurrency",
    "race": "concurrency", "leak": "resource-leak", "resource": "resource-leak",
    "error": "error-handling", "correctness": "correctness", "logic": "correctness",
}


def _normalize_category(finding: dict[str, Any]) -> str:
    """旧格式 findings（自由文本 type）映射到受控 category；已有 category 直接用。"""
    if finding.get("category"):
        return str(finding["category"])[:32]
    raw = str(finding.get("type") or "").lower()
    for kw, cat in _CATEGORY_KEYWORDS.items():
        if kw in raw:
            return cat
    return "other"


def _fallback_group_findings(state: EngineState, group_files: list[str]) -> dict[str, Any]:
    """LLM 不可用（mock/未配置）→ 降级到既有规则版 agent，系统保持可用。"""
    from app.agents import performance_agent, security_agent, test_agent
    from app.schemas.agents import AgentContext

    parsed = [d for d in state.get("parsed_diff", []) if d.get("file") in group_files]
    ctx = AgentContext(pr_info=state.get("pr_info", {}), parsed_diff=parsed)
    findings: list[dict[str, Any]] = []
    for module in (security_agent, performance_agent, test_agent):
        res = module.run(ctx)
        findings.extend([f.model_dump(mode="json") for f in res.findings])
    return {"findings": findings, "group_summaries": [], "tokens_used": [0], "budget_exceeded": [False]}


async def grace_round(state: EngineState, config: RunnableConfig) -> dict[str, Any]:
    """预算耗尽标记节点：抢救提交已在 ReviewAgent 的 FINAL ROUND 机制内完成，
    此处负责可观测（warning 事件）——预算耗尽≠失败，产出部分结果。"""
    store: ReviewJobStore = _deps(config)["store"]
    await store.add_progress_event(state["job_id"], {
        "type": "warning", "code": "BUDGET_EXCEEDED",
        "message": "token 预算耗尽，已执行最后一轮抢救，产出部分结果",
    })
    return {}


async def finish_failed(state: EngineState, config: RunnableConfig) -> dict[str, Any]:
    store: ReviewJobStore = _deps(config)["store"]
    await store.add_progress_event(state["job_id"], {
        "type": "warning", "code": "PIPELINE_FAILED",
        "message": state.get("error") or "unknown error",
    })
    return {}


# ---------------------------------------------------------------------------
# 聚合（确定性）
# ---------------------------------------------------------------------------

async def aggregate(state: EngineState, config: RunnableConfig) -> dict[str, Any]:
    store: ReviewJobStore = _deps(config)["store"]
    job_id = state["job_id"]
    # 幂等护栏：多次调度时已完成则直接返回
    try:
        existing = await store.get(job_id)
        if existing.status == ReviewJobStatus.completed:
            return {}
    except Exception:
        pass

    changed_files = [
        ChangedFile(
            filename=f["filename"], status=f.get("status", "unknown"),
            additions=f.get("additions", 0), deletions=f.get("deletions", 0),
            changes=f.get("changes", f.get("additions", 0) + f.get("deletions", 0)),
            patch=f.get("patch"), risk_count=0,
        )
        for f in state.get("included_files", [])
    ]
    changed_symbols = [
        ChangedSymbol(
            file=ctx["file"], symbol=ctx["symbol"], language=ctx.get("language", "unknown"),
            start_line=ctx.get("start_line", 0), end_line=ctx.get("end_line", 0),
            changed_lines=ctx.get("changed_lines", []), code=ctx.get("code"),
        )
        for ctx in state.get("ast_contexts", []) if ctx.get("symbol")
    ]

    await store.add_progress_event(job_id, {"type": "progress", "step": "FINDING_VALIDATOR", "percent": 80, "message": "正在过滤误报"})
    raw_findings = state.get("findings", [])
    findings = _dedupe_findings(_apply_confidence_floor(raw_findings))
    if len(findings) < len(raw_findings):
        await store.add_progress_event(job_id, {
            "type": "warning", "code": "LOW_CONFIDENCE_DROPPED",
            "message": f"置信度地板（≥{settings.review_min_confidence}）过滤 {len(raw_findings) - len(findings)} 条低置信 findings",
        })
    await store.add_progress_event(job_id, {"type": "progress", "step": "RISK_JUDGE", "percent": 88, "message": "风险聚合与去重"})
    stats = _stats_of(findings)
    risk = _risk_level(stats)
    summary = "\n\n".join(s for s in state.get("group_summaries", []) if s) or \
        f"共审查 {len(state.get('parsed_diff', []))} 个文件，发现 {len(findings)} 个问题。"

    report = ReviewReport(
        summary=summary[:2000], risk_level=risk, stats=ReviewReportStats(**stats),
        changed_files=changed_files, changed_symbols=changed_symbols,
        findings=_to_review_findings(findings, state["job_id"]),
        review_comment=_render_comment(findings, risk, stats, summary),
    )
    await store.add_progress_event(job_id, {"type": "progress", "step": "REPORT_AGENT", "percent": 95, "message": "生成报告中"})
    # Phase 4：LLM 请求明细落库（观测切片：job/组/阶段/token/延迟）
    records = state.get("llm_records", []) or []
    saved = await store.save_llm_requests(job_id, records)
    if saved:
        total_tokens = sum(int(r.get("total_tokens", 0) or 0) for r in records)
        logger.info("[ENGINE] LLM usage job=%s requests=%d total_tokens=%d", job_id, saved, total_tokens)

    await store.update_status(job_id, ReviewJobStatus.completed, report=report)
    # Phase 2.5：审查记忆写入（失败降级日志，不阻塞完成）
    if settings.review_enable_memory:
        try:
            from app.services.review_memory import save_findings_to_memory
            repo_full = "/".join(str(state.get("pr_url") or "").split("/")[3:5]) if "/" in str(state.get("pr_url")) else ""
            pr_info = state.get("pr_info") or {}
            repo_full = f"{pr_info.get('owner', '')}/{pr_info.get('repo', '')}" or repo_full
            n = await save_findings_to_memory(repo_full, findings)
            if n:
                logger.info("[ENGINE] memory saved %d findings (repo=%s)", n, repo_full)
        except Exception as exc:
            logger.warning("[ENGINE] memory save failed: %s", exc)
    await store.add_progress_event(job_id, {
        "type": "progress", "step": "DONE", "percent": 100, "message": "Agent 引擎审查完成",
    })
    return {}


def _to_review_findings(findings: list[dict[str, Any]], job_id: str) -> list[ReviewFinding]:
    out: list[ReviewFinding] = []
    for i, f in enumerate(findings):
        try:
            out.append(ReviewFinding(
                id=str(f.get("id") or f"{job_id}_f{i}"),
                agent=str(f.get("agent") or "review_agent"),
                file=str(f.get("file") or ""),
                line=int(f.get("line") or 0),
                level=str(f.get("level") or "INFO").upper(),
                type=str(f.get("category") or f.get("type") or "review"),
                confidence=float(f.get("confidence") or 0.5),
                description=str(f.get("description") or ""),
                suggestion=str(f.get("suggestion") or ""),
                symbol=f.get("symbol"),
                code_snippet=f.get("existing_code") or f.get("code_snippet"),
            ))
        except Exception:
            logger.warning("[ENGINE] skip malformed finding: %s", f)
    return out


def _anchor_span(f: dict[str, Any]) -> tuple[int, int]:
    """锚点起止行：起 = anchored_line，长 = existing_code 行数。"""
    start = int(f.get("anchored_line") or f.get("line") or 0)
    n_lines = len([ln for ln in str(f.get("existing_code") or "").splitlines() if ln.strip()]) or 1
    return start, start + n_lines - 1


def _overlap_tokens(text: str) -> set[str]:
    """中英混合重叠特征：英文词 + 中文 bigram（中文无词边界，字符对是标准做法）。"""
    import re as _re
    stop = set("the a an to of in for is are with on at this that it and or not".split())
    tokens = {t for t in _re.findall(r"[a-zA-Z_]\w*", text.lower()) if t not in stop}
    han = _re.findall("[\u4e00-\u9fff]", text)
    han_str = "".join(han)
    tokens.update(han_str[i : i + 2] for i in range(len(han_str) - 1))
    return tokens


def _shared_features(a: str, b: str) -> int:
    return len(_overlap_tokens(a) & _overlap_tokens(b))


def _similar_description(a: str, b: str, *, min_shared: int = 3, ratio: float = 0.5) -> bool:
    """同一问题的两种说法：共享 ≥3 个核心特征（英文词+中文 bigram），或重叠比率 ≥0.5。
    "共享三个核心词"（如 判空/None/崩溃）业务上可解释，比单一比率阈值稳健。"""
    ta, tb = _overlap_tokens(a), _overlap_tokens(b)
    if not ta or not tb:
        return False
    return len(ta & tb) >= min_shared or len(ta & tb) / min(len(ta), len(tb)) >= ratio


_LEVEL_ORDER = {"CRITICAL": 0, "HIGH": 1, "MEDIUM": 2, "WARNING": 2, "LOW": 3, "INFO": 4}
_ANCHOR_RANK = {"hunk": 0, "fulltext": 1, "relocated": 2, "unanchored": 3}


def _apply_confidence_floor(findings: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """聚合阶段置信度地板（确定性，P1 校准第二环）。

    产出侧 prompt 允许低置信提交（换召回），噪声由本地板在聚合期统一回收——
    工作点 0.4 经 49 标签评测扫描定标；未锚定 findings 的 confidence 已在
    锚点阶段 ×0.5，地板对其实际值生效（两级降权协同）。
    """
    floor = settings.review_min_confidence
    if floor <= 0:
        return findings
    return [f for f in findings if float(f.get("confidence") or 0) >= floor]


def _dedupe_findings(findings: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """三级去重（§16.3）：精确三元组 → 锚点区间聚类 → 簇内软合并。

    软合并条件（任一满足）：category 相同 或 描述词重叠 ≥0.5——
    "或"兜住模型粗分类不一致；合并保留 level 更高/锚定更好者。
    """
    # 归一 category（旧格式兼容）
    for f in findings:
        f["category"] = _normalize_category(f)

    # 第 1 级：精确三元组
    seen: set[tuple] = set()
    stage1: list[dict[str, Any]] = []
    for f in findings:
        key = (f.get("file"), int(f.get("line") or 0), f.get("category"))
        if key in seen:
            continue
        seen.add(key)
        stage1.append(f)

    # 第 2 级：锚点区间聚类（同文件 + 区间重叠 → 同簇）
    by_file: dict[str, list[dict[str, Any]]] = {}
    for f in stage1:
        by_file.setdefault(str(f.get("file") or ""), []).append(f)

    out: list[dict[str, Any]] = []
    for file, items in by_file.items():
        items.sort(key=lambda f: _anchor_span(f)[0])
        clusters: list[list[dict[str, Any]]] = []
        for f in items:
            placed = False
            for cluster in clusters:
                # 与簇内任一成员区间重叠 → 入簇
                for member in cluster:
                    ms, me = _anchor_span(member)
                    fs, fe = _anchor_span(f)
                    if fs <= me and ms <= fe:
                        cluster.append(f)
                        placed = True
                        break
                if placed:
                    break
            if not placed:
                clusters.append([f])

        # 第 3 级：簇内软合并（同 category 或 词重叠 ≥0.5）
        for cluster in clusters:
            kept = sorted(
                cluster,
                key=lambda f: (_LEVEL_ORDER.get(str(f.get("level")).upper(), 5),
                               _ANCHOR_RANK.get(str(f.get("anchor_status")), 3)),
            )
            primary = kept[0]
            for other in kept[1:]:
                if (other.get("category") == primary.get("category")
                        or _similar_description(str(other.get("description") or ""), str(primary.get("description") or ""))):
                    continue  # 视为同一问题，被 primary 覆盖
                out.append(other)  # 不同问题，保留
            out.append(primary)
    return out


def _stats_of(findings: list[dict[str, Any]]) -> dict[str, int]:
    stats = {"critical": 0, "high": 0, "medium": 0, "low": 0, "suggestion": 0}
    for f in findings:
        lv = str(f.get("level", "INFO")).upper()
        if lv == "CRITICAL":
            stats["critical"] += 1
        elif lv == "HIGH":
            stats["high"] += 1
        elif lv in ("MEDIUM", "WARNING"):
            stats["medium"] += 1
        elif lv == "LOW":
            stats["low"] += 1
        else:
            stats["suggestion"] += 1
    return stats


def _risk_level(stats: dict[str, int]) -> str:
    if stats["critical"]:
        return "CRITICAL"
    if stats["high"]:
        return "HIGH"
    if stats["medium"]:
        return "MEDIUM"
    return "LOW"


def _render_comment(findings: list[dict[str, Any]], risk: str, stats: dict[str, int], summary: str) -> str:
    lines = [
        "## AI 审查摘要（Agent 引擎）", "", summary, "",
        f"**风险等级：** {risk}", f"**风险发现数：** {sum(stats.values())}", "",
        f"CRITICAL {stats['critical']} / HIGH {stats['high']} / MEDIUM {stats['medium']} / LOW {stats['low']} / INFO {stats['suggestion']}",
        "", "### 主要发现",
    ]
    for f in findings[:10]:
        lines.append(f"- [{f.get('level')}] `{f.get('file')}:{f.get('line', 0)}` {(f.get('description') or '')[:120]}")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# 装配与入口
# ---------------------------------------------------------------------------

def build_graph(checkpointer: Any = None) -> Any:
    g = StateGraph(EngineState)
    g.add_node("pre", pre_node)
    g.add_node("review_group", review_group)
    g.add_node("post_review", post_review)
    g.add_node("grace_round", grace_round)
    g.add_node("aggregate", aggregate)
    g.add_node("finish_failed", finish_failed)

    g.add_edge(START, "pre")
    # 条件边 1：PRE 失败短路 / Send() fan-out 每组一个 review_group 实例
    g.add_conditional_edges("pre", route_pre, ["review_group", "finish_failed"])
    # 汇聚：普通边（多分支条件边会逐分支评估路由，导致 aggregate 重复调度）
    g.add_edge("review_group", "post_review")
    # 条件边 2：预算闸路由（预算耗尽 ≠ 失败，走 grace_round 后正常产出部分结果）
    g.add_conditional_edges("post_review", route_after_review, {
        "aggregate": "aggregate", "grace_round": "grace_round", "finish_failed": "finish_failed",
    })
    g.add_edge("grace_round", "aggregate")
    g.add_edge("aggregate", END)
    g.add_edge("finish_failed", END)
    return g.compile(checkpointer=checkpointer)


_checkpointer: Any = None
_checkpointer_ready = False


async def _get_checkpointer() -> Any:
    global _checkpointer, _checkpointer_ready
    if _checkpointer is not None and _checkpointer_ready:
        return _checkpointer
    from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver

    url = settings.database_url.replace("postgresql+asyncpg://", "postgresql://")
    _checkpointer = AsyncPostgresSaver.from_conn_string(url)
    await _checkpointer.__aenter__()
    await _checkpointer.setup()
    _checkpointer_ready = True
    return _checkpointer


async def run_engine(
    store: ReviewJobStore, github_client: GitHubClient, job: ReviewJob,
) -> dict[str, Any]:
    """引擎入口：ReviewGraph.run 在 review_use_agent_loop=True 时委托到此处。"""
    initial: EngineState = {
        "job_id": job.job_id,
        "pr_url": job.pr_url,
        "findings": [],
        "group_summaries": [],
        "tokens_used": [],
        "budget_exceeded": [],
        "warnings": [],
        "error": None,
    }

    checkpointer = None
    if settings.database_url.startswith("postgresql"):
        try:
            checkpointer = await _get_checkpointer()
        except Exception as exc:  # checkpointer 不可用不阻塞审查（可观测降级）
            logger.warning("[ENGINE] checkpointer init failed, run without persistence: %s", exc)

    compiled = build_graph(checkpointer)
    final_state = await compiled.ainvoke(
        initial,
        {"configurable": {
            "thread_id": job.job_id,
            "store": store, "github_client": github_client, "job": job,
        }},
    )
    findings = _dedupe_findings(_apply_confidence_floor(final_state.get("findings", []) or []))
    return {
        "pr_info": final_state.get("pr_info", {}),
        "filtered_files": {
            "included_files": final_state.get("included_files", []),
            "excluded_files": final_state.get("excluded_files", []),
        },
        "parsed_diff": final_state.get("parsed_diff", []),
        "warnings": final_state.get("warnings", []),
        # Phase 4 观测元数据：由外层 review_pipeline 合入 pipeline_result（不再被覆盖丢失）
        "engine_meta": {
            "findings": findings,
            "tokens_used": sum(final_state.get("tokens_used", []) or [0]),
            "group_summaries": final_state.get("group_summaries", []) or [],
            "llm_request_count": len(final_state.get("llm_records", []) or []),
        },
    }
