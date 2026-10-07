"""Review Graph 节点：每个节点读写 ReviewGraphState 的一部分。"""

import asyncio
import logging
from typing import Any, Callable, Coroutine

from app.agents import (
    performance_agent,
    report_agent,
    risk_judge_agent,
    security_agent,
    summary_agent,
    test_agent,
)
from app.graph.state import ReviewGraphState
from app.schemas.agents import AgentContext
from app.schemas.ast_context import AstContext
from app.schemas.diff import PullRequestFile
from app.schemas.github import GitHubPullRequestFile
from app.services.ast_context import extract_ast_context
from app.services.code_retriever import CodeRetriever
from app.services.diff_filter import filter_diff_files
from app.services.diff_parser import parse_diff_file
from app.services.github_client import GitHubClient
from app.services.github_url_parser import parse_github_pr_url
from app.services.rag_trigger import (
    RagTriggerLevel,
    evaluate_rag_trigger,
    get_risk_file_paths,
)
from app.services.review_job_store import ReviewJobStore

from app.services.tech_stack_analyzer import analyze_tech_stack

logger = logging.getLogger(__name__)


def _run_async(coro: Callable[..., Coroutine], *args, **kwargs) -> Any:
    """在同步上下文中安全地执行异步函数。"""
    result_holder: list[Any] = []
    exception_holder: list[Exception] = []

    async def _runner() -> None:
        try:
            result_holder.append(await coro(*args, **kwargs))
        except Exception as exc:
            exception_holder.append(exc)

    try:
        loop = asyncio.get_event_loop()
    except RuntimeError:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)

    if loop.is_running():
        # 已在事件循环中，创建 task 并等待
        import concurrent.futures
        future = asyncio.run_coroutine_threadsafe(_runner(), loop)
        future.result(timeout=120)
    else:
        loop.run_until_complete(_runner())

    if exception_holder:
        raise exception_holder[0]
    return result_holder[0] if result_holder else None


async def node_fetch_pr(state: ReviewGraphState, github_client: GitHubClient, store: ReviewJobStore) -> ReviewGraphState:
    """节点 1：解析 PR URL 并拉取 GitHub PR 基本信息。"""
    try:
        pr_ref = parse_github_pr_url(state.pr_url)
        pr_info = await github_client.fetch_pull_request(pr_ref)
        state.pr_info = pr_info.model_dump(mode="json")
        await store.save_pr_info(state.job_id, state.pr_info)
    except Exception as exc:
        state.error = f"FETCH_PR: {exc}"
    return state


async def node_fetch_pr_async(state: ReviewGraphState, github_client: GitHubClient, store: ReviewJobStore) -> ReviewGraphState:
    """节点 1（异步版）：解析 PR URL 并拉取 GitHub PR 基本信息。"""
    try:
        pr_ref = parse_github_pr_url(state.pr_url)
        pr_info = await github_client.fetch_pull_request(pr_ref)
        state.pr_info = pr_info.model_dump(mode="json")
        await store.save_pr_info(state.job_id, state.pr_info)
    except Exception as exc:
        state.error = f"FETCH_PR: {exc}"
    return state


async def node_fetch_files_async(state: ReviewGraphState, github_client: GitHubClient, _store: ReviewJobStore) -> ReviewGraphState:
    """节点 2：拉取 PR changed files。"""
    if state.error:
        return state
    try:
        pr_ref = parse_github_pr_url(state.pr_url)
        files = await github_client.fetch_pull_request_files(pr_ref)
        state.github_files = [f.model_dump(mode="json") for f in files]
    except Exception as exc:
        state.error = f"FETCH_FILES: {exc}"
    return state


def node_diff_filter(state: ReviewGraphState, _github_client: GitHubClient, _store: ReviewJobStore) -> ReviewGraphState:
    """节点 3：过滤无意义的 diff 文件。"""
    if state.error:
        return state
    try:
        pull_request_files = [
            PullRequestFile(
                filename=f["filename"],
                status=f["status"],
                additions=f.get("additions", 0),
                deletions=f.get("deletions", 0),
                patch=f.get("patch"),
            )
            for f in state.github_files
        ]
        filtered = filter_diff_files(pull_request_files)
        state.filtered_files = filtered.model_dump(mode="json")
    except Exception as exc:
        state.error = f"DIFF_FILTER: {exc}"
    return state


def node_parse_diff(state: ReviewGraphState, _github_client: GitHubClient, _store: ReviewJobStore) -> ReviewGraphState:
    """节点 4：解析每个 included file 的 patch 为结构化 hunks 和 changed lines。"""
    if state.error:
        return state
    try:
        included = state.filtered_files.get("included_files", [])
        parsed = []
        for f in included:
            pf = PullRequestFile(
                filename=f["filename"],
                status=f["status"],
                additions=f.get("additions", 0),
                deletions=f.get("deletions", 0),
                patch=f.get("patch"),
            )
            parsed.append(parse_diff_file(pf).model_dump(mode="json"))
        state.parsed_diff = parsed
    except Exception as exc:
        state.error = f"DIFF_PARSE: {exc}"
    return state


async def node_ast_context(state: ReviewGraphState, github_client: GitHubClient, _store: ReviewJobStore) -> ReviewGraphState:
    """节点 5：对支持的源码文件提取 AST 上下文。非关键节点，失败降级。

    优先拉取 head 版本真实文件内容做 AST（changed_lines 即新文件行号，天然对齐）；
    拉取失败时降级为 patch 重建源码——对修改型文件重建文本不完整，ast.parse
    常失败，是 AST 命中率不稳的历史主因。
    """
    if state.error:
        return state
    try:
        from app.schemas.github import GitHubPullRequestRef
        from app.services.ast_context import detect_language
        from app.services.github_url_parser import parse_github_pr_url

        contexts: list[dict] = []
        included = state.filtered_files.get("included_files", [])
        file_map = {f["filename"]: f for f in included}

        pr_ref = None
        head_sha = (state.pr_info.get("head") or {}).get("sha") if state.pr_info else None

        for diff in state.parsed_diff:
            filename = diff.get("file", "")
            changed_lines = diff.get("changed_lines", [])
            f_info = file_map.get(filename, {})
            patch = f_info.get("patch", "")
            if not changed_lines or detect_language(filename) == "unknown":
                continue

            source: str | None = None
            # 路径 1：拉取 head 真实源码
            if head_sha:
                try:
                    if pr_ref is None:
                        pr_ref = parse_github_pr_url(state.pr_url)
                    source = await github_client.fetch_file_content(pr_ref, filename, head_sha)
                except Exception:
                    source = None
            # 路径 2：patch 重建（降级）
            if not source:
                source = _extract_source_from_patch(patch)
            if not source:
                continue

            try:
                ctx_list = extract_ast_context(filename, source, changed_lines)
                contexts.extend([c.model_dump(mode="json") for c in ctx_list])
            except Exception:
                pass  # AST 解析失败降级为无上下文
        state.ast_contexts = contexts
    except Exception as exc:
        state.warnings.append(f"AST_CONTEXT: {exc}")
    return state


async def node_summary_agent(state: ReviewGraphState, _github_client: GitHubClient, _store: ReviewJobStore) -> ReviewGraphState:
    """节点 6：Summary Agent 生成总体摘要（调用 LLM）。"""
    if state.error:
        return state
    try:
        ctx = AgentContext(
            pr_info=state.pr_info,
            parsed_diff=state.parsed_diff,
            ast_contexts=state.ast_contexts,
            rag_contexts=state.rag_contexts,
            tech_stack_prompt=state.tech_stack_prompt,
        )
        result = await summary_agent.run_async(ctx)
        state.summary_text = result.summary
        state.agent_results.append(result)
        # SSE 推送摘要文本流
        if result.summary:
            await _store.add_progress_event(
                state.job_id,
                {"type": "chunk", "target": "summary", "content": result.summary},
            )
    except Exception as exc:
        try:
            result = summary_agent.run(ctx)
            state.summary_text = result.summary
            state.agent_results.append(result)
        except Exception:
            pass
        state.warnings.append(f"SUMMARY_AGENT: {exc}")
        if not state.summary_text:
            state.summary_text = f"PR 涉及 {len(state.parsed_diff)} 个文件变更。"
    return state


async def node_security_agent(state: ReviewGraphState, _github_client: GitHubClient, _store: ReviewJobStore) -> ReviewGraphState:
    """节点 7：Security Agent（调用 LLM 分析安全风险）。"""
    if state.error:
        return state
    try:
        ctx = AgentContext(
            pr_info=state.pr_info,
            parsed_diff=state.parsed_diff,
            ast_contexts=state.ast_contexts,
            rag_contexts=state.rag_contexts,
            tech_stack_prompt=state.tech_stack_prompt,
        )
        result = await security_agent.run_async(ctx)
        state.agent_results.append(result)
        # SSE 推送 finding 事件（"type":"finding" 放最后，避免被 f.type 覆盖）
        for f in result.findings:
            await _store.add_progress_event(
                state.job_id,
                {
                    "id": f.id, "agent": f.agent, "file": f.file, "line": f.line,
                    "symbol": f.symbol or "", "level": f.level, "finding_type": f.type,
                    "confidence": f.confidence, "description": f.description,
                    "suggestion": f.suggestion, "type": "finding",
                },
            )
    except Exception as exc:
        try:
            result = security_agent.run(ctx)
            state.agent_results.append(result)
        except Exception:
            pass
        state.warnings.append(f"SECURITY_AGENT: {exc}")
    return state


async def node_performance_agent(state: ReviewGraphState, _github_client: GitHubClient, _store: ReviewJobStore) -> ReviewGraphState:
    """节点 8：Performance Agent（调用 LLM 分析性能风险）。"""
    if state.error:
        return state
    try:
        ctx = AgentContext(
            pr_info=state.pr_info,
            parsed_diff=state.parsed_diff,
            ast_contexts=state.ast_contexts,
            rag_contexts=state.rag_contexts,
            tech_stack_prompt=state.tech_stack_prompt,
        )
        result = await performance_agent.run_async(ctx)
        state.agent_results.append(result)
        for f in result.findings:
            await _store.add_progress_event(
                state.job_id,
                {
                    "id": f.id, "agent": f.agent, "file": f.file, "line": f.line,
                    "symbol": f.symbol or "", "level": f.level, "finding_type": f.type,
                    "confidence": f.confidence, "description": f.description,
                    "suggestion": f.suggestion, "type": "finding",
                },
            )
    except Exception as exc:
        try:
            result = performance_agent.run(ctx)
            state.agent_results.append(result)
        except Exception:
            pass
        state.warnings.append(f"PERFORMANCE_AGENT: {exc}")
    return state


async def node_test_agent(state: ReviewGraphState, _github_client: GitHubClient, _store: ReviewJobStore) -> ReviewGraphState:
    """节点 9：Test Agent（调用 LLM 分析测试覆盖）。"""
    if state.error:
        return state
    try:
        ctx = AgentContext(
            pr_info=state.pr_info,
            parsed_diff=state.parsed_diff,
            ast_contexts=state.ast_contexts,
            rag_contexts=state.rag_contexts,
            tech_stack_prompt=state.tech_stack_prompt,
        )
        result = await test_agent.run_async(ctx)
        state.agent_results.append(result)
        for f in result.findings:
            await _store.add_progress_event(
                state.job_id,
                {
                    "id": f.id, "agent": f.agent, "file": f.file, "line": f.line,
                    "symbol": f.symbol or "", "level": f.level, "finding_type": f.type,
                    "confidence": f.confidence, "description": f.description,
                    "suggestion": f.suggestion, "type": "finding",
                },
            )
    except Exception as exc:
        try:
            result = test_agent.run(ctx)
            state.agent_results.append(result)
        except Exception:
            pass
        state.warnings.append(f"TEST_AGENT: {exc}")
    return state


async def node_risk_judge(state: ReviewGraphState, _github_client: GitHubClient, _store: ReviewJobStore) -> ReviewGraphState:
    """节点 10：Risk Judge 聚合所有 agent findings。"""
    if state.error:
        return state
    try:
        state.aggregated_risk = risk_judge_agent.aggregate(state.agent_results)
        # SSE 推送最终风险统计
        if state.aggregated_risk:
            await _store.add_progress_event(
                state.job_id,
                {
                    "type": "chunk", "target": "report",
                    "content": f"Risk Judge 完成：{len(state.aggregated_risk.findings)} 个风险，去重 {state.aggregated_risk.dedup_count} 个，风险等级 {state.aggregated_risk.risk_level}",
                },
            )
    except Exception as exc:
        state.warnings.append(f"RISK_JUDGE: {exc}")
    return state


def node_report_agent(state: ReviewGraphState, _github_client: GitHubClient, _store: ReviewJobStore) -> ReviewGraphState:
    """节点 11：Report Agent 生成最终报告。"""
    if state.error:
        return state
    try:
        if state.aggregated_risk is None:
            return state
        state.report_output = report_agent.generate_report(
            state.job_id,
            state.aggregated_risk,
            state.summary_text,
        )
    except Exception as exc:
        state.warnings.append(f"REPORT_AGENT: {exc}")
    return state


def _extract_source_from_patch(patch: str) -> str | None:
    """尝试从 patch 中提取全部 "+" 行作为源码的近似值（降级方案）。"""
    lines = []
    for line in patch.splitlines():
        if line.startswith("+") and not line.startswith("+++"):
            lines.append(line[1:])
        elif not line.startswith("-") and not line.startswith("@@") and not line.startswith("---"):
            lines.append(line)
    return "\n".join(lines) if lines else None


async def node_rag_context(state: ReviewGraphState, _github_client: GitHubClient, store: ReviewJobStore) -> ReviewGraphState:
    """节点 5.5：RAG 分级触发 + 语义检索（在 AST 之后、agents 之前执行）。

    非关键节点，失败降级为跳过 RAG。
    """
    if state.error:
        return state

    try:
        # 1. 读取 enable_rag 配置
        enable_rag = state.config.get("enable_rag", True)
        if not enable_rag:
            state.rag_level = 0
            state.rag_trigger_reason = "RAG disabled by config"
            logger.info("[RAG_CONTEXT] Skipped: %s", state.rag_trigger_reason)
            return state

        # 2. 评估触发级别
        level, reason = evaluate_rag_trigger(
            pr_info=state.pr_info,
            filtered_files=state.filtered_files,
            parsed_diff=state.parsed_diff,
            enable_rag=enable_rag,
        )
        state.rag_level = level.value
        state.rag_trigger_reason = reason

        await store.add_progress_event(
            state.job_id,
            {"type": "progress", "step": "RAG_CONTEXT", "percent": 64, "message": f"RAG 级别: {level.name} | {reason}"},
        )

        if level == RagTriggerLevel.NONE:
            logger.info("[RAG_CONTEXT] Skipped (NONE): %s", reason)
            return state

        # 3. 提取 repo_url 用于检索
        repo_url = ""
        if state.pr_info:
            owner = state.pr_info.get("owner", "")
            repo = state.pr_info.get("repo", "")
            if owner and repo:
                repo_url = f"https://github.com/{owner}/{repo}"

        if not repo_url:
            logger.warning("[RAG_CONTEXT] Cannot determine repo_url, skipping")
            return state

        # 4. 提取查询文本：对每个变更文件取其 patch/AST 代码作为查询
        from app.core.config import settings as app_settings
        top_k = app_settings.rag_top_k_light if level == RagTriggerLevel.LIGHT else app_settings.rag_top_k_full

        retriever = CodeRetriever(
            repo_url=repo_url,
            top_k=top_k,
            max_snippet_chars=app_settings.rag_max_snippet_chars,
            cache_ttl_seconds=app_settings.rag_cache_ttl_seconds,
        )

        # 收集需要检索的文件路径
        all_file_paths = [diff.get("file", "") for diff in state.parsed_diff if diff.get("file")]
        if level == RagTriggerLevel.LIGHT:
            # 仅对风险文件检索
            query_files = get_risk_file_paths(all_file_paths)
            if not query_files:
                logger.info("[RAG_CONTEXT] LIGHT mode: no risk files found, skipping retrieval")
                return state
        else:
            query_files = all_file_paths

        # 5. 对每个文件执行语义检索
        all_rag_contexts: list[dict] = []
        for file_path in query_files[:10]:  # 最多检索 10 个文件
            # 构建查询：文件路径 + AST symbol
            query = file_path
            # 尝试找到对应的 AST context
            for ctx in state.ast_contexts:
                if ctx.get("file") == file_path:
                    symbol = ctx.get("symbol", "")
                    if symbol:
                        query = f"{file_path}:{symbol}"
                    code = ctx.get("code", "")
                    if code:
                        query = code[:500]  # 用实际代码作为查询更准确
                    break

            try:
                docs = await retriever.aretrieve(query)
                for doc in docs:
                    all_rag_contexts.append({
                        "file_path": doc.metadata.get("file_path", file_path),
                        "symbol": doc.metadata.get("symbol"),
                        "language": doc.metadata.get("language", ""),
                        "code": doc.page_content,
                        "similarity": doc.metadata.get("similarity", 0),
                        "query_file": file_path,
                    })
            except Exception as exc:
                logger.warning("[RAG_CONTEXT] Retrieve failed for %s: %s", file_path, exc)
                continue

        # 按相似度降序排序，去重
        seen_keys: set[tuple[str, str]] = set()
        unique_contexts: list[dict] = []
        for ctx in sorted(all_rag_contexts, key=lambda x: x.get("similarity", 0), reverse=True):
            key = (ctx["file_path"], ctx.get("symbol") or "")
            if key not in seen_keys:
                seen_keys.add(key)
                unique_contexts.append(ctx)

        state.rag_contexts = unique_contexts
        logger.info("[RAG_CONTEXT] Retrieved %d unique contexts (level=%s, files=%d)",
                    len(unique_contexts), level.name, len(query_files))

        await store.add_progress_event(
            state.job_id,
            {"type": "progress", "step": "RAG_CONTEXT_DONE", "percent": 65, "message": f"RAG 检索完成: {len(unique_contexts)} 条相似代码"},
        )

    except Exception as exc:
        logger.warning("[RAG_CONTEXT] Failed, degrading: %s", exc)
        state.warnings.append(f"RAG_CONTEXT: {exc}")
        state.rag_level = 0
        state.rag_trigger_reason = f"RAG degraded due to error: {exc}"

    return state


# ---------------------------------------------------------------------------
# Tech Stack Analysis — 技术栈推断 → 框架安全上下文
# ---------------------------------------------------------------------------

def node_tech_stack_analysis(state: ReviewGraphState, _github_client: GitHubClient, _store: ReviewJobStore) -> ReviewGraphState:
    """节点 5.6：从 changed files 推断技术栈，生成框架安全上下文 prompt。

    在 rag_context 之后、agents 之前执行。
    非关键节点，失败降级为空 prompt。
    """
    if state.error:
        return state
    try:
        profile = analyze_tech_stack(state.github_files)
        prompt = profile.to_prompt_block()
        state.tech_stack_prompt = prompt
        if prompt:
            logger.info("[TECH_STACK] Detected framework context: %s", prompt[:200])
        else:
            logger.info("[TECH_STACK] No framework-specific security context detected")
    except Exception as exc:
        logger.warning("[TECH_STACK] Analysis failed, degrading: %s", exc)
        state.warnings.append(f"TECH_STACK: {exc}")
    return state


# ---------------------------------------------------------------------------
# Finding Validator — 基于技术栈过滤明显误报
# ---------------------------------------------------------------------------

# 每条规则：(tech_stack_prompt 中的关键词, finding.type 匹配模式列表, 可选: file 扩展名限制)
_FALSE_POSITIVE_RULES: list[tuple[str, list[str], list[str] | None]] = [
    # React JSX 自动转义 → 前端文件中的 XSS 误报
    ("React", ["xss", "cross-site scripting"], [".tsx", ".jsx", ".js", ".ts"]),
    # SQLAlchemy ORM 不拼接 SQL → SQL 注入误报
    ("SQLAlchemy ORM", ["sql_injection", "sql injection", "sqli"], None),
    # Pydantic 自动类型验证 → 输入校验误报
    ("Pydantic", ["input_validation", "input validation"], None),
    # Django 模板自动转义 → XSS 误报
    ("Django 模板", ["xss", "cross-site scripting"], [".html", ".django"]),
]


def node_finding_validator(state: ReviewGraphState, _github_client: GitHubClient, _store: ReviewJobStore) -> ReviewGraphState:
    """节点 10.5：基于技术栈上下文，过滤 Agent findings 中的明显误报。

    在所有 Agent 之后、risk_judge 之前执行。
    只做确定性规则过滤，不做 LLM 调用。
    """
    if state.error or not state.tech_stack_prompt:
        return state

    total_before = 0
    total_after = 0

    for result in state.agent_results:
        original = result.findings
        total_before += len(original)
        filtered: list[Any] = []
        for f in original:
            if _is_false_positive(f, state.tech_stack_prompt):
                logger.info(
                    "[FINDING_VALIDATOR] Dropped finding: agent=%s file=%s type=%s desc=%.80s",
                    f.agent, f.file, f.type, f.description,
                )
                continue
            filtered.append(f)
        result.findings = filtered
        total_after += len(filtered)

    dropped = total_before - total_after
    state.validated_findings_dropped = dropped
    if dropped > 0:
        logger.info("[FINDING_VALIDATOR] Dropped %d/%d findings as false positives", dropped, total_before)
    return state


def _is_false_positive(finding: Any, tech_stack_prompt: str) -> bool:
    """判断单个 finding 是否为基于技术栈的误报。"""
    fp_type = finding.type.lower()
    fp_file = finding.file.lower()
    for keyword, type_patterns, ext_restrictions in _FALSE_POSITIVE_RULES:
        if keyword.lower() not in tech_stack_prompt.lower():
            continue
        # finding.type 是否匹配任一模式
        if not any(pat in fp_type for pat in type_patterns):
            continue
        # 如果有扩展名限制，检查 file 后缀
        if ext_restrictions is not None:
            if not any(fp_file.endswith(ext) for ext in ext_restrictions):
                continue
        return True
    return False