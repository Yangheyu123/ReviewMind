"""ReviewAgent：组内原生 tool-calling 审查循环。

机制（对齐改进方案 §4 / 阿里 OCR llmloop 范式）：
- 循环：LLM 发 tool_calls → 工具执行 → 结果回灌 → 直到 task_done 或上限；
- 无 tool_calls 的回复 → 注入重试提示（最多 2 次）后终止；
- 连续失败三级降级（tool_failure_streak）：第 1 次原样报错、第 2 次点名警告、
  第 3 次起伪装"已跳过"终结该工具；
- 预算闸：每轮累加 usage token，超限触发 grace round——工具面收缩为
  submit_finding/task_done，注入 "FINAL ROUND" 抢救未提交发现；
- 硬上限：单组工具调用 ≤ MAX_TOOL_CALLS。
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any

from app.agent_loop.agent_tools_v2 import build_tool_registry
from app.agent_loop.tool_context import ToolContext
from app.agent_loop.tools import ToolRegistry
from app.core.llm import LLMClient, LLMClientError, ToolCallResponse

logger = logging.getLogger(__name__)

MAX_TOOL_CALLS = 100          # 单组工具调用硬上限
# 三区上下文压缩阈值（Phase 4，对齐 OCR compression.go 范式）：
# 软阈 60% 触发压缩；硬阈 80% 压缩后仍超则终止（产出已提交的 findings）
COMPRESSION_SOFT_RATIO = 0.6
COMPRESSION_HARD_RATIO = 0.8
ACTIVE_ZONE_MESSAGES = 6      # active 区保留的最近消息条数（含完整工具往返）


def estimate_tokens(messages: list[dict[str, Any]]) -> int:
    """粗估消息列表 token 数（字符/4）。有真实 usage 时由调用方取 max 校准。"""
    total_chars = 0
    for m in messages:
        total_chars += len(str(m.get("content") or ""))
        for tc in m.get("tool_calls") or []:
            total_chars += len(str(tc.get("function", {}).get("arguments") or ""))
    return total_chars // 4


def compression_action(est_tokens: int, budget: int) -> str:
    """判定压缩动作：none / soft（60%）/ hard（80%）。"""
    if est_tokens > budget * COMPRESSION_HARD_RATIO:
        return "hard"
    if est_tokens > budget * COMPRESSION_SOFT_RATIO:
        return "soft"
    return "none"


def compress_messages(messages: list[dict[str, Any]], keep_last: int = ACTIVE_ZONE_MESSAGES) -> list[dict[str, Any]]:
    """三区压缩：frozen（system+首 user，永不动）+ 中段摘要 + active（最近完整轮次）。

    中段摘要为确定性摘要（工具调用名 + 结果截断），不额外消耗 LLM 调用——
    在审查场景里中间轮次的价值主要是"做过什么"，细节已在 findings 里。
    """
    if len(messages) <= keep_last + 2:
        return messages
    frozen = messages[:2]
    active = messages[-keep_last:]
    middle = messages[2:-keep_last]
    digest: list[str] = []
    for m in middle:
        role = m.get("role")
        if role == "assistant" and m.get("tool_calls"):
            names = [str(tc.get("function", {}).get("name", "?")) for tc in m["tool_calls"]]
            digest.append(f"- assistant 调用工具: {', '.join(names)}")
        elif role == "tool":
            digest.append(f"- 工具结果（已归档）: {str(m.get('content', ''))[:150]}")
        elif role == "assistant":
            digest.append(f"- assistant: {str(m.get('content', ''))[:150]}")
        elif role == "user":
            digest.append(f"- 补充提示: {str(m.get('content', ''))[:100]}")
    summary_msg = {
        "role": "user",
        "content": (
            "<previous_review_summary>（历史轮次已压缩，细节见已提交 findings）"
            + "\n" + "\n".join(digest[:80])
        ),
    }
    return frozen + [summary_msg] + active


MAX_TOOL_CALLS_GUARD = MAX_TOOL_CALLS  # 兼容引用
MAX_EMPTY_ROUNDS = 2          # 连续无 tool_calls 的重试上限
GRACE_TOOLS_ALLOWLIST = ("submit_finding", "task_done")  # 最后一轮只留提交/终止


@dataclass
class AgentRunResult:
    findings: list[dict[str, Any]] = field(default_factory=list)
    summary: str = ""
    tokens_used: int = 0
    tool_calls_made: int = 0
    stop_reason: str = ""          # task_done / max_tool_calls / empty_rounds / budget_exceeded / llm_error
    transcript: list[dict[str, Any]] = field(default_factory=list)  # 取证轨迹（前端展示用）


class _FailureStreak:
    """按工具名计连续失败：1 原样报错 → 2 点名警告 → 3+ 伪装已跳过。"""

    def __init__(self) -> None:
        self._counts: dict[str, int] = {}

    def record(self, tool_name: str, failed: bool) -> None:
        if failed:
            self._counts[tool_name] = self._counts.get(tool_name, 0) + 1
        else:
            self._counts.pop(tool_name, None)

    def wrap_result(self, tool_name: str, success: bool, error_text: str) -> tuple[bool, str]:
        """返回 (最终 success, 给 LLM 的 error 文案)。"""
        streak = self._counts.get(tool_name, 0)
        if success:
            return True, ""
        if streak == 2:
            return False, f"{error_text} — 这是连续第二次失败，请修正参数，或改用其它工具，或调用 task_done 结束"
        if streak >= 3:
            logger.warning("[AGENT] tool %s failed %d times consecutively, skipping", tool_name, streak)
            return True, ""  # 伪装成功（observation 由调用方写成"已跳过"）
        return False, error_text


class ReviewAgent:
    """单组审查员：在其上下文内自主决定取证路径。"""

    def __init__(
        self,
        llm: LLMClient,
        ctx: ToolContext,
        *,
        model: str | None = None,
        budget_tokens: int = 200_000,
        group_label: str = "",
        context_budget: int | None = None,
    ) -> None:
        self._llm = llm
        self._ctx = ctx
        self._model = model
        self._budget = budget_tokens
        from app.core.config import settings as _settings
        self._context_budget = context_budget or _settings.agent_context_budget_tokens
        self._registry, self._tools = build_tool_registry(ctx, group_label)

    # ------------------------------------------------------------------

    async def run(self, system_prompt: str, user_prompt: str) -> AgentRunResult:
        result = AgentRunResult()
        messages: list[dict[str, Any]] = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]
        streak = _FailureStreak()
        empty_rounds = 0
        grace_triggered = False

        while result.tool_calls_made < MAX_TOOL_CALLS:
            # Phase 4 三区压缩闸：60% 软压 / 80% 压缩后仍超则终止（产出已提交 findings）
            est = max(estimate_tokens(messages), result.tokens_used)
            action = compression_action(est, self._context_budget)
            if action in ("soft", "hard"):
                before = len(messages)
                messages = compress_messages(messages)
                result.transcript.append({"type": "compression", "trigger": action,
                                          "messages_before": before, "messages_after": len(messages)})
                if action == "hard" and compression_action(estimate_tokens(messages), self._context_budget) == "hard":
                    result.stop_reason = "compression_exceeded"
                    break

            # 预算闸：超限且尚未触发过 grace → 收缩工具面 + FINAL ROUND 提示
            schemas = self._registry.to_openai_schemas()
            if result.tokens_used >= self._budget and not grace_triggered:
                grace_triggered = True
                schemas = [
                    s for s in schemas
                    if s["function"]["name"] in GRACE_TOOLS_ALLOWLIST
                ]
                messages.append({
                    "role": "user",
                    "content": (
                        "TOKEN BUDGET EXCEEDED. This is your FINAL round: only submit_finding and "
                        "task_done are available. Submit any findings you have not yet reported, then "
                        "call task_done.",
                    ),
                })
                result.transcript.append({"type": "grace_round", "tokens_used": result.tokens_used})

            try:
                resp: ToolCallResponse = await self._llm.chat_with_tools(
                    messages, tools=schemas, model=self._model,
                )
            except LLMClientError as exc:
                result.stop_reason = f"llm_error: {exc}"
                break

            if resp.usage:
                result.tokens_used += resp.total_tokens

            # 无 tool_calls → 重试提示，连续 2 次则终止
            if not resp.has_tool_calls:
                empty_rounds += 1
                if empty_rounds >= MAX_EMPTY_ROUNDS:
                    result.stop_reason = "empty_rounds"
                    break
                messages.append({"role": "assistant", "content": resp.content or ""})
                messages.append({
                    "role": "user",
                    "content": "You did not successfully call any tools. Continue the review using the tools.",
                })
                continue
            empty_rounds = 0

            # 回灌 assistant 消息（原生 tool_calls + 推理模型 reasoning_content）
            assistant_msg: dict[str, Any] = {"role": "assistant", "content": resp.content or ""}
            if resp.reasoning_content:
                assistant_msg["reasoning_content"] = resp.reasoning_content
            assistant_msg["tool_calls"] = [
                {
                    "id": tc.id,
                    "type": "function",
                    "function": {"name": tc.name, "arguments": _dump_args(tc.arguments)},
                }
                for tc in resp.tool_calls
            ]
            messages.append(assistant_msg)

            done_requested = False
            for tc in resp.tool_calls:
                result.tool_calls_made += 1
                if result.tool_calls_made > MAX_TOOL_CALLS:
                    result.stop_reason = "max_tool_calls"
                    break

                tool = self._registry.get(tc.name)
                if tool is None:
                    observation = f"[tool:{tc.name}] ERROR: unknown tool"
                    streak.record(tc.name, True)
                    _append_tool_result(messages, tc.id, tc.name, observation)
                    continue

                tr = await tool.invoke(tc.arguments)
                logger.info("[AGENT] tool %s ok=%s args=%s", tc.name, tr.success, str(tc.arguments)[:120])
                ok, err = streak.wrap_result(tc.name, tr.success, tr.error)
                if tr.success:
                    observation = tr.to_llm_friendly()
                elif ok and not err:
                    observation = f"[tool:{tc.name}] SKIPPED: this tool has failed repeatedly and its result is skipped; move on."
                else:
                    observation = f"[tool:{tc.name}] ERROR: {err}"
                result.transcript.append({
                    "type": "tool_call", "name": tc.name, "args": tc.arguments,
                    "ok": tr.success, "output_brief": _brief(tr.output),
                })

                if tc.name == "task_done":
                    done_requested = True
                    result.summary = getattr(tool, "summary", "") or result.summary
                _append_tool_result(messages, tc.id, tc.name, observation)

            if done_requested:
                result.stop_reason = "task_done"
                break
            if result.tool_calls_made >= MAX_TOOL_CALLS:
                result.stop_reason = "max_tool_calls"
                break
            if grace_triggered:
                # FINAL ROUND 只给一轮提交机会
                result.stop_reason = "budget_exceeded"
                break

        submit_tool = self._tools.get("submit_finding")
        if submit_tool is not None and submit_tool.last_summary:
            result.summary = submit_tool.last_summary
        result.findings = list(self._ctx.findings)
        return result


def _dump_args(args: dict[str, Any]) -> str:
    import json
    try:
        return json.dumps(args, ensure_ascii=False)
    except (TypeError, ValueError):
        return "{}"


def _append_tool_result(messages: list[dict[str, Any]], call_id: str, name: str, observation: str) -> None:
    messages.append({"role": "tool", "tool_call_id": call_id or name, "content": observation[:4000]})


def _brief(output: Any) -> str:
    import json
    try:
        s = json.dumps(output, ensure_ascii=False, default=str)
    except (TypeError, ValueError):
        s = str(output)
    return s[:200]
