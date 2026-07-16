"""Debate Agent：对单条 finding 做解释或重新评估。

开发者通过 /explain 命令对某条 finding 提出异议后，本 Agent 基于
「代码上下文 + 最佳实践 + 历史对话」给出 keep/downgrade/dismiss 裁决。

复用简单 LLM 调用模式（仿 security_agent），不走 Agent Loop。
"""

from __future__ import annotations

import logging
from typing import Any

from app.agents.json_utils import try_parse_json
from app.agents.prompts import DEBATE_SYSTEM, build_debate_user_prompt
from app.core.llm import LLMClientError, llm_client
from app.schemas.review import DebateResult, DebateVerdict

logger = logging.getLogger(__name__)

# level 排序，用于校验 downgrade 的新等级确实更低
_LEVEL_ORDER = {"INFO": 0, "LOW": 1, "MEDIUM": 2, "HIGH": 3, "CRITICAL": 4}


async def run_async(
    *,
    finding: dict[str, Any],
    challenge: str,
    code_context: str,
    tech_stack_prompt: str = "",
    history: list[dict[str, Any]] | None = None,
) -> DebateResult:
    """异步版本：优先 LLM，失败降级为 verdict=keep。"""
    history = history or []
    if llm_client.is_configured and not llm_client._mock_mode:
        try:
            messages = [
                {"role": "system", "content": DEBATE_SYSTEM},
                {
                    "role": "user",
                    "content": build_debate_user_prompt(
                        finding=finding,
                        challenge=challenge,
                        code_context=code_context,
                        tech_stack_prompt=tech_stack_prompt,
                        history=history,
                    ),
                },
            ]
            logger.info(
                "[DEBATE_AGENT] Calling LLM... finding=%s history=%d",
                finding.get("id"),
                len(history),
            )
            raw = await llm_client.chat(messages, model=None, temperature=0.0)
            result = _parse_debate_result(raw, original_level=str(finding.get("level", "")))
            logger.info(
                "[DEBATE_AGENT] LLM OK | verdict=%s revised_level=%s",
                result.verdict,
                result.revised_level,
            )
            return result
        except (LLMClientError, Exception) as exc:  # noqa: BLE001
            logger.warning(
                "[DEBATE_AGENT] LLM failed, fallback to keep | %s: %s",
                type(exc).__name__,
                exc,
            )

    # 降级：不阻塞闭环，默认维持原结论
    return DebateResult(
        explanation="辩论 Agent 暂时不可用，已维持原审查结论。请在稍后重试 `/explain`。",
        verdict=DebateVerdict.keep,
        confidence=0.0,
    )


def _parse_debate_result(raw: str, *, original_level: str) -> DebateResult:
    """解析 LLM 输出为 DebateResult，对非法值做容错。"""
    parsed = try_parse_json(raw)
    verdict_raw = str(parsed.get("verdict", "keep")).lower()
    try:
        verdict = DebateVerdict(verdict_raw)
    except ValueError:
        verdict = DebateVerdict.keep

    revised_level = parsed.get("revised_level")
    revised_level = str(revised_level).upper() if revised_level else None

    # downgrade 必须给出比原等级更低的新等级，否则退化为 keep
    if verdict == DebateVerdict.downgrade:
        if not revised_level or _LEVEL_ORDER.get(revised_level, -1) >= _LEVEL_ORDER.get(
            original_level.upper(), -1
        ):
            verdict = DebateVerdict.keep
            revised_level = None

    # 非 downgrade 时强制清空 revised_level
    if verdict != DebateVerdict.downgrade:
        revised_level = None

    explanation = str(parsed.get("explanation", "")).strip()
    if not explanation:
        explanation = "辩论 Agent 未返回有效解释，已维持原审查结论。"

    try:
        confidence = float(parsed.get("confidence", 0.5))
    except (TypeError, ValueError):
        confidence = 0.5
    confidence = max(0.0, min(1.0, confidence))

    return DebateResult(
        explanation=explanation,
        verdict=verdict,
        revised_level=revised_level,
        confidence=confidence,
    )
