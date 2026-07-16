import pytest

from app.agents import debate_agent
from app.schemas.review import DebateVerdict


def _finding():
    return {
        "id": "sec_abc",
        "file": "src/db.py",
        "line": 10,
        "level": "HIGH",
        "type": "sql_injection",
        "description": "疑似 SQL 注入",
        "suggestion": "使用参数化查询",
    }


def test_parse_keep_verdict(monkeypatch) -> None:
    raw = """
    {"explanation":"异议不成立，字符串拼接确实存在注入风险。","verdict":"keep","revised_level":null,"confidence":0.8}
    """
    result = debate_agent._parse_debate_result(raw, original_level="HIGH")
    assert result.verdict == DebateVerdict.keep
    assert result.revised_level is None
    assert "异议不成立" in result.explanation


def test_parse_dismiss_verdict(monkeypatch) -> None:
    raw = """
    {"explanation":"该处已使用参数化查询，属误报。","verdict":"dismiss","revised_level":null,"confidence":0.9}
    """
    result = debate_agent._parse_debate_result(raw, original_level="HIGH")
    assert result.verdict == DebateVerdict.dismiss
    assert result.revised_level is None


def test_parse_downgrade_verdict_lower_level(monkeypatch) -> None:
    raw = """
    {"explanation":"风险存在但影响有限。","verdict":"downgrade","revised_level":"LOW","confidence":0.7}
    """
    result = debate_agent._parse_debate_result(raw, original_level="HIGH")
    assert result.verdict == DebateVerdict.downgrade
    assert result.revised_level == "LOW"


def test_downgrade_without_lower_level_falls_back_to_keep() -> None:
    # revised_level 不比原等级低 → 退化为 keep
    raw = """
    {"explanation":"x","verdict":"downgrade","revised_level":"CRITICAL","confidence":0.7}
    """
    result = debate_agent._parse_debate_result(raw, original_level="HIGH")
    assert result.verdict == DebateVerdict.keep
    assert result.revised_level is None


def test_invalid_verdict_falls_back_to_keep() -> None:
    raw = '{"explanation":"x","verdict":"maybe","revised_level":null,"confidence":0.5}'
    result = debate_agent._parse_debate_result(raw, original_level="HIGH")
    assert result.verdict == DebateVerdict.keep


@pytest.mark.anyio
async def test_run_async_uses_llm_when_configured(monkeypatch) -> None:
    captured = {}

    class FakeLLM:
        is_configured = True
        _mock_mode = False

        async def chat(self, messages, model=None, temperature=0.0):
            captured["called"] = True
            return '{"explanation":"成立","verdict":"keep","revised_level":null,"confidence":0.6}'

    monkeypatch.setattr(debate_agent, "llm_client", FakeLLM())

    result = await debate_agent.run_async(
        finding=_finding(),
        challenge="这是误报",
        code_context="@@ code @@",
    )
    assert captured.get("called") is True
    assert result.verdict == DebateVerdict.keep


@pytest.mark.anyio
async def test_run_async_fallback_on_llm_failure(monkeypatch) -> None:
    class FakeLLM:
        is_configured = True
        _mock_mode = False

        async def chat(self, messages, model=None, temperature=0.0):
            raise debate_agent.LLMClientError("timeout")

    monkeypatch.setattr(debate_agent, "llm_client", FakeLLM())

    result = await debate_agent.run_async(
        finding=_finding(),
        challenge="误报",
        code_context="",
    )
    assert result.verdict == DebateVerdict.keep
    assert "不可用" in result.explanation
