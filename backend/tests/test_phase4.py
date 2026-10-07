"""Phase 4 测试：三区上下文压缩、LLM 用量记录器、观测落库。"""

import pytest

from app.agent_loop.review_agent import (
    ReviewAgent,
    compression_action,
    compress_messages,
    estimate_tokens,
)
from app.agent_loop.tool_context import ToolContext
from app.core.llm import ToolCallItem, ToolCallResponse
from app.core.llm_usage import llm_context, record
from app.schemas.github import GitHubPullRequestRef


def _ref():
    return GitHubPullRequestRef(owner="o", repo="r", pull_number=1,
                                html_url="https://github.com/o/r/pull/1")


def make_ctx() -> ToolContext:
    return ToolContext(
        github_client=None, pr_ref=_ref(), head_sha="h",
        changed_files=[{"filename": "a.py", "status": "modified", "patch": "@@ -1 +1 @@\n-a\n+b"}],
    )


# ---------------------------------------------------------------------------
# 压缩纯函数
# ---------------------------------------------------------------------------

def test_compression_action_thresholds():
    assert compression_action(50, 100) == "none"
    assert compression_action(61, 100) == "soft"
    assert compression_action(81, 100) == "hard"


def test_compress_messages_structure():
    msgs = [{"role": "system", "content": "sys"}, {"role": "user", "content": "task"}]
    for i in range(10):
        msgs.append({"role": "assistant", "content": "", "tool_calls": [
            {"id": f"t{i}", "type": "function", "function": {"name": "file_read", "arguments": "{}"}}]})
        msgs.append({"role": "tool", "tool_call_id": f"t{i}", "content": f"result {i} " * 50})

    compressed = compress_messages(msgs, keep_last=6)
    assert compressed[0] is msgs[0]            # frozen：system 原对象不动
    assert compressed[1] is msgs[1]            # frozen：首条 user 不动
    assert compressed[-6:] == msgs[-6:]        # active：最近 6 条完整保留
    assert len(compressed) < len(msgs)         # 中段被压缩
    assert any("<previous_review_summary>" in str(m.get("content", "")) for m in compressed)
    assert estimate_tokens(compressed) < estimate_tokens(msgs)


def test_compress_short_messages_noop():
    msgs = [{"role": "system", "content": "s"}, {"role": "user", "content": "u"},
            {"role": "assistant", "content": "a"}]
    assert compress_messages(msgs) == msgs


# ---------------------------------------------------------------------------
# 循环内压缩闸
# ---------------------------------------------------------------------------

class FakeLLM:
    _mock_mode = False
    is_configured = True

    def __init__(self, responses):
        self._responses = list(responses)

    async def chat_with_tools(self, messages, *, tools, model=None, temperature=0.0, tool_choice="auto"):
        if not self._responses:
            return ToolCallResponse(content="done", tool_calls=[])
        return self._responses.pop(0)


def _resp(*calls, usage=100):
    return ToolCallResponse(
        content="", tool_calls=[ToolCallItem(id=f"c{i}", name=n, arguments=a) for i, (n, a) in enumerate(calls)],
        finish_reason="tool_calls",
        usage={"prompt_tokens": usage, "completion_tokens": 0, "total_tokens": usage},
    )


async def test_agent_hard_compression_terminates_gracefully():
    """上下文超硬阈且压缩后仍超 → 有界终止，stop_reason=compression_exceeded。"""
    big_user = "x" * 20_000  # ~5000 tokens
    llm = FakeLLM([
        _resp(("task_done", {"state": "DONE"}), usage=99_000),
    ])
    agent = ReviewAgent(llm, make_ctx(), context_budget=1000)  # 极小预算
    result = await agent.run("sys", big_user)
    assert result.stop_reason in ("compression_exceeded",)
    assert any(t.get("type") == "compression" for t in result.transcript)


async def test_agent_soft_compression_continues():
    """软阈触发压缩但任务继续完成。"""
    user_prompt = "x" * 800  # ~200 tokens
    llm = FakeLLM([
        _resp(("file_read", {"path": "a.py"}), usage=100),
        _resp(("task_done", {"state": "DONE"}), usage=100),
    ])
    agent = ReviewAgent(llm, make_ctx(), context_budget=300)  # soft=180, hard=240
    result = await agent.run("sys", user_prompt)
    assert result.stop_reason == "task_done"


# ---------------------------------------------------------------------------
# LLM 用量记录器
# ---------------------------------------------------------------------------

def test_record_without_context_only_logs():
    rec = record("glm-4.5-air", protocol="anthropic", prompt_tokens=10, completion_tokens=5)
    assert rec["total_tokens"] == 15 and rec["job_id"] == ""


def test_record_with_context_collects():
    with llm_context("job_1", group="g0", phase="review") as ctx:
        record("glm-4.5-air", protocol="anthropic", prompt_tokens=100, completion_tokens=20)
        record("glm-4.5-air", protocol="anthropic", prompt_tokens=50, completion_tokens=10)
    assert len(ctx["records"]) == 2
    assert ctx["records"][0]["phase"] == "review"
    assert ctx["records"][1]["total_tokens"] == 60


async def test_store_save_llm_requests_persists():
    from tests.memory_store import MemoryReviewJobStore
    store = MemoryReviewJobStore()
    n = await store.save_llm_requests("job_1", [
        {"job_id": "job_1", "group": "g0", "phase": "review", "model": "glm-4.5-air",
         "protocol": "anthropic", "prompt_tokens": 100, "completion_tokens": 20,
         "total_tokens": 120, "latency_ms": 500, "status": "ok"},
    ])
    assert n == 1 and len(store.llm_requests) == 1
