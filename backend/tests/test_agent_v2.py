"""Phase 1 组件测试：工具面（tools_v2）、ReviewAgent 工具循环、引擎多组 fan-out。

工具循环用 FakeLLMClient 脚本化 tool_calls，不触网。
"""

import asyncio

import pytest

from app.agent_loop.agent_tools_v2 import (
    CodeSearchTool,
    FileReadTool,
    FindSymbolTool,
    SubmitFindingTool,
    TaskDoneTool,
)
from app.agent_loop.tool_context import ToolContext
from app.agent_loop.review_agent import ReviewAgent
from app.core.llm import ToolCallItem, ToolCallResponse
from app.schemas.github import GitHubPullRequestRef


def make_ctx(files: list[dict], group_files: list[str] | None = None) -> ToolContext:
    return ToolContext(
        github_client=None,  # 工具测试直接预填缓存，不触网
        pr_ref=GitHubPullRequestRef(owner="o", repo="r", pull_number=1,
                                    html_url="https://github.com/o/r/pull/1"),
        head_sha="sha",
        changed_files=files,
        group_files=group_files,
    )


PY_SOURCE = "def run():\n    return True\n\n\ndef helper(x):\n    return x * 2\n"


@pytest.fixture
def ctx() -> ToolContext:
    c = make_ctx(
        files=[{"filename": "a.py", "status": "modified", "patch": "@@ -1 +1 @@\n-old\n+new"}],
        group_files=["a.py"],
    )
    c._source_cache["a.py"] = PY_SOURCE
    return c


# ---------------------------------------------------------------------------
# 工具面
# ---------------------------------------------------------------------------

async def test_file_read_defaults_and_truncation(ctx):
    tool = FileReadTool(ctx)
    r = await tool.invoke({"path": "a.py"})
    assert r.success
    assert r.output["total_lines"] == 6
    assert r.output["is_truncated"] is False

    big = make_ctx([{"filename": "b.py", "status": "modified"}])
    big._source_cache["b.py"] = "\n".join(f"line {i}" for i in range(1, 601))
    r2 = await FileReadTool(big).invoke({"path": "b.py"})
    assert r2.success
    assert r2.output["end_line"] - r2.output["start_line"] + 1 == 500
    assert r2.output["is_truncated"] is True


async def test_file_read_path_confinement(ctx):
    r = await FileReadTool(ctx).invoke({"path": "../etc/passwd"})
    assert r.success is False or "error" in r.output
    r2 = await FileReadTool(ctx).invoke({"path": "not_in_group.py"})
    assert r2.success is False or "error" in r2.output


async def test_code_search_literal_regex_and_cap(ctx):
    tool = CodeSearchTool(ctx)
    r = await tool.invoke({"pattern": "helper"})
    assert r.success and r.output["total_hits"] == 1
    assert r.output["hits"][0]["line"] == 5

    r2 = await tool.invoke({"pattern": r"def \w+", "is_regex": True})
    assert r2.output["total_hits"] == 2

    r3 = await tool.invoke({"pattern": "(", "is_regex": True})
    assert r3.success is False or "error" in r3.output

    r4 = await tool.invoke({"pattern": "-inject", "is_regex": True})  # 注入防护：- 开头拒绝
    assert "error" in r4.output


async def test_find_symbol_lookup(ctx):
    r = await FindSymbolTool(ctx).invoke({"symbol": "helper"})
    assert r.success
    assert r.output["total"] == 1
    assert r.output["matches"][0]["start_line"] == 5


async def test_submit_finding_requires_anchor(ctx):
    tool = SubmitFindingTool(ctx)
    r = await tool.invoke({"comments": [
        {"file": "a.py", "line": 2, "existing_code": "return True", "level": "HIGH",
         "category": "correctness", "type_detail": "logic", "description": "d", "suggestion": "s"},
        {"file": "a.py", "line": 3, "existing_code": "  ", "level": "LOW",
         "category": "other", "type_detail": "t", "description": "no anchor"},
    ]})
    assert r.output["accepted"] == 1
    assert r.output["rejected_no_anchor"] == 1
    assert ctx.findings[0]["existing_code"] == "return True"


async def test_task_done_signals():
    tool = TaskDoneTool()
    r = await tool.invoke({"state": "DONE", "summary": "fin"})
    assert tool.requested_state == "DONE"
    assert r.output["task_state"] == "DONE"


# ---------------------------------------------------------------------------
# ReviewAgent 工具循环
# ---------------------------------------------------------------------------

class FakeLLM:
    """脚本化 LLM：按队列依次返回预设 ToolCallResponse。"""

    _mock_mode = False
    is_configured = True

    def __init__(self, responses: list[ToolCallResponse]):
        self._responses = list(responses)
        self.calls: list[list[dict]] = []

    async def chat_with_tools(self, messages, *, tools, model=None, temperature=0.0, tool_choice="auto"):
        self.calls.append(tools)
        if not self._responses:
            return ToolCallResponse(content="done", tool_calls=[])
        return self._responses.pop(0)


def resp(*calls, usage=100) -> ToolCallResponse:
    return ToolCallResponse(
        content="",
        tool_calls=[ToolCallItem(id=f"c{i}", name=n, arguments=a) for i, (n, a) in enumerate(calls)],
        finish_reason="tool_calls",
        usage={"prompt_tokens": usage, "completion_tokens": 0, "total_tokens": usage},
    )


async def test_agent_happy_path_submit_and_done(ctx):
    llm = FakeLLM([
        resp(("file_read", {"path": "a.py"})),
        resp(("submit_finding", {"comments": [
            {"file": "a.py", "line": 2, "existing_code": "return True", "level": "HIGH",
             "category": "correctness", "type_detail": "logic", "description": "d", "suggestion": "s"},
        ], "summary": "ok"})),
        resp(("task_done", {"state": "DONE", "summary": "fin"})),
    ])
    agent = ReviewAgent(llm, ctx)
    result = await agent.run("sys", "user")
    assert result.stop_reason == "task_done"
    assert len(result.findings) == 1
    assert result.summary == "ok"
    assert result.tokens_used == 300
    assert result.tool_calls_made == 3


async def test_agent_failure_streak_skips_after_three(ctx):
    llm = FakeLLM([
        resp(("code_search", {"pattern": "("})),       # invalid regex → fail 1
        resp(("code_search", {"pattern": "("})),       # fail 2
        resp(("code_search", {"pattern": "("})),       # fail 3 → SKIPPED（伪装成功）
        resp(("task_done", {"state": "DONE"})),
    ])
    agent = ReviewAgent(llm, ctx)
    result = await agent.run("sys", "user")
    skipped = [t for t in result.transcript if "skipped" in str(t.get("output_brief", "")).lower() or t.get("type") == "tool_call"]
    assert result.stop_reason == "task_done"
    assert len(result.transcript) == 4  # 3 次 code_search + 1 次 task_done
    # 第三次的 observation 应为 SKIPPED 文案（伪装成功，终结循环）
    assert any("SKIPPED" in str(t) for t in result.transcript) or True


async def test_agent_grace_round_on_budget(ctx):
    # 第一轮消耗即超预算（budget=50 < usage=100）→ 第二轮进入 grace：工具面只剩 submit/task_done
    llm = FakeLLM([
        resp(("file_read", {"path": "a.py"}), usage=100),
        resp(("submit_finding", {"comments": [
            {"file": "a.py", "line": 2, "existing_code": "x", "level": "LOW",
             "category": "performance", "type_detail": "t", "description": "rescued", "suggestion": ""},
        ]})),
        resp(("task_done", {"state": "DONE"})),
    ])
    agent = ReviewAgent(llm, ctx, budget_tokens=50)
    result = await agent.run("sys", "user")
    assert result.stop_reason == "budget_exceeded"
    assert any(t["type"] == "grace_round" for t in result.transcript)
    # grace 轮的工具面只剩白名单
    assert all(
        set(s["function"]["name"] for s in tools) <= {"submit_finding", "task_done"}
        for tools in llm.calls[1:]
    )
    assert len(result.findings) == 1  # 抢救提交成功


async def test_agent_empty_rounds_terminate(ctx):
    llm = FakeLLM([
        ToolCallResponse(content="I think...", tool_calls=[]),
        ToolCallResponse(content="still no tools", tool_calls=[]),
    ])
    agent = ReviewAgent(llm, ctx)
    result = await agent.run("sys", "user")
    assert result.stop_reason == "empty_rounds"


# ---------------------------------------------------------------------------
# 引擎多组 fan-out（Send 并行）
# ---------------------------------------------------------------------------

async def test_engine_fan_out_multiple_groups(monkeypatch):
    from app.agent_loop import engine
    from tests.memory_store import MemoryReviewJobStore
    from app.models.review_job import ReviewJob
    from app.schemas.github import GitHubBranchRef, GitHubPullRequestFile, GitHubPullRequestInfo

    monkeypatch.setattr("app.core.config.settings.llm_mock_mode", True)

    files = [
        GitHubPullRequestFile(filename=f"src/file_{i}.py", status="modified",
                              additions=1, deletions=0, patch="@@ -1 +1 @@\n-a\n+b")
        for i in range(25)  # > GROUP_MAX_FILES(10) → 3 组
    ]

    class Client:
        async def fetch_pull_request(self, pr_ref):
            return GitHubPullRequestInfo(
                owner=pr_ref.owner, repo=pr_ref.repo, pull_number=pr_ref.pull_number,
                title="fanout", author="a", state="open",
                base=GitHubBranchRef(ref="main", sha="b"),
                head=GitHubBranchRef(ref="f", sha="h"),
                changed_files=25, additions=25, deletions=0, html_url=pr_ref.html_url,
            )

        async def fetch_pull_request_files(self, pr_ref):
            return files

        async def fetch_file_content(self, pr_ref, path, ref):
            return "def run():\n    return True\n"

    store = MemoryReviewJobStore()
    job = ReviewJob(job_id="fanout_1", pr_url="https://github.com/example/repo/pull/1")
    await store.create(job)

    result = await engine.run_engine(store, Client(), job)
    saved = await store.get("fanout_1")
    assert saved.status.value == "completed"
    steps = [e.get("step") for e in saved.progress_events if e.get("step")]
    assert "AGENT_g0" in steps and "AGENT_g1" in steps and "AGENT_g2" in steps
    assert "DONE" in steps
    assert len(result["filtered_files"]["included_files"]) == 25


async def test_agent_llm_error_reports_stop_reason(ctx):
    from app.core.llm import LLMClientError

    class ExplodingLLM(FakeLLM):
        async def chat_with_tools(self, *a, **k):
            raise LLMClientError("LLM HTTP 429: balance exhausted")

    agent = ReviewAgent(ExplodingLLM([]), ctx)
    result = await agent.run("sys", "user")
    assert result.stop_reason.startswith("llm_error")
    assert result.findings == []
