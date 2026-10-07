"""Tests for ReviewOrchestrator：端到端 Agent Loop 集成。

通过 mock GitHubClient + ReviewJobStore + PlannerAgent，验证：
- 预处理 → Planner → Executor → Finalizer 全链路跑通；
- 开关关闭时 ReviewGraph 仍走原路径（兼容性）；
- 开关开启时 ReviewGraph 委托 orchestrator（委托正确）。
"""

from __future__ import annotations

import pytest

from app.agent_loop.orchestrator import ReviewOrchestrator
from app.core.config import settings
from app.graph.review_graph import ReviewGraph
from app.models.review_job import ReviewJob
from app.schemas.github import GitHubBranchRef, GitHubPullRequestFile, GitHubPullRequestInfo
from app.schemas.review import ReviewJobStatus
from app.services.github_client import GitHubClientError
from app.services.review_job_store import ReviewJobStore


class MockGitHubClient:
    """与现有 test_review_graph.py 一致的 mock。"""

    def __init__(self, *, fail=False):
        self._fail = fail

    async def fetch_pull_request(self, pr_ref):
        if self._fail:
            raise GitHubClientError("network down")
        return GitHubPullRequestInfo(
            owner=pr_ref.owner, repo=pr_ref.repo, pull_number=pr_ref.pull_number,
            title="T", author="alice", state="open",
            base=GitHubBranchRef(ref="main", sha="b"), head=GitHubBranchRef(ref="f", sha="h"),
            changed_files=1, additions=3, deletions=1, html_url=pr_ref.html_url,
        )

    async def fetch_pull_request_files(self, pr_ref):
        return [
            GitHubPullRequestFile(
                filename="backend/app/x.py", status="modified",
                additions=3, deletions=1,
                patch="@@ -1,2 +1,3 @@\n def run():\n-    return False\n+    return True\n+\n",
            ),
        ]


class MockStore:
    """轻量 mock store，记录所有调用。"""

    def __init__(self):
        self.status_updates: list[tuple[str, str]] = []
        self.events: list[dict] = []
        self.saved_reports: list = []
        self.pr_infos: list = []

    async def update_status(self, job_id, status, **kwargs):
        self.status_updates.append((job_id, status))
        if "report" in kwargs:
            self.saved_reports.append(kwargs["report"])

    async def add_progress_event(self, job_id, event):
        self.events.append(event)

    async def save_pr_info(self, job_id, pr_info):
        self.pr_infos.append((job_id, pr_info))

    async def save_pipeline_result(self, job_id, result):
        pass


def _make_job():
    return ReviewJob(job_id="job-test", pr_url="https://github.com/o/r/pull/1")


# ---------------------------------------------------------------------------
# 端到端：Planner 走降级路径（避免依赖真实 LLM）
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_orchestrator_end_to_end_with_planner_fallback(monkeypatch):
    """Planner 因 LLM 不可用降级为全维度计划 → 全链路仍跑通并产出报告。

    通过让 PlannerAgent.plan 直接返回 fallback_plan 来模拟（不依赖网络）。
    """
    store = MockStore()
    gc = MockGitHubClient()
    orchestrator = ReviewOrchestrator(store, gc)

    # 让 Planner 跳过真实 LLM，直接用 fallback_plan（全维度）
    async def fake_plan(self):
        return self.fallback_plan(reason="test")
    monkeypatch.setattr("app.agent_loop.planner.PlannerAgent.plan", fake_plan)

    # 让各 agent 也走 mock（summary_agent 等内部会降级为规则路径，无需真实 LLM）
    result = await orchestrator.run(_make_job())

    # 状态最终为 completed
    assert ("job-test", ReviewJobStatus.completed) in store.status_updates
    assert ("job-test", ReviewJobStatus.running) in store.status_updates
    # 产出了报告
    assert len(store.saved_reports) == 1
    report = store.saved_reports[0]
    assert report.risk_level in ("LOW", "MEDIUM", "HIGH", "CRITICAL")
    # 推送了 DONE 进度
    assert any(e.get("step") == "DONE" for e in store.events)
    # 返回结构兼容
    assert result.pr_info["title"] == "T"
    assert len(result.parsed_diff) >= 0


@pytest.mark.asyncio
async def test_orchestrator_pushes_planner_and_executor_progress(monkeypatch):
    """验证 orchestrator 推送了 PLANNER / EXECUTOR / FINALIZER 进度事件。"""
    store = MockStore()
    orchestrator = ReviewOrchestrator(store, MockGitHubClient())

    async def fake_plan(self):
        return self.fallback_plan(reason="test")
    monkeypatch.setattr("app.agent_loop.planner.PlannerAgent.plan", fake_plan)

    await orchestrator.run(_make_job())
    steps = {e.get("step") for e in store.events}
    assert "PLANNER" in steps
    assert "EXECUTOR" in steps
    assert "FINALIZER" in steps


@pytest.mark.asyncio
async def test_orchestrator_critical_failure_marks_job_failed():
    """PR 拉取失败（关键节点）应将 job 标记为 failed。"""
    store = MockStore()
    orchestrator = ReviewOrchestrator(store, MockGitHubClient(fail=True))
    result = await orchestrator.run(_make_job())
    assert ("job-test", ReviewJobStatus.failed) in store.status_updates
    assert result.pr_info == {}  # 拉取失败，无 pr_info


# ---------------------------------------------------------------------------
# 开关兼容性：ReviewGraph 在两种模式下的行为
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_graph_delegates_to_orchestrator_when_enabled(monkeypatch):
    """开关开启时，ReviewGraph.run 应委托 LangGraph 引擎（Phase 1 转正后的默认路径）。"""
    monkeypatch.setattr(settings, "review_use_agent_loop", True)

    delegated: list[bool] = []

    async def fake_engine_run(store, github_client, job):
        delegated.append(True)
        return {
            "pr_info": {"title": "via-engine"},
            "filtered_files": {"included_files": [], "excluded_files": []},
            "parsed_diff": [],
            "warnings": [],
        }

    monkeypatch.setattr("app.agent_loop.engine.run_engine", fake_engine_run)

    store = MockStore()
    graph = ReviewGraph(ReviewJobStore.__new__(ReviewJobStore))  # 不实际用 store
    # MockStore 兼容 ReviewGraph 对 store 的调用
    graph._store = store
    result = await graph.run(_make_job())
    assert delegated == [True]
    assert result.pr_info == {"title": "via-engine"}


@pytest.mark.asyncio
async def test_graph_original_path_when_disabled(monkeypatch):
    """开关关闭时，ReviewGraph.run 走原有编排路径（不调 orchestrator）。"""
    monkeypatch.setattr(settings, "review_use_agent_loop", False)

    called: list[bool] = []

    async def fake_orchestrator_run(self, *a, **k):  # noqa: ARG002
        called.append(True)
        return None

    monkeypatch.setattr("app.agent_loop.orchestrator.ReviewOrchestrator.run", fake_orchestrator_run)

    # 原路径需要真实 store 行为；用一个会触发 fetch 失败的 mock 让它早退
    store = MockStore()
    gc = MockGitHubClient(fail=True)
    graph = ReviewGraph(store, gc)
    await graph.run(_make_job())
    # orchestrator 不应被调用
    assert called == []
    # 走的是原路径 → fetch 失败标记 failed
    assert ("job-test", ReviewJobStatus.failed) in store.status_updates


def test_config_default_is_enabled():
    """Phase 1 转正：LangGraph 引擎（agent_loop）为默认路径，旧 graph 流水线保留可回滚。"""
    from app.core.config import Settings

    fresh = Settings()
    assert fresh.review_use_agent_loop is True
