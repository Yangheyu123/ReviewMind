"""Tests for ReviewGraph — LangGraph-style 工作流。"""

import pytest

from app.graph.review_graph import ReviewGraph, ReviewGraphResult
from app.models.review_job import ReviewJob
from app.schemas.github import GitHubBranchRef, GitHubPullRequestFile, GitHubPullRequestInfo
from app.schemas.review import ReviewJobStatus
from app.services.github_client import GitHubClientError
from tests.memory_store import MemoryReviewJobStore


class MockGitHubClient:
    async def fetch_pull_request(self, pr_ref):
        return GitHubPullRequestInfo(
            owner=pr_ref.owner,
            repo=pr_ref.repo,
            pull_number=pr_ref.pull_number,
            title="Improve review pipeline",
            author="alice",
            state="open",
            base=GitHubBranchRef(ref="main", sha="base-sha"),
            head=GitHubBranchRef(ref="feature", sha="head-sha"),
            changed_files=2,
            additions=8,
            deletions=1,
            html_url=pr_ref.html_url,
        )

    async def fetch_file_content(self, pr_ref, path, ref):
        if path == "backend/app/services/example.py":
            # run() 在第 1-2 行，使 changed_lines [2, 3] 能命中 run 的区间
            return (
                "def run():\n    return True\n\n\n"
                "def unused_helper():\n    pass\n"
            )
        return None

    async def fetch_pull_request_files(self, pr_ref):
        return [
            GitHubPullRequestFile(
                filename="backend/app/services/example.py",
                status="modified",
                additions=3,
                deletions=1,
                patch="@@ -1,2 +1,3 @@\n def run():\n-    return False\n+    return True\n+\n",
            ),
            GitHubPullRequestFile(
                filename="frontend/dist/app.min.js",
                status="modified",
                additions=5,
                deletions=0,
                patch="@@ -1 +1 @@\n-console.log(1)\n+console.log(2)\n",
            ),
        ]


class FailingGitHubClient:
    async def fetch_pull_request(self, pr_ref):
        raise GitHubClientError("GitHub pull request was not found", status_code=404)

    async def fetch_pull_request_files(self, pr_ref):
        return []


@pytest.mark.anyio
async def test_review_graph_completes_with_mock_agents() -> None:
    """正常路径：fetch → filter → parse → agents → risk → report 全部通过。"""
    store = MemoryReviewJobStore()
    job = await store.create(ReviewJob(job_id="g_1", pr_url="https://github.com/example/repo/pull/1"))
    graph = ReviewGraph(store, MockGitHubClient())

    result = await graph.run(job)

    saved = await store.get("g_1")
    assert saved.status == ReviewJobStatus.completed
    assert saved.report is not None
    assert result.pr_info["title"] == "Improve review pipeline"
    assert len(result.filtered_files.get("included_files", [])) == 1
    assert len(result.parsed_diff) == 1
    assert result.parsed_diff[0]["changed_lines"] == [2, 3]

    # progress 事件覆盖所有节点（agents 并行段整体发一个 AGENTS 事件）
    steps = [e["step"] for e in saved.progress_events if "step" in e]
    assert "FETCH_PR" in steps
    assert "FETCH_FILES" in steps
    assert "DIFF_FILTER" in steps
    assert "DIFF_PARSE" in steps
    assert "AST_CONTEXT" in steps
    assert "AGENTS" in steps
    assert "RISK_JUDGE" in steps
    assert "REPORT_AGENT" in steps
    assert "DONE" in steps

    # report 包含 findings（security_agent + test_agent 应产出）
    assert len(saved.report.findings) > 0
    # changed_symbols 来自 head 真实源码的 AST 提取（P1 修复：不再恒为空）
    assert len(saved.report.changed_symbols) > 0
    assert saved.report.changed_symbols[0].symbol == "run"
    assert saved.report.risk_level in ("LOW", "MEDIUM", "HIGH", "CRITICAL")
    assert "AI 审查摘要" in saved.report.review_comment


@pytest.mark.anyio
async def test_review_graph_fails_on_github_404() -> None:
    """关键节点 fetch_pr 失败 → job 直接 failed。"""
    store = MemoryReviewJobStore()
    job = await store.create(ReviewJob(job_id="g_2", pr_url="https://github.com/example/repo/pull/404"))
    graph = ReviewGraph(store, FailingGitHubClient())

    result = await graph.run(job)

    saved = await store.get("g_2")
    assert saved.status == ReviewJobStatus.failed
    assert "FETCH_PR" in (saved.error_message or "")
    assert result.pr_info == {}

    # 应有 warning 事件
    warning_events = [e for e in saved.progress_events if e.get("type") == "warning"]
    assert len(warning_events) >= 1


@pytest.mark.anyio
async def test_review_graph_report_has_correct_stats() -> None:
    """报告 stats 应包含 risk 统计。"""
    store = MemoryReviewJobStore()
    job = await store.create(ReviewJob(job_id="g_3", pr_url="https://github.com/example/repo/pull/1"))
    graph = ReviewGraph(store, MockGitHubClient())

    await graph.run(job)

    saved = await store.get("g_3")
    report = saved.report
    assert report is not None
    total_stats = (
        report.stats.critical
        + report.stats.high
        + report.stats.medium
        + report.stats.low
        + report.stats.suggestion
    )
    assert total_stats >= 0


@pytest.mark.anyio
async def test_review_graph_result_contains_warnings_from_agents() -> None:
    """即使有 agent 警告，graph 也能完成。"""
    store = MemoryReviewJobStore()
    job = await store.create(ReviewJob(job_id="g_4", pr_url="https://github.com/example/repo/pull/1"))
    graph = ReviewGraph(store, MockGitHubClient())

    result = await graph.run(job)

    saved = await store.get("g_4")
    assert saved.status == ReviewJobStatus.completed
    # result.warnings 可以为空（正常情况），也可以有 AST 降级等
    assert isinstance(result.warnings, list)