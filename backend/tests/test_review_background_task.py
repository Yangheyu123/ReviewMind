"""Tests for ReviewTaskRunner."""

import asyncio

import pytest

from app.models.review_job import ReviewJob
from app.schemas.github import GitHubBranchRef, GitHubPullRequestFile, GitHubPullRequestInfo
from app.schemas.review import CreateReviewJobRequest, ReviewJobStatus
from app.services.github_client import GitHubClientError
from app.services.review_job_service import ReviewJobService
from tests.memory_store import MemoryReviewJobStore
from app.services.review_pipeline import ReviewPipeline
from app.services.review_task_runner import ReviewTaskRunner


class SlowMockGitHubClient:
    def __init__(self, delay: float = 0.05):
        self._delay = delay

    async def fetch_pull_request(self, pr_ref):
        await asyncio.sleep(self._delay)
        return GitHubPullRequestInfo(
            owner=pr_ref.owner,
            repo=pr_ref.repo,
            pull_number=pr_ref.pull_number,
            title="Test PR",
            author="tester",
            state="open",
            base=GitHubBranchRef(ref="main", sha="abc"),
            head=GitHubBranchRef(ref="feature", sha="def"),
            changed_files=1,
            additions=5,
            deletions=2,
            html_url=pr_ref.html_url,
        )

    async def fetch_pull_request_files(self, pr_ref):
        await asyncio.sleep(self._delay)
        return [
            GitHubPullRequestFile(
                filename="src/example.py",
                status="modified",
                additions=5,
                deletions=2,
                patch="@@ -1,3 +1,6 @@\n def hello():\n-    return 1\n+    return 2\n+    x = 3\n",
            ),
        ]


class FailingMockGitHubClient:
    async def fetch_pull_request(self, pr_ref):
        raise GitHubClientError("Not found", status_code=404)

    async def fetch_pull_request_files(self, pr_ref):
        return []


@pytest.mark.anyio
async def test_task_runner_submits_and_completes_background_task() -> None:
    store = MemoryReviewJobStore()
    runner = ReviewTaskRunner(store)
    pipeline = ReviewPipeline(store, SlowMockGitHubClient(delay=0.02))

    job = await store.create(ReviewJob(job_id="bg_1", pr_url="https://github.com/example/repo/pull/1"))

    assert not runner.is_running("bg_1")

    await runner.submit(job, lambda j: pipeline.run(j))

    assert runner.is_running("bg_1")
    assert runner.running_count() == 1

    await asyncio.sleep(1.5)  # 引擎含 checkpointer/分组开销，等待预算放宽

    assert not runner.is_running("bg_1")
    assert runner.running_count() == 0

    saved = await store.get("bg_1")
    assert saved.status == ReviewJobStatus.completed
    assert saved.report is not None


@pytest.mark.anyio
async def test_task_runner_prevents_duplicate_submission() -> None:
    store = MemoryReviewJobStore()
    runner = ReviewTaskRunner(store)
    pipeline = ReviewPipeline(store, SlowMockGitHubClient(delay=0.1))

    job = await store.create(ReviewJob(job_id="bg_dup", pr_url="https://github.com/example/repo/pull/1"))
    await runner.submit(job, lambda j: pipeline.run(j))

    with pytest.raises(RuntimeError, match="already running"):
        await runner.submit(job, lambda j: pipeline.run(j))

    await asyncio.sleep(1.5)


@pytest.mark.anyio
async def test_task_runner_handles_pipeline_failure_gracefully() -> None:
    store = MemoryReviewJobStore()
    runner = ReviewTaskRunner(store)
    pipeline = ReviewPipeline(store, FailingMockGitHubClient())

    job = await store.create(ReviewJob(job_id="bg_fail", pr_url="https://github.com/example/repo/pull/404"))
    await runner.submit(job, lambda j: pipeline.run(j))

    await asyncio.sleep(0.1)

    assert not runner.is_running("bg_fail")
    saved = await store.get("bg_fail")
    assert saved.status == ReviewJobStatus.failed


@pytest.mark.anyio
async def test_create_job_returns_pending_immediately() -> None:
    store = MemoryReviewJobStore()
    runner = ReviewTaskRunner(store)
    pipeline = ReviewPipeline(store, SlowMockGitHubClient(delay=0.1))

    service = ReviewJobService(store, pipeline, task_runner=runner)
    request = CreateReviewJobRequest(pr_url="https://github.com/example/repo/pull/99")

    start = asyncio.get_event_loop().time()
    response = await service.create_job(request)
    elapsed = asyncio.get_event_loop().time() - start

    assert elapsed < 0.05
    assert response.status == ReviewJobStatus.pending
    assert response.job_id.startswith("rev_")

    await asyncio.sleep(1.5)
    saved = await store.get(response.job_id)
    assert saved.status == ReviewJobStatus.completed

@pytest.mark.anyio
async def test_cancel_job_stops_running_task() -> None:
    """P1 回归：cancel_job 必须真正取消后台任务，而非只改状态
    （旧实现任务跑完后 cancelled->completed 转移非法抛异常被吞，报告丢失）。"""
    store = MemoryReviewJobStore()
    runner = ReviewTaskRunner(store)

    class HangingPipeline:
        """pipeline 卡在 LLM 调用模拟点，直到被 cancel。"""
        def __init__(self):
            self.cancelled = False

        async def run(self, job, config=None, github_token=None):
            try:
                await asyncio.sleep(30)
            except asyncio.CancelledError:
                self.cancelled = True
                raise

    pipeline = HangingPipeline()
    job = await store.create(ReviewJob(job_id="bg_cancel", pr_url="https://github.com/example/repo/pull/1"))
    await store.update_status("bg_cancel", ReviewJobStatus.running)
    await runner.submit(job, lambda j: pipeline.run(j))

    assert runner.is_running("bg_cancel")
    await asyncio.sleep(0.05)  # 让任务真正进入 sleep(30) 再取消（复现真实场景）
    assert runner.cancel("bg_cancel") is True

    for _ in range(50):
        await asyncio.sleep(0.02)
        if pipeline.cancelled:
            break
    assert pipeline.cancelled, "cancel 后后台任务应停止执行"
    assert not runner.is_running("bg_cancel")
