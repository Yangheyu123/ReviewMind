import pytest

from app.models.review_job import ReviewJob
from app.schemas.github import GitHubBranchRef, GitHubPullRequestFile, GitHubPullRequestInfo
from app.schemas.review import ReviewJobStatus
from app.services.github_client import GitHubClientError
from tests.memory_store import MemoryReviewJobStore
from app.services.review_pipeline import ReviewPipeline


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
async def test_review_pipeline_saves_intermediate_result_and_completes_job() -> None:
    store = MemoryReviewJobStore()
    job = await store.create(ReviewJob(job_id="rev_1", pr_url="https://github.com/example/repo/pull/1"))
    pipeline = ReviewPipeline(store, MockGitHubClient())

    result = await pipeline.run(job)
    saved_job = await store.get("rev_1")

    assert saved_job.status == ReviewJobStatus.completed
    assert saved_job.report is not None
    assert result.pr_info["title"] == "Improve review pipeline"
    assert len(result.filtered_files["included_files"]) == 1
    assert len(result.filtered_files["excluded_files"]) == 1
    assert result.parsed_diff[0]["changed_lines"] == [2, 3]
    assert saved_job.pipeline_result is not None
    steps = [event["step"] for event in saved_job.progress_events if "step" in event]
    for expected in ["FETCH_PR", "FETCH_FILES", "DIFF_FILTER", "DIFF_PARSE", "DONE"]:
        assert expected in steps


@pytest.mark.anyio
async def test_review_pipeline_marks_job_failed_when_github_fetch_fails() -> None:
    store = MemoryReviewJobStore()
    job = await store.create(ReviewJob(job_id="rev_2", pr_url="https://github.com/example/repo/pull/404"))
    pipeline = ReviewPipeline(store, FailingGitHubClient())

    result = await pipeline.run(job)
    saved_job = await store.get("rev_2")

    assert saved_job.status == ReviewJobStatus.failed
    assert "GitHub pull request was not found" in saved_job.error_message
    assert result.pr_info == {}
    assert saved_job.progress_events[-1]["type"] == "warning"