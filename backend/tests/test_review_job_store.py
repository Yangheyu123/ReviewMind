"""ReviewJobStore（DB 版）与 ReviewJobService 的行为测试。"""

import pytest
from fastapi import HTTPException
from fastapi.testclient import TestClient

from app.main import app
from app.models.review_job import ReviewJob
from app.schemas.review import CreateReviewJobRequest, ReviewJobStatus, ReviewReport, ReviewReportStats
from app.services.review_job_service import ReviewJobService, review_job_service
from app.services.review_pipeline import ReviewPipelineResult
from app.services.review_job_store import InvalidReviewJobTransitionError, ReviewJobNotFoundError, ReviewJobStore


def make_request() -> CreateReviewJobRequest:
    return CreateReviewJobRequest(pr_url="https://github.com/example/repo/pull/1")


class NoopPipeline:
    async def run(self, job: ReviewJob, config: dict | None = None, github_token: str | None = None) -> ReviewPipelineResult:
        return ReviewPipelineResult(pr_info={}, filtered_files={}, parsed_diff=[])


@pytest.mark.anyio
async def test_create_job_can_be_queried_from_store() -> None:
    store = ReviewJobStore()
    service = ReviewJobService(store, NoopPipeline())

    response = await service.create_job(make_request())
    job = await store.get(response.job_id)

    assert response.status == ReviewJobStatus.pending
    assert job.pr_url == "https://github.com/example/repo/pull/1"
    assert job.progress_events[0]["step"] == "JOB_CREATED"


@pytest.mark.anyio
async def test_missing_job_raises_store_error() -> None:
    store = ReviewJobStore()

    with pytest.raises(ReviewJobNotFoundError):
        await store.get("missing")


@pytest.mark.anyio
async def test_job_status_transition_boundaries() -> None:
    store = ReviewJobStore()
    await store.create(ReviewJob(job_id="rev_1", pr_url="https://github.com/example/repo/pull/1"))

    await store.update_status("rev_1", ReviewJobStatus.running)
    report = ReviewReport(
        summary="done",
        risk_level="LOW",
        stats=ReviewReportStats(),
        changed_files=[],
        changed_symbols=[],
        findings=[],
        review_comment="done",
    )
    completed = await store.update_status("rev_1", ReviewJobStatus.completed, report=report)

    assert completed.status == ReviewJobStatus.completed
    assert completed.report is not None
    assert completed.report.risk_level == "LOW"

    with pytest.raises(InvalidReviewJobTransitionError):
        await store.update_status("rev_1", ReviewJobStatus.failed)


@pytest.mark.anyio
async def test_service_maps_missing_job_to_404() -> None:
    service = ReviewJobService(ReviewJobStore())

    with pytest.raises(HTTPException) as exc_info:
        await service.get_report("missing")

    assert exc_info.value.status_code == 404


def test_review_job_state_endpoint_returns_created_job() -> None:
    from app.api import review as review_api
    from tests.memory_store import MemoryReviewJobStore

    store = MemoryReviewJobStore()
    original_pipeline = review_job_service._pipeline
    original_service_store = review_job_service._store
    original_api_store = review_api.review_job_store
    review_job_service._pipeline = NoopPipeline()
    review_job_service._store = store
    review_api.review_job_store = store
    client = TestClient(app)
    try:
        create_response = client.post(
            "/api/v1/review/jobs",
            json={"pr_url": "https://github.com/example/repo/pull/1"},
        )
        job_id = create_response.json()["data"]["job_id"]

        state_response = client.get(f"/api/v1/review/jobs/{job_id}/state")
    finally:
        review_job_service._pipeline = original_pipeline
        review_job_service._store = original_service_store
        review_api.review_job_store = original_api_store

    assert state_response.status_code == 200
    body = state_response.json()
    assert body["job_id"] == job_id
    assert body["status"] == "pending"
    assert body["progress_events"][0]["step"] == "JOB_CREATED"


def test_review_report_endpoint_returns_404_for_missing_job() -> None:
    from app.api import review as review_api
    from tests.memory_store import MemoryReviewJobStore

    store = MemoryReviewJobStore()
    original_service_store = review_job_service._store
    original_api_store = review_api.review_job_store
    review_job_service._store = store
    review_api.review_job_store = store
    client = TestClient(app)
    try:
        response = client.get("/api/v1/review/jobs/missing")
    finally:
        review_job_service._store = original_service_store
        review_api.review_job_store = original_api_store

    assert response.status_code == 404
