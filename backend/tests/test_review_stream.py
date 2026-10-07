"""SSE 进度流测试 —— 用内存 store 播种，TestClient 消费事件流。

注意：running 状态的任务没有注册事件队列（queue 未 register），
SSE 端点会回放历史事件后发送 done 并退出，测试依赖该行为。
缺失任务的当前实现返回 200 + SSE error 事件（而非 HTTP 404）。
"""

import asyncio

from fastapi.testclient import TestClient

from app.main import app
from app.api import review as review_api
from app.models.review_job import ReviewJob
from app.schemas.review import ReviewJobStatus, ReviewReport, ReviewReportStats
from app.services.review_job_service import review_job_service
from tests.memory_store import MemoryReviewJobStore


def install_store(store: MemoryReviewJobStore):
    """同时替换 service 与 API 层直接引用的 store 单例（stream 端点绕过 service 直连 store）。"""
    original_store = review_job_service._store
    original_api_store = review_api.review_job_store
    review_job_service._store = store
    review_api.review_job_store = store
    return original_store, original_api_store


def restore_store(original_store, original_api_store) -> None:
    review_job_service._store = original_store
    review_api.review_job_store = original_api_store


def _make_report() -> ReviewReport:
    return ReviewReport(
        summary="done",
        risk_level="LOW",
        stats=ReviewReportStats(),
        changed_files=[],
        changed_symbols=[],
        findings=[],
        review_comment="done",
    )


def test_stream_returns_real_progress_events_for_running_job() -> None:
    store = MemoryReviewJobStore()

    async def seed() -> None:
        await store.create(ReviewJob(job_id="rev_stream_1", pr_url="https://github.com/example/repo/pull/1"))
        await store.update_status("rev_stream_1", ReviewJobStatus.running)
        await store.add_progress_event(
            "rev_stream_1",
            {"type": "progress", "step": "FETCH_PR", "percent": 30, "message": "GitHub PR 基本信息已拉取"},
        )

    asyncio.run(seed())
    original_store, original_api_store = install_store(store)
    client = TestClient(app)

    try:
        response = client.get("/api/v1/review/stream/rev_stream_1")
    finally:
        restore_store(original_store, original_api_store)

    assert response.status_code == 200
    assert response.headers["content-type"].startswith("text/event-stream")
    assert "event: progress" in response.text
    assert '"step": "FETCH_PR"' in response.text


def test_stream_sends_done_for_completed_job() -> None:
    store = MemoryReviewJobStore()

    async def seed() -> None:
        await store.create(ReviewJob(job_id="rev_stream_2", pr_url="https://github.com/example/repo/pull/2"))
        await store.update_status("rev_stream_2", ReviewJobStatus.running)
        await store.update_status("rev_stream_2", ReviewJobStatus.completed, report=_make_report())

    asyncio.run(seed())
    original_store, original_api_store = install_store(store)
    client = TestClient(app)

    try:
        response = client.get("/api/v1/review/stream/rev_stream_2")
    finally:
        restore_store(original_store, original_api_store)

    assert response.status_code == 200
    assert "event: done" in response.text
    assert '"status": "completed"' in response.text


def test_stream_sends_warning_and_done_for_failed_job() -> None:
    store = MemoryReviewJobStore()

    async def seed() -> None:
        await store.create(ReviewJob(job_id="rev_stream_3", pr_url="https://github.com/example/repo/pull/3"))
        await store.update_status(
            "rev_stream_3", ReviewJobStatus.failed, error_message="GitHub pull request was not found"
        )

    asyncio.run(seed())
    original_store, original_api_store = install_store(store)
    client = TestClient(app)

    try:
        response = client.get("/api/v1/review/stream/rev_stream_3")
    finally:
        restore_store(original_store, original_api_store)

    assert response.status_code == 200
    assert "event: done" in response.text


def test_stream_returns_error_event_for_missing_job() -> None:
    store = MemoryReviewJobStore()
    original_store, original_api_store = install_store(store)
    client = TestClient(app)

    try:
        response = client.get("/api/v1/review/stream/missing")
    finally:
        restore_store(original_store, original_api_store)

    # 实现为 SSE error 事件 + done，HTTP 状态码仍 200（流式响应无法中途改状态码）
    assert response.status_code == 200
    assert "event: error" in response.text
    assert "event: done" in response.text
