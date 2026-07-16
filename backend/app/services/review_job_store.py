"""ReviewJob 数据库持久化存储。

提供与原内存 ReviewJobStore 相同的接口，内部改用 SQLAlchemy 操作数据库。
"""

import asyncio
import json
import logging
from datetime import UTC, datetime

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.database import async_session
from app.models.db_models import ReviewJobModel
from app.models.review_job import ReviewJob, _QUEUE_DONE
from app.schemas.review import ReviewJobStatus, ReviewReport

logger = logging.getLogger(__name__)


class ReviewJobNotFoundError(Exception):
    def __init__(self, job_id: str) -> None:
        super().__init__(f"Review job not found: {job_id}")
        self.job_id = job_id


class InvalidReviewJobTransitionError(Exception):
    def __init__(self, current: ReviewJobStatus, target: ReviewJobStatus) -> None:
        super().__init__(f"Invalid review job transition: {current} -> {target}")
        self.current = current
        self.target = target


class ReviewJobStore:
    """基于 SQLAlchemy 的 ReviewJob 持久化存储。"""

    allowed_transitions: dict[ReviewJobStatus, set[ReviewJobStatus]] = {
        ReviewJobStatus.pending: {ReviewJobStatus.running, ReviewJobStatus.failed, ReviewJobStatus.cancelled},
        ReviewJobStatus.running: {ReviewJobStatus.completed, ReviewJobStatus.failed, ReviewJobStatus.cancelled},
        ReviewJobStatus.completed: set(),
        ReviewJobStatus.failed: set(),
        ReviewJobStatus.cancelled: set(),
    }

    async def create(self, job: ReviewJob) -> ReviewJob:
        async with async_session() as session:
            model = ReviewJobModel(
                job_id=job.job_id,
                pr_url=job.pr_url,
                status=job.status.value,
                created_at=job.created_at,
                updated_at=job.updated_at,
                completed_at=job.completed_at,
                error_message=job.error_message,
                pr_info=json.dumps(job.pr_info, ensure_ascii=False) if job.pr_info else None,
                progress_events=json.dumps(job.progress_events, ensure_ascii=False),
                pipeline_result=json.dumps(job.pipeline_result, ensure_ascii=False) if job.pipeline_result else None,
                report=job.report.model_dump_json() if job.report else None,
            )
            session.add(model)
            await session.commit()
        return job

    async def get(self, job_id: str) -> ReviewJob:
        async with async_session() as session:
            result = await session.execute(
                select(ReviewJobModel).where(ReviewJobModel.job_id == job_id)
            )
            model = result.scalar_one_or_none()
        if model is None:
            raise ReviewJobNotFoundError(job_id)
        return self._to_job(model)

    async def get_all(self) -> list[ReviewJob]:
        async with async_session() as session:
            result = await session.execute(select(ReviewJobModel))
            models = result.scalars().all()
        return [self._to_job(m) for m in models]

    async def update_status(
        self,
        job_id: str,
        status: ReviewJobStatus,
        *,
        error_message: str | None = None,
        report: ReviewReport | None = None,
    ) -> ReviewJob:
        async with async_session() as session:
            result = await session.execute(
                select(ReviewJobModel).where(ReviewJobModel.job_id == job_id)
            )
            model = result.scalar_one_or_none()
            if model is None:
                raise ReviewJobNotFoundError(job_id)

            current_status = ReviewJobStatus(model.status)
            allowed = self.allowed_transitions[current_status]
            if status not in allowed:
                raise InvalidReviewJobTransitionError(current_status, status)

            model.status = status.value
            if error_message is not None:
                model.error_message = error_message
            if report is not None:
                model.report = report.model_dump_json()
            if status == ReviewJobStatus.completed:
                model.completed_at = datetime.now(UTC)
            model.updated_at = datetime.now(UTC)

            await session.commit()
            await session.refresh(model)

        job = self._to_job(model)
        if status in {ReviewJobStatus.completed, ReviewJobStatus.failed, ReviewJobStatus.cancelled}:
            eq = event_queue_registry.get(job_id)
            if eq is not None:
                try:
                    eq.put_nowait(_QUEUE_DONE)
                except asyncio.QueueFull:
                    pass
            event_queue_registry.remove(job_id)
        return job

    async def add_progress_event(self, job_id: str, event: dict[str, object]) -> ReviewJob:
        async with async_session() as session:
            result = await session.execute(
                select(ReviewJobModel).where(ReviewJobModel.job_id == job_id)
            )
            model = result.scalar_one_or_none()
            if model is None:
                raise ReviewJobNotFoundError(job_id)

            events = json.loads(model.progress_events) if model.progress_events else []
            events.append(event)
            model.progress_events = json.dumps(events, ensure_ascii=False)
            model.updated_at = datetime.now(UTC)
            await session.commit()

        # 推送到内存事件队列（供 SSE 实时流）
        eq = event_queue_registry.get(job_id)
        if eq is not None:
            try:
                eq.put_nowait(event)
            except asyncio.QueueFull:
                pass
        return self._to_job(model)

    async def get_progress_events(self, job_id: str) -> list[dict[str, object]]:
        async with async_session() as session:
            result = await session.execute(
                select(ReviewJobModel).where(ReviewJobModel.job_id == job_id)
            )
            model = result.scalar_one_or_none()
        if model is None:
            raise ReviewJobNotFoundError(job_id)
        return json.loads(model.progress_events) if model.progress_events else []

    async def save_pr_info(self, job_id: str, pr_info: dict[str, object]) -> ReviewJob:
        async with async_session() as session:
            result = await session.execute(
                select(ReviewJobModel).where(ReviewJobModel.job_id == job_id)
            )
            model = result.scalar_one_or_none()
            if model is None:
                raise ReviewJobNotFoundError(job_id)
            model.pr_info = json.dumps(pr_info, ensure_ascii=False)
            model.updated_at = datetime.now(UTC)
            await session.commit()
        return self._to_job(model)

    async def save_pipeline_result(self, job_id: str, result: object) -> ReviewJob:
        async with async_session() as session:
            res = await session.execute(
                select(ReviewJobModel).where(ReviewJobModel.job_id == job_id)
            )
            model = res.scalar_one_or_none()
            if model is None:
                raise ReviewJobNotFoundError(job_id)
            if hasattr(result, "__dict__"):
                data = dict(result.__dict__)
            else:
                data = {"result": result}
            model.pipeline_result = json.dumps(data, ensure_ascii=False, default=str)
            model.updated_at = datetime.now(UTC)
            await session.commit()
        return self._to_job(model)

    async def save_report(self, job_id: str, report: ReviewReport) -> ReviewJob:
        """原地写回 report（不改状态机）。

        供 accept/reject/explain 等命令在 job 终态后修改 finding 后持久化使用。
        """
        async with async_session() as session:
            result = await session.execute(
                select(ReviewJobModel).where(ReviewJobModel.job_id == job_id)
            )
            model = result.scalar_one_or_none()
            if model is None:
                raise ReviewJobNotFoundError(job_id)
            model.report = report.model_dump_json()
            model.updated_at = datetime.now(UTC)
            await session.commit()
        return self._to_job(model)

    async def get_latest_job_by_pr_url(self, pr_url: str) -> ReviewJob | None:
        """按 pr_url 倒序取最近一条 job（命令路由定位 job 用）。"""
        async with async_session() as session:
            result = await session.execute(
                select(ReviewJobModel)
                .where(ReviewJobModel.pr_url == pr_url)
                .order_by(ReviewJobModel.created_at.desc())
                .limit(1)
            )
            model = result.scalar_one_or_none()
        return self._to_job(model) if model is not None else None

    def _to_job(self, model: ReviewJobModel) -> ReviewJob:
        """将 ORM 模型转换为领域对象 ReviewJob。"""
        report = None
        if model.report:
            try:
                report = ReviewReport.model_validate_json(model.report)
            except Exception:
                logger.warning("Failed to parse report for job %s", model.job_id)

        return ReviewJob(
            job_id=model.job_id,
            pr_url=model.pr_url,
            status=ReviewJobStatus(model.status),
            created_at=model.created_at,
            updated_at=model.updated_at,
            completed_at=model.completed_at,
            error_message=model.error_message,
            pr_info=json.loads(model.pr_info) if model.pr_info else None,
            progress_events=json.loads(model.progress_events) if model.progress_events else [],
            pipeline_result=json.loads(model.pipeline_result) if model.pipeline_result else None,
            report=report,
        )


# 全局单例，兼容现有导入
review_job_store = ReviewJobStore()


class _EventQueueRegistry:
    """内存事件队列注册表：SSE 实时流需要内存队列，DB 持久化不含队列引用。"""

    def __init__(self) -> None:
        self._queues: dict[str, asyncio.Queue] = {}

    def register(self, job_id: str, queue: asyncio.Queue) -> None:
        self._queues[job_id] = queue

    def get(self, job_id: str) -> asyncio.Queue | None:
        return self._queues.get(job_id)

    def remove(self, job_id: str) -> None:
        self._queues.pop(job_id, None)


event_queue_registry = _EventQueueRegistry()