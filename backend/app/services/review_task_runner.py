"""后台任务运行器：管理 Review Pipeline 的异步后台执行与防重复保护。"""

from __future__ import annotations

import asyncio
import logging
from typing import Any

from app.models.review_job import ReviewJob
from app.schemas.review import ReviewJobStatus
from app.services.review_job_store import ReviewJobStore

logger = logging.getLogger(__name__)


class ReviewTaskRunner:
    """管理后台 review 任务的生命周期。

    职责：
    - 在后台 asyncio.Task 中执行 pipeline
    - 防止同一 job 重复启动
    - 提供运行状态查询
    """

    def __init__(self, store: ReviewJobStore) -> None:
        self._store = store
        self._running_tasks: dict[str, asyncio.Task[Any]] = {}
        self._running_job_ids: set[str] = set()

    def is_running(self, job_id: str) -> bool:
        """检查指定 job 是否正在后台执行。"""
        return job_id in self._running_job_ids

    def running_count(self) -> int:
        """当前后台运行中的任务数量。"""
        return len(self._running_job_ids)

    async def submit(self, job: ReviewJob, coro_factory) -> None:
        """提交后台任务。

        Args:
            job: ReviewJob 实例
            coro_factory: 接收 job 作为参数的 async callable，返回 pipeline result

        Raises:
            RuntimeError: 同一 job 已在运行中
        """
        if job.job_id in self._running_job_ids:
            raise RuntimeError(f"Review job {job.job_id} is already running")

        self._running_job_ids.add(job.job_id)

        task = asyncio.create_task(
            self._execute(job.job_id, coro_factory(job)),
            name=f"review_job_{job.job_id}",
        )
        self._running_tasks[job.job_id] = task
        task.add_done_callback(lambda _: self._cleanup(job.job_id))

    async def _execute(self, job_id: str, coro) -> Any:
        """包装执行，确保异常被捕获并记录。

        注意：asyncio.CancelledError 继承自 BaseException，不会被这里吞掉，
        cancel() 后任务会真正终止，pipeline 的 _finish 不再执行。
        """
        try:
            return await coro
        except Exception:
            logger.exception("[TaskRunner] Pipeline execution failed for job=%s", job_id)

    def cancel(self, job_id: str) -> bool:
        """取消运行中的后台任务。

        Returns:
            True 表示存在运行中任务并已请求取消；False 表示该 job 当前没有后台任务。
        """
        task = self._running_tasks.get(job_id)
        if task is None or task.done():
            return False
        task.cancel()
        logger.info("[TaskRunner] Cancelled background task for job=%s", job_id)
        return True

    def _cleanup(self, job_id: str) -> None:
        """任务完成后清理运行状态。"""
        self._running_job_ids.discard(job_id)
        self._running_tasks.pop(job_id, None)