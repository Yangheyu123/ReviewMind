"""异步内存版 ReviewJobStore —— 供单测快速运行，不依赖数据库。

接口与 app.services.review_job_store.ReviewJobStore 完全一致（async），
行为语义对齐 DB 版：状态转移校验、终态推送 _QUEUE_DONE、事件双写。
"""

from datetime import UTC, datetime
from typing import Any

from app.models.review_job import ReviewJob, _QUEUE_DONE
from app.schemas.review import ReviewJobStatus, ReviewReport
from app.services.review_job_store import (
    InvalidReviewJobTransitionError,
    ReviewJobNotFoundError,
    ReviewJobStore,
    event_queue_registry,
)

_TERMINAL_STATUSES = {ReviewJobStatus.completed, ReviewJobStatus.failed, ReviewJobStatus.cancelled}


class MemoryReviewJobStore:
    """与 ReviewJobStore 同接口的内存实现（原同步内存 store 的异步复刻）。"""

    allowed_transitions = ReviewJobStore.allowed_transitions

    def __init__(self) -> None:
        self._jobs: dict[str, ReviewJob] = {}

    async def create(self, job: ReviewJob) -> ReviewJob:
        self._jobs[job.job_id] = job
        return job

    async def get(self, job_id: str) -> ReviewJob:
        job = self._jobs.get(job_id)
        if job is None:
            raise ReviewJobNotFoundError(job_id)
        return job

    async def get_all(self) -> list[ReviewJob]:
        return list(self._jobs.values())

    async def update_status(
        self,
        job_id: str,
        status: ReviewJobStatus,
        *,
        error_message: str | None = None,
        report: ReviewReport | None = None,
    ) -> ReviewJob:
        job = await self.get(job_id)
        allowed = self.allowed_transitions[job.status]
        if status not in allowed:
            raise InvalidReviewJobTransitionError(job.status, status)

        job.status = status
        if error_message is not None:
            job.error_message = error_message
        if report is not None:
            job.report = report
        if status == ReviewJobStatus.completed:
            job.completed_at = datetime.now(UTC)
        job.updated_at = datetime.now(UTC)

        if status in _TERMINAL_STATUSES:
            eq = event_queue_registry.get(job_id)
            if eq is not None:
                try:
                    eq.put_nowait(_QUEUE_DONE)
                except Exception:
                    pass
            event_queue_registry.remove(job_id)
        return job

    async def add_progress_event(self, job_id: str, event: dict[str, object]) -> ReviewJob:
        job = await self.get(job_id)
        job.progress_events.append(event)
        job.updated_at = datetime.now(UTC)
        eq = event_queue_registry.get(job_id)
        if eq is not None:
            try:
                eq.put_nowait(event)
            except Exception:
                pass
        return job

    async def get_progress_events(self, job_id: str) -> list[dict[str, object]]:
        job = await self.get(job_id)
        return job.progress_events

    async def save_pr_info(self, job_id: str, pr_info: dict[str, object]) -> ReviewJob:
        job = await self.get(job_id)
        job.pr_info = dict(pr_info)
        job.updated_at = datetime.now(UTC)
        return job

    async def save_review_memory(self, repo: str, findings: list, min_chars: int = 10) -> int:
        if not hasattr(self, "review_memory"):
            self.review_memory: list[dict] = []
        saved = 0
        for f in findings:
            if len(str(f.get("description") or "")) < min_chars:
                continue
            self.review_memory.append({**f, "repo": repo, "created_at": "2026-09-20T00:00:00"})
            saved += 1
        return saved

    async def recall_review_memory(self, repo: str, changed_files: list[str], window_days: int = 180) -> list[dict]:
        names = {f.rsplit("/", 1)[-1] for f in changed_files if "/" in f}
        dirs = {f.rsplit("/", 1)[0] for f in changed_files if "/" in f}
        out = []
        for m in getattr(self, "review_memory", []):
            if m.get("repo") != repo:
                continue
            f = m.get("file") or ""
            f_name = f.rsplit("/", 1)[-1] if "/" in f else f
            f_dir = f.rsplit("/", 1)[0] if "/" in f else ""
            if f_name in names or (f_dir and f_dir in dirs):
                out.append(m)
        return out[:50]

    async def save_llm_requests(self, job_id: str, records: list) -> int:
        if not hasattr(self, "llm_requests"):
            self.llm_requests = []
        self.llm_requests.extend(records)
        return len(records)

    async def save_pipeline_result(self, job_id: str, result: Any) -> ReviewJob:
        job = await self.get(job_id)
        data = dict(result.__dict__) if hasattr(result, "__dict__") else {"result": result}
        job.pipeline_result = data
        job.updated_at = datetime.now(UTC)
        return job
