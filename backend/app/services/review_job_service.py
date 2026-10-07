"""ReviewJobService：创建、查询、取消 ReviewJob。

create_job 采用 async fire-and-forget 模式：
  1. 立即创建 job（状态 pending）
  2. 通过 asyncio.create_task 后台执行 pipeline
  3. 接口快速返回 CreateReviewJobResponse
"""

import asyncio
import logging
from uuid import uuid4

from fastapi import HTTPException, status

from app.core.cache import redis_cache
from app.models.review_job import ReviewJob
from app.schemas.review import (
    ChangedFile,
    CreateReviewJobRequest,
    CreateReviewJobResponse,
    JobListItem,
    JobListResponse,
    PrInfo,
    ReviewJobDetailResponse,
    ReviewJobSnapshot,
    ReviewJobStatus,
    ReviewReport,
    ReviewReportStats,
)
from app.services.review_job_store import (
    InvalidReviewJobTransitionError,
    ReviewJobNotFoundError,
    ReviewJobStore,
    event_queue_registry,
    review_job_store,
)
from app.services.review_pipeline import ReviewPipeline
from app.services.review_task_runner import ReviewTaskRunner

# 缓存 TTL（秒）
_CACHE_TTL_JOB_DETAIL = 1800  # 30 分钟

logger = logging.getLogger(__name__)


class ReviewJobService:
    def __init__(
        self,
        store: ReviewJobStore,
        pipeline: ReviewPipeline | None = None,
        task_runner: ReviewTaskRunner | None = None,
    ) -> None:
        self._store = store
        self._pipeline = pipeline or ReviewPipeline(store)
        self._task_runner = task_runner

    async def create_job(self, request: CreateReviewJobRequest) -> CreateReviewJobResponse:
        job_id = f"rev_{uuid4().hex[:8]}"
        job = ReviewJob(job_id=job_id, pr_url=str(request.pr_url), github_token=request.github_token)
        eq = asyncio.Queue(maxsize=256)
        event_queue_registry.register(job_id, eq)
        job.event_queue = eq
        await self._store.create(job)
        await self._store.add_progress_event(
            job_id,
            {"step": "JOB_CREATED", "percent": 0, "message": "Review Job 已创建"},
        )
        # 提取 config 用于 pipeline，同时传入用户 token
        config = request.config.model_dump() if request.config else {}
        github_token = request.github_token
        if self._task_runner is not None:
            await self._task_runner.submit(job, lambda j: self._pipeline.run(j, config=config, github_token=github_token))
        else:
            asyncio.create_task(self._pipeline.run(job, config=config, github_token=github_token))

        return CreateReviewJobResponse(
            job_id=job_id,
            status=ReviewJobStatus.pending,
            stream_url=f"/api/v1/review/stream/{job_id}",
            report_url=f"/api/v1/review/jobs/{job_id}",
        )

    async def _make_pipeline_coro(self, job: ReviewJob):
        """构建 pipeline 协程，供 task_runner 调用。"""
        return await self._pipeline.run(job)

    async def get_report(self, job_id: str) -> ReviewReport:
        job = await self._get_job_or_404(job_id)
        if job.report is not None:
            return job.report

        return ReviewReport(
            summary="ReviewMind 已创建任务。真实 PR 拉取、Diff 解析和多 Agent 审查将在后续 PR 中接入。",
            risk_level="LOW",
            stats=ReviewReportStats(),
            changed_files=[],
            changed_symbols=[],
            findings=[],
            review_comment="## AI Review Summary\n\n当前任务已进入内存状态仓库，暂无真实风险发现。",
        )

    async def get_job_detail(self, job_id: str) -> ReviewJobDetailResponse:
        # 尝试从缓存读取（仅对 completed/failed/cancelled 的终态 job 缓存）
        cache_key = f"job:detail:{job_id}"
        cached = await redis_cache.get(cache_key)
        if cached is not None:
            return ReviewJobDetailResponse.model_validate(cached)

        job = await self._get_job_or_404(job_id)
        pr = _build_pr_info(job.pr_info) if job.pr_info else None
        progress = _last_progress(job)
        findings = _collect_findings(job)

        report: ReviewReport | None = None
        if job.report is not None:
            report = job.report
            logger.info("[SERVICE] get_job_detail | job=%s source=job.report findings=%d status=%s",
                        job_id, len(report.findings) if report.findings else 0, job.status)
        elif job.status == ReviewJobStatus.completed and job.pipeline_result:
            report = _build_report_from_pipeline(job)
            logger.info("[SERVICE] get_job_detail | job=%s source=_build_report_from_pipeline (fallback) findings=%d",
                        job_id, len(report.findings) if report.findings else 0)
        else:
            logger.info("[SERVICE] get_job_detail | job=%s source=no_report status=%s pipeline_result=%s",
                        job_id, job.status, "present" if job.pipeline_result else "None")

        detail = ReviewJobDetailResponse(
            job_id=job.job_id,
            status=job.status,
            pr=pr,
            progress=progress,
            findings=findings,
            report=report,
            created_at=job.created_at,
            completed_at=job.completed_at,
            updated_at=job.updated_at,
            error_message=job.error_message,
        )

        # 终态 job 缓存报告（避免重复构建）
        if job.status in {ReviewJobStatus.completed, ReviewJobStatus.failed, ReviewJobStatus.cancelled}:
            await redis_cache.set(cache_key, detail.model_dump(mode="json"), ttl_seconds=_CACHE_TTL_JOB_DETAIL)

        return detail

    async def get_job(self, job_id: str) -> ReviewJobSnapshot:
        job = await self._get_job_or_404(job_id)
        return ReviewJobSnapshot(
            job_id=job.job_id,
            pr_url=job.pr_url,
            status=job.status,
            created_at=job.created_at,
            updated_at=job.updated_at,
            error_message=job.error_message,
            progress_events=job.progress_events,
            pipeline_result=job.pipeline_result,
        )

    async def cancel_job(self, job_id: str) -> ReviewJobDetailResponse:
        await self._get_job_or_404(job_id)
        # 先取消后台执行（若在运行），避免状态改为 cancelled 后 pipeline 继续
        # 烧完剩余 LLM 调用、并在 _finish 里触发 cancelled->completed 非法转移
        if self._task_runner is not None:
            self._task_runner.cancel(job_id)
        try:
            await self._store.update_status(job_id, ReviewJobStatus.cancelled)
        except InvalidReviewJobTransitionError as exc:
            raise HTTPException(
                status_code=status.HTTP_409_CONFLICT,
                detail=str(exc),
            ) from exc
        return await self.get_job_detail(job_id)

    async def get_job_list(self, page: int = 1, page_size: int = 10) -> JobListResponse:
        all_jobs = sorted(
            await self._store.get_all(),
            key=lambda j: j.created_at,
            reverse=True,
        )
        total = len(all_jobs)
        offset = (page - 1) * page_size
        page_jobs = all_jobs[offset:offset + page_size]
        items = [_to_job_list_item(j) for j in page_jobs]
        return JobListResponse(items=items, page=page, page_size=page_size, total=total)

    async def _get_job_or_404(self, job_id: str) -> ReviewJob:
        try:
            return await self._store.get(job_id)
        except ReviewJobNotFoundError as exc:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Review job not found: {job_id}",
            ) from exc


def _build_pr_info(pr_info: dict[str, object]) -> PrInfo:
    return PrInfo(
        owner=str(pr_info.get("owner", "")),
        repo=str(pr_info.get("repo", "")),
        number=int(pr_info.get("pull_number", 0)),
        title=str(pr_info.get("title", "")),
        author=str(pr_info.get("author", "")),
        base_branch=str(pr_info.get("base", {}).get("ref", "") if isinstance(pr_info.get("base"), dict) else ""),
        head_branch=str(pr_info.get("head", {}).get("ref", "") if isinstance(pr_info.get("head"), dict) else ""),
        changed_files=int(pr_info.get("changed_files", 0)),
        additions=int(pr_info.get("additions", 0)),
        deletions=int(pr_info.get("deletions", 0)),
        html_url=str(pr_info.get("html_url", "")),
    )


def _last_progress(job: ReviewJob) -> "ReviewProgressEvent | None":
    from app.schemas.review import ReviewProgressEvent
    if not job.progress_events:
        return None
    last = job.progress_events[-1]
    return ReviewProgressEvent(
        step=str(last.get("step", "UNKNOWN")),
        percent=int(last.get("percent", 0)),
        message=str(last.get("message", "")),
        type=str(last.get("type", "progress")),
    )


def _collect_findings(job: ReviewJob) -> list["ReviewFinding"]:
    from app.schemas.review import ReviewFinding
    if job.report and job.report.findings:
        return job.report.findings
    return []


def _build_report_from_pipeline(job: ReviewJob) -> ReviewReport:
    pipe = job.pipeline_result or {}
    filtered = pipe.get("filtered_files", {})
    included = filtered.get("included_files", []) if isinstance(filtered, dict) else []
    logger.info("[SERVICE] _build_report_from_pipeline | job=%s included_files=%d",
                job.job_id, len(included))
    changed_files = [
        ChangedFile(
            filename=f.get("filename", ""),
            status=f.get("status", ""),
            additions=f.get("additions", 0),
            deletions=f.get("deletions", 0),
            changes=f.get("changes", f.get("additions", 0) + f.get("deletions", 0)),
            patch=f.get("patch"),
            risk_count=0,
        )
        for f in included
        if isinstance(f, dict)
    ]
    total_additions = sum(cf.additions for cf in changed_files)
    return ReviewReport(
        summary=f"已完成基础 Diff 分析，共保留 {len(changed_files)} 个文件，+{total_additions} 行。",
        risk_level="LOW",
        stats=ReviewReportStats(),
        changed_files=changed_files,
        changed_symbols=[],
        findings=[],
        review_comment="## AI Review Summary\n\n基础 Review Pipeline 已完成。",
    )


# 生产装配：注入 TaskRunner 管理后台任务（强引用 + 异常日志 + 可取消），
# 替代裸 asyncio.create_task（异常只在 GC 时打日志、cancel 无法停止执行）
review_task_runner = ReviewTaskRunner(review_job_store)
review_job_service = ReviewJobService(review_job_store, task_runner=review_task_runner)


def _to_job_list_item(job: ReviewJob) -> JobListItem:
    report = job.report
    return JobListItem(
        job_id=job.job_id,
        status=job.status,
        pr_title=str(job.pr_info.get("title", "")) if job.pr_info else "",
        pr_url=job.pr_url,
        risk_level=report.risk_level if report else "LOW",
        finding_count=len(report.findings) if report else 0,
        created_at=job.created_at,
    )