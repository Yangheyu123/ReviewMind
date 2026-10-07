"""ReviewPipeline：委托 ReviewGraph 执行完整的 AI Review 工作流。"""

import json
from dataclasses import dataclass
from typing import Any

from app.graph.review_graph import ReviewGraph
from app.models.review_job import ReviewJob
from app.services.github_client import GitHubClient
from app.services.review_job_store import ReviewJobStore


@dataclass(frozen=True)
class ReviewPipelineResult:
    pr_info: dict[str, Any]
    filtered_files: dict[str, Any]
    parsed_diff: list[dict[str, Any]]


class ReviewPipeline:
    def __init__(self, store: ReviewJobStore, github_client: GitHubClient | None = None) -> None:
        self._store = store
        self._github_client = github_client or GitHubClient()

    async def run(self, job: ReviewJob, config: dict[str, Any] | None = None, github_token: str | None = None) -> ReviewPipelineResult:
        """委托 ReviewGraph 执行完整的多 Agent + LLM Review 工作流。"""
        client = GitHubClient(token=github_token) if github_token else self._github_client
        graph = ReviewGraph(self._store, client)
        result = await graph.run(job, config=config)

        # SSE 推送文件列表
        included = result.filtered_files.get("included_files", [])
        file_names = [
            f.get("filename", f.get("file", "unknown")) if isinstance(f, dict) else getattr(f, "filename", "unknown")
            for f in included
        ]
        if file_names:
            await self._store.add_progress_event(
                job.job_id,
                {
                    "type": "chunk",
                    "target": "files",
                    "content": json.dumps(file_names, ensure_ascii=False),
                },
            )

        # Phase 4：引擎观测元数据（findings/tokens/llm 请求数）合入持久化，
        # 替代此前"外层覆盖引擎保存"导致观测数据丢失的缺口
        from types import SimpleNamespace
        meta = getattr(result, "engine_meta", None) or {}
        payload = SimpleNamespace(
            pr_info=result.pr_info,
            filtered_files=result.filtered_files,
            parsed_diff=result.parsed_diff,
            findings=meta.get("findings", []),
            tokens_used=meta.get("tokens_used", 0),
            group_summaries=meta.get("group_summaries", []),
            llm_request_count=meta.get("llm_request_count", 0),
        )
        await self._store.save_pipeline_result(job.job_id, payload)
        return ReviewPipelineResult(
            pr_info=result.pr_info,
            filtered_files=result.filtered_files,
            parsed_diff=result.parsed_diff,
        )