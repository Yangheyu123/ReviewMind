"""辩论 / Finding 处置服务。

承载 /explain /accept /reject 命令的业务逻辑：
- 定位 job 与 finding
- 调用辩论 Agent（/explain）或直接改 status（/accept /reject）
- 持久化对话历史与更新后的报告
- 重生成 review_comment 摘要
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

from app.agents import debate_agent
from app.agents.report_agent import rebuild_review_comment
from app.schemas.review import (
    ConversationRole,
    ConversationTurn,
    DebateResult,
    DebateVerdict,
    FindingStatus,
    ReviewFinding,
    ReviewReport,
)
from app.services.conversation_store import conversation_store
from app.services.github_client import GitHubClient
from app.schemas.github import GitHubPullRequestRef
from app.services.review_job_store import ReviewJobNotFoundError, review_job_store

logger = logging.getLogger(__name__)


class FindingNotFoundError(LookupError):
    def __init__(self, job_id: str, finding_id: str) -> None:
        super().__init__(f"Finding {finding_id} not found in job {job_id}")
        self.job_id = job_id
        self.finding_id = finding_id


@dataclass(frozen=True)
class DebateOutcome:
    """一次 /explain 的最终产出（含可能更新后的报告摘要）。"""

    finding: ReviewFinding
    result: DebateResult
    review_comment: str
    report_changed: bool


class DebateService:
    """辩论与 finding 处置。"""

    def __init__(self, github_client: GitHubClient | None = None) -> None:
        self._github_client = github_client

    async def explain_finding(
        self,
        job_id: str,
        finding_id: str,
        challenge: str,
    ) -> DebateOutcome:
        """对单条 finding 发起辩论，必要时更新报告。"""
        job = await review_job_store.get(job_id)
        report, finding = _resolve_finding(job.report, job_id, finding_id)

        code_context = await self._build_code_context(job, report, finding)
        history = await conversation_store.list(job_id, finding_id=finding_id)

        result: DebateResult = await debate_agent.run_async(
            finding=finding.model_dump(mode="json"),
            challenge=challenge,
            code_context=code_context,
            tech_stack_prompt="",
            history=[t.model_dump(mode="json") for t in history],
        )

        # 持久化对话：开发者异议 + Agent 解释
        await conversation_store.append(
            ConversationTurn(
                job_id=job_id,
                finding_id=finding_id,
                role=ConversationRole.user,
                content=challenge,
            )
        )
        await conversation_store.append(
            ConversationTurn(
                job_id=job_id,
                finding_id=finding_id,
                role=ConversationRole.assistant,
                content=_format_assistant_reply(result),
            )
        )

        report_changed = False
        if result.verdict == DebateVerdict.dismiss:
            finding.status = FindingStatus.dismissed.value
            report_changed = True
        elif result.verdict == DebateVerdict.downgrade and result.revised_level:
            finding.level = result.revised_level
            report_changed = True

        if report_changed:
            report.review_comment = rebuild_review_comment(report)
            await review_job_store.save_report(job_id, report)
            logger.info(
                "[DEBATE_SERVICE] report updated | job=%s finding=%s verdict=%s",
                job_id,
                finding_id,
                result.verdict,
            )

        return DebateOutcome(
            finding=finding,
            result=result,
            review_comment=report.review_comment,
            report_changed=report_changed,
        )

    async def set_finding_status(
        self,
        job_id: str,
        finding_id: str,
        status: FindingStatus,
    ) -> ReviewReport:
        """accept/reject 公共路径：改 status → 重生成摘要 → 持久化。"""
        job = await review_job_store.get(job_id)
        report, finding = _resolve_finding(job.report, job_id, finding_id)

        finding.status = status.value
        report.review_comment = rebuild_review_comment(report)
        await review_job_store.save_report(job_id, report)
        logger.info(
            "[DEBATE_SERVICE] finding status set | job=%s finding=%s status=%s",
            job_id,
            finding_id,
            status.value,
        )
        return report

    async def _build_code_context(
        self,
        job,
        report: ReviewReport,
        finding: ReviewFinding,
    ) -> str:
        """优先从 report.changed_files 取 patch（零网络），缺失时回退 GitHub 拉取。"""
        for changed in report.changed_files:
            if changed.filename == finding.file and changed.patch:
                return _format_patch_context(finding, changed.patch)

        # 回退：从 GitHub 拉取该 PR 的文件 diff
        patch = await self._fetch_patch_fallback(job, finding.file)
        return _format_patch_context(finding, patch) if patch else ""

    async def _fetch_patch_fallback(self, job, filename: str) -> str | None:
        pr_info = job.pr_info or {}
        owner = pr_info.get("owner")
        repo = pr_info.get("repo")
        number = pr_info.get("pull_number") or pr_info.get("number")
        if not owner or not repo or not number:
            return None
        try:
            client = self._github_client or GitHubClient(token=job.github_token)
            ref = GitHubPullRequestRef(
                owner=str(owner),
                repo=str(repo),
                pull_number=int(number),
                html_url=job.pr_url,
            )
            files = await client.fetch_pull_request_files(ref)
        except Exception as exc:  # noqa: BLE001
            logger.warning("[DEBATE_SERVICE] fetch patch fallback failed | %s: %s", type(exc).__name__, exc)
            return None
        for f in files:
            if f.filename == filename and f.patch:
                return f.patch
        return None


def _resolve_finding(
    report: ReviewReport | None,
    job_id: str,
    finding_id: str,
) -> tuple[ReviewReport, ReviewFinding]:
    if report is None:
        raise FindingNotFoundError(job_id, finding_id)
    for finding in report.findings:
        if finding.id == finding_id:
            return report, finding
    raise FindingNotFoundError(job_id, finding_id)


def _format_patch_context(finding: ReviewFinding, patch: str) -> str:
    """从完整 patch 中截取 finding 附近片段。"""
    if len(patch) <= 4000:
        return patch
    return patch[:4000] + "\n... (truncated)"


def _format_assistant_reply(result: DebateResult) -> str:
    verdict_zh = {
        DebateVerdict.keep: "维持原结论",
        DebateVerdict.downgrade: "下调风险等级",
        DebateVerdict.dismiss: "撤销该发现",
    }.get(result.verdict, result.verdict.value)
    prefix = f"【裁决：{verdict_zh}】"
    if result.verdict == DebateVerdict.downgrade and result.revised_level:
        prefix += f"（新等级 {result.revised_level}）"
    return f"{prefix}\n{result.explanation}"


# 全局单例
debate_service = DebateService()
