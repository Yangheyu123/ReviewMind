"""GitHub Webhook 触发 Review Job 的服务层。

支持三类事件：
- ``issue_comment``：解析斜杠命令（/review、/explain、/accept、/reject）或触发词后分发。
- ``pull_request``（action=opened）：可选自动触发评审（受开关与白名单约束）。
- 其它事件：忽略。

所有命令的处置结果通过 issue 评论回写到 PR，形成人机协作闭环。
"""

from __future__ import annotations

import asyncio
import hashlib
import hmac
import json
import logging
from dataclasses import dataclass
from typing import Any

from app.core.cache import redis_cache
from app.core.config import settings
from app.schemas.review import (
    CreateReviewJobRequest,
    DebateVerdict,
    FindingStatus,
    ReviewJobStatus,
)
from app.services.debate_service import DebateService, FindingNotFoundError, debate_service as default_debate_service
from app.services.github_client import GitHubClient
from app.services.github_comment import GitHubCommentError, post_pr_comment
from app.services.github_commands import (
    AcceptCommand,
    CommentCommand,
    ExplainCommand,
    RejectCommand,
    ReviewCommand,
    parse_comment_command,
)
from app.services.review_job_service import ReviewJobService, review_job_service
from app.services.review_job_store import review_job_store

logger = logging.getLogger(__name__)

_DELIVERY_CACHE_TTL_SECONDS = 24 * 60 * 60


class GitHubWebhookError(RuntimeError):
    def __init__(self, message: str, status_code: int = 400) -> None:
        super().__init__(message)
        self.status_code = status_code


@dataclass(frozen=True)
class GitHubWebhookResult:
    accepted: bool
    ignored: bool
    reason: str
    job_id: str | None = None
    pr_url: str | None = None
    start_comment_url: str | None = None


@dataclass(frozen=True)
class PrCommentContext:
    """从 webhook payload 提取的 PR 评论上下文。"""

    owner: str
    repo: str
    pull_number: int
    pr_url: str
    commenter: str
    body: str


async def handle_github_webhook(
    *,
    event: str | None,
    delivery_id: str | None,
    signature: str | None,
    raw_body: bytes,
    service: ReviewJobService = review_job_service,
    debate_service: DebateService | None = None,
    github_client: GitHubClient | None = None,
) -> GitHubWebhookResult:
    """处理 GitHub Webhook。"""
    verify_github_signature(raw_body=raw_body, signature=signature)

    if not delivery_id:
        raise GitHubWebhookError("Missing X-GitHub-Delivery header", status_code=400)
    if await _is_duplicate_delivery(delivery_id):
        return GitHubWebhookResult(accepted=False, ignored=True, reason="duplicate_delivery")

    payload = _decode_payload(raw_body)

    if event == "issue_comment":
        return await _handle_issue_comment(payload, service, debate_service)
    if event == "pull_request":
        return await _handle_pull_request(payload, service)

    return GitHubWebhookResult(accepted=False, ignored=True, reason="unsupported_event")


# ---------------------------------------------------------------------------
# issue_comment：命令路由
# ---------------------------------------------------------------------------


async def _handle_issue_comment(
    payload: dict[str, Any],
    service: ReviewJobService,
    debate_service: DebateService | None,
) -> GitHubWebhookResult:
    context = _extract_pr_comment_context(payload)
    if context is None:
        return GitHubWebhookResult(accepted=False, ignored=True, reason="not_a_pr_comment")

    command = parse_comment_command(
        context.body, context.commenter, trigger=settings.github_review_trigger
    )
    if command is None:
        return GitHubWebhookResult(accepted=False, ignored=True, reason="no_review_trigger")

    if isinstance(command, ReviewCommand):
        return await _start_review(context, service)

    # 命令型操作（explain/accept/reject）：需要先定位已完成 job
    debate = debate_service or default_debate_service
    return await _dispatch_finding_command(command, context, debate)


async def _start_review(
    context: PrCommentContext,
    service: ReviewJobService,
) -> GitHubWebhookResult:
    """触发一次完整审查（/review 或触发词）。"""
    request = CreateReviewJobRequest(pr_url=context.pr_url, github_token=settings.github_token)
    response = await service.create_job(request)

    start_comment_url = await _post_start_comment(context, response.job_id)
    asyncio.create_task(_post_final_comment_when_done(context, response.job_id, service))

    return GitHubWebhookResult(
        accepted=True,
        ignored=False,
        reason="review_job_created",
        job_id=response.job_id,
        pr_url=context.pr_url,
        start_comment_url=start_comment_url,
    )


async def _dispatch_finding_command(
    command: CommentCommand,
    context: PrCommentContext,
    debate: DebateService,
) -> GitHubWebhookResult:
    """分发 /explain /accept /reject 到辩论服务，并回写结果评论。"""
    job = await review_job_store.get_latest_job_by_pr_url(context.pr_url)
    if job is None or job.report is None:
        await _post_text_comment(
            context,
            "### ReviewMind\n\n尚未找到该 PR 的审查报告，请先评论 `/review` 触发审查。",
        )
        return GitHubWebhookResult(accepted=False, ignored=True, reason="no_review_job")

    try:
        if isinstance(command, ExplainCommand):
            outcome = await debate.explain_finding(
                job.job_id, command.finding_id, command.message
            )
            body = _build_explain_comment(command, outcome.result, outcome.report_changed)
        elif isinstance(command, AcceptCommand):
            await debate.set_finding_status(
                job.job_id, command.finding_id, FindingStatus.accepted
            )
            body = _build_status_comment("accept", command.finding_id)
        elif isinstance(command, RejectCommand):
            await debate.set_finding_status(
                job.job_id, command.finding_id, FindingStatus.rejected
            )
            body = _build_status_comment("reject", command.finding_id)
        else:  # pragma: no cover - 不可达
            return GitHubWebhookResult(accepted=False, ignored=True, reason="unknown_command")
    except FindingNotFoundError:
        await _post_text_comment(
            context,
            f"### ReviewMind\n\n未找到 finding `{command.finding_id}`，请检查 ID 后重试。",
        )
        return GitHubWebhookResult(accepted=False, ignored=True, reason="finding_not_found")

    await _post_text_comment(context, body)
    return GitHubWebhookResult(
        accepted=True,
        ignored=False,
        reason="command_handled",
        job_id=job.job_id,
        pr_url=context.pr_url,
    )


# ---------------------------------------------------------------------------
# pull_request：opened 自动触发
# ---------------------------------------------------------------------------


async def _handle_pull_request(
    payload: dict[str, Any],
    service: ReviewJobService,
) -> GitHubWebhookResult:
    if not settings.github_auto_review_on_pr_opened:
        return GitHubWebhookResult(accepted=False, ignored=True, reason="auto_review_disabled")
    if payload.get("action") != "opened":
        return GitHubWebhookResult(accepted=False, ignored=True, reason="pr_action_not_opened")

    repository = payload.get("repository")
    pr = payload.get("pull_request")
    if not isinstance(repository, dict) or not isinstance(pr, dict):
        return GitHubWebhookResult(accepted=False, ignored=True, reason="invalid_payload")

    owner_payload = repository.get("owner", {})
    owner = str(owner_payload.get("login", "")) if isinstance(owner_payload, dict) else ""
    repo = str(repository.get("name", ""))
    if not _is_repo_allowed(owner, repo):
        return GitHubWebhookResult(accepted=False, ignored=True, reason="repo_not_allowed")

    pull_number = int(pr.get("number", 0))
    pr_url = str(
        pr.get("html_url")
        or f"https://github.com/{owner}/{repo}/pull/{pull_number}"
    )
    if pull_number <= 0:
        return GitHubWebhookResult(accepted=False, ignored=True, reason="invalid_payload")

    context = PrCommentContext(
        owner=owner,
        repo=repo,
        pull_number=pull_number,
        pr_url=pr_url,
        commenter=str(pr.get("user", {}).get("login", "")) if isinstance(pr.get("user"), dict) else "",
        body="",
    )
    return await _start_review(context, service)


# ---------------------------------------------------------------------------
# payload 解析
# ---------------------------------------------------------------------------


def _extract_pr_comment_context(payload: dict[str, Any]) -> PrCommentContext | None:
    """从 issue_comment payload 提取 PR 评论上下文；非 PR 评论或 Bot 评论返回 None。"""
    if payload.get("action") != "created":
        return None

    issue = payload.get("issue")
    comment = payload.get("comment")
    repository = payload.get("repository")
    if not isinstance(issue, dict) or not isinstance(comment, dict) or not isinstance(repository, dict):
        return None
    if "pull_request" not in issue:
        return None

    user = comment.get("user")
    commenter = str(user.get("login", "")) if isinstance(user, dict) else ""
    user_type = str(user.get("type", "")) if isinstance(user, dict) else ""
    if _is_bot_comment(commenter, user_type):
        return None

    body = str(comment.get("body", ""))
    owner_payload = repository.get("owner", {})
    owner = str(owner_payload.get("login", "")) if isinstance(owner_payload, dict) else ""
    repo = str(repository.get("name", ""))
    try:
        pull_number = int(issue.get("number", 0))
    except (TypeError, ValueError):
        return None
    if not owner or not repo or pull_number <= 0:
        return None
    if not _is_repo_allowed(owner, repo):
        return None

    pr_url = str(
        issue.get("html_url")
        or f"https://github.com/{owner}/{repo}/pull/{pull_number}"
    )
    return PrCommentContext(
        owner=owner,
        repo=repo,
        pull_number=pull_number,
        pr_url=pr_url,
        commenter=commenter,
        body=body,
    )


async def _is_duplicate_delivery(delivery_id: str) -> bool:
    cache_key = f"github:webhook:delivery:{delivery_id}"
    if await redis_cache.get(cache_key) is not None:
        return True
    await redis_cache.set(cache_key, True, ttl_seconds=_DELIVERY_CACHE_TTL_SECONDS)
    return False


def _decode_payload(raw_body: bytes) -> dict[str, Any]:
    try:
        payload = json.loads(raw_body.decode("utf-8"))
    except json.JSONDecodeError as exc:
        raise GitHubWebhookError("Invalid GitHub webhook JSON payload", status_code=400) from exc
    if not isinstance(payload, dict):
        raise GitHubWebhookError("GitHub webhook payload must be an object", status_code=400)
    return payload


def _is_bot_comment(commenter: str, user_type: str) -> bool:
    bot_login = settings.github_bot_login.strip().lower()
    return user_type.lower() == "bot" or (bool(bot_login) and commenter.lower() == bot_login)


def _is_repo_allowed(owner: str, repo: str) -> bool:
    allowed = [item.lower() for item in settings.github_allowed_repos]
    return not allowed or f"{owner}/{repo}".lower() in allowed


# ---------------------------------------------------------------------------
# 评论回写
# ---------------------------------------------------------------------------


async def _post_start_comment(context: PrCommentContext, job_id: str) -> str | None:
    body = (
        "### ReviewMind 已开始审查\n\n"
        f"- PR: #{context.pull_number}\n"
        f"- Job: `{job_id}`\n"
        f"- Triggered by: @{context.commenter}\n\n"
        "完成后我会在本 PR 下追加审查报告。"
    )
    try:
        result = await post_pr_comment(
            owner=context.owner,
            repo=context.repo,
            pull_number=context.pull_number,
            body=body,
            github_token=settings.github_token,
        )
        return result.html_url
    except GitHubCommentError as exc:
        logger.warning("[WEBHOOK] Failed to post start comment for job=%s: %s", job_id, exc)
        return None


async def _post_text_comment(context: PrCommentContext, body: str) -> None:
    try:
        await post_pr_comment(
            owner=context.owner,
            repo=context.repo,
            pull_number=context.pull_number,
            body=body,
            github_token=settings.github_token,
        )
    except GitHubCommentError as exc:
        logger.warning("[WEBHOOK] Failed to post comment: %s", exc)


def _build_explain_comment(
    command: ExplainCommand,
    result,
    report_changed: bool,
) -> str:
    verdict_zh = {
        DebateVerdict.keep: "维持原结论",
        DebateVerdict.downgrade: "下调风险等级",
        DebateVerdict.dismiss: "撤销该发现（误报）",
    }.get(result.verdict, result.verdict.value)
    lines = [
        f"### ReviewMind 辩论回复 · finding `{command.finding_id}`",
        "",
        f"**裁决：** {verdict_zh}" + (
            f"（新等级 {result.revised_level}）"
            if result.verdict == DebateVerdict.downgrade and result.revised_level
            else ""
        ),
        "",
        result.explanation,
    ]
    if report_changed:
        lines.append("")
        lines.append("_报告已据此更新。_")
    return "\n".join(lines)


def _build_status_comment(action: str, finding_id: str) -> str:
    label = "已接受" if action == "accept" else "已驳回"
    emoji = "✅" if action == "accept" else "❌"
    return (
        f"### ReviewMind 处置确认 · finding `{finding_id}`\n\n"
        f"{emoji} 该发现 {label}，报告摘要已更新。"
    )


async def _post_final_comment_when_done(
    context: PrCommentContext,
    job_id: str,
    service: ReviewJobService,
) -> None:
    timeout_seconds = max(settings.github_webhook_result_timeout_seconds, 1)
    poll_seconds = max(settings.github_webhook_result_poll_seconds, 0.2)
    deadline = asyncio.get_event_loop().time() + timeout_seconds

    while asyncio.get_event_loop().time() < deadline:
        detail = await service.get_job_detail(job_id)
        if detail.status in {ReviewJobStatus.completed, ReviewJobStatus.failed, ReviewJobStatus.cancelled}:
            body = _build_final_comment_body(detail)
            try:
                await post_pr_comment(
                    owner=context.owner,
                    repo=context.repo,
                    pull_number=context.pull_number,
                    body=body,
                    github_token=settings.github_token,
                )
            except GitHubCommentError as exc:
                logger.warning("[WEBHOOK] Failed to post final comment for job=%s: %s", job_id, exc)
            return
        await asyncio.sleep(poll_seconds)

    logger.warning("[WEBHOOK] Timed out waiting for review job=%s", job_id)


def _build_final_comment_body(detail) -> str:
    if detail.status == ReviewJobStatus.completed and detail.report and detail.report.review_comment:
        return detail.report.review_comment
    if detail.status == ReviewJobStatus.failed:
        reason = detail.error_message or "unknown error"
        return f"### ReviewMind 审查失败\n\nJob `{detail.job_id}` 执行失败：{reason}"
    if detail.status == ReviewJobStatus.cancelled:
        return f"### ReviewMind 审查已取消\n\nJob `{detail.job_id}` 已被取消。"
    return f"### ReviewMind 审查结束\n\nJob `{detail.job_id}` 状态：{detail.status}"


def verify_github_signature(*, raw_body: bytes, signature: str | None) -> None:
    """校验 GitHub Webhook 的 X-Hub-Signature-256。"""
    secret = settings.github_webhook_secret
    if not secret:
        raise GitHubWebhookError("GITHUB_WEBHOOK_SECRET is not configured", status_code=500)
    if not signature:
        raise GitHubWebhookError("Missing X-Hub-Signature-256 header", status_code=401)

    expected = "sha256=" + hmac.new(
        secret.encode("utf-8"),
        raw_body,
        hashlib.sha256,
    ).hexdigest()
    if not hmac.compare_digest(expected, signature):
        raise GitHubWebhookError("Invalid GitHub webhook signature", status_code=401)
