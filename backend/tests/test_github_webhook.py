import hashlib
import hmac
import json
from dataclasses import dataclass

import pytest
from fastapi.testclient import TestClient

from app.core.config import settings
from app.main import app
from app.schemas.review import (
    CreateReviewJobRequest,
    CreateReviewJobResponse,
    DebateResult,
    DebateVerdict,
    FindingStatus,
    ReviewFinding,
    ReviewJobStatus,
    ReviewReport,
)
from app.services.debate_service import DebateOutcome
from app.services.github_webhook import GitHubWebhookResult
from app.services.github_webhook import (
    GitHubWebhookError,
    handle_github_webhook,
    verify_github_signature,
)


SECRET = "webhook-secret"


@pytest.fixture(autouse=True)
def _webhook_settings(monkeypatch):
    monkeypatch.setattr(settings, "github_webhook_secret", SECRET)
    monkeypatch.setattr(settings, "github_review_trigger", "@reviewmind review")
    monkeypatch.setattr(settings, "github_bot_login", "reviewmind")
    monkeypatch.setattr(settings, "github_allowed_repos", [])
    monkeypatch.setattr(settings, "github_token", "test-token")
    monkeypatch.setattr(settings, "github_auto_review_on_pr_opened", False)
    yield


def test_verify_github_signature_accepts_valid_signature() -> None:
    raw = b'{"ok":true}'
    verify_github_signature(raw_body=raw, signature=_signature(raw))


def test_verify_github_signature_rejects_invalid_signature() -> None:
    with pytest.raises(GitHubWebhookError) as exc:
        verify_github_signature(raw_body=b"{}", signature="sha256=bad")

    assert exc.value.status_code == 401
    assert str(exc.value) == "Invalid GitHub webhook signature"


@pytest.mark.anyio
async def test_handle_github_webhook_creates_job_and_start_comment(monkeypatch) -> None:
    raw = _raw_payload(_payload())
    service = FakeReviewJobService()
    posted_comments: list[dict] = []

    async def fake_post_pr_comment(**kwargs):
        posted_comments.append(kwargs)
        return FakeCommentResult(html_url="https://github.com/owner/repo/pull/12#issuecomment-1")

    monkeypatch.setattr("app.services.github_webhook.post_pr_comment", fake_post_pr_comment)
    monkeypatch.setattr("app.services.github_webhook._post_final_comment_when_done", _noop)
    monkeypatch.setattr("app.services.github_webhook._is_duplicate_delivery", _not_duplicate)

    result = await handle_github_webhook(
        event="issue_comment",
        delivery_id="delivery-1",
        signature=_signature(raw),
        raw_body=raw,
        service=service,
    )

    assert result.accepted is True
    assert result.job_id == "rev_webhook"
    assert result.pr_url == "https://github.com/owner/repo/pull/12"
    assert service.requests[0].pr_url.unicode_string() == "https://github.com/owner/repo/pull/12"
    assert "ReviewMind 已开始审查" in posted_comments[0]["body"]


@pytest.mark.anyio
async def test_handle_github_webhook_ignores_non_pr_or_non_command_comment(monkeypatch) -> None:
    monkeypatch.setattr("app.services.github_webhook._is_duplicate_delivery", _not_duplicate)

    # 非 PR issue 评论
    raw = _raw_payload(_payload(is_pull_request=False))
    result = await handle_github_webhook(
        event="issue_comment",
        delivery_id="d-np",
        signature=_signature(raw),
        raw_body=raw,
        service=FakeReviewJobService(),
    )
    assert result.ignored is True
    assert result.reason == "not_a_pr_comment"

    # PR 评论但无命令/触发词
    raw2 = _raw_payload(_payload(body="looks good to me"))
    result2 = await handle_github_webhook(
        event="issue_comment",
        delivery_id="d-nocmd",
        signature=_signature(raw2),
        raw_body=raw2,
        service=FakeReviewJobService(),
    )
    assert result2.ignored is True
    assert result2.reason == "no_review_trigger"


@pytest.mark.anyio
async def test_pull_request_event_disabled_by_default(monkeypatch) -> None:
    monkeypatch.setattr("app.services.github_webhook._is_duplicate_delivery", _not_duplicate)
    raw = _raw_payload(_pull_request_payload())

    result = await handle_github_webhook(
        event="pull_request",
        delivery_id="d-pr-1",
        signature=_signature(raw),
        raw_body=raw,
        service=FakeReviewJobService(),
    )
    assert result.ignored is True
    assert result.reason == "auto_review_disabled"


@pytest.mark.anyio
async def test_pull_request_opened_creates_job_when_enabled(monkeypatch) -> None:
    monkeypatch.setattr(settings, "github_auto_review_on_pr_opened", True)
    monkeypatch.setattr("app.services.github_webhook.post_pr_comment", _fake_post_comment([]))
    monkeypatch.setattr("app.services.github_webhook._post_final_comment_when_done", _noop)
    monkeypatch.setattr("app.services.github_webhook._is_duplicate_delivery", _not_duplicate)

    service = FakeReviewJobService()
    raw = _raw_payload(_pull_request_payload())
    result = await handle_github_webhook(
        event="pull_request",
        delivery_id="d-pr-2",
        signature=_signature(raw),
        raw_body=raw,
        service=service,
    )
    assert result.accepted is True
    assert result.reason == "review_job_created"
    assert len(service.requests) == 1


@pytest.mark.anyio
async def test_pull_request_synchronize_ignored_even_when_enabled(monkeypatch) -> None:
    monkeypatch.setattr(settings, "github_auto_review_on_pr_opened", True)
    monkeypatch.setattr("app.services.github_webhook._is_duplicate_delivery", _not_duplicate)

    raw = _raw_payload(_pull_request_payload(action="synchronize"))
    result = await handle_github_webhook(
        event="pull_request",
        delivery_id="d-pr-3",
        signature=_signature(raw),
        raw_body=raw,
        service=FakeReviewJobService(),
    )
    assert result.ignored is True
    assert result.reason == "pr_action_not_opened"


@pytest.mark.anyio
async def test_explain_command_dispatches_to_debate_service(monkeypatch) -> None:
    posted: list[dict] = []
    monkeypatch.setattr("app.services.github_webhook.post_pr_comment", _fake_post_comment(posted))
    monkeypatch.setattr("app.services.github_webhook._is_duplicate_delivery", _not_duplicate)

    fake_job = _fake_job_with_finding()
    async def fake_get_latest(pr_url: str):
        return fake_job

    monkeypatch.setattr(
        "app.services.github_webhook.review_job_store.get_latest_job_by_pr_url",
        fake_get_latest,
    )

    debate = FakeDebateService()
    raw = _raw_payload(_payload(body="/explain sec_abc 这是误报，已有参数化查询"))
    result = await handle_github_webhook(
        event="issue_comment",
        delivery_id="d-explain",
        signature=_signature(raw),
        raw_body=raw,
        service=FakeReviewJobService(),
        debate_service=debate,
    )

    assert result.accepted is True
    assert result.reason == "command_handled"
    assert debate.explain_calls == [("rev_x", "sec_abc", "这是误报，已有参数化查询")]
    assert any("辩论回复" in c["body"] for c in posted)


@pytest.mark.anyio
async def test_accept_command_updates_status_and_posts_comment(monkeypatch) -> None:
    posted: list[dict] = []
    monkeypatch.setattr("app.services.github_webhook.post_pr_comment", _fake_post_comment(posted))
    monkeypatch.setattr("app.services.github_webhook._is_duplicate_delivery", _not_duplicate)

    fake_job = _fake_job_with_finding()
    monkeypatch.setattr(
        "app.services.github_webhook.review_job_store.get_latest_job_by_pr_url",
        lambda pr_url: _async_return(fake_job),
    )

    debate = FakeDebateService()
    raw = _raw_payload(_payload(body="/accept sec_abc"))
    result = await handle_github_webhook(
        event="issue_comment",
        delivery_id="d-accept",
        signature=_signature(raw),
        raw_body=raw,
        service=FakeReviewJobService(),
        debate_service=debate,
    )

    assert result.accepted is True
    assert debate.status_calls == [("rev_x", "sec_abc", FindingStatus.accepted)]
    assert any("已接受" in c["body"] for c in posted)


@pytest.mark.anyio
async def test_command_without_existing_job_posts_hint(monkeypatch) -> None:
    posted: list[dict] = []
    monkeypatch.setattr("app.services.github_webhook.post_pr_comment", _fake_post_comment(posted))
    monkeypatch.setattr("app.services.github_webhook._is_duplicate_delivery", _not_duplicate)

    async def none_job(pr_url: str):
        return None

    monkeypatch.setattr(
        "app.services.github_webhook.review_job_store.get_latest_job_by_pr_url",
        none_job,
    )

    raw = _raw_payload(_payload(body="/reject sec_abc"))
    result = await handle_github_webhook(
        event="issue_comment",
        delivery_id="d-nojob",
        signature=_signature(raw),
        raw_body=raw,
        service=FakeReviewJobService(),
        debate_service=FakeDebateService(),
    )

    assert result.ignored is True
    assert result.reason == "no_review_job"
    assert any("请先评论" in c["body"] for c in posted)


@pytest.mark.anyio
async def test_command_with_unknown_finding_posts_hint(monkeypatch) -> None:
    posted: list[dict] = []
    monkeypatch.setattr("app.services.github_webhook.post_pr_comment", _fake_post_comment(posted))
    monkeypatch.setattr("app.services.github_webhook._is_duplicate_delivery", _not_duplicate)

    monkeypatch.setattr(
        "app.services.github_webhook.review_job_store.get_latest_job_by_pr_url",
        lambda pr_url: _async_return(_fake_job_with_finding()),
    )

    from app.services.debate_service import FindingNotFoundError

    class RaisingDebate(FakeDebateService):
        async def set_finding_status(self, job_id, finding_id, status):
            raise FindingNotFoundError(job_id, finding_id)

    raw = _raw_payload(_payload(body="/accept missing_id"))
    result = await handle_github_webhook(
        event="issue_comment",
        delivery_id="d-nofinding",
        signature=_signature(raw),
        raw_body=raw,
        service=FakeReviewJobService(),
        debate_service=RaisingDebate(),
    )

    assert result.ignored is True
    assert result.reason == "finding_not_found"
    assert any("未找到 finding" in c["body"] for c in posted)


@pytest.mark.anyio
async def test_handle_github_webhook_ignores_duplicate_delivery(monkeypatch) -> None:
    raw = _raw_payload(_payload())

    async def duplicate(_delivery_id: str) -> bool:
        return True

    monkeypatch.setattr("app.services.github_webhook._is_duplicate_delivery", duplicate)

    result = await handle_github_webhook(
        event="issue_comment",
        delivery_id="delivery-3",
        signature=_signature(raw),
        raw_body=raw,
        service=FakeReviewJobService(),
    )

    assert result.ignored is True
    assert result.reason == "duplicate_delivery"


def test_webhook_route_returns_structured_response(monkeypatch) -> None:
    async def fake_handle_github_webhook(**_kwargs):
        return GitHubWebhookResult(
            accepted=True,
            ignored=False,
            reason="review_job_created",
            job_id="rev_route",
            pr_url="https://github.com/owner/repo/pull/12",
            start_comment_url="https://github.com/owner/repo/pull/12#issuecomment-1",
        )

    monkeypatch.setattr("app.api.github_webhook.handle_github_webhook", fake_handle_github_webhook)

    response = TestClient(app).post(
        "/api/v1/github/webhook",
        content=_raw_payload(_payload()),
        headers={
            "X-GitHub-Event": "issue_comment",
            "X-GitHub-Delivery": "delivery-route",
            "X-Hub-Signature-256": "sha256=test",
        },
    )

    assert response.status_code == 200
    body = response.json()
    assert body["code"] == 20200
    assert body["data"]["accepted"] is True


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------


async def _noop(*_args, **_kwargs):
    return None


async def _not_duplicate(_delivery_id: str) -> bool:
    return False


def _fake_post_comment(store: list[dict]):
    async def fake_post_pr_comment(**kwargs):
        store.append(kwargs)
        return FakeCommentResult(html_url="https://github.com/owner/repo/pull/12#issuecomment-x")

    return fake_post_pr_comment


async def _async_return(value):
    return value


@dataclass
class FakeCommentResult:
    html_url: str
    comment_id: int = 1


@dataclass
class FakeJob:
    job_id: str
    pr_url: str
    report: ReviewReport


def _fake_job_with_finding() -> FakeJob:
    finding = ReviewFinding(
        id="sec_abc",
        agent="security_agent",
        file="src/db.py",
        line=10,
        level="HIGH",
        type="sql_injection",
        confidence=0.8,
        description="疑似 SQL 注入",
        suggestion="使用参数化查询",
    )
    report = ReviewReport(
        summary="测试摘要",
        risk_level="HIGH",
        findings=[finding],
        review_comment="## AI 审查摘要\n测试",
    )
    return FakeJob(job_id="rev_x", pr_url="https://github.com/owner/repo/pull/12", report=report)


class FakeReviewJobService:
    def __init__(self) -> None:
        self.requests: list[CreateReviewJobRequest] = []

    async def create_job(self, request: CreateReviewJobRequest) -> CreateReviewJobResponse:
        self.requests.append(request)
        return CreateReviewJobResponse(
            job_id="rev_webhook",
            status=ReviewJobStatus.pending,
            stream_url="/api/v1/review/stream/rev_webhook",
            report_url="/api/v1/review/jobs/rev_webhook",
        )

    async def get_job_detail(self, job_id: str):
        raise AssertionError("unexpected get_job_detail in unit tests")


class FakeDebateService:
    def __init__(self) -> None:
        self.explain_calls: list[tuple] = []
        self.status_calls: list[tuple] = []

    async def explain_finding(self, job_id: str, finding_id: str, challenge: str) -> DebateOutcome:
        self.explain_calls.append((job_id, finding_id, challenge))
        finding = _fake_job_with_finding().report.findings[0]
        return DebateOutcome(
            finding=finding,
            result=DebateResult(
                explanation="异议成立，该处已使用参数化查询。",
                verdict=DebateVerdict.dismiss,
                confidence=0.9,
            ),
            review_comment="## AI 审查摘要\n更新",
            report_changed=True,
        )

    async def set_finding_status(self, job_id: str, finding_id: str, status: FindingStatus):
        self.status_calls.append((job_id, finding_id, status))
        return _fake_job_with_finding().report


def _payload(
    *,
    action: str = "created",
    body: str = "@reviewmind review",
    commenter: str = "alice",
    user_type: str = "User",
    is_pull_request: bool = True,
) -> dict:
    issue = {
        "number": 12,
        "html_url": "https://github.com/owner/repo/pull/12",
    }
    if is_pull_request:
        issue["pull_request"] = {"url": "https://api.github.com/repos/owner/repo/pulls/12"}

    return {
        "action": action,
        "issue": issue,
        "comment": {
            "body": body,
            "user": {
                "login": commenter,
                "type": user_type,
            },
        },
        "repository": {
            "name": "repo",
            "owner": {
                "login": "owner",
            },
        },
    }


def _pull_request_payload(*, action: str = "opened") -> dict:
    return {
        "action": action,
        "pull_request": {
            "number": 12,
            "html_url": "https://github.com/owner/repo/pull/12",
            "user": {"login": "alice"},
        },
        "repository": {
            "name": "repo",
            "owner": {"login": "owner"},
        },
    }


def _raw_payload(payload: dict) -> bytes:
    return json.dumps(payload, separators=(",", ":")).encode("utf-8")


def _signature(raw: bytes) -> str:
    return "sha256=" + hmac.new(SECRET.encode("utf-8"), raw, hashlib.sha256).hexdigest()
