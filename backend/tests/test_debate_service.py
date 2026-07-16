import pytest

from app.agents import debate_agent
from app.models.review_job import ReviewJob
from app.schemas.review import (
    ChangedFile,
    DebateResult,
    DebateVerdict,
    FindingStatus,
    ReviewFinding,
    ReviewReport,
)
from app.services.debate_service import DebateService, FindingNotFoundError
from app.services.conversation_store import conversation_store
from app.services.review_job_store import review_job_store


def _finding(level: str = "HIGH") -> ReviewFinding:
    return ReviewFinding(
        id="sec_abc",
        agent="security_agent",
        file="src/db.py",
        line=10,
        level=level,
        type="sql_injection",
        confidence=0.8,
        description="疑似 SQL 注入",
        suggestion="使用参数化查询",
    )


def _report(finding: ReviewFinding) -> ReviewReport:
    return ReviewReport(
        summary="测试摘要",
        risk_level="HIGH",
        findings=[finding],
        changed_files=[
            ChangedFile(
                filename="src/db.py",
                status="modified",
                additions=2,
                deletions=1,
                patch="@@ -10 +10 @@\n+query = f'SELECT * FROM t WHERE id={uid}'",
            )
        ],
        review_comment="## AI 审查摘要\n测试",
    )


async def _seed_job(report: ReviewReport) -> str:
    job = ReviewJob(job_id="rev_debate_1", pr_url="https://github.com/owner/repo/pull/12", report=report)
    await review_job_store.create(job)
    return job.job_id


@pytest.mark.anyio
async def test_explain_dismiss_updates_finding_status_and_persists_conversation(monkeypatch):
    job_id = await _seed_job(_report(_finding()))

    async def fake_run(**kwargs):
        return DebateResult(
            explanation="异议成立，已使用参数化查询，属误报。",
            verdict=DebateVerdict.dismiss,
            confidence=0.9,
        )

    monkeypatch.setattr(debate_agent, "run_async", fake_run)

    outcome = await DebateService().explain_finding(job_id, "sec_abc", "这是误报")

    assert outcome.result.verdict == DebateVerdict.dismiss
    assert outcome.report_changed is True

    # 报告已持久化：finding.status 变为 dismissed
    reloaded = await review_job_store.get(job_id)
    assert reloaded.report.findings[0].status == FindingStatus.dismissed.value

    # 对话历史落库 2 条（user + assistant）
    turns = await conversation_store.list(job_id, finding_id="sec_abc")
    assert len(turns) == 2


@pytest.mark.anyio
async def test_explain_keep_does_not_mutate_report(monkeypatch):
    job_id = await _seed_job(_report(_finding()))

    async def fake_run(**kwargs):
        return DebateResult(
            explanation="异议不成立。",
            verdict=DebateVerdict.keep,
            confidence=0.7,
        )

    monkeypatch.setattr(debate_agent, "run_async", fake_run)

    outcome = await DebateService().explain_finding(job_id, "sec_abc", "看看？")
    assert outcome.report_changed is False

    reloaded = await review_job_store.get(job_id)
    assert reloaded.report.findings[0].status == FindingStatus.open.value
    assert reloaded.report.findings[0].level == "HIGH"


@pytest.mark.anyio
async def test_set_finding_status_accept_updates_report():
    job_id = await _seed_job(_report(_finding()))

    report = await DebateService().set_finding_status(job_id, "sec_abc", FindingStatus.accepted)

    assert report.findings[0].status == FindingStatus.accepted.value
    assert "✅ 已接受" in report.review_comment

    reloaded = await review_job_store.get(job_id)
    assert reloaded.report.findings[0].status == FindingStatus.accepted.value


@pytest.mark.anyio
async def test_explain_unknown_finding_raises():
    job_id = await _seed_job(_report(_finding()))
    with pytest.raises(FindingNotFoundError):
        await DebateService().explain_finding(job_id, "missing", "x")
