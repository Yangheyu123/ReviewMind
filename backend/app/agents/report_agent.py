from app.schemas.agents import AggregatedRisk, ReviewReportOutput
from app.schemas.review import FindingStatus, ReviewFinding, ReviewJobStatus, ReviewReport

# finding status → 摘要里的展示徽章
_STATUS_BADGE = {
    FindingStatus.accepted.value: "✅ 已接受",
    FindingStatus.rejected.value: "❌ 已驳回",
    FindingStatus.dismissed.value: "🚫 已撤销（误报）",
    FindingStatus.open.value: "🟰 未处置",
}


def generate_report(
    job_id: str,
    risk: AggregatedRisk,
    summary_text: str,
) -> ReviewReportOutput:
    findings = risk.findings
    risk_level = risk.risk_level

    comment_lines = ["## AI 审查摘要", ""]
    comment_lines.append(summary_text)
    comment_lines.append("")
    comment_lines.append(f"**风险等级：** {risk_level}")
    comment_lines.append(f"**风险发现数：** {len(findings)}")
    comment_lines.append("")

    if findings:
        comment_lines.append("### 主要发现")
        for finding in findings[:5]:
            comment_lines.append(f"- [{finding.level}] {finding.description}")
        if len(findings) > 5:
            comment_lines.append(f"- ……还有 {len(findings) - 5} 项")
    else:
        comment_lines.append("未发现明显问题。")

    return ReviewReportOutput(
        summary=summary_text,
        risk_level=risk_level,
        findings=findings,
        review_comment="\n".join(comment_lines),
        stats={
            "total_findings": len(findings),
            "risk_level": risk_level,
            "dedup_count": risk.dedup_count,
        },
    )


def rebuild_review_comment(report: ReviewReport) -> str:
    """基于当前 findings（含人机协作 status）重生成 review_comment 摘要。

    供 /accept /reject /explain 命令处置 finding 后刷新 PR 摘要评论使用。
    风格与 generate_report 一致，额外标注每条 finding 的处置徽章。
    """
    findings = report.findings
    open_findings = [f for f in findings if f.status == FindingStatus.open.value]
    resolved_findings = [f for f in findings if f.status != FindingStatus.open.value]

    lines = ["## AI 审查摘要", ""]
    lines.append(report.summary.strip() or "（无摘要）")
    lines.append("")
    lines.append(f"**风险等级：** {report.risk_level}")
    lines.append(f"**风险发现数：** {len(findings)}（未处置 {len(open_findings)} / 已处置 {len(resolved_findings)}）")
    lines.append("")

    if open_findings:
        lines.append("### 主要发现")
        for finding in open_findings[:5]:
            lines.append(f"- [{finding.level}] {finding.description}")
        if len(open_findings) > 5:
            lines.append(f"- ……还有 {len(open_findings) - 5} 项")
    else:
        lines.append("所有发现均已处置。")

    if resolved_findings:
        lines.append("")
        lines.append("### 已处置发现")
        for finding in resolved_findings[:10]:
            badge = _STATUS_BADGE.get(finding.status, finding.status)
            lines.append(f"- {badge} [{finding.level}] {finding.description}")
        if len(resolved_findings) > 10:
            lines.append(f"- ……还有 {len(resolved_findings) - 10} 项")

    lines.append("")
    lines.append("> 可在评论中使用 `/explain <id> <异议>`、`/accept <id>`、`/reject <id>` 与我进一步协作。")
    return "\n".join(lines)