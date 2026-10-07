from app.agents import performance_agent, report_agent, risk_judge_agent, security_agent, summary_agent, test_agent
from app.schemas.agents import AgentContext


SAMPLE_DIFF = [
    {
        "file": "src/services/order_service.py",
        "status": "modified",
        "additions": 12,
        "deletions": 4,
        "changed_lines": [18, 19, 20, 45],
    },
    {
        "file": "tests/test_order_service.py",
        "status": "modified",
        "additions": 5,
        "deletions": 0,
        "changed_lines": [10, 11],
    },
]

LARGE_DIFF = [
    {
        "file": "src/services/big_module.py",
        "status": "modified",
        "additions": 60,
        "deletions": 10,
        "changed_lines": [1, 2, 3],
    },
]


def make_context(parsed_diff=None):
    if parsed_diff is None:
        parsed_diff = SAMPLE_DIFF
    return AgentContext(parsed_diff=parsed_diff)


def test_summary_agent_returns_expected_structure():
    result = summary_agent.run(make_context())

    assert result.agent == "summary_agent"
    assert result.summary
    assert isinstance(result.findings, list)


def test_security_agent_produces_findings():
    result = security_agent.run(make_context())

    assert result.agent == "security_agent"
    assert len(result.findings) == 2
    assert result.findings[0].agent == "security_agent"
    assert result.findings[0].file == "src/services/order_service.py"


def test_performance_agent_flags_large_changes():
    result = performance_agent.run(make_context(LARGE_DIFF))

    assert result.agent == "performance_agent"
    assert len(result.findings) == 1
    assert result.findings[0].level == "WARNING"
    assert result.findings[0].type == "large_change"


def test_performance_agent_ignores_small_changes():
    result = performance_agent.run(make_context())

    assert result.agent == "performance_agent"
    assert len(result.findings) == 0


def test_test_agent_detects_test_files():
    result = test_agent.run(make_context())

    assert result.agent == "test_agent"
    assert len(result.findings) == 1
    assert result.findings[0].file == "tests/test_order_service.py"


def test_risk_judge_aggregates_and_dedupes():
    security = security_agent.run(make_context())
    performance = performance_agent.run(make_context(LARGE_DIFF))
    test = test_agent.run(make_context())

    risk = risk_judge_agent.aggregate([security, performance, test])

    assert risk.risk_level == "MEDIUM"
    assert len(risk.findings) > 0
    all_keys = [(f.agent, f.file, f.line, f.type) for f in risk.findings]
    assert len(all_keys) == len(set(all_keys))


def test_risk_judge_returns_low_when_no_findings():
    empty = summary_agent.run(make_context([]))
    risk = risk_judge_agent.aggregate([empty])

    assert risk.risk_level == "LOW"


def test_risk_judge_deduplicates_identical_findings():
    security = security_agent.run(make_context())

    risk = risk_judge_agent.aggregate([security, security])

    assert len(risk.findings) == 2
    assert risk.dedup_count == 2


def test_report_agent_generates_structured_comment():
    security = security_agent.run(make_context())
    test = test_agent.run(make_context())
    risk = risk_judge_agent.aggregate([security, test])

    report = report_agent.generate_report("rev_test1", risk, "Test summary")

    assert report.risk_level == risk.risk_level
    assert "AI 审查摘要" in report.review_comment
    assert "Test summary" in report.review_comment
    assert report.stats["total_findings"] == len(risk.findings)


def test_report_agent_handles_no_findings():
    empty = summary_agent.run(make_context([]))
    risk = risk_judge_agent.aggregate([empty])

    report = report_agent.generate_report("rev_test2", risk, "Clean PR")

    assert report.risk_level == "LOW"
    assert "未发现明显问题" in report.review_comment
    assert report.stats["total_findings"] == 0


def test_agent_context_accepts_empty_diff():
    context = make_context([])

    result = summary_agent.run(context)
    assert result.agent == "summary_agent"

    result = security_agent.run(context)
    assert len(result.findings) == 0

def test_user_prompt_contains_patch_content():
    """回归测试（P0#1）：parse_diff_file → build_user_prompt 全链路必须把 diff 代码
    注入 LLM prompt。此前 ParsedDiffFile 缺 patch 字段导致审查 agent 只能看到文件名
    （盲审），且旧测试手工给 parsed_diff 塞 "patch" 键掩盖了该缺陷。
    """
    from app.agents.prompts import build_user_prompt
    from app.schemas.agents import AgentContext
    from app.schemas.diff import PullRequestFile
    from app.services.diff_parser import parse_diff_file

    pf = PullRequestFile(
        filename="services/auth.py",
        status="modified",
        additions=2,
        deletions=1,
        patch=(
            "@@ -8,2 +8,3 @@\n"
            " def login(username, password):\n"
            "-    user = db.query(f\"SELECT * FROM users WHERE name='{username}'\")\n"
            "+    user = db.query(\"SELECT * FROM users WHERE name=%s\", (username,))\n"
        ),
    )
    parsed = parse_diff_file(pf).model_dump(mode="json")
    prompt = build_user_prompt(AgentContext(parsed_diff=[parsed]))

    assert "Patch:" in prompt
    assert "db.query" in prompt, "diff 代码内容必须进入 prompt，否则 agent 盲审"
    assert "services/auth.py" in prompt
