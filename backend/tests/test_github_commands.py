from app.services.github_commands import (
    AcceptCommand,
    ExplainCommand,
    RejectCommand,
    ReviewCommand,
    parse_comment_command,
)

TRIGGER = "@reviewmind review"


def test_trigger_word_returns_review_command() -> None:
    cmd = parse_comment_command("@ReviewMind review please", "alice", trigger=TRIGGER)
    assert isinstance(cmd, ReviewCommand)
    assert cmd.commenter == "alice"


def test_slash_review_returns_review_command() -> None:
    cmd = parse_comment_command("/review", "bob", trigger=TRIGGER)
    assert isinstance(cmd, ReviewCommand)


def test_explain_command_splits_finding_id_and_message() -> None:
    cmd = parse_comment_command("/explain sec_abc 这是误报，已有防护", "alice", trigger=TRIGGER)
    assert isinstance(cmd, ExplainCommand)
    assert cmd.finding_id == "sec_abc"
    assert cmd.message == "这是误报，已有防护"


def test_explain_command_without_message_keeps_empty() -> None:
    cmd = parse_comment_command("/explain sec_abc", "alice", trigger=TRIGGER)
    assert isinstance(cmd, ExplainCommand)
    assert cmd.finding_id == "sec_abc"
    assert cmd.message == ""


def test_explain_command_without_finding_id_returns_none() -> None:
    assert parse_comment_command("/explain", "alice", trigger=TRIGGER) is None


def test_accept_and_reject_commands() -> None:
    accept = parse_comment_command("/accept sec_abc", "alice", trigger=TRIGGER)
    assert isinstance(accept, AcceptCommand)
    assert accept.finding_id == "sec_abc"

    reject = parse_comment_command("/reject sec_abc", "alice", trigger=TRIGGER)
    assert isinstance(reject, RejectCommand)
    assert reject.finding_id == "sec_abc"


def test_accept_without_finding_id_returns_none() -> None:
    assert parse_comment_command("/accept", "alice", trigger=TRIGGER) is None


def test_unknown_command_returns_none() -> None:
    assert parse_comment_command("looks good to me", "alice", trigger=TRIGGER) is None
    assert parse_comment_command("/unknown x", "alice", trigger=TRIGGER) is None


def test_slash_in_prose_is_not_a_command() -> None:
    # 正文中出现的 /review 不应被当作命令（只识别首行首个 token）
    assert parse_comment_command("see /review guidelines below", "alice", trigger=TRIGGER) is None


def test_empty_body_returns_none() -> None:
    assert parse_comment_command("", "alice", trigger=TRIGGER) is None
    assert parse_comment_command("   \n  ", "alice", trigger=TRIGGER) is None


def test_trigger_case_insensitive() -> None:
    cmd = parse_comment_command("@REVIEWMIND REVIEW", "alice", trigger=TRIGGER)
    assert isinstance(cmd, ReviewCommand)
