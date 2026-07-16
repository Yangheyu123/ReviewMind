"""Issue Comment 命令解析。

把 PR 评论正文解析为判别式命令对象，供 webhook 路由分发：

- ``ReviewCommand``    : ``/review`` 或命中触发词（向后兼容）
- ``ExplainCommand``   : ``/explain <finding_id> [异议说明]`` → 触发辩论
- ``AcceptCommand``    : ``/accept <finding_id>``
- ``RejectCommand``    : ``/reject <finding_id>``

设计要点：
- 只识别「首行首个 /token」形式的斜杠命令，避免误吞正文里出现的 /xxx。
- 触发词命中仍走 ReviewCommand，保持对既有 ``@reviewmind review`` 的兼容。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Union


@dataclass(frozen=True)
class ReviewCommand:
    """触发一次完整审查。"""

    commenter: str
    raw_body: str


@dataclass(frozen=True)
class ExplainCommand:
    """对单条 finding 发起辩论。"""

    finding_id: str
    message: str
    commenter: str
    raw_body: str


@dataclass(frozen=True)
class AcceptCommand:
    """接受单条 finding。"""

    finding_id: str
    commenter: str
    raw_body: str


@dataclass(frozen=True)
class RejectCommand:
    """驳回单条 finding。"""

    finding_id: str
    commenter: str
    raw_body: str


CommentCommand = Union[ReviewCommand, ExplainCommand, AcceptCommand, RejectCommand]


def parse_comment_command(
    body: str,
    commenter: str,
    *,
    trigger: str,
) -> CommentCommand | None:
    """解析评论正文为命令对象。

    Args:
        body: 评论正文。
        commenter: 评论者 login。
        trigger: 触发词（来自 ``settings.github_review_trigger``），大小写不敏感。

    Returns:
        命令对象；未命中任何命令返回 None。
    """
    normalized_body = body or ""
    trigger_norm = (trigger or "").strip().lower()
    if trigger_norm and trigger_norm in " ".join(normalized_body.lower().split()):
        return ReviewCommand(commenter=commenter, raw_body=normalized_body)

    # 取首行首个 token 判断是否斜杠命令
    first_line = normalized_body.splitlines()[0].strip() if normalized_body.strip() else ""
    tokens = first_line.split()
    if not tokens:
        return None

    head = tokens[0].lower()
    rest = tokens[1:]

    if head == "/review":
        return ReviewCommand(commenter=commenter, raw_body=normalized_body)
    if head == "/explain":
        finding_id, message = _split_finding_and_message(rest)
        if not finding_id:
            return None
        return ExplainCommand(
            finding_id=finding_id,
            message=message,
            commenter=commenter,
            raw_body=normalized_body,
        )
    if head in ("/accept", "/reject"):
        if not rest:
            return None
        finding_id = rest[0]
        cls = AcceptCommand if head == "/accept" else RejectCommand
        return cls(  # type: ignore[operator]
            finding_id=finding_id,
            commenter=commenter,
            raw_body=normalized_body,
        )
    return None


def _split_finding_and_message(rest: list[str]) -> tuple[str, str]:
    """从 /explain 的剩余 token 中拆出 finding_id 与异议说明。"""
    if not rest:
        return "", ""
    finding_id = rest[0]
    message = " ".join(rest[1:]).strip()
    return finding_id, message
