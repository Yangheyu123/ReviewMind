from datetime import datetime
from enum import StrEnum
from typing import Any

from pydantic import BaseModel, Field, HttpUrl


class ReviewJobStatus(StrEnum):
    pending = "pending"
    running = "running"
    completed = "completed"
    failed = "failed"
    cancelled = "cancelled"


class FindingStatus(StrEnum):
    """单条 finding 的人机协作处置状态。"""

    open = "open"            # 未处置
    accepted = "accepted"    # 开发者接受该发现
    rejected = "rejected"    # 开发者驳回该发现
    dismissed = "dismissed"  # 辩论后被判定为误报并撤销


class DebateVerdict(StrEnum):
    """辩论 Agent 对单条 finding 的裁决。"""

    keep = "keep"              # 维持原结论
    downgrade = "downgrade"   # 下调风险等级
    dismiss = "dismiss"       # 撤销（判定为误报）


class ReviewConfig(BaseModel):
    enable_ast: bool = True
    enable_rag: bool = False
    strict_mode: bool = True


class CreateReviewJobRequest(BaseModel):
    pr_url: HttpUrl = Field(description="GitHub Pull Request URL")
    config: ReviewConfig = Field(default_factory=ReviewConfig)
    github_token: str | None = Field(default=None, description="用户自定义 GitHub Token，不传则使用服务器默认配置")


class CreateReviewJobResponse(BaseModel):
    job_id: str
    status: ReviewJobStatus
    stream_url: str
    report_url: str


class ReviewFinding(BaseModel):
    id: str
    agent: str
    file: str
    line: int
    level: str
    type: str
    confidence: float
    description: str
    suggestion: str
    symbol: str | None = None
    code_snippet: str | None = None
    status: str = Field(default=FindingStatus.open.value, description="人机协作处置状态")


class ReviewProgressEvent(BaseModel):
    step: str
    percent: int = Field(ge=0, le=100)
    message: str
    type: str = "progress"


class ReviewJobSnapshot(BaseModel):
    job_id: str
    pr_url: HttpUrl
    status: ReviewJobStatus
    created_at: datetime
    updated_at: datetime
    error_message: str | None = None
    progress_events: list[dict[str, Any]] = Field(default_factory=list)
    pipeline_result: dict[str, Any] | None = None


# --- 报告相关结构（与 API 文档对齐） ---

class PrInfo(BaseModel):
    owner: str
    repo: str
    number: int
    title: str
    author: str
    base_branch: str
    head_branch: str
    changed_files: int = 0
    additions: int = 0
    deletions: int = 0
    html_url: str


class ReviewReportStats(BaseModel):
    critical: int = 0
    high: int = 0
    medium: int = 0
    low: int = 0
    suggestion: int = 0


class ChangedFile(BaseModel):
    filename: str
    status: str
    additions: int = 0
    deletions: int = 0
    changes: int = 0
    patch: str | None = None
    risk_count: int = 0


class ChangedSymbol(BaseModel):
    file: str
    symbol: str
    language: str
    start_line: int
    end_line: int
    changed_lines: list[int] = Field(default_factory=list)
    code: str | None = None


class ReviewReport(BaseModel):
    summary: str
    risk_level: str
    stats: ReviewReportStats = Field(default_factory=ReviewReportStats)
    changed_files: list[ChangedFile] = Field(default_factory=list)
    changed_symbols: list[ChangedSymbol] = Field(default_factory=list)
    findings: list[ReviewFinding] = Field(default_factory=list)
    review_comment: str = ""


class ReviewJobDetailResponse(BaseModel):
    job_id: str
    status: ReviewJobStatus
    pr: PrInfo | None = None
    progress: ReviewProgressEvent | None = None
    findings: list[ReviewFinding] = Field(default_factory=list)
    report: ReviewReport | None = None
    created_at: datetime | None = None
    completed_at: datetime | None = None
    updated_at: datetime | None = None
    error_message: str | None = None


class JobListItem(BaseModel):
    job_id: str
    status: ReviewJobStatus
    pr_title: str = ""
    pr_url: str = ""
    risk_level: str = "LOW"
    finding_count: int = 0
    created_at: datetime | None = None


class JobListResponse(BaseModel):
    items: list[JobListItem] = Field(default_factory=list)
    page: int = 1
    page_size: int = 10
    total: int = 0


class PostCommentRequest(BaseModel):
    comment_body: str | None = Field(default=None, description="自定义评论内容，不传则使用 report.review_comment")
    github_token: str | None = Field(default=None, description="用户自定义 GitHub Token")


class PostCommentResponse(BaseModel):
    comment_id: int
    html_url: str


class MergeRequest(BaseModel):
    commit_title: str | None = Field(default=None, description="合并 commit 标题")
    commit_message: str | None = Field(default=None, description="合并 commit 消息")
    merge_method: str = Field(default="merge", description="合并方式：merge / squash / rebase")
    github_token: str | None = Field(default=None, description="用户自定义 GitHub Token")


class MergeResponse(BaseModel):
    merged: bool
    message: str
    sha: str | None = None
    html_url: str | None = None


# --- 多轮辩论 / 命令交互相关结构 ---

class ConversationRole(StrEnum):
    """对话轮次的角色。"""

    user = "user"          # 开发者（命令发起方）
    assistant = "assistant"  # ReviewMind Agent
    system = "system"


class ConversationTurn(BaseModel):
    """一条对话记录（持久化到 review_conversations 表）。"""

    job_id: str
    finding_id: str | None = None
    role: ConversationRole
    content: str
    created_at: datetime | None = None


class DebateResult(BaseModel):
    """辩论 Agent 对单次 /explain 的产出。"""

    explanation: str = Field(description="给开发者的解释说明（中文）")
    verdict: DebateVerdict = Field(default=DebateVerdict.keep, description="裁决：keep/downgrade/dismiss")
    revised_level: str | None = Field(default=None, description="downgrade 时的新等级，如 LOW/MEDIUM")
    confidence: float = Field(default=0.5, ge=0.0, le=1.0)
