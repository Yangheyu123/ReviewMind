"""数据库 ORM 模型。"""

from datetime import datetime, UTC
from typing import Any

from sqlalchemy import Column, DateTime, Float, Integer, String, Text
from sqlalchemy.orm import DeclarativeBase

from app.core.config import settings

# 仅在 PostgreSQL 环境下导入 pgvector 类型
if "postgresql" in settings.database_url:
    from pgvector.sqlalchemy import Vector
else:
    # SQLite 兼容：使用 Text 存储向量 JSON，降级但不报错
    Vector = None  # type: ignore[assignment,misc]


class Base(DeclarativeBase):
    pass


def _utcnow() -> datetime:
    return datetime.now(UTC)


class ReviewJobModel(Base):
    """ReviewJob 持久化模型，所有结构化数据用 JSON 文本存储。"""

    __tablename__ = "review_jobs"

    job_id = Column(String(64), primary_key=True)
    pr_url = Column(String(512), nullable=False)
    status = Column(String(16), nullable=False, default="pending")
    created_at = Column(DateTime(timezone=True), nullable=False, default=_utcnow)
    updated_at = Column(DateTime(timezone=True), nullable=False, default=_utcnow, onupdate=_utcnow)
    completed_at = Column(DateTime(timezone=True), nullable=True)
    error_message = Column(Text, nullable=True)

    # 结构化数据以 JSON 文本存储（兼容 SQLite 和 PostgreSQL）
    pr_info = Column(Text, nullable=True)           # JSON 字符串
    progress_events = Column(Text, nullable=True)    # JSON 数组字符串
    pipeline_result = Column(Text, nullable=True)    # JSON 字符串
    report = Column(Text, nullable=True)             # ReviewReport JSON


class CodeEmbeddingModel(Base):
    """代码片段向量存储，用于 pgvector 语义检索。

    仅在 PostgreSQL + pgvector 环境下可用。
    SQLite 环境下该表仍会创建，但 embedding 列降级为 Text（存 JSON 数组）。
    """

    __tablename__ = "code_embeddings"

    id = Column(Integer, primary_key=True, autoincrement=True)
    repo_url = Column(String(512), nullable=False, index=True)
    file_path = Column(String(1024), nullable=False)
    symbol = Column(String(256), nullable=True)
    language = Column(String(32), nullable=True)
    code = Column(Text, nullable=False)
    chunk_index = Column(Integer, nullable=False, default=0)

    # pgvector 环境使用 Vector 列，SQLite 降级为 Text
    embedding = (
        Column(Vector(settings.embedding_dimensions), nullable=False)
        if Vector is not None
        else Column(Text, nullable=False)
    )

    created_at = Column(DateTime(timezone=True), nullable=False, default=_utcnow)

class LlmRequestLogModel(Base):
    """LLM 请求明细（Phase 4 观测）：按 job/组/阶段切片 token 成本与延迟。

    数据来源：core/llm_usage 的 contextvar 记录器（双协议钩子统一埋点）。
    """

    __tablename__ = "llm_request_logs"

    id = Column(Integer, primary_key=True, autoincrement=True)
    job_id = Column(String(64), nullable=False, index=True)
    group = Column(String(64), nullable=False, default="")
    phase = Column(String(32), nullable=False, default="")
    model = Column(String(128), nullable=False, default="")
    protocol = Column(String(16), nullable=False, default="openai")
    prompt_tokens = Column(Integer, nullable=False, default=0)
    completion_tokens = Column(Integer, nullable=False, default=0)
    total_tokens = Column(Integer, nullable=False, default=0)
    latency_ms = Column(Integer, nullable=False, default=0)
    status = Column(String(16), nullable=False, default="ok")
    error = Column(Text, nullable=True)
    created_at = Column(DateTime(timezone=True), nullable=False, default=_utcnow)


class ReviewMemoryModel(Base):
    """审查记忆（Phase 2.5）：跨任务的历史 findings 积累。

    检索层当前为关键词级（repo + 文件路径匹配 + 词重叠排序）；
    embedding 额度具备后将 description/existing_code 向量化升级语义检索（接口不变）。
    """

    __tablename__ = "review_memory"

    id = Column(Integer, primary_key=True, autoincrement=True)
    repo = Column(String(256), nullable=False, index=True)
    file = Column(String(1024), nullable=False)
    line = Column(Integer, nullable=False, default=0)
    level = Column(String(16), nullable=False, default="INFO")
    category = Column(String(32), nullable=False, default="other")
    type_detail = Column(String(256), nullable=True)
    description = Column(Text, nullable=False)
    suggestion = Column(Text, nullable=True)
    existing_code = Column(Text, nullable=True)   # 锚点代码段（回归比对的依据）
    anchor_status = Column(String(16), nullable=True)
    created_at = Column(DateTime(timezone=True), nullable=False, default=_utcnow)
