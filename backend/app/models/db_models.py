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


class ReviewConversationModel(Base):
    """多轮辩论对话历史，按 (job_id, finding_id) 维度追加存储。

    通过 Base.metadata.create_all 自动建表（无需 Alembic 迁移）。
    """

    __tablename__ = "review_conversations"

    id = Column(Integer, primary_key=True, autoincrement=True)
    job_id = Column(String(64), nullable=False, index=True)
    finding_id = Column(String(64), nullable=True, index=True)
    role = Column(String(16), nullable=False)
    content = Column(Text, nullable=False)
    created_at = Column(DateTime(timezone=True), nullable=False, default=_utcnow)