"""多轮辩论对话历史持久化。

操作 ``ReviewConversationModel``，按 ``(job_id, finding_id)`` 维度追加。
供 /explain 辩论闭环存取人机对话记录。
"""

from __future__ import annotations

import logging
from datetime import UTC, datetime

from sqlalchemy import select

from app.core.database import async_session
from app.models.db_models import ReviewConversationModel
from app.schemas.review import ConversationTurn

logger = logging.getLogger(__name__)


class ConversationStore:
    """对话历史的数据库读写。"""

    async def append(self, turn: ConversationTurn) -> ConversationTurn:
        """追加一条对话记录，回填 id 与 created_at。"""
        created_at = turn.created_at or datetime.now(UTC)
        model = ReviewConversationModel(
            job_id=turn.job_id,
            finding_id=turn.finding_id,
            role=turn.role.value if hasattr(turn.role, "value") else str(turn.role),
            content=turn.content,
            created_at=created_at,
        )
        async with async_session() as session:
            session.add(model)
            await session.commit()
            await session.refresh(model)

        return ConversationTurn(
            job_id=model.job_id,
            finding_id=model.finding_id,
            role=model.role,  # type: ignore[arg-type]
            content=model.content,
            created_at=model.created_at,
        )

    async def list(
        self,
        job_id: str,
        finding_id: str | None = None,
    ) -> list[ConversationTurn]:
        """按时间正序返回对话记录；指定 finding_id 则只返回该 finding 的对话。"""
        async with async_session() as session:
            stmt = (
                select(ReviewConversationModel)
                .where(ReviewConversationModel.job_id == job_id)
                .order_by(ReviewConversationModel.created_at.asc(), ReviewConversationModel.id.asc())
            )
            if finding_id is not None:
                stmt = stmt.where(ReviewConversationModel.finding_id == finding_id)
            result = await session.execute(stmt)
            models = result.scalars().all()

        return [
            ConversationTurn(
                job_id=m.job_id,
                finding_id=m.finding_id,
                role=m.role,  # type: ignore[arg-type]
                content=m.content,
                created_at=m.created_at,
            )
            for m in models
        ]


# 全局单例
conversation_store = ConversationStore()
