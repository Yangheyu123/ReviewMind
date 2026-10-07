"""测试全局 fixtures：为每个测试重置数据库。"""

import os

# 必须在导入 app（读取 settings）之前设置：测试永远走 LLM mock 模式，
# 防止本地 backend/.env 的真实 key（LLM_MOCK_MODE=false）泄漏进测试、
# 导致单测发起真实 LLM 请求（慢且偶发 429）。
os.environ["LLM_MOCK_MODE"] = "true"

import pytest

from app.core.config import settings


@pytest.fixture(autouse=True)
async def _reset_db():
    """每个测试前清空数据库表，保证测试隔离。

    teardown 时 dispose 连接池：pytest-asyncio 每个测试使用独立事件循环，
    模块级共享 engine 的池化连接会绑定到首个循环，跨循环复用会触发
    asyncpg "another operation is in progress"。逐测试重建池保证隔离。
    """
    from app.core.database import engine
    from app.models.db_models import Base

    # 重建表（drop all + create all）
    async with engine.begin() as conn:
        # 仅 PostgreSQL 需要注册 pgvector 扩展
        if "postgresql" in settings.database_url:
            from sqlalchemy import text
            await conn.execute(text("CREATE EXTENSION IF NOT EXISTS vector"))
        await conn.run_sync(Base.metadata.drop_all)
        await conn.run_sync(Base.metadata.create_all)

    yield

    await engine.dispose()