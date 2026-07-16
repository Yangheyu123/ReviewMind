from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    app_name: str = "ReviewMind API"
    api_prefix: str = "/api/v1"
    cors_origins: list[str] = ["http://localhost:5173"]
    github_token: str | None = None
    github_api_base_url: str = "https://api.github.com"
    github_timeout_seconds: float = 10.0
    github_webhook_secret: str | None = None
    github_review_trigger: str = "@reviewmind review"
    github_bot_login: str = "reviewmind"
    github_allowed_repos: list[str] = []
    github_webhook_result_timeout_seconds: int = 1800
    github_webhook_result_poll_seconds: float = 2.0
    # PR opened 自动触发评审（仅 opened，synchronize 等更新不触发）；默认关闭，受 allowed_repos 白名单约束。
    github_auto_review_on_pr_opened: bool = False

    llm_api_key: str | None = None
    llm_api_base: str = "https://ai.sxuan.top/v1"
    llm_model_summary: str = "gpt-4o-mini"
    llm_model_review: str = "gpt-4o"
    llm_timeout_seconds: float = 30.0
    llm_mock_mode: bool = True

    # PostgreSQL 为默认数据库（含 pgvector 扩展），本地无 PG 时可通过
    # DATABASE_URL=sqlite+aiosqlite:///./reviewmind.db 切回 SQLite
    database_url: str = "postgresql+asyncpg://reviewmind:reviewmind@localhost:5432/reviewmind"
    redis_url: str = "redis://localhost:6379/0"

    # Embedding 配置（用于 pgvector 向量检索）
    embedding_api_key: str | None = None
    embedding_api_base: str = "https://ai.sxuan.top/v1"
    embedding_model: str = "text-embedding-3-small"
    embedding_dimensions: int = 1536
    # Embedding 供应商专属 header（JSON 字符串），如 Gitee 容灾：'{"X-Failover-Enabled":"true"}'
    embedding_extra_headers: dict[str, str] = {}

    # RAG 分级触发阈值
    rag_light_min_files: int = 5
    rag_light_min_lines: int = 200
    rag_full_min_files: int = 15
    rag_full_min_lines: int = 1000
    # RAG 检索参数
    rag_top_k_light: int = 3
    rag_top_k_full: int = 5
    rag_max_snippet_chars: int = 2000
    rag_cache_ttl_seconds: int = 3600

    # Agent Loop 开关：启用后 ReviewGraph 委托给 ReviewOrchestrator（Planner→Executor→Finalizer）。
    # 默认关闭，保留原有编排路径，便于灰度验证与回滚。
    review_use_agent_loop: bool = False

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )


settings = Settings()
