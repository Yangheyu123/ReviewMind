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

    llm_api_key: str | None = None
    llm_api_base: str = "https://ai.sxuan.top/v1"
    llm_model_summary: str = "gpt-4o-mini"
    llm_model_review: str = "gpt-4o"
    llm_timeout_seconds: float = 30.0
    llm_api_protocol: str = ""       # 空=自动（/api/anthropic 自动识别）；可显式 "openai"/"anthropic"
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
    review_use_agent_loop: bool = True
    review_enable_filter: bool = True
    # 置信度地板（P1 校准第二环）：聚合阶段确定性过滤低置信 findings——
    # 工作点 0.4 由评测扫描定标（召回 32.6% / 3.6 条/PR / 单条效率≈直读基线）
    review_min_confidence: float = 0.4
    # 消融实验开关（评测体系 v3）：single = 全部文件一组、单 reviewer + 仓内检索工具
    # （Repository Context 档，剥离 §16.1 分层分组路由）；auto = 生产默认
    review_grouping_mode: str = "auto"
    # 输出语言（评测适配）：zh=生产默认；en=AACR-Bench 等英文 GT 基准跑分用
    # （其确定性语义匹配依赖英文词重叠，跨语言会系统性误判——适配需在报告中披露）
    review_output_language: str = "zh"
    review_enable_memory: bool = True     # Phase 2.5：审查记忆（空库零成本）    # Phase 5 评测 C/D 对照开关（filter 反思节点）   # Phase 1 转正：LangGraph 引擎为默认路径（旧 graph 流水线保留可回滚）
    # Phase 4：三区上下文压缩阈值（token 估算）
    agent_context_budget_tokens: int = 120_000
    # Phase 2：源码快照缓存（tarball 内容寻址）
    snapshot_cache_dir: str = "./data/snapshots"
    snapshot_cache_max_bytes: int = 2 * 1024 * 1024 * 1024  # 缓存总量 2GB，超限 LRU 逐出   # Phase 1 转正：LangGraph 引擎为默认路径（旧 graph 流水线保留可回滚）

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )


settings = Settings()
