"""LLM 请求用量记录器（Phase 4 观测）。

设计：
- contextvar 携带 (job_id, group, phase) 上下文——引擎在 review_group/filter
  处包裹上下文，LLMClient 双协议（openai/anthropic）统一埋点；
- 每次调用记录 model/protocol/tokens/latency/status，无上下文时仅打日志（兜底）；
- 记录随分组节点返回值汇入引擎 state（reducer），aggregate 落 llm_request_logs 表。
"""

from __future__ import annotations

import logging
import time
from contextlib import contextmanager
from contextvars import ContextVar
from typing import Any, Iterator

logger = logging.getLogger(__name__)

_ctx: ContextVar[dict[str, Any] | None] = ContextVar("llm_usage_ctx", default=None)


@contextmanager
def llm_context(job_id: str, group: str = "", phase: str = "") -> Iterator[dict[str, Any]]:
    """为一段 LLM 调用建立 (job, group, phase) 记录上下文。"""
    ctx: dict[str, Any] = {"job_id": job_id, "group": group, "phase": phase, "records": []}
    token = _ctx.set(ctx)
    try:
        yield ctx
    finally:
        _ctx.reset(token)


def record(
    model: str,
    *,
    protocol: str,
    prompt_tokens: int = 0,
    completion_tokens: int = 0,
    total_tokens: int = 0,
    latency_ms: int = 0,
    status: str = "ok",
    error: str = "",
) -> dict[str, Any]:
    """记录一次 LLM 调用。返回记录 dict（无上下文时不上浮，仅日志）。"""
    ctx = _ctx.get()
    rec: dict[str, Any] = {
        "job_id": ctx["job_id"] if ctx else "",
        "group": ctx["group"] if ctx else "",
        "phase": ctx["phase"] if ctx else "",
        "model": model,
        "protocol": protocol,
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
        "total_tokens": total_tokens or (prompt_tokens + completion_tokens),
        "latency_ms": latency_ms,
        "status": status,
        "error": error[:300],
    }
    logger.info(
        "[LLM-USAGE] model=%s phase=%s group=%s in=%s out=%s %sms status=%s",
        model, rec["phase"], rec["group"], prompt_tokens, completion_tokens, latency_ms, status,
    )
    if ctx is not None:
        ctx["records"].append(rec)
    return rec


class latency_timer:
    """简单计时器：`with latency_timer() as t: ...` 后 `t.ms`。"""

    def __init__(self) -> None:
        # 自启动：允许不进 with 直接 snapshot_ms()（埋点处更简洁）
        self._start = time.perf_counter()
        self.ms = 0

    def __enter__(self) -> "latency_timer":
        self._start = time.perf_counter()
        self.ms = 0
        return self

    def __exit__(self, *exc: object) -> None:
        self.ms = self.snapshot_ms()

    def snapshot_ms(self) -> int:
        """不退出计时器读取当前耗时。"""
        return int((time.perf_counter() - self._start) * 1000)
