"""Phase 2.5 审查记忆测试：存取、召回匹配、prompt 注入、空库零成本。

按目标约定：不验证多任务积累效果（需真实持续使用），只验证机制正确与面试可解释。
"""

import pytest

from app.services.review_memory import (
    recall_memory,
    render_memory_hints,
    save_findings_to_memory,
)
from tests.memory_store import MemoryReviewJobStore


@pytest.fixture
async def mem_store(monkeypatch):
    store = MemoryReviewJobStore()
    import app.services.review_memory as rm
    monkeypatch.setattr(rm, "_store", lambda: store)
    return store


FINDINGS = [
    {"file": "src/payment/retry.py", "line": 42, "level": "HIGH", "category": "correctness",
     "type_detail": "retry without idempotency", "description": "支付重试缺少幂等校验，重试会重复扣款",
     "suggestion": "按订单号做幂等键", "existing_code": "for attempt in range(3):"},
    {"file": "src/payment/retry.py", "line": 60, "level": "LOW", "category": "other",
     "description": "短",  # 低于 min_chars，不入库
    },
]


async def test_save_and_recall_roundtrip(mem_store):
    n = await save_findings_to_memory("acme/shop", FINDINGS)
    assert n == 1  # 短描述被过滤

    # 同仓库同文件 → 召回
    hits = await recall_memory("acme/shop", ["src/payment/retry.py"])
    assert len(hits) == 1
    assert hits[0]["description"].startswith("支付重试")

    # 同仓库不同目录且文件名不同 → 不召回
    miss = await recall_memory("acme/shop", ["docs/readme.md"])
    assert miss == []

    # 其它仓库 → 不召回（仓库隔离）
    other = await recall_memory("other/repo", ["src/payment/retry.py"])
    assert other == []


async def test_recall_matches_same_directory(mem_store):
    await save_findings_to_memory("acme/shop", FINDINGS)
    # 目录亲和：同目录的另一个文件也召回
    hits = await recall_memory("acme/shop", ["src/payment/webhook.py"])
    assert len(hits) == 1


async def test_empty_memory_returns_nothing(mem_store):
    hits = await recall_memory("acme/shop", ["src/anything.py"])
    assert hits == []
    assert render_memory_hints(hits) == ""  # 空库零注入


def test_render_memory_hints_format():
    text = render_memory_hints([{
        "file": "src/payment/retry.py", "line": 42, "category": "correctness",
        "description": "支付重试缺少幂等校验", "created_at": "2026-09-01T12:00:00",
    }])
    assert "本仓库历史审查记忆" in text
    assert "src/payment/retry.py:42" in text
    assert "[correctness]" in text
    assert "历史发现而非本次结论" in text  # 防误用提示


async def test_engine_prompt_injects_memory(monkeypatch, mem_store):
    """引擎组 prompt 包含记忆段（机制接线验证，不跑真实积累）。"""
    await save_findings_to_memory("example/repo", FINDINGS)
    from app.agent_loop.engine import _group_user_prompt
    prompt = _group_user_prompt(
        {"parsed_diff": [], "memory_hints": render_memory_hints(
            await recall_memory("example/repo", ["src/payment/retry.py"]))},
        ["src/payment/retry.py"],
    )
    assert "本仓库历史审查记忆" in prompt
