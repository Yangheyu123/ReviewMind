"""P1 校准第二环：聚合阶段置信度地板（确定性工作点 0.4）。"""

from app.agent_loop.engine import _apply_confidence_floor
from app.core.config import settings


def test_floor_drops_low_confidence_keeps_rest():
    findings = [
        {"id": "a", "confidence": 0.9},
        {"id": "b", "confidence": 0.4},   # 恰好等于地板 → 保留
        {"id": "c", "confidence": 0.35},  # 地板之下 → 过滤
        {"id": "d"},                      # 无 confidence 字段（None→0）→ 过滤
    ]
    kept = _apply_confidence_floor(findings)
    assert [f["id"] for f in kept] == ["a", "b"]


def test_floor_zero_disables():
    settings.review_min_confidence = 0.0
    try:
        findings = [{"id": "x", "confidence": 0.1}]
        assert _apply_confidence_floor(findings) == findings
    finally:
        settings.review_min_confidence = 0.4


def test_floor_interacts_with_unanchored_halving():
    """未锚定 confidence×0.5 已在锚点阶段生效，地板对其实际值判定。"""
    unanchored = {"id": "u", "confidence": 0.7 * 0.5}  # 0.35 < 0.4 → 过滤
    anchored = {"id": "k", "confidence": 0.7}
    kept = _apply_confidence_floor([unanchored, anchored])
    assert [f["id"] for f in kept] == ["k"]
