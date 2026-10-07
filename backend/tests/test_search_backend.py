"""P1 自适应检索后端测试：rg 加速档 / 纯 Python 兜底档的口径一致性与护栏。"""

from pathlib import Path

import pytest

from app.agent_loop import tool_context
from app.agent_loop.tool_context import (
    MAX_LINE_CHARS,
    MAX_PATTERN_CHARS,
    ToolContext,
    search_in_sources,
)
from app.schemas.github import GitHubPullRequestRef


def _make_ctx(snapshot_root=None, snapshot_files=None, cache=None):
    ctx = ToolContext(
        github_client=None,
        pr_ref=GitHubPullRequestRef(owner="o", repo="r", pull_number=1,
                                    html_url="https://github.com/o/r/pull/1"),
        head_sha="h",
        snapshot_root=snapshot_root,
        snapshot_files=snapshot_files or [],
    )
    for k, v in (cache or {}).items():
        ctx._source_cache[k] = v
    return ctx


@pytest.fixture()
def snapshot(tmp_path: Path):
    (tmp_path / "a.py").write_text("def fa():\n    return needle_one\n", encoding="utf-8")
    sub = tmp_path / "sub"
    sub.mkdir()
    (sub / "b.py").write_text("NEEDLE_two = 1\nplain = 2\n", encoding="utf-8")
    (tmp_path / "c.txt").write_text("needle_three\n", encoding="utf-8")
    return tmp_path


def _hit_set(result):
    return {(h["file"], h["line"], h["text"].strip()) for h in result["hits"]}


def test_pattern_length_guard():
    ctx = _make_ctx()
    result = search_in_sources(ctx, "a" * (MAX_PATTERN_CHARS + 1), is_regex=False, file_glob=None)
    assert "error" in result and "too long" in result["error"]


def test_rg_backend_and_python_parity(snapshot: Path):
    """rg 档启用（本机有 rg）；强制关掉 rg 后两档命中集合完全一致。"""
    ctx = _make_ctx(snapshot_root=snapshot,
                    snapshot_files=["a.py", "sub/b.py", "c.txt"])
    result = search_in_sources(ctx, "needle", is_regex=False, file_glob=None)
    if tool_context._rg_binary() is None:  # 环境无 rg：只验证兜底档可用
        assert result["backend"] == "python"
        return
    assert result["backend"] == "rg"
    rg_hits = _hit_set(result)
    # sub/b.py 是大写 NEEDLE_two——两档都区分大小写，不在命中集
    assert {f for f, _, _ in rg_hits} == {"a.py", "c.txt"}

    tool_context._RG_CHECKED = True
    tool_context._RG_BIN = None  # 模拟 rg 缺席
    try:
        fallback = search_in_sources(_make_ctx(snapshot_root=snapshot,
                                               snapshot_files=["a.py", "sub/b.py", "c.txt"]),
                                     "needle", is_regex=False, file_glob=None)
        assert fallback["backend"] == "python"
        assert _hit_set(fallback) == rg_hits  # 两档口径一致
    finally:
        tool_context._RG_BIN = None
        tool_context._RG_CHECKED = False  # 恢复自动探测


def test_rg_timeout_falls_back(snapshot: Path, monkeypatch):
    def _timeout(*args, **kwargs):
        raise tool_context.subprocess.TimeoutExpired(cmd="rg", timeout=1)
    monkeypatch.setattr(tool_context.subprocess, "run", _timeout)
    ctx = _make_ctx(snapshot_root=snapshot, snapshot_files=["a.py"])
    result = search_in_sources(ctx, "needle_one", is_regex=False, file_glob=None)
    assert result["backend"] == "python" and len(result["hits"]) == 1


def test_long_line_skipped():
    """Python 档跳过超长行——限制病态 pattern 的回溯爆炸半径。"""
    long_line = "x" * (MAX_LINE_CHARS + 100) + " needle"
    ctx = _make_ctx(cache={"big.py": long_line})
    result = search_in_sources(ctx, "needle", is_regex=False, file_glob=None)
    assert result["hits"] == []


def test_custom_glob_always_uses_python(snapshot: Path):
    """自定义 glob 语义与 rg glob 不等价 → 固定走 Python 保证口径。"""
    ctx = _make_ctx(snapshot_root=snapshot, snapshot_files=["a.py", "sub/b.py", "c.txt"])
    result = search_in_sources(ctx, "needle", is_regex=False, file_glob="*.py")
    assert result["backend"] == "python"
    assert {h["file"] for h in result["hits"]} == {"a.py"}  # glob 限定根目录 .py


def test_regex_mode_both_backends(snapshot: Path):
    ctx = _make_ctx(snapshot_root=snapshot, snapshot_files=["a.py", "sub/b.py"])
    result = search_in_sources(ctx, r"needle_\w+", is_regex=True, file_glob=None)
    assert result["hits"] and all("needle_" in h["text"] for h in result["hits"])
