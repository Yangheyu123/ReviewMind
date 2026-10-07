"""Phase 2 测试：源码快照（tarball 下载/解压/内容寻址缓存/LRU）与全仓检索面。"""

import io
import tarfile

import pytest

from app.agent_loop.tool_context import (
    ToolContext,
    find_symbol_in_sources,
    search_in_sources,
)
from app.services.source_snapshot import (
    ensure_snapshot,
    iter_snapshot_files,
    read_snapshot_file,
)
from app.schemas.github import GitHubPullRequestRef


def make_tarball(files: dict[str, str]) -> bytes:
    """构造 GitHub 风格 tarball（顶层目录 owner-repo-sha/）。"""
    buf = io.BytesIO()
    with tarfile.open(fileobj=buf, mode="w:gz") as tar:
        for rel, content in files.items():
            data = content.encode()
            info = tarfile.TarInfo(name=f"owner-repo-deadbeef/{rel}")
            info.size = len(data)
            tar.addfile(info, io.BytesIO(data))
    return buf.getvalue()


class TarballClient:
    """返回固定 tarball 的假 GitHubClient。"""

    def __init__(self, files: dict[str, str], fail: bool = False):
        self._bytes = make_tarball(files)
        self._fail = fail
        self.downloads = 0

    async def download_tarball(self, pr_ref, ref):
        self.downloads += 1
        if self._fail:
            raise RuntimeError("boom")
        return self._bytes


@pytest.fixture
async def snapshot_root(tmp_path, monkeypatch):
    """临时快照缓存目录 + 解压一份仓库（async fixture，避免 asyncio.run 污染线程 loop）。"""
    from app.services import source_snapshot
    monkeypatch.setattr(source_snapshot.settings, "snapshot_cache_dir", str(tmp_path))
    client = TarballClient({
        "src/app.py": "def run():\n    return True\n",
        "src/util/helpers.py": "def helper(x):\n    return x * 2\n",
        "src/web/index.ts": "export function render(): void {}\n",
        "docs/readme.md": "# readme\n",
    })
    return await ensure_snapshot(client, _ref(), "deadbeef")


def _ref():
    return GitHubPullRequestRef(owner="owner", repo="repo", pull_number=1,
                                html_url="https://github.com/owner/repo/pull/1")


# ---------------------------------------------------------------------------
# 快照服务
# ---------------------------------------------------------------------------

async def test_snapshot_extract_and_content_addressing(snapshot_root):
    files = iter_snapshot_files(snapshot_root)
    assert "src/app.py" in files and "src/web/index.ts" in files
    assert read_snapshot_file(snapshot_root, "src/app.py").startswith("def run()")

    # 同 base_sha 二次 ensure → 零下载（内容寻址缓存命中）
    client = TarballClient({"src/app.py": "x"})
    await ensure_snapshot(client, _ref(), "deadbeef")
    assert client.downloads == 0


async def test_snapshot_failure_returns_none(tmp_path, monkeypatch):
    from app.services import source_snapshot
    monkeypatch.setattr(source_snapshot.settings, "snapshot_cache_dir", str(tmp_path))
    assert await ensure_snapshot(TarballClient({}, fail=True), _ref(), "sha1") is None


def test_snapshot_path_escape_rejected(snapshot_root):
    assert read_snapshot_file(snapshot_root, "../etc/passwd") is None
    assert read_snapshot_file(snapshot_root, "/abs/path") is None


# ---------------------------------------------------------------------------
# 全仓检索面（ToolContext + 快照）
# ---------------------------------------------------------------------------

def make_ctx(snapshot_root) -> ToolContext:
    from pathlib import Path
    return ToolContext(
        github_client=None, pr_ref=_ref(), head_sha="h",
        changed_files=[{"filename": "src/app.py", "status": "modified", "patch": "x"}],
        snapshot_root=snapshot_root,
        snapshot_files=iter_snapshot_files(snapshot_root),
    )


async def test_code_search_spans_repository(snapshot_root):
    ctx = make_ctx(snapshot_root)
    # helper 只存在于快照（不在变更文件集）——全仓检索面生效的判据
    r = search_in_sources(ctx, "helper", is_regex=False, file_glob=None)
    assert r["total_hits"] >= 1
    assert any(h["file"] == "src/util/helpers.py" for h in r["hits"])


async def test_find_symbol_repo_wide(snapshot_root):
    ctx = make_ctx(snapshot_root)
    r = find_symbol_in_sources(ctx, "render")
    assert r["total"] == 1
    assert r["matches"][0]["file"] == "src/web/index.ts"


async def test_file_read_from_snapshot(snapshot_root):
    from app.agent_loop.agent_tools_v2 import FileReadTool
    ctx = make_ctx(snapshot_root)
    r = await FileReadTool(ctx).invoke({"path": "src/util/helpers.py"})
    assert r.success
    assert "return x * 2" in r.output["content"]


async def test_confinement_still_blocks_outside(snapshot_root):
    from app.agent_loop.agent_tools_v2 import FileReadTool
    ctx = make_ctx(snapshot_root)
    r = await FileReadTool(ctx).invoke({"path": "../outside.py"})
    assert "error" in r.output or r.success is False
