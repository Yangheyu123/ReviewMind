"""源码快照：base 分支 tarball 的下载、解压与内容寻址缓存（Phase 2）。

设计（对齐改进方案 §5.1）：
- 内容寻址：缓存键 = ``{owner}_{repo}_{base_sha}``——base 不变零重复下载，
  base 变则键变，永远不会读到陈旧代码；
- 三消费者共用：code_search / find_symbol / file_read 全仓检索面；
- 护栏：单快照文件数上限、tarball 体积上限、缓存总量上限 + LRU 逐出（防磁盘膨胀，§14 墙 3）；
- 任何失败都不阻塞审查管线（返回 None，调用方降级为"仅变更文件"检索面）。
"""

from __future__ import annotations

import io
import logging
import shutil
import tarfile
from pathlib import Path

from app.core.config import settings

logger = logging.getLogger(__name__)

# 护栏常量（§5.1 大仓库护栏）
MAX_TARBALL_BYTES = 300 * 1024 * 1024   # 单 tarball 300MB
MAX_FILES_PER_SNAPSHOT = 20_000         # 单快照文件数上限
MAX_SNAPSHOT_FILE_BYTES = 2 * 1024 * 1024  # 单文件读取上限（防超大文件进检索）


def snapshot_cache_dir() -> Path:
    root = Path(settings.snapshot_cache_dir)
    root.mkdir(parents=True, exist_ok=True)
    return root


def _safe_extract(tar: tarfile.TarFile, dest: Path) -> int:
    """安全解压：剥离 GitHub tarball 的顶层目录、拒绝绝对路径与 ..、跳过非常规文件。

    Returns:
        解压出的文件数。
    """
    dest.mkdir(parents=True, exist_ok=True)
    count = 0
    for member in tar.getmembers():
        if count >= MAX_FILES_PER_SNAPSHOT:
            logger.warning("[SNAPSHOT] file cap reached (%d), rest skipped", MAX_FILES_PER_SNAPSHOT)
            break
        if not member.isfile():
            continue
        name = member.name
        # GitHub tarball 顶层形如 owner-repo-sha/，剥离之
        parts = name.split("/", 1)
        if len(parts) < 2 or not parts[1]:
            continue
        rel = parts[1]
        rel_path = Path(rel)
        if rel_path.is_absolute() or ".." in rel_path.parts:
            continue  # 路径逃逸防护
        target = dest / rel_path
        target.parent.mkdir(parents=True, exist_ok=True)
        with tar.extractfile(member) as src, open(target, "wb") as out:
            shutil.copyfileobj(src, out)
        count += 1
    return count


def evict_cache(keep_dir: Path | None = None) -> None:
    """缓存总量超限时按 mtime LRU 逐出（保留当前快照与最近使用）。"""
    root = snapshot_cache_dir()
    limit = settings.snapshot_cache_max_bytes
    entries = [d for d in root.iterdir() if d.is_dir()]
    total = sum(_dir_size(d) for d in entries)
    if total <= limit:
        return
    for d in sorted(entries, key=lambda p: p.stat().st_mtime):
        if total <= limit:
            break
        if keep_dir is not None and d.resolve() == keep_dir.resolve():
            continue
        size = _dir_size(d)
        shutil.rmtree(d, ignore_errors=True)
        total -= size
        logger.info("[SNAPSHOT] evicted %s (%.1f MB)", d.name, size / 1e6)


def _dir_size(d: Path) -> int:
    return sum(f.stat().st_size for f in d.rglob("*") if f.is_file())


async def ensure_snapshot(
    github_client, pr_ref, base_sha: str,
) -> Path | None:
    """确保 base_sha 的源码快照存在，返回快照根目录（失败返回 None）。

    同 base_sha 已有快照直接复用（内容寻址，零重复下载）。
    """
    if not base_sha:
        return None
    root = snapshot_cache_dir()
    # 目录名用 12 位短哈希：Windows MAX_PATH=260，长仓库名+40 位 sha+深层相对路径
    # 会超限导致解压/读取失败（dbeaver 等 Java 单体仓实测踩坑）
    dest = root / f"{pr_ref.owner}_{pr_ref.repo}_{base_sha[:12]}"
    if (dest / ".complete").exists():
        return dest

    try:
        tar_bytes = await github_client.download_tarball(pr_ref, base_sha)
    except Exception as exc:
        logger.warning("[SNAPSHOT] tarball download failed: %s", exc)
        return None
    if not tar_bytes:
        return None
    if len(tar_bytes) > MAX_TARBALL_BYTES:
        logger.warning("[SNAPSHOT] tarball too large (%d bytes), skipped", len(tar_bytes))
        return None

    tmp = dest.with_name(dest.name + ".tmp")
    shutil.rmtree(tmp, ignore_errors=True)
    try:
        with tarfile.open(fileobj=io.BytesIO(tar_bytes), mode="r:gz") as tar:
            count = _safe_extract(tar, tmp)
        (tmp / ".complete").write_text(str(count))
        shutil.rmtree(dest, ignore_errors=True)
        tmp.rename(dest)
    except Exception as exc:
        logger.warning("[SNAPSHOT] extract failed: %s", exc)
        shutil.rmtree(tmp, ignore_errors=True)
        return None

    evict_cache(keep_dir=dest)
    logger.info("[SNAPSHOT] ready %s (%d files)", dest.name, count)
    return dest


def read_snapshot_file(root: Path, rel_path: str) -> str | None:
    """读快照内文件（带路径逃逸防护与大小上限），返回文本或 None。"""
    p = Path(rel_path)
    if p.is_absolute() or ".." in p.parts:
        return None
    target = root / p
    try:
        if not target.is_file() or target.stat().st_size > MAX_SNAPSHOT_FILE_BYTES:
            return None
        return target.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return None


def iter_snapshot_files(root: Path, suffixes: set[str] | None = None, limit: int = 5000) -> list[str]:
    """列出快照内文件（可选按后缀过滤），供检索面建立索引。"""
    out: list[str] = []
    for f in root.rglob("*"):
        if len(out) >= limit:
            break
        if not f.is_file() or f.name == ".complete":
            continue
        rel = f.relative_to(root).as_posix()
        if suffixes and not any(rel.endswith(s) for s in suffixes):
            continue
        out.append(rel)
    return out
