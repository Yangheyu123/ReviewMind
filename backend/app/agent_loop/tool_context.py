"""Agent 工具的执行上下文：变更文件源码缓存、findings 收集、预算记账。

Phase 1 的检索面限定为「PR 变更文件集」（懒拉取 head 版本内容并缓存）；
Phase 2 引入 tarball 源码快照后，同一上下文接口可扩展到全仓检索。
"""

from __future__ import annotations

import re
import shutil
import subprocess
import time
from dataclasses import dataclass, field
from typing import Any

from app.services.ast_context import detect_language
from app.services.github_client import GitHubClient
from app.schemas.github import GitHubPullRequestRef


# 单文件读取上限（行）——防止把超大文件整段灌进上下文
FILE_READ_MAX_LINES = 500

# ---------------------------------------------------------------------------
# 检索后端（P1 自适应）：快照树优先 rg（列表传参，无 shell，无命令注入面；
# 线性时间正则引擎天然免疫 ReDoS），rg 缺席/超时/出错回退纯 Python——
# 零外部依赖保证任何环境可跑。双护栏对两档同时生效：
#   模式复杂度（长度上限）+ 扫描面/时间预算（限制病态 pattern 的爆炸半径）
# ---------------------------------------------------------------------------

MAX_PATTERN_CHARS = 300      # 模式复杂度护栏：超长 pattern 直接拒绝
MAX_LINE_CHARS = 5000        # 行长护栏：Python 逐行匹配跳过超长行（ReDoS 爆炸半径）
SEARCH_TIME_BUDGET_S = 8.0   # 单次检索墙钟预算（Python 后端软限）
_RG_TIMEOUT_S = 10           # rg 子进程硬超时
_RG_BIN: str | None = None
_RG_CHECKED = False


def _rg_binary() -> str | None:
    """能力探测（进程内缓存）：PATH 上找得到 rg 才启用加速档。"""
    global _RG_BIN, _RG_CHECKED
    if not _RG_CHECKED:
        try:
            _RG_BIN = shutil.which("rg")
        except OSError:
            _RG_BIN = None
        _RG_CHECKED = True
    return _RG_BIN


def _rg_search_snapshot(
    ctx: "ToolContext", pattern: str, is_regex: bool, max_hits: int,
) -> dict[str, Any] | None:
    """rg 档：快照树全仓检索。失败/超时/出错返回 None，调用方回退纯 Python。"""
    rg = _rg_binary()
    if rg is None or ctx.snapshot_root is None:
        return None
    args = [rg, "--line-number", "--no-heading", "--color", "never",
            "--path-separator", "/", "--no-ignore", "--hidden",
            "--max-filesize", "2M", "-m", "25"]
    if not is_regex:
        args.append("--fixed-strings")
    args += ["-e", pattern, "."]
    try:
        proc = subprocess.run(
            args, cwd=str(ctx.snapshot_root), capture_output=True,
            text=True, encoding="utf-8", errors="replace", timeout=_RG_TIMEOUT_S,
        )
    except (subprocess.TimeoutExpired, OSError):
        return None
    if proc.returncode not in (0, 1):  # 1 = 无匹配；其余（如非法正则）回退统一口径
        return None
    hits: list[dict[str, Any]] = []
    truncated = False
    for line in proc.stdout.splitlines():
        parts = line.split(":", 2)
        if len(parts) < 3 or not parts[1].isdigit():
            continue
        path = parts[0][2:] if parts[0].startswith("./") else parts[0]
        hits.append({"file": path, "line": int(parts[1]), "text": parts[2][:200]})
        if len(hits) >= max_hits:
            truncated = True
            break
    return {"hits": hits, "truncated": truncated,
            "files_searched": len({h["file"] for h in hits})}


@dataclass
class ToolContext:
    """一次审查任务（或一个分组）的工具执行上下文。

    检索面两级：变更文件集（head 懒加载）→ 源码快照（Phase 2，base tarball 全仓）。
    """

    github_client: GitHubClient
    pr_ref: GitHubPullRequestRef
    head_sha: str
    # Phase 2：base tarball 快照根目录（None = 无快照，检索面退化为变更文件集）
    snapshot_root: Any = None
    snapshot_files: list[str] = field(default_factory=list)
    # 变更文件清单（included，post-filter）：{"filename","status","patch",...}
    changed_files: list[dict[str, Any]] = field(default_factory=list)
    # head 版本源码缓存（懒加载）
    _source_cache: dict[str, str | None] = field(default_factory=dict)
    # submit_finding 的结果汇聚处（引擎聚合时取走）
    findings: list[dict[str, Any]] = field(default_factory=list)
    # 分组内允许访问的文件子集（None = 全部变更文件）
    group_files: list[str] | None = None
    _snapshot_index: set[str] | None = field(default=None, repr=False, compare=False)

    @property
    def accessible_files(self) -> list[dict[str, Any]]:
        if self.group_files is None:
            return self.changed_files
        allow = set(self.group_files)
        return [f for f in self.changed_files if f.get("filename") in allow]

    def _confine(self, path: str) -> dict[str, Any] | None:
        """路径约束：本组变更文件，或快照内任意文件（Phase 2 全仓检索面）。

        绝对路径与 .. 一律拒绝（防逃逸）。
        """
        if not path or path.startswith("/") or ".." in path.split("/"):
            return None
        for f in self.accessible_files:
            if f.get("filename") == path:
                return f
        if self.snapshot_root is not None and path in self._snapshot_set():
            return {"filename": path, "status": "snapshot", "patch": None}
        return None

    def _snapshot_set(self) -> set[str]:
        if self._snapshot_index is None:
            self._snapshot_index = set(self.snapshot_files)
        return self._snapshot_index

    async def get_source(self, path: str) -> str | None:
        """取源码：变更文件走 head 懒加载缓存；其余走 base 快照（Phase 2）。"""
        if self._confine(path) is None:
            return None
        if path not in self._source_cache:
            content: str | None = None
            if any(f.get("filename") == path for f in self.changed_files):
                try:
                    content = await self.github_client.fetch_file_content(self.pr_ref, path, self.head_sha)
                except Exception:
                    content = None
            if content is None and self.snapshot_root is not None:
                from app.services.source_snapshot import read_snapshot_file
                content = read_snapshot_file(self.snapshot_root, path)
            self._source_cache[path] = content
        return self._source_cache[path]

    def patch_of(self, path: str) -> str | None:
        f = self._confine(path)
        return (f or {}).get("patch") or None


def search_in_sources(
    ctx: ToolContext,
    pattern: str,
    is_regex: bool,
    file_glob: str | None,
    max_hits: int = 100,
) -> dict[str, Any]:
    """全仓检索：缓存源码走内存匹配；快照树优先 rg 档、纯 Python 懒扫兜底。

    自定义 file_glob（`**/x*.py` 形态）与 rg glob 语义不等价 → 一律走 Python
    保证两档口径一致；无 glob 时快照树走 rg（能力探测 + 超时回退）。
    """
    if len(pattern) > MAX_PATTERN_CHARS:
        return {"error": f"pattern too long (>{MAX_PATTERN_CHARS} chars)"}
    if is_regex:
        try:
            matcher = re.compile(pattern, re.MULTILINE)
        except re.error as exc:
            return {"error": f"invalid regex: {exc}"}
    else:
        matcher = re.compile(re.escape(pattern))

    glob_matcher = None
    if file_glob:
        seg = file_glob.replace(".", r"\.").replace("**/", "(?:.*/)?").replace("*", "[^/]*")
        try:
            glob_matcher = re.compile(f"^{seg}$")
        except re.error as exc:
            return {"error": f"invalid file_glob: {exc}"}

    hits: list[dict[str, Any]] = []
    seen: set[tuple[str, int]] = set()
    truncated = False
    files_searched = 0
    backend = "python"
    deadline = time.monotonic() + SEARCH_TIME_BUDGET_S

    def _add(path: str, lineno: int, line: str) -> None:
        nonlocal truncated
        if (path, lineno) in seen:
            return
        seen.add((path, lineno))
        hits.append({"file": path, "line": lineno, "text": line[:200]})
        if len(hits) >= max_hits:
            truncated = True

    def _scan_source(path: str, source: str) -> None:
        nonlocal truncated
        files_searched_local[0] += 1
        for lineno, line in enumerate(source.splitlines(), start=1):
            if len(line) > MAX_LINE_CHARS:
                continue  # 超长行跳过：限制病态 pattern 的回溯爆炸半径
            if matcher.search(line):
                _add(path, lineno, line)
                if truncated:
                    return
    files_searched_local = [0]

    # 1) 已缓存源码（变更文件 head 版本等）——内存匹配
    for path in list(ctx._source_cache):
        source = ctx._source_cache.get(path)
        if not source:
            continue
        if glob_matcher is not None and not glob_matcher.match(path):
            continue
        _scan_source(path, source)
        if truncated or time.monotonic() > deadline:
            truncated = True
            break

    # 2) 快照树：无自定义 glob 时走 rg 档；缺席/超时/出错回退纯 Python 懒扫
    if not truncated and ctx.snapshot_root is not None:
        rg_result = None
        if glob_matcher is None:
            rg_result = _rg_search_snapshot(ctx, pattern, is_regex, max_hits - len(hits))
        if rg_result is not None:
            backend = "rg"
            files_searched_local[0] += rg_result["files_searched"]
            for h in rg_result["hits"]:
                _add(h["file"], h["line"], h["text"])
                if truncated:
                    break
            truncated = truncated or rg_result["truncated"]
        else:
            snapshot_paths = [p for p in ctx.snapshot_files if p not in ctx._source_cache]
            if glob_matcher is not None:
                snapshot_paths = [p for p in snapshot_paths if glob_matcher.match(p)]
            for path in snapshot_paths[:300]:
                if time.monotonic() > deadline:
                    truncated = True
                    break
                source = ctx._source_cache.get(path)
                if source is None:
                    from app.services.source_snapshot import read_snapshot_file
                    source = read_snapshot_file(ctx.snapshot_root, path)
                    if source is None:
                        continue
                    ctx._source_cache[path] = source
                if not source:
                    continue
                if glob_matcher is not None and not glob_matcher.match(path):
                    continue
                _scan_source(path, source)
                if truncated:
                    break

    return {
        "hits": hits,
        "total_hits": len(hits),
        "files_searched": files_searched_local[0],
        "cached_files": len([p for p in ctx._source_cache.values() if p]),
        "truncated": truncated,
        "backend": backend,
        "note": f"检索后端={backend}；搜索面含缓存源码与 base 快照"
                f"（Python 兜底单次懒扫 ≤300 文件 / {SEARCH_TIME_BUDGET_S:.0f}s 预算）",
    }


def find_symbol_in_sources(ctx: ToolContext, symbol: str) -> dict[str, Any]:
    """在已缓存源码的 AST 符号表中查定义位置与行区间。"""
    from app.services.ast_context import (
        collect_java_symbol_ranges,
        collect_js_ts_symbol_ranges,
        collect_python_symbol_ranges,
    )

    results: list[dict[str, Any]] = []
    # 解析面 = 已缓存源码 + 快照内可解析文件（Phase 2 全仓，封顶 200 个懒解析）
    scan: list[tuple[str, str | None]] = [(p, src) for p, src in ctx._source_cache.items() if src]
    if ctx.snapshot_root is not None:
        _exts = (".py", ".js", ".jsx", ".ts", ".tsx", ".java")
        extra = [p for p in ctx.snapshot_files
                 if p.endswith(_exts) and p not in ctx._source_cache][:200]
        for p in extra:
            from app.services.source_snapshot import read_snapshot_file
            src = read_snapshot_file(ctx.snapshot_root, p)
            if src:
                ctx._source_cache[p] = src
                scan.append((p, src))
    for path, source in scan:
        if not source:
            continue
        language = detect_language(path)
        try:
            if language == "python":
                import ast as pyast
                ranges = collect_python_symbol_ranges(pyast.parse(source))
            elif language in ("javascript", "typescript"):
                ranges = collect_js_ts_symbol_ranges(source)
            elif language == "java":
                ranges = collect_java_symbol_ranges(source)
            else:
                continue
        except Exception:
            continue
        for r in ranges:
            if r.symbol == symbol or r.symbol.endswith(f".{symbol}"):
                lines = source.splitlines()
                snippet = "\n".join(lines[r.start_line - 1 : min(r.end_line, r.start_line + 9)])
                results.append({
                    "file": path, "symbol": r.symbol, "language": language,
                    "start_line": r.start_line, "end_line": r.end_line,
                    "snippet": snippet,
                })
    return {"matches": results, "total": len(results)}


def read_file_lines(ctx: ToolContext, path: str, start_line: int | None, end_line: int | None) -> dict[str, Any]:
    """按行区间读文件，带截断标记与行号前缀。"""
    source = ctx._source_cache.get(path)
    if source is None:
        return {"error": f"file not cached: {path}（先调用 file_read 无区间版本预热）"}
    lines = source.splitlines()
    total = len(lines)
    start = max(1, start_line or 1)
    end = min(total, end_line or (start + FILE_READ_MAX_LINES - 1))
    if end - start + 1 > FILE_READ_MAX_LINES:
        end = start + FILE_READ_MAX_LINES - 1
    selected = [f"{i}: {lines[i - 1]}" for i in range(start, end + 1)]
    return {
        "file": path,
        "start_line": start,
        "end_line": end,
        "total_lines": total,
        "is_truncated": end < total or (end_line or 0) > end,
        "content": "\n".join(selected),
    }
