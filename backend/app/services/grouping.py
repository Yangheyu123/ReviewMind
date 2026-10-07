"""分层分组（§16.1 定稿实现）：import 依赖图 → 目录亲和 → 顺序兜底。

三层瀑布，误差单调有界：任何一层失灵，结果不劣于顺序切块。
- 第 1 层 import 依赖图：变更文件为节点，A 的 import 引用 B（两端均在变更集）
  则连边；连通分量即分组——调用链完整。枢纽治理：__init__.py 与变更行数 <3
  的纯被引文件不作为合并节点（防 utils/__init__ 把无关文件焊成一锅）。
  超大分量拆分双护栏：token 预算（diff 字符估算）+ 文件数上限，均按目录前缀切。
- 第 2 层目录亲和：无依赖边的文件同目录优先同包。
- 第 3 层顺序装包：兜底，等价旧顺序切块。
"""

from __future__ import annotations

import ast
import logging
import re
from typing import Any, Callable

logger = logging.getLogger(__name__)

GROUP_MAX_FILES = 10                 # 文件护栏（辅助）
GROUP_MAX_DIFF_CHARS = 40_000        # token 护栏：组内 diff 总量（≈10k token）
HUB_MIN_CHANGED_LINES = 3            # 变更行数低于此值的纯被引文件不作合并节点

_JS_IMPORT_RE = re.compile(
    r"""(?:from\s+['"]([^'"]+)['"]|import\s+['"]([^'"]+)['"]|require\(\s*['"]([^'"]+)['"]\s*\))""",
    re.VERBOSE,
)


class _UnionFind:
    def __init__(self, n: int) -> None:
        self.parent = list(range(n))

    def find(self, x: int) -> int:
        while self.parent[x] != x:
            self.parent[x] = self.parent[self.parent[x]]
            x = self.parent[x]
        return x

    def union(self, a: int, b: int) -> None:
        ra, rb = self.find(a), self.find(b)
        if ra != rb:
            self.parent[rb] = ra


def _parse_python_imports(source: str, own_path: str) -> set[str]:
    """提取 Python import 的可解析目标路径（相对仓库根的点分路径 + 相对导入展开）。"""
    targets: set[str] = set()
    own_dir_parts = own_path.rsplit("/", 1)[0].split("/") if "/" in own_path else []
    try:
        tree = ast.parse(source)
    except SyntaxError:
        return targets
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                targets.add(alias.name.replace(".", "/"))
        elif isinstance(node, ast.ImportFrom):
            base = (node.module or "").replace(".", "/")
            level = node.level or 0
            if level > 0:  # 相对导入：from .x import y → 同目录展开
                prefix = "/".join(own_dir_parts[: len(own_dir_parts) - (level - 1)]) if level > 1 else "/".join(own_dir_parts)
                base = f"{prefix}/{base}".strip("/")
            if base:
                targets.add(base)  # 模块本体可能是文件（from x.y import Z → x/y.py）
            for alias in node.names:
                targets.add(f"{base}/{alias.name}".strip("/"))
    return targets


def _parse_js_imports(source: str, own_path: str) -> set[str]:
    """提取 JS/TS 的相对导入目标（相对 own_path 目录展开）。"""
    targets: set[str] = set()
    own_dir = own_path.rsplit("/", 1)[0] if "/" in own_path else ""
    for m in _JS_IMPORT_RE.finditer(source):
        spec = m.group(1) or m.group(2) or m.group(3) or ""
        if not spec.startswith("."):
            continue  # 仅相对导入可解析
        parts = (own_dir.split("/") if own_dir else []) + spec.split("/")
        resolved: list[str] = []
        for part in parts:
            if part == "." or part == "":
                continue
            if part == "..":
                if resolved:
                    resolved.pop()
                continue
            resolved.append(part)
        targets.add("/".join(resolved))
    return targets


def _match_targets(targets: set[str], path_index: dict[str, int]) -> set[int]:
    """import 目标（无扩展名路径）匹配到变更文件索引（后缀容错 + 尾段匹配）。"""
    matched: set[int] = set()
    for t in targets:
        if not t:
            continue
        if t in path_index:
            matched.add(path_index[t])
            continue
        # 尾段匹配：import a.b.c 可匹配 src/a/b/c.py（取尾段最长的唯一命中）
        best: int | None = None
        for path, idx in path_index.items():
            stem = path.rsplit(".", 1)[0] if "." in path.rsplit("/", 1)[-1] else path
            if stem == t or stem.endswith(f"/{t}"):
                if best is None or idx == best:
                    best = idx
        if best is not None:
            matched.add(best)
    return matched


def build_groups_v2(
    files: list[dict[str, Any]],
    read_source: Callable[[str], str | None],
) -> tuple[list[list[str]], str]:
    """分层分组主入口。

    Args:
        files: included 文件（含 filename/patch/additions/deletions）
        read_source: 取 head 源码的回调（快照/懒拉取），失败返回 None
    Returns:
        (分组结果, 采用的层级说明)——异常时回退顺序切块。
    """
    filenames = [f["filename"] for f in files]
    if len(filenames) <= 4:
        return [filenames] if filenames else [], "local-bundle(≤4)"

    try:
        path_index = {p.rsplit(".", 1)[0] if "." in p.rsplit("/", 1)[-1] else p: i
                      for i, p in enumerate(filenames)}
        # 枢纽集合：不作合并节点
        hubs = set()
        for i, f in enumerate(files):
            name = f["filename"].rsplit("/", 1)[-1]
            changed_lines = int(f.get("additions", 0) or 0) + int(f.get("deletions", 0) or 0)
            if name == "__init__.py" or (changed_lines < HUB_MIN_CHANGED_LINES):
                hubs.add(i)

        uf = _UnionFind(len(filenames))
        edges = 0
        for i, f in enumerate(files):
            path = f["filename"]
            if i in hubs:
                continue
            source = read_source(path)
            if not source:
                continue
            if path.endswith(".py"):
                targets = _parse_python_imports(source, path)
            elif path.endswith((".js", ".jsx", ".ts", ".tsx", ".mjs", ".cjs")):
                targets = _parse_js_imports(source, path)
            else:
                continue
            for j in _match_targets(targets, path_index):
                if j != i and j not in hubs:
                    uf.union(i, j)
                    edges += 1

        # 连通分量收集
        components: dict[int, list[int]] = {}
        for i in range(len(filenames)):
            components.setdefault(uf.find(i), []).append(i)

        # 第 2 层目录亲和：单文件分量按目录聚拢（同一目录的孤儿合并进同一候选包）
        dir_map: dict[str, list[int]] = {}
        for root, members in components.items():
            if len(members) == 1:
                i = members[0]
                d = filenames[i].rsplit("/", 1)[0] if "/" in filenames[i] else ""
                dir_map.setdefault(d, []).append(i)

        # 装包：先放大分量（护栏拆分），再放目录亲和包，最后顺序兜底
        groups: list[list[str]] = []

        def _diff_chars(members: list[int]) -> int:
            return sum(len(files[i].get("patch") or "") for i in members)

        def _split_by_guard(members: list[int]) -> list[list[int]]:
            """双护栏拆分：token（diff 字符）与文件数；按目录前缀切再顺序装包。"""
            if len(members) <= GROUP_MAX_FILES and _diff_chars(members) <= GROUP_MAX_DIFF_CHARS:
                return [members]
            by_dir: dict[str, list[int]] = {}
            for i in members:
                d = filenames[i].rsplit("/", 1)[0] if "/" in filenames[i] else ""
                by_dir.setdefault(d, []).append(i)
            packs: list[list[int]] = []
            current: list[int] = []
            current_chars = 0
            for d in sorted(by_dir):
                for i in by_dir[d]:
                    if (len(current) >= GROUP_MAX_FILES or
                            current_chars + len(files[i].get("patch") or "") > GROUP_MAX_DIFF_CHARS):
                        if current:
                            packs.append(current)
                        current, current_chars = [], 0
                    current.append(i)
                    current_chars += len(files[i].get("patch") or "")
            if current:
                packs.append(current)
            return packs

        for root, members in sorted(components.items(), key=lambda kv: -len(kv[1])):
            if len(members) > 1:
                for pack in _split_by_guard(sorted(members)):
                    groups.append([filenames[i] for i in pack])
        for d, members in sorted(dir_map.items()):
            for pack in _split_by_guard(sorted(members)):
                groups.append([filenames[i] for i in pack])

        # 第 3 层兜底：任何遗漏文件顺序补包（不应发生，防御性）
        seen = {f for g in groups for f in g}
        leftover = [p for p in filenames if p not in seen]
        for i in range(0, len(leftover), GROUP_MAX_FILES):
            groups.append(leftover[i : i + GROUP_MAX_FILES])

        layer = f"import-graph({edges} edges, {len(components)} components)"
        return groups, layer
    except Exception as exc:
        logger.warning("[GROUPING] v2 failed, fallback to sequential: %s", exc)
        return _sequential(filenames), "fallback-sequential"


def _sequential(filenames: list[str]) -> list[list[str]]:
    return [filenames[i : i + GROUP_MAX_FILES] for i in range(0, len(filenames), GROUP_MAX_FILES)]
