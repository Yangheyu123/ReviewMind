"""语言审查清单（Phase 3）：按文件后缀选择高信号 checklist 注入组 prompt。

清单是初始版本，Phase 5 评测后按误报/漏报分桶迭代（改进方案 §14 墙 7）。
风格参考 alibaba/open-code-review rule_docs（Apache-2.0）的"高信号+反例约束"范式，
内容为本项目按自身技术栈重写。
"""

from __future__ import annotations

from pathlib import Path

_RULES_DIR = Path(__file__).parent

# glob/后缀 → 清单文件（顺序即优先级，first-match）
_SUFFIX_MAP: list[tuple[tuple[str, ...], str]] = [
    ((".py",), "python.md"),
    ((".ts", ".tsx", ".js", ".jsx", ".mjs", ".cjs"), "typescript_javascript.md"),
    ((".java",), "java.md"),
]


def resolve_rule_docs(filenames: list[str]) -> list[str]:
    """按组内文件后缀选择语言清单（去重保序）。"""
    picked: list[str] = []
    for name in filenames:
        for suffixes, doc in _SUFFIX_MAP:
            if any(name.endswith(s) for s in suffixes) and doc not in picked:
                picked.append(doc)
    if not picked:
        picked.append("default.md")
    return picked


def load_rules_for_group(filenames: list[str], max_chars: int = 4000) -> str:
    """拼装组内适用的审查清单文本（超长截断）。"""
    parts: list[str] = []
    total = 0
    for doc in resolve_rule_docs(filenames):
        path = _RULES_DIR / doc
        if not path.is_file():
            continue
        text = path.read_text(encoding="utf-8").strip()
        if total + len(text) > max_chars:
            text = text[: max_chars - total] + "\n... (truncated)"
        parts.append(text)
        total += len(text)
        if total >= max_chars:
            break
    return "\n\n".join(parts)
