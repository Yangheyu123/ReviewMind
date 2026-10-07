import ast
import re
from dataclasses import dataclass

from app.schemas.ast_context import AstContext

# 语言后缀映射表
LANGUAGE_EXTENSIONS: dict[str, str] = {
    ".py": "python",
    ".js": "javascript",
    ".jsx": "javascript",
    ".ts": "typescript",
    ".tsx": "typescript",
    ".java": "java",
}

# JS/TS 函数声明正则：function / async function / 箭头函数 / class method
# 注意 1：行首空白只匹配 [ \t]，\s 会吞掉上一行的换行符导致 start_line 偏移一行
# 注意 2：class method 分支用负向前瞻排除 if/for/while 等语句关键字（for 循环头长得像方法签名）
_JS_TS_FUNC_RE = re.compile(
    r"^[ \t]*(?:(?:export\s+)?(?:default\s+)?(?:async\s+)?)?"
    r"(?:function\s+(\w+)"
    r"|const\s+(\w+)\s*=\s*(?:async\s+)?(?:\([^)]*\)|[\w]+)\s*=>"
    r"|const\s+(\w+)\s*=\s*(?:async\s+)?function"
    r"|(?!(?:if|for|while|switch|catch|return|function)\b)(\w+)\s*\([^)]*\)\s*\{)",
    re.MULTILINE,
)

# Java 方法正则：[access] [static] Type methodName(...)
# 类型字符类不含 \s（换成 空格+Tab），避免跨行匹配把 start_line 拖到上一行
_JAVA_METHOD_RE = re.compile(
    r"^[ \t]*(?:(?:public|protected|private)\s+)?"
    r"(?:static\s+)?"
    r"(?:[\w<>\[\], \t]+?)\s+(\w+)\s*\([^)]*\)\s*\{",
    re.MULTILINE,
)

# Java 类声明正则
_JAVA_CLASS_RE = re.compile(
    r"^\s*(?:public\s+)?(?:abstract\s+)?(?:class|interface|enum)\s+(\w+)",
    re.MULTILINE,
)


@dataclass(frozen=True)
class SymbolRange:
    symbol: str
    start_line: int
    end_line: int


def extract_ast_context(
    file_path: str, source_code: str, changed_lines: list[int]
) -> list[AstContext]:
    """提取变更行对应的 AST 上下文，支持 Python/JS/TS/Java。"""
    language = detect_language(file_path)
    normalized_lines = sorted(set(line for line in changed_lines if line > 0))

    if language == "python":
        return _extract_python(file_path, source_code, normalized_lines, language)

    if language in ("javascript", "typescript"):
        return _extract_by_regex(
            file_path, source_code, normalized_lines, language,
            collect_js_ts_symbol_ranges,
        )

    if language == "java":
        return _extract_by_regex(
            file_path, source_code, normalized_lines, language,
            collect_java_symbol_ranges,
        )

    return [_file_context(file_path, source_code, normalized_lines, language, reason="unsupported language")]


def detect_language(file_path: str) -> str:
    """根据文件后缀检测编程语言。"""
    for ext, lang in LANGUAGE_EXTENSIONS.items():
        if file_path.endswith(ext):
            return lang
    return "unknown"


def collect_python_symbol_ranges(tree: ast.AST) -> list[SymbolRange]:
    """从 Python AST 收集函数/方法范围。"""
    ranges: list[SymbolRange] = []
    for node in tree.body if isinstance(tree, ast.Module) else []:
        if isinstance(node, ast.ClassDef):
            ranges.extend(_class_method_ranges(node))
        elif isinstance(node, ast.FunctionDef | ast.AsyncFunctionDef):
            ranges.append(_node_range(node.name, node))
    return sorted(ranges, key=lambda item: (item.start_line, item.end_line, item.symbol))


def collect_js_ts_symbol_ranges(source_code: str) -> list[SymbolRange]:
    """用正则从 JS/TS 源码收集函数/方法范围。"""
    lines = source_code.splitlines()
    ranges: list[SymbolRange] = []
    for match in _JS_TS_FUNC_RE.finditer(source_code):
        name = next((g for g in match.groups() if g), None)
        if not name:
            continue
        start_line = source_code[: match.start()].count("\n") + 1
        end_line = _find_block_end(lines, start_line - 1)
        ranges.append(SymbolRange(symbol=name, start_line=start_line, end_line=end_line))
    return sorted(ranges, key=lambda item: (item.start_line, item.end_line, item.symbol))


def collect_java_symbol_ranges(source_code: str) -> list[SymbolRange]:
    """用正则从 Java 源码收集方法范围。"""
    lines = source_code.splitlines()
    ranges: list[SymbolRange] = []
    class_match = _JAVA_CLASS_RE.search(source_code)
    class_name = class_match.group(1) if class_match else None
    for match in _JAVA_METHOD_RE.finditer(source_code):
        method_name = match.group(1)
        start_line = source_code[: match.start()].count("\n") + 1
        end_line = _find_block_end(lines, start_line - 1)
        symbol = f"{class_name}.{method_name}" if class_name else method_name
        ranges.append(SymbolRange(symbol=symbol, start_line=start_line, end_line=end_line))
    return sorted(ranges, key=lambda item: (item.start_line, item.end_line, item.symbol))


def _extract_python(
    file_path: str, source_code: str, normalized_lines: list[int], language: str
) -> list[AstContext]:
    """Python AST 解析路径。"""
    try:
        tree = ast.parse(source_code)
    except SyntaxError:
        return [_file_context(file_path, source_code, normalized_lines, language, reason="python ast parse failed")]

    symbol_ranges = collect_python_symbol_ranges(tree)
    matched = [sr for sr in symbol_ranges if _overlaps(sr, normalized_lines)]
    if not matched:
        return [_file_context(file_path, source_code, normalized_lines, language, reason="changed lines are outside symbols")]

    return _build_contexts(file_path, source_code, normalized_lines, language, _dedupe_ranges(matched))


def _extract_by_regex(
    file_path: str, source_code: str, normalized_lines: list[int],
    language: str, collector,
) -> list[AstContext]:
    """正则解析路径（JS/TS/Java 共用）。"""
    try:
        symbol_ranges = collector(source_code)
    except Exception:
        return [_file_context(file_path, source_code, normalized_lines, language, reason=f"{language} regex parse failed")]

    matched = [sr for sr in symbol_ranges if _overlaps(sr, normalized_lines)]
    if not matched:
        return [_file_context(file_path, source_code, normalized_lines, language, reason="changed lines are outside symbols")]

    return _build_contexts(file_path, source_code, normalized_lines, language, _dedupe_ranges(matched))


def _build_contexts(
    file_path: str, source_code: str, normalized_lines: list[int],
    language: str, ranges: list[SymbolRange],
) -> list[AstContext]:
    """将匹配的符号范围构建为 AstContext 列表。"""
    source_lines = source_code.splitlines()
    return [
        AstContext(
            file=file_path,
            symbol=sr.symbol,
            start_line=sr.start_line,
            end_line=sr.end_line,
            changed_lines=[ln for ln in normalized_lines if sr.start_line <= ln <= sr.end_line],
            language=language,
            code="\n".join(source_lines[sr.start_line - 1 : sr.end_line]),
        )
        for sr in ranges
    ]


def _find_block_end(lines: list[str], start_idx: int) -> int:
    """从起始行索引出发，通过花括号配对找到代码块结束行。"""
    depth = 0
    found_open = False
    for i in range(start_idx, len(lines)):
        for ch in lines[i]:
            if ch == "{":
                depth += 1
                found_open = True
            elif ch == "}":
                depth -= 1
                if found_open and depth <= 0:
                    return i + 1  # 转为 1-based 行号
    return len(lines)  # 未找到闭合则返回文件末尾


def _class_method_ranges(class_node: ast.ClassDef) -> list[SymbolRange]:
    ranges: list[SymbolRange] = []
    for node in class_node.body:
        if isinstance(node, ast.FunctionDef | ast.AsyncFunctionDef):
            ranges.append(_node_range(f"{class_node.name}.{node.name}", node))
    return ranges


def _node_range(symbol: str, node: ast.FunctionDef | ast.AsyncFunctionDef) -> SymbolRange:
    return SymbolRange(
        symbol=symbol,
        start_line=node.lineno,
        end_line=getattr(node, "end_lineno", node.lineno),
    )


def _overlaps(symbol_range: SymbolRange, changed_lines: list[int]) -> bool:
    return any(symbol_range.start_line <= line <= symbol_range.end_line for line in changed_lines)


def _dedupe_ranges(ranges: list[SymbolRange]) -> list[SymbolRange]:
    seen: set[tuple[str, int, int]] = set()
    unique: list[SymbolRange] = []
    for item in ranges:
        key = (item.symbol, item.start_line, item.end_line)
        if key not in seen:
            seen.add(key)
            unique.append(item)
    return unique


def _file_context(
    file_path: str, source_code: str, changed_lines: list[int],
    language: str, reason: str,
) -> AstContext:
    source_lines = source_code.splitlines()
    return AstContext(
        file=file_path,
        symbol=None,
        start_line=1 if source_lines else 0,
        end_line=len(source_lines),
        changed_lines=changed_lines,
        language=language,
        code=source_code,
        degraded=True,
        reason=reason,
    )