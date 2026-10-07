"""Filter 反思节点（Phase 3）——工作流步骤，非 agent（刻意设计）。

对齐改进方案 §7.2 / 阿里 OCR review_filter 范式：
- 只删"diff 能证明错误"的 findings——非对称成本：删掉正确评论 = 无声丢失真实问题，
  保留错误评论 = 浪费几秒注意力，前者代价远高于后者；
- 受保护主题一票豁免——优先按受控 category 字段判定（concurrency/security/resource-leak），
  自由文本关键词兜底（"On a protected subject you do not get to be confident"）；
- 删除决策须通过工程校验（P0 反误杀）：Ground B 的 evidence 必须逐字命中该文件的
  diff 内容、Ground A 仅当评论指向的文件不在 diff 变更清单时成立，验不过强制保留
  ——与锚点同一哲学：行号是匹配出来的，删除也必须是验证出来的，不是模型说了算；
- 强制二选一工具调用：report_incorrect_comments（analysis 先于结论） / approve_all_comments；
- LLM 不可用/解析失败 → 全部保留（保守方向），不阻塞管线。
"""

from __future__ import annotations

import json
import logging
from typing import Any

from app.core.llm import LLMClient, LLMClientError

logger = logging.getLogger(__name__)

# 受保护主题：宁可多留不错杀
PROTECTED_CATEGORIES = ("concurrency", "security", "resource-leak")
PROTECTED_KEYWORDS = (
    "concurrency", "race", "race condition", "thread-safe", "thread safety",
    "lock", "deadlock",
    "memory-safety", "memory safety", "use-after-free", "overflow", "leak",
    "behavior-change", "behavior change", "breaking", "regression",
    "security", "injection", "xss", "csrf", "ssrf", "auth", "permission",
    "并发", "线程", "竞态", "竞争条件", "数据竞争", "死锁",
    "内存安全", "泄漏", "泄露", "行为变更", "回归", "安全", "注入", "越权",
)

FILTER_SYSTEM_PROMPT = """你是代码审查的事实核查员。下面是审查员提交的 findings 与对应的 diff 证据。
你的唯一任务：删除那些"diff 证据能够直接证明是错误"的评论。

成本是非对称的：
- 删掉一条正确的评论 = 真实问题被无声丢失，代价极高；
- 保留一条错误的评论 = 浏览者浪费几秒注意力，代价很低。
因此只在你有把握时删除。不确定 = 保留。风格偏好、可以争论的取舍 = 保留。

合法的删除理由只有两种（工程侧会逐条校验你的证据，验不过自动保留该评论）：
- Ground A：评论指向的文件根本不在本 diff 的变更文件中；
- Ground B：diff 的字面内容直接反驳了评论的说法（如评论说"缺少判空"而 diff 新增行里就有判空）
  ——必须在 evidence 中逐字摘录 diff 里存在的那段反驳代码。

你必须调用且只调用一个工具：
- report_incorrect_comments：先写 analysis（逐条推理），再在 comments 里逐条给出
  id、ground（A 或 B）、evidence（Ground B 必填：从 diff 逐字摘录的反驳代码段）；
- approve_all_comments：全部保留。"""

_FILTER_TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "report_incorrect_comments",
            "description": "报告应删除的错误评论（每条附 ground 与可验证的 evidence）",
            "parameters": {
                "type": "object",
                "properties": {
                    "analysis": {
                        "type": "string",
                        "description": "逐条分析：comment id → 适用 Ground A 还是 B 的具体依据（先写分析再给结论）",
                    },
                    "comments": {
                        "type": "array",
                        "description": "要删除的评论列表（工程侧逐条校验证据）",
                        "items": {
                            "type": "object",
                            "properties": {
                                "id": {"type": "string", "description": "要删除的评论 id"},
                                "ground": {"type": "string", "enum": ["A", "B"],
                                           "description": "A=文件不在 diff 中；B=diff 内容直接反驳"},
                                "evidence": {"type": "string",
                                             "description": "Ground B 必填：从 diff 逐字摘录的反驳代码段"},
                            },
                            "required": ["id", "ground"],
                        },
                    },
                },
                "required": ["analysis", "comments"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "approve_all_comments",
            "description": "全部评论保留，无需删除",
            "parameters": {"type": "object", "properties": {}},
        },
    },
]


def _is_protected(finding: dict[str, Any]) -> bool:
    if str(finding.get("category") or "").strip().lower() in PROTECTED_CATEGORIES:
        return True
    text = f"{finding.get('type', '')} {finding.get('description', '')}".lower()
    return any(kw in text for kw in PROTECTED_KEYWORDS)


def _build_review_payload(findings: list[dict[str, Any]], patches: dict[str, str]) -> str:
    lines = ["## 待核查的 findings", ""]
    for i, f in enumerate(findings):
        fid = f.get("id") or f"f{i}"
        f["_filter_id"] = fid
        lines.append(
            f"[{fid}] file={f.get('file')} line={f.get('line')} level={f.get('level')} "
            f"type={f.get('type')} category={f.get('category')}"
        )
        lines.append(f"  锚点代码: {str(f.get('existing_code', ''))[:200]}")
        lines.append(f"  描述: {str(f.get('description', ''))[:300]}")
        lines.append("")
    lines.append("## diff 证据")
    for path, patch in patches.items():
        lines.append(f"### {path}")
        lines.append(f"```diff\n{patch[:4000]}\n```")
    return "\n".join(lines)


def parse_filter_decision(response) -> tuple[list[dict[str, Any]], str]:
    """从 ToolCallResponse 解析删除决策。返回 (删除项列表, analysis)。

    每个删除项形如 {"id", "ground", "evidence"}；旧格式 comment_ids（无 ground/evidence）
    归一为无 ground 的项——工程校验不过即保留，等效保守 no-op。
    """
    for tc in response.tool_calls:
        if tc.name == "approve_all_comments":
            return [], "approve_all"
        if tc.name == "report_incorrect_comments":
            analysis = str(tc.arguments.get("analysis") or "")
            comments = tc.arguments.get("comments") or []
            items = [
                {
                    "id": str(c.get("id") or ""),
                    "ground": str(c.get("ground") or ""),
                    "evidence": str(c.get("evidence") or ""),
                }
                for c in comments if isinstance(c, dict) and c.get("id")
            ]
            if not items:  # 旧格式兼容：comment_ids -> 无 ground 项（校验不过，保守保留）
                items = [{"id": str(i), "ground": "", "evidence": ""}
                         for i in (tc.arguments.get("comment_ids") or []) if i]
            return items, analysis
    # 无工具调用 → 兜底解析纯 JSON 文本
    try:
        data = json.loads(response.content or "")
        comments = data.get("comments") or []
        items = [
            {"id": str(c.get("id") or ""), "ground": str(c.get("ground") or ""),
             "evidence": str(c.get("evidence") or "")}
            for c in comments if isinstance(c, dict) and c.get("id")
        ]
        if not items:
            items = [{"id": str(i), "ground": "", "evidence": ""}
                     for i in (data.get("comment_ids") or []) if i]
        return items, str(data.get("analysis", ""))
    except (json.JSONDecodeError, TypeError, AttributeError):
        return [], ""


def _norm(text: Any) -> str:
    """证据比对用的归一化：去除全部空白，避免缩进/换行差异导致误判。"""
    return "".join(str(text).split())


def _patch_for_file(file: str, patches: dict[str, str]) -> str | None:
    """取评论所指文件的 patch：精确路径优先，退化到尾缀匹配（跨目录前缀差异）。"""
    if not file:
        return None
    if file in patches:
        return patches[file]
    for key, patch in patches.items():
        if key.endswith(file) or file.endswith(key):
            return patch
    return None


def _validate_removal(finding: dict[str, Any], item: dict[str, Any],
                      patches: dict[str, str]) -> bool:
    """工程校验单条删除决策。返回 True 表示该删除可信。

    - Ground B：evidence 归一化后 ≥8 字符，且逐字命中该文件 patch（归一化比对）；
    - Ground A：评论指向的文件不在 diff 变更清单中（唯一可确定性验证的形态）；
    - 其余（无 ground / 证据过短 / 引用不存在）一律不可信。
    """
    ground = str(item.get("ground") or "").strip().upper()
    patch = _patch_for_file(str(finding.get("file") or ""), patches)
    if ground == "B":
        evidence = _norm(item.get("evidence"))
        if len(evidence) < 8:
            return False
        return patch is not None and evidence in _norm(patch)
    if ground == "A":
        return patch is None
    return False


async def filter_findings(
    findings: list[dict[str, Any]],
    patches: dict[str, str],
    llm: LLMClient | None = None,
    *,
    all_patches: dict[str, str] | None = None,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """反思过滤入口。返回 (保留的 findings, 元信息)。

    LLM 不可用/失败 → 原样返回（保守方向），元信息标记 skipped。
    受保护主题（category/关键词）无条件保留；删除决策逐条过工程校验，
    验不过强制保留——all_patches 为全量变更文件的 patch（校验证据源），
    patches 仅用于构造核查 prompt（组内补丁，控制上下文）。
    """
    meta: dict[str, Any] = {"status": "skipped", "removed": [], "protected_kept": 0,
                            "unverified_kept": 0, "analysis": ""}

    if not findings:
        meta["status"] = "empty"
        return findings, meta

    llm = llm or LLMClient()
    if llm._mock_mode or not llm.is_configured:
        return findings, meta

    payload = _build_review_payload(findings, patches)
    try:
        response = await llm.chat_with_tools(
            [{"role": "system", "content": FILTER_SYSTEM_PROMPT},
             {"role": "user", "content": payload}],
            tools=_FILTER_TOOLS,
            tool_choice="required",
        )
    except LLMClientError as exc:
        logger.warning("[FILTER] LLM failed, keep all findings: %s", exc)
        meta["status"] = "llm_error"
        return findings, meta

    items, analysis = parse_filter_decision(response)
    evidence_pool = all_patches if all_patches is not None else patches
    removal_map = {it["id"]: it for it in items if it.get("id")}

    kept: list[dict[str, Any]] = []
    removed: list[dict[str, Any]] = []
    protected_kept = 0
    unverified_kept = 0
    for f in findings:
        fid = f.get("_filter_id")
        item = removal_map.get(str(fid))
        if item is None:
            kept.append(f)
        elif _is_protected(f):
            protected_kept += 1
            kept.append(f)
        elif not _validate_removal(f, item, evidence_pool):
            unverified_kept += 1
            kept.append(f)
        else:
            removed.append(f)
        f.pop("_filter_id", None)

    meta = {
        "status": "applied",
        "removed": [
            {"id": r.get("id") or f"f{i}", "file": r.get("file"),
             "type": r.get("type"), "ground": (removal_map.get(str(r.get("id"))) or {}).get("ground")}
            for i, r in enumerate(removed)
        ],
        "protected_kept": protected_kept,
        "unverified_kept": unverified_kept,
        "analysis": analysis[:500],
    }
    logger.info("[FILTER] removed %d findings (protected kept: %s, unverified kept: %s)",
                len(removed), protected_kept, unverified_kept)
    return kept, meta
