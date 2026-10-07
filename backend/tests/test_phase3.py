"""Phase 3 测试：锚点三级流水、filter 反思、语言规则清单。"""

import pytest

from app.agent_loop.tool_context import ToolContext
from app.services.comment_anchor import anchor_finding
from app.agents.filter_agent import filter_findings, parse_filter_decision
from app.rules import load_rules_for_group, resolve_rule_docs
from app.schemas.github import GitHubPullRequestRef
from app.core.llm import ToolCallItem, ToolCallResponse


def _ref():
    return GitHubPullRequestRef(owner="o", repo="r", pull_number=1,
                                html_url="https://github.com/o/r/pull/1")


PATCH_A = """@@ -10,3 +10,5 @@ def run():
     old_context_line()
-    user = db.query(f"SELECT {name}")
+    user = db.query("SELECT ...", (name,))
+    logger.info("done")
"""


def make_ctx(patch: str, source: str | None = None) -> ToolContext:
    ctx = ToolContext(
        github_client=None, pr_ref=_ref(), head_sha="h",
        changed_files=[{"filename": "a.py", "status": "modified", "patch": patch}],
    )
    if source is not None:
        ctx._source_cache["a.py"] = source
    return ctx


# ---------------------------------------------------------------------------
# 锚点三级流水
# ---------------------------------------------------------------------------

async def test_anchor_hunk_match_on_added_line():
    ctx = make_ctx(PATCH_A)
    f = {"file": "a.py", "line": 999, "existing_code": 'user = db.query("SELECT ...", (name,))',
         "confidence": 0.8}
    f = await anchor_finding(f, ctx)
    assert f["anchor_status"] == "hunk"
    assert f["anchored_line"] == 11  # hunk +10,5：新文件 10=context、11=替换行
    assert f["confidence"] == 0.8  # 锚定成功不降权


async def test_anchor_fulltext_when_not_in_hunk():
    ctx = make_ctx(PATCH_A, source="x\ny\n\ndef helper():\n    return 42\n")
    f = {"file": "a.py", "line": 0, "existing_code": "def helper():\n    return 42",
         "confidence": 0.7}
    f = await anchor_finding(f, ctx)
    assert f["anchor_status"] == "fulltext"
    assert f["anchored_line"] == 4


async def test_anchor_deleted_line_not_matched():
    # 锚点只出现在删除行 → L1 返回 None → L2 无全文 → unanchored 降权
    ctx = make_ctx(PATCH_A, source="totally unrelated")
    f = {"file": "a.py", "line": 11, "existing_code": 'user = db.query(f"SELECT {name}")',
         "confidence": 0.9}
    f = await anchor_finding(f, ctx)
    assert f["anchor_status"] == "unanchored"
    assert f["confidence"] == pytest.approx(0.45)


async def test_anchor_relocate_across_files(tmp_path, monkeypatch):
    from app.services import source_snapshot
    from tests.test_source_snapshot import TarballClient

    monkeypatch.setattr(source_snapshot.settings, "snapshot_cache_dir", str(tmp_path))
    root = await source_snapshot.ensure_snapshot(
        TarballClient({"src/legacy.py": "a\nb\n\nlegacy_impl(x)\n"}), _ref(), "sha"
    )
    ctx = make_ctx(PATCH_A)
    ctx.snapshot_root = root
    ctx.snapshot_files = source_snapshot.iter_snapshot_files(root)
    # 触发快照文件进入缓存（模拟 code_search 预热）
    ctx._source_cache["src/legacy.py"] = source_snapshot.read_snapshot_file(root, "src/legacy.py")

    f = {"file": "a.py", "line": 0, "existing_code": "legacy_impl(x)", "confidence": 0.6}
    f = await anchor_finding(f, ctx)
    assert f["anchor_status"] == "relocated"
    assert f["file"] == "src/legacy.py"
    assert f["anchored_line"] == 4


# ---------------------------------------------------------------------------
# Filter 反思
# ---------------------------------------------------------------------------

class FilterFakeLLM:
    _mock_mode = False
    is_configured = True

    def __init__(self, response):
        self._response = response

    async def chat_with_tools(self, messages, *, tools, model=None, temperature=0.0, tool_choice="auto"):
        return self._response


FINDINGS = [
    {"id": "f0", "file": "src/a.py", "line": 3, "type": "style", "description": "命名不规范",
     "existing_code": "data = open(path).read()"},
    {"id": "f1", "file": "src/a.py", "line": 2, "type": "concurrency", "description": "并发竞态 risk",
     "existing_code": "return open(path).read()"},
    {"id": "f2", "file": "src/a.py", "line": 4, "type": "other", "category": "security",
     "description": "路径未校验", "existing_code": "return data.strip()"},
    {"id": "f3", "file": "ghost.py", "line": 1, "type": "style", "description": "指向 diff 外文件",
     "existing_code": "zz"},
]
PATCHES = {
    "src/a.py": "@@ -1,3 +1,4 @@\n def load(path):\n-    return open(path).read()\n"
                "+    data = open(path).read()\n+    return data.strip()\n",
}


def _resp_items(items, analysis="逐条分析"):
    return ToolCallResponse(
        content="", finish_reason="tool_calls",
        tool_calls=[ToolCallItem(id="c0", name="report_incorrect_comments",
                                 arguments={"analysis": analysis, "comments": items})],
    )


async def test_filter_removal_requires_verified_evidence():
    items = [
        {"id": "f0", "ground": "B", "evidence": "return data.strip()"},  # 引用逐字命中 diff
        {"id": "f3", "ground": "A", "evidence": ""},                     # 文件不在 diff 清单
    ]
    kept, meta = await filter_findings(
        [dict(f) for f in FINDINGS], PATCHES, FilterFakeLLM(_resp_items(items)))
    assert meta["status"] == "applied"
    assert [f["id"] for f in kept] == ["f1", "f2"]
    assert len(meta["removed"]) == 2 and meta["unverified_kept"] == 0


async def test_filter_unverified_evidence_forces_keep():
    items = [
        {"id": "f0", "ground": "B", "evidence": "no such line in this patch"},  # 引用不存在
        {"id": "f3", "ground": "B", "evidence": "anything at all here"},        # 文件不在 diff，B 不成立
    ]
    kept, meta = await filter_findings(
        [dict(f) for f in FINDINGS], PATCHES, FilterFakeLLM(_resp_items(items)))
    assert len(kept) == 4 and meta["removed"] == []
    assert meta["unverified_kept"] == 2


async def test_filter_protected_by_category_and_keyword():
    valid_evidence = {"ground": "B", "evidence": "return data.strip()"}
    items = [{"id": "f1", **valid_evidence},   # 关键词：并发/竞态
             {"id": "f2", **valid_evidence}]   # 受控 category=security（描述无关键词）
    kept, meta = await filter_findings(
        [dict(f) for f in FINDINGS], PATCHES, FilterFakeLLM(_resp_items(items)))
    assert [f["id"] for f in kept] == ["f0", "f1", "f2", "f3"]
    assert meta["protected_kept"] == 2 and meta["removed"] == []


async def test_filter_ground_a_invalid_when_file_in_diff():
    items = [{"id": "f0", "ground": "A", "evidence": ""}]  # f0 文件在 diff 中，A 不成立
    kept, meta = await filter_findings(
        [dict(f) for f in FINDINGS], PATCHES, FilterFakeLLM(_resp_items(items)))
    assert len(kept) == 4 and meta["unverified_kept"] == 1


async def test_filter_short_evidence_rejected():
    items = [{"id": "f0", "ground": "B", "evidence": "x = 1"}]  # 归一化后 <8 字符
    kept, meta = await filter_findings(
        [dict(f) for f in FINDINGS], PATCHES, FilterFakeLLM(_resp_items(items)))
    assert len(kept) == 4 and meta["unverified_kept"] == 1


async def test_filter_legacy_comment_ids_is_conservative_noop():
    resp = ToolCallResponse(
        content="", finish_reason="tool_calls",
        tool_calls=[ToolCallItem(id="c0", name="report_incorrect_comments",
                                 arguments={"analysis": "Ground A: not in diff", "comment_ids": ["f0", "f3"]})],
    )
    kept, meta = await filter_findings([dict(f) for f in FINDINGS], PATCHES, FilterFakeLLM(resp))
    assert len(kept) == 4  # 旧格式无 ground/evidence → 全部校验不过 → 保守保留
    assert meta["unverified_kept"] == 2


async def test_filter_approve_all():
    resp = ToolCallResponse(content="", finish_reason="tool_calls",
                            tool_calls=[ToolCallItem(id="c0", name="approve_all_comments", arguments={})])
    kept, meta = await filter_findings([dict(f) for f in FINDINGS], PATCHES, FilterFakeLLM(resp))
    assert len(kept) == 4 and meta["status"] == "applied"


async def test_filter_llm_unavailable_keeps_all():
    class Unavailable:
        _mock_mode = True
        is_configured = False
    kept, meta = await filter_findings([dict(f) for f in FINDINGS], PATCHES, Unavailable())
    assert len(kept) == 4 and meta["status"] == "skipped"


def test_parse_filter_decision_text_fallback():
    resp = ToolCallResponse(content='{"comment_ids": ["f9"], "analysis": "Ground B"}', tool_calls=[])
    items, analysis = parse_filter_decision(resp)
    assert [i["id"] for i in items] == ["f9"] and items[0]["ground"] == ""
    assert "Ground B" in analysis


# ---------------------------------------------------------------------------
# 语言规则清单
# ---------------------------------------------------------------------------

def test_rule_resolution_by_suffix():
    assert resolve_rule_docs(["a.py", "b.py"]) == ["python.md"]
    assert resolve_rule_docs(["a.ts", "b.js"]) == ["typescript_javascript.md"]
    assert resolve_rule_docs(["a.java"]) == ["java.md"]
    assert resolve_rule_docs(["README.md"]) == ["default.md"]
    assert resolve_rule_docs(["a.py", "b.ts"]) == ["python.md", "typescript_javascript.md"]


def test_rule_text_injected_into_group_prompt():
    from app.agent_loop.engine import _group_user_prompt
    prompt = _group_user_prompt(
        {"parsed_diff": [{"file": "a.py", "additions": 1, "deletions": 0, "patch": "+x"}]},
        ["a.py"],
    )
    assert "语言审查清单" in prompt
    assert "可变默认参数" in prompt  # python.md 的内容
