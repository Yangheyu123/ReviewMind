"""§16 实现测试：分层分组（import 图/枢纽/双护栏/降级）、双字段 type、三级去重。"""

import pytest

from app.services.grouping import build_groups_v2
from app.agent_loop.agent_tools_v2 import FINDING_CATEGORIES
from app.agent_loop.engine import _dedupe_findings


# ---------------------------------------------------------------------------
# 分层分组
# ---------------------------------------------------------------------------

SRC_SERVICE = """
from app.dao.order_dao import OrderDAO
from app.util import helper

def run():
    return OrderDAO().get()
"""

SRC_DAO = """
class OrderDAO:
    def get(self):
        return 1
"""

SRC_UNRELATED = """
import zipfile

def other():
    return 2
"""


def _files(names_adds: list[tuple[str, int]]) -> list[dict]:
    # (filename, additions)——additions 大于阈值避免被当枢纽
    return [{"filename": n, "status": "modified", "additions": a, "deletions": 0,
             "patch": "+x\n" * min(a, 10)} for n, a in names_adds]


def test_import_graph_groups_call_chain():
    files = _files([
        ("src/app/services/order_service.py", 20),
        ("src/app/dao/order_dao.py", 20),
        ("src/app/other/standalone.py", 20),
        ("src/app/other/standalone2.py", 20),
        ("docs/readme.md", 20),
    ])
    src = {
        "src/app/services/order_service.py": SRC_SERVICE,
        "src/app/dao/order_dao.py": SRC_DAO,
    }
    groups, layer = build_groups_v2(files, lambda p: src.get(p))
    # service ↔ dao（import 边）必须同组；调用链完整
    flat_groups = [set(g) for g in groups]
    chain = next(g for g in flat_groups if "src/app/services/order_service.py" in g)
    assert "src/app/dao/order_dao.py" in chain
    assert "src/app/other/standalone.py" not in chain
    assert "import-graph" in layer


def test_hub_files_do_not_weld_components():
    # __init__.py 被两者 import——不作合并节点，两个分量不该被焊在一起
    files = _files([
        ("pkg/a/__init__.py", 5),
        ("pkg/a/mod1.py", 20),
        ("pkg/a/mod2.py", 20),
        ("pkg/b/mod3.py", 20),
        ("pkg/b/mod4.py", 20),
        ("pkg/extra.py", 20),
    ])
    src = {
        "pkg/a/mod1.py": "from pkg.a import something\n",
        "pkg/b/mod3.py": "from pkg.a import other\n",
        "pkg/extra.py": "import os\n",
    }
    groups, _ = build_groups_v2(files, lambda p: src.get(p))
    flat_groups = [set(g) for g in groups]
    g1 = next(g for g in flat_groups if "pkg/a/mod1.py" in g)
    g2 = next(g for g in flat_groups if "pkg/b/mod3.py" in g)
    assert g1 is not g2  # __init__ 枢纽未焊接两个分量


def test_guard_splits_oversized_component():
    files = _files([(f"src/big/mod_{i}.py", 30) for i in range(15)])
    # mod_i import mod_0 → 全连成一锅 → 双护栏拆分
    src = {f"src/big/mod_{i}.py": f"from src.big.mod_0 import x\n" for i in range(1, 15)}
    src["src/big/mod_0.py"] = "x = 1\n"
    groups, _ = build_groups_v2(files, lambda p: src.get(p))
    assert all(len(g) <= 10 for g in groups)
    assert sum(len(g) for g in groups) == 15  # 不丢文件


def test_small_pr_local_bundle_no_parsing():
    files = _files([("a.py", 5), ("b.py", 5), ("c.py", 5)])
    calls = []
    groups, layer = build_groups_v2(files, lambda p: calls.append(p) or None)
    assert groups == [["a.py", "b.py", "c.py"]]
    assert layer.startswith("local-bundle")
    assert calls == []  # ≤4 文件零解析零源码读取


def test_grouping_exception_falls_back_sequential():
    files = _files([(f"f{i}.py", 20) for i in range(8)])

    def boom(path):
        raise RuntimeError("no snapshot")

    groups, layer = build_groups_v2(files, boom)
    assert layer == "fallback-sequential"
    assert sum(len(g) for g in groups) == 8


# ---------------------------------------------------------------------------
# 双字段 type
# ---------------------------------------------------------------------------

def test_categories_are_controlled_vocabulary():
    assert "correctness" in FINDING_CATEGORIES and "other" in FINDING_CATEGORIES
    assert len(FINDING_CATEGORIES) == 9


@pytest.fixture
def ctx():
    from app.agent_loop.tool_context import ToolContext
    from app.schemas.github import GitHubPullRequestRef
    from tests.test_agent_v2 import PY_SOURCE
    c = ToolContext(
        github_client=None,
        pr_ref=GitHubPullRequestRef(owner="o", repo="r", pull_number=1,
                                    html_url="https://github.com/o/r/pull/1"),
        head_sha="h",
        changed_files=[{"filename": "a.py", "status": "modified",
                        "patch": "@@ -1 +1 @@\n-a\n+b"}],
    )
    c._source_cache["a.py"] = PY_SOURCE
    return c


async def test_submit_rejects_invalid_category(ctx):
    from app.agent_loop.agent_tools_v2 import SubmitFindingTool
    tool = SubmitFindingTool(ctx)
    r = await tool.invoke({"comments": [
        {"file": "a.py", "line": 2, "existing_code": "x", "level": "HIGH",
         "category": "made-up-category", "description": "d"},
    ]})
    assert r.output["accepted"] == 0  # 非法粗分类拒收
    r2 = await tool.invoke({"comments": [
        {"file": "a.py", "line": 2, "existing_code": "x", "level": "HIGH",
         "category": "security", "type_detail": "sql-injection via f-string", "description": "d"},
    ]})
    assert r2.output["accepted"] == 1
    assert ctx.findings[-1]["category"] == "security"
    assert ctx.findings[-1]["type_detail"] == "sql-injection via f-string"


# ---------------------------------------------------------------------------
# 三级去重
# ---------------------------------------------------------------------------

def f1() -> dict:
    return {"file": "a.py", "line": 10, "anchored_line": 10, "existing_code": "x = 1",
            "level": "HIGH", "category": "correctness", "type_detail": "null path",
            "description": "缺少对 x 的判空，空值会崩溃", "anchor_status": "hunk"}


def test_level1_exact_triple_dedup():
    dup = {**f1(), "description": "重复提交的同一条"}
    assert len(_dedupe_findings([f1(), dup])) == 1


def test_level2_anchor_overlap_clusters_then_soft_merges():
    # 同文件、锚点区间重叠、category 不同——词重叠高（共享核心词 判空/None/崩溃）→ 软合并
    other = {**f1(), "category": "error-handling", "type_detail": "None crash",
             "description": "x 未判空，None 时直接崩溃"}
    out = _dedupe_findings([f1(), other])
    assert len(out) == 1
    assert out[0]["level"] == "HIGH"  # 保留 level 更高者


def test_level3_same_cluster_different_problem_kept():
    # 同文件、锚点重叠、但词重叠低且 category 不同 → 两个不同问题都保留
    diff_problem = {**f1(), "line": 12, "anchored_line": 12, "category": "performance",
                    "type_detail": "loop query", "description": "循环内逐条查询数据库性能差",
                    "existing_code": "for i in items:\n    q(i)"}
    out = _dedupe_findings([f1(), diff_problem])
    assert len(out) == 2


def test_legacy_type_normalized_to_category():
    legacy = {"file": "a.py", "line": 1, "type": "sql-injection", "description": "注入"}
    out = _dedupe_findings([legacy])
    assert out[0]["category"] == "security"
