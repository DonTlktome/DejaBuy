"""T4 稳定性硬化验收（A 写的用例，C 改写为 unittest 以便真正被执行）。

背景（evaluator/local_evaluator.py 核实）：评测器只保护 respond()（异常 = 空响应丢一轮）；
Agent 构造和 reset() 抛异常 = 整场评测直接崩。正式评测还可能把超时/非法输出记 miss。

⚠️ 为什么改写：原版是 pytest 风格（模块级 `def test_*(tmp_path)` + pytest 的 tmp_path fixture）。
本项目基础路径零第三方依赖、用 stdlib unittest，`unittest discover` 只收集 TestCase 子类，
所以这 5 个用例**一次都没被执行过**（`discover` 报 15 tests，全部来自 test_evaluator/
test_lexicon/test_signals）。逐个手动执行验证过：**5 个断言逻辑本身全部通过**，
A 的稳定性硬化是对的，问题只在收集不到。

现已并入 scripts/check_guards.py 的护栏：任何 tests/test_*.py 里定义了用例却被
discover 漏掉，自检会直接报错——同类问题不会再悄悄发生。
"""

from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from dejabuy.agent import DejaBuyAgent as ShoppingAgent  # noqa: E402

_VALID_ASK = {"category", "material", "color", "size", "style", "brand",
              "budget", "feature", "use_case", "other"}

_PRODUCTS = [
    {"parent_asin": "A001", "title": "Blue Cotton Socks", "price": 9.99,
     "features": ["100% Cotton", "Machine Wash"], "categories": ["Clothing", "Socks"]},
    {"parent_asin": "A002", "title": "Leather Belt", "price": 19.99,
     "features": ["100% Leather"], "categories": ["Clothing", "Belts"]},
    {"parent_asin": "A003", "title": "Wool Scarf", "price": 14.5,
     "features": ["Warm wool blend"], "categories": ["Clothing", "Scarves"]},
]


class StabilityTest(unittest.TestCase):
    """评测器保护圈外的两个入口（构造 / reset）+ 畸形输入。"""

    def setUp(self) -> None:
        self.tmp_path = Path(tempfile.mkdtemp())

    # ---- 辅助 ----
    def _write_catalog(self, corrupt: bool = False) -> str:
        lines = [json.dumps(p) for p in _PRODUCTS]
        if corrupt:
            lines.insert(1, "{not valid json!!!")              # 坏 JSON
            lines.insert(3, json.dumps({"title": "no asin"}))  # 缺 parent_asin
            lines.append("")                                   # 空行
        path = self.tmp_path / "catalog.jsonl"
        path.write_text("\n".join(lines) + "\n", encoding="utf-8")
        return str(path)

    def _assert_legal(self, resp: object, top_k: int = 10) -> None:
        self.assertIsInstance(resp, dict)
        self.assertIsInstance(resp["message"], str)
        self.assertIn(resp["ask_attribute"], _VALID_ASK)
        recs = resp["recommendations"]
        self.assertIsInstance(recs, list)
        self.assertLessEqual(len(recs), top_k)
        for rec in recs:
            self.assertIsInstance(rec["parent_asin"], str)

    # ---- 用例 ----
    def test_corrupt_catalog_lines_skipped(self) -> None:
        """畸形 catalog 行必须跳过而不是让构造函数崩——构造抛异常 = 整场评测崩。"""
        agent = ShoppingAgent(self._write_catalog(corrupt=True))
        self.assertEqual(len(agent.retriever.products), len(_PRODUCTS))
        self.assertEqual(agent.retriever.skipped_lines, 3)  # 坏 JSON + 缺 asin + 空行
        agent.reset("s", {})
        self._assert_legal(
            agent.respond("s", "I'm looking for Socks. A key requirement is: 100% Cotton.", 1, 10))

    def test_reset_never_raises(self) -> None:
        """reset() 不在评测器的保护圈内，任何 profile 形态都不能抛。"""
        agent = ShoppingAgent(self._write_catalog())
        for profile in (None, [], "junk", 42, {"preference_tags": None}):
            with self.subTest(profile=profile):
                agent.reset("s1", profile)  # 不抛即过
        self._assert_legal(
            agent.respond("s1", "I'm looking for Belts, but I'm still exploring.", 1, 10))

    def test_respond_malformed_inputs(self) -> None:
        """畸形消息一律返回合法响应（顶层 try/except 兜底）。"""
        agent = ShoppingAgent(self._write_catalog())
        agent.reset("s2", {})
        for msg in ("", "   ", None, "🧦" * 500, "x" * 200_000,
                    "Actually, please ignore my earlier preference.",
                    "For that, what matters is: .", "I'm looking for "):
            with self.subTest(msg=repr(msg)[:40]):
                self._assert_legal(agent.respond("s2", msg, 1, 10))

    def test_respond_without_reset(self) -> None:
        """评测器保证先 reset，但正式环境不保证——未 reset 的 session 也要出合法响应。"""
        agent = ShoppingAgent(self._write_catalog())
        self._assert_legal(
            agent.respond("ghost",
                          "I'm looking for Scarves. A key requirement is: Warm wool blend.", 1, 10))

    def test_fallback_recommends_last_ranked(self) -> None:
        """兜底路径必须带上一轮的推荐，而不是空手——空响应等于白丢一轮。"""
        agent = ShoppingAgent(self._write_catalog())
        agent.reset("s3", {})
        first = agent.respond("s3", "I'm looking for Socks. A key requirement is: 100% Cotton.", 1, 10)
        self.assertTrue(first["recommendations"])
        # None 消息触发内部异常 → 走兜底路径
        fallback = agent.respond("s3", None, 2, 10)
        self.assertEqual([r["parent_asin"] for r in fallback["recommendations"]],
                         [r["parent_asin"] for r in first["recommendations"]])


if __name__ == "__main__":
    unittest.main()
