"""第三层防线（LLM 兜底解析）单测——全部 mock，零网络零 key。

unittest.TestCase 风格：本项目基础路径零第三方依赖，`unittest discover` 只收集
TestCase 子类（教训见 team/留言板.md T-007，check_guards.py 有护栏钉着）。
"""

from __future__ import annotations

import sys
import unittest
from pathlib import Path
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from dejabuy.understand import llm_extract  # noqa: E402
from dejabuy.understand import rules as parser  # noqa: E402
from dejabuy.state import DialogState  # noqa: E402

# 一句不含冒号、不匹配任何模板/salvage 规则的改写消息（会走到 LLM 层）
_MSG = "Hey so the thing is 100% Ring Spun Cotton matters a lot to me here"


class LLMParseTest(unittest.TestCase):
    def _arm(self, reply):
        """开 LLM_PARSE、mock chat_json、清熔断计数（addCleanup 自动还原）。"""
        calls: list[str] = []

        def fake_chat_json(system, user, max_tokens=200, timeout=None):
            # timeout= 是解析层专用超时（config.LLM_PARSE_TIMEOUT）传进来的；
            # 替身必须跟着真签名走，否则 7 个用例全部 TypeError 而不是失败——
            # 报错和"测过了"在 CI 摘要里长得不一样，但在"没人看摘要"时一样危险。
            calls.append(user)
            return reply(user) if callable(reply) else reply

        for patcher in (
            mock.patch.object(llm_extract.config, "LLM_PARSE", True),
            mock.patch.object(llm_extract.llm_client, "chat_json", fake_chat_json),
            mock.patch.object(llm_extract, "_failures", 0),
        ):
            patcher.start()
            self.addCleanup(patcher.stop)
        return DialogState(session_id="t", profile={}), calls

    def test_extracts_verbatim_constraints(self) -> None:
        state, calls = self._arm({
            "category": None, "override": False,
            "constraints": ["100% Ring Spun Cotton"],
        })
        parser.update_state(state, _MSG, 1)
        self.assertTrue(calls, "应触发 LLM")
        self.assertEqual([s.value for s in state.slots], ["100% Ring Spun Cotton"])
        self.assertFalse(state.slots[0].hard)
        self.assertTrue(state.slots[0].terms)

    def test_rejects_paraphrased_output(self) -> None:
        """LLM 擅自改写（ring-spun 加了连字符）→ verbatim 校验拒绝；
        且 LLM 已应答即接管——不再退回整句切分撒垃圾槽。"""
        state, _ = self._arm({
            "category": None, "override": False,
            "constraints": ["100% ring-spun cotton fabric"],
        })
        parser.update_state(state, _MSG, 1)
        self.assertEqual(state.slots, [])

    def test_override_promotes(self) -> None:
        state, _ = self._arm({
            "category": None, "override": True,
            "constraints": ["Water Resistant"],
        })
        parser.update_state(state, "For that, what matters is: Water Resistant; Steel Band.", 1)
        parser.update_state(state, "Hmm scrap all that -- Water Resistant is the only thing", 2)
        self.assertEqual(state.scenario, "override")
        self.assertEqual([s.value for s in state.slots if s.hard], ["Water Resistant"])

    def test_override_hint_fallthrough(self) -> None:
        """salvage 的改需求信号命中、但载荷规则没捞到"新需求"→ 第三层接手抽取。"""
        state, calls = self._arm({
            "category": None, "override": True,
            "constraints": ["Water Resistant"],
        })
        parser.update_state(state, "For that, what matters is: Water Resistant; Steel Band.", 1)
        parser.update_state(
            state,
            "Hold on — scratch what I said earlier, Water Resistant is the thing that counts.", 2)
        self.assertTrue(calls, "载荷落空时应触发 LLM")
        self.assertEqual([s.value for s in state.slots if s.hard], ["Water Resistant"])

    def test_circuit_breaker(self) -> None:
        state, calls = self._arm(lambda user: None)
        for turn in (1, 2, 3, 4):
            parser.update_state(state, _MSG, turn)
        self.assertEqual(len(calls), 2, "连续两次失败后熔断，不再调用")

    def test_off_by_default(self) -> None:
        calls: list[int] = []
        for patcher in (
            mock.patch.object(llm_extract.config, "LLM_PARSE", False),
            mock.patch.object(llm_extract.llm_client, "chat_json",
                              lambda *a, **k: calls.append(1)),
        ):
            patcher.start()
            self.addCleanup(patcher.stop)
        state = DialogState(session_id="t", profile={})
        parser.update_state(state, _MSG, 1)
        self.assertEqual(calls, [])

    def test_colon_in_value_picks_first_colon(self) -> None:
        """值内冒号 badcase（实验 33）：'Department: womens; color: pink' 这类约束
        不能被"末冒号"规则劈碎——目录校验应让首冒号候选胜出。"""
        known = {"department womens", "color pink"}
        parser.set_catalog_verifier(lambda phrase: phrase in known)
        self.addCleanup(parser.set_catalog_verifier, None)
        state = DialogState(session_id="t", profile={})
        parser.update_state(
            state, "On that front, here's what counts for me: Department: womens; color: pink", 1)
        self.assertEqual([s.value for s in state.slots], ["Department: womens", "color: pink"])

    def test_garbage_colon_extraction_falls_through_to_llm(self) -> None:
        """粘连口水话的冒号提取全目录查无此文 → 不算成功，第三层必须能出场。"""
        state, calls = self._arm({
            "category": None, "override": False, "constraints": ["100% Acrylic"],
        })
        parser.set_catalog_verifier(lambda phrase: phrase == "100 acrylic")
        self.addCleanup(parser.set_catalog_verifier, None)
        parser.update_state(
            state, "On that front I'd say 100% Acrylic is the thing: honestly whatever works", 1)
        self.assertTrue(calls, "零验证通过时应放行 LLM")
        self.assertEqual([s.value for s in state.slots], ["100% Acrylic"])

    def test_llm_failure_keeps_colon_fragments(self) -> None:
        """LLM 失败时冒号片段保底入槽——信息只多不少。"""
        state, _ = self._arm(lambda user: None)
        parser.set_catalog_verifier(lambda phrase: False)
        self.addCleanup(parser.set_catalog_verifier, None)
        parser.update_state(state, "Anyway the key thing: Warm wool blend", 1)
        self.assertEqual([s.value for s in state.slots], ["Warm wool blend"])

    def test_strict_template_never_reaches_llm(self) -> None:
        """严格模板命中时 LLM 层零触发——公开集零影响的构造保证。"""
        state, calls = self._arm({"category": None, "override": False, "constraints": []})
        parser.update_state(state, "I'm looking for Socks. A key requirement is: 100% Cotton.", 1)
        parser.update_state(state, "For that, what matters is: Machine Wash.", 2)
        self.assertEqual(calls, [])


if __name__ == "__main__":
    unittest.main()
