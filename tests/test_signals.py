from __future__ import annotations

import unittest

from dejabuy.state import DialogState, Slot
from dejabuy.memory import session as signals


class SignalsTest(unittest.TestCase):
    """每个 test 用独立 session_id——signals._STORE 是模块级字典，避免互相污染。"""

    def test_gate_skips_rejection_tracking_while_scenario_unknown(self) -> None:
        state = DialogState(session_id="sig-gate", profile={})
        # override 触发句之前，scenario 仍是默认值 "unknown"
        state.last_ranked = ["A", "B"]
        signals.update(state)
        self.assertEqual(signals.rejected_asins("sig-gate"), frozenset())

    def test_rejection_tracked_once_scenario_known(self) -> None:
        state = DialogState(session_id="sig-reject", profile={})
        state.scenario = "buying"
        # 第 1 轮：distill() 调用时 last_ranked 还是初始默认值 []（还没人展示过东西）
        signals.update(state)
        self.assertEqual(signals.rejected_asins("sig-reject"), frozenset())
        # 模拟第 1 轮 rank() 算出的结果，供第 2 轮的 distill() 读到
        state.last_ranked = ["A", "B", "C"]
        signals.update(state)
        self.assertEqual(signals.rejected_asins("sig-reject"), frozenset({"A", "B", "C"}))

    def test_stagnation_increments_on_no_new_info(self) -> None:
        state = DialogState(session_id="sig-stagnant", profile={})
        state.scenario = "browsing"
        state.last_ranked = ["A", "B", "C", "D", "E"]
        signals.update(state)  # 建立基线：scenario 刚从 unknown 变化，计数清零
        self.assertEqual(signals.stagnant_turns("sig-stagnant"), 0)
        signals.update(state)  # 无新槽位、top-5 完全重合
        self.assertEqual(signals.stagnant_turns("sig-stagnant"), 1)
        signals.update(state)
        self.assertEqual(signals.stagnant_turns("sig-stagnant"), 2)

    def test_stagnation_resets_on_new_slot(self) -> None:
        state = DialogState(session_id="sig-reset", profile={})
        state.scenario = "browsing"
        state.last_ranked = ["A", "B", "C", "D", "E"]
        signals.update(state)
        signals.update(state)
        self.assertEqual(signals.stagnant_turns("sig-reset"), 1)
        state.slots.append(Slot(attribute="color", value="blue", hard=True, turn_added=3))
        signals.update(state)
        self.assertEqual(signals.stagnant_turns("sig-reset"), 0)

    def test_actionable_rejections_withheld_until_stagnation_threshold(self) -> None:
        """诊断实证（public_0087，team/experiments.md #10）：还没停滞时新信息可能让排序
        改善、把之前没命中的目标送回 top-10；这时候把它拉黑是误伤，必须等真的停滞了再启用。"""
        state = DialogState(session_id="sig-actionable", profile={})
        state.scenario = "browsing"
        state.last_ranked = ["A", "B", "C", "D", "E"]
        signals.update(state)  # 第1轮：gate 刚打开，rejected 还是空
        self.assertEqual(signals.actionable_rejections("sig-actionable"), frozenset())

        signals.update(state)  # 第2轮：rejected 已有内容，但 stagnant_turns 只有 1
        self.assertEqual(signals.rejected_asins("sig-actionable"), {"A", "B", "C", "D", "E"})
        self.assertEqual(signals.actionable_rejections("sig-actionable"), frozenset())

        signals.update(state)  # 第3轮：连续两轮无新信息，stagnant_turns=2，达到门槛
        self.assertEqual(signals.stagnant_turns("sig-actionable"), 2)
        self.assertEqual(
            signals.actionable_rejections("sig-actionable"),
            frozenset({"A", "B", "C", "D", "E"}),
        )

    def test_stagnation_resets_on_low_overlap(self) -> None:
        state = DialogState(session_id="sig-overlap", profile={})
        state.scenario = "browsing"
        state.last_ranked = ["A", "B", "C", "D", "E"]
        signals.update(state)
        state.last_ranked = ["F", "G", "H", "I", "J"]  # 候选池完全换了一批
        signals.update(state)
        self.assertEqual(signals.stagnant_turns("sig-overlap"), 0)


if __name__ == "__main__":
    unittest.main()
