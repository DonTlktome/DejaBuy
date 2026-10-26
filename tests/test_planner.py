"""Expected-utility planner: question choice and slate composition over a support set."""

import unittest

from dejabuy import config
from dejabuy.decide import planner, utility
from dejabuy.state import DialogState


def _products(cards):
    return {pid: {"card_hard": h, "card_soft": s, "coarse_exact": "C"} for pid, (h, s) in cards.items()}


class PlannerTests(unittest.TestCase):
    def setUp(self):
        self._mode = config.WIDTH_MODE
        config.WIDTH_MODE = "benchmark"

    def tearDown(self):
        config.WIDTH_MODE = self._mode

    def test_value_function_matches_benchmark_formula(self):
        self.assertAlmostEqual(utility.session_score(True, 1, 1), 0.5 + 0.3 + 0.2)
        self.assertAlmostEqual(utility.session_score(True, 7, 1), 0.5 + 0.3 / 7 + 0.2)
        self.assertAlmostEqual(utility.session_score(True, 1, 2), 0.5 + 0.3 + 0.18)

    def test_indistinguishable_group_is_walked_one_per_turn(self):
        # four twins: identical cards, no question separates them -> show the leader only
        products = _products({f"p{i}": (["cotton"], ["Imported"]) for i in range(4)})
        support = {pid: {"cotton"} for pid in products}
        beliefs = [("p0", 0.4), ("p1", 0.3), ("p2", 0.2), ("p3", 0.1)]
        state = DialogState(session_id="s", profile={}, opening="buying")
        plan = planner.plan_strict(state, products, support, beliefs, turn=2, top_k=10)
        self.assertEqual(plan.shown, ["p0"])
        self.assertEqual(plan.deferred, ["p1", "p2", "p3"])

    def test_separable_candidates_pick_the_splitting_question(self):
        products = _products({
            "a": (["cotton", "color: red"], ["Imported", "Pull On closure"]),
            "b": (["cotton", "color: blue"], ["Imported", "Pull On closure"]),
            "c": (["cotton", "color: green"], ["Imported", "Pull On closure"]),
        })
        support = {pid: {"cotton"} for pid in products}
        beliefs = [("a", 0.34), ("b", 0.33), ("c", 0.33)]
        state = DialogState(session_id="s", profile={}, opening="buying")
        plan = planner.plan_strict(state, products, support, beliefs, turn=1, top_k=10)
        # asking `color` (or `other`, which discloses the colour too) separates all three
        self.assertIn(plan.ask, ("color", "other"))
        self.assertEqual(plan.shown, ["a"])          # leader now, the rest at rank one next turn

    def test_last_turn_shows_everything(self):
        products = _products({f"p{i}": (["cotton"], ["Imported"]) for i in range(3)})
        support = {pid: {"cotton"} for pid in products}
        beliefs = [("p0", 0.5), ("p1", 0.3), ("p2", 0.2)]
        state = DialogState(session_id="s", profile={}, opening="buying")
        plan = planner.plan_strict(state, products, support, beliefs, turn=10, top_k=10)
        self.assertEqual(plan.shown, ["p0", "p1", "p2"])

    def test_pre_override_phase_is_information_only(self):
        products = _products({"a": (["cotton"], ["Imported"]), "b": (["wool"], ["Imported"])})
        support = {pid: set() for pid in products}
        state = DialogState(session_id="s", profile={}, opening="override_open")
        plan = planner.plan_strict(state, products, support, [("a", 0.5), ("b", 0.5)], turn=1, top_k=10)
        self.assertTrue(plan.info_only)
        self.assertEqual(plan.shown, ["a", "b"])     # showing is free before the override

    def test_product_mode_shows_the_belief_order(self):
        config.WIDTH_MODE = "product"
        products = _products({f"p{i}": (["cotton"], ["Imported"]) for i in range(4)})
        support = {pid: {"cotton"} for pid in products}
        beliefs = [("p0", 0.4), ("p1", 0.3), ("p2", 0.2), ("p3", 0.1)]
        state = DialogState(session_id="s", profile={}, opening="buying")
        plan = planner.plan_strict(state, products, support, beliefs, turn=2, top_k=10)
        self.assertEqual(plan.shown, ["p0", "p1", "p2", "p3"])


if __name__ == "__main__":
    unittest.main()
