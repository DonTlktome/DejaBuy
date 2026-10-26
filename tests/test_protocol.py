"""Strict environment model: card replay must mirror the reference simulator exactly."""

import unittest

from dejabuy.env import protocol


def _info(hard, soft, cat="Shoes Loafers & Slip-Ons"):
    return {"card_hard": list(hard), "card_soft": list(soft), "coarse_exact": cat}


class ReplayTests(unittest.TestCase):
    def test_buying_opening_requires_first_hard_constraint(self):
        info = _info(["leather", "100% Leather"], ["Rubber sole", "Imported"])
        ok, disclosed = protocol.replay(info, [{"type": "open", "scenario": "buying",
                                                "category": "Shoes Loafers & Slip-Ons", "value": "leather"}])
        self.assertTrue(ok)
        self.assertEqual(disclosed, {"leather"})
        ok, _ = protocol.replay(info, [{"type": "open", "scenario": "buying",
                                        "category": "Shoes Loafers & Slip-Ons", "value": "Imported"}])
        self.assertFalse(ok)

    def test_category_mismatch_is_inconsistent(self):
        info = _info(["leather"], ["Imported"], cat="Boots Mid-Calf")
        ok, _ = protocol.replay(info, [{"type": "open", "scenario": "browsing", "category": "Shoes Loafers & Slip-Ons"}])
        self.assertFalse(ok)

    def test_other_discloses_two_undisclosed_values_in_card_order(self):
        info = _info(["leather", "100% Leather"], ["Rubber sole", "Imported"])
        events = [{"type": "open", "scenario": "buying", "category": "Shoes Loafers & Slip-Ons", "value": "leather"},
                  {"type": "disclose", "ask": "other", "body": "100% Leather; Rubber sole"}]
        ok, disclosed = protocol.replay(info, events)
        self.assertTrue(ok)
        self.assertEqual(disclosed, {"leather", "100% Leather", "Rubber sole"})
        self.assertEqual(protocol.reply_key(info, disclosed, "other"), "Imported")
        self.assertEqual(protocol.reply_key(info, disclosed | {"Imported"}, "other"), protocol.EXHAUSTED)

    def test_specific_attribute_filters_by_classifier(self):
        info = _info(["cotton", "color: grey"], ["Pull On closure", "Imported"], cat="Novelty Women")
        events = [{"type": "open", "scenario": "browsing", "category": "Novelty Women"},
                  {"type": "disclose", "ask": "color", "body": "color: grey"}]
        ok, disclosed = protocol.replay(info, events)
        self.assertTrue(ok)
        self.assertEqual(protocol.predicted_reply(info, disclosed, "material"), ["cotton"])
        self.assertEqual(protocol.predicted_reply(info, disclosed, "budget"), [])

    def test_exhausted_reply_must_match(self):
        info = _info(["cotton"], ["Imported"], cat="Novelty Women")
        events = [{"type": "open", "scenario": "browsing", "category": "Novelty Women"},
                  {"type": "exhausted", "ask": "budget"}]
        self.assertTrue(protocol.replay(info, events)[0])
        events[1] = {"type": "exhausted", "ask": "material"}
        self.assertFalse(protocol.replay(info, events)[0])

    def test_override_session(self):
        info = _info(["leather", "100% Leather"], ["Rubber sole", "Imported"])
        events = [{"type": "open", "scenario": "override_open", "category": "Shoes Loafers & Slip-Ons", "value": "Imported"},
                  {"type": "disclose", "ask": "other", "body": "leather; 100% Leather"},
                  {"type": "override", "value": "leather"}]
        ok, disclosed = protocol.replay(info, events)
        self.assertTrue(ok)
        self.assertIn("leather", disclosed)
        # the old (soft[-1]) value is never marked disclosed by the simulator, so `other` can re-surface it
        self.assertEqual(protocol.predicted_reply(info, disclosed, "other"), ["Rubber sole", "Imported"])
        bad = [{"type": "open", "scenario": "override_open", "category": "Shoes Loafers & Slip-Ons", "value": "Rubber sole"}]
        self.assertFalse(protocol.replay(info, bad)[0])

    def test_unknown_event_breaks_recognition(self):
        events = [{"type": "open", "scenario": "browsing", "category": "X"}, {"type": "unknown"}]
        self.assertFalse(protocol.recognised(events))
        self.assertTrue(protocol.recognised(events[:1]))


if __name__ == "__main__":
    unittest.main()
