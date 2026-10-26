"""Second-tier verification: LLM fragments aligned to card values become set-valued events
that narrow the strict support set without ever pruning the true target."""

import unittest
from unittest import mock

from dejabuy.env import protocol
from dejabuy.state import DialogState
from dejabuy.understand import canonical, llm_extract, rules, verify

_CAT = "Shoes Loafers & Slip-Ons"
_PRODUCTS = {
    "A": {"card_hard": ["leather", "Zipper closure"], "card_soft": ["Rubber sole", "budget around $40.0"], "coarse_exact": _CAT},
    "B": {"card_hard": ["leather", "Buckle closure"], "card_soft": ["Imported", "budget around $55.0"], "coarse_exact": _CAT},
    "C": {"card_hard": ["75% Polyester, 20% Rayon, 5% Spandex", "color: blue"], "card_soft": ["Machine Wash"], "coarse_exact": _CAT},
    "D": {"card_hard": ["cotton"], "card_soft": ["Imported"], "coarse_exact": "Boots Mid-Calf"},
}


def _vocab(category):
    out = {}
    for info in _PRODUCTS.values():
        if info["coarse_exact"] != category:
            continue
        for value in (*info["card_hard"], *info["card_soft"]):
            out.setdefault(verify.normalize(value), set()).add(value)
    return out


class AlignmentTests(unittest.TestCase):
    def setUp(self):
        verify.set_card_vocab(_vocab, lambda: [_CAT, "Boots Mid-Calf"])
        verify.set_catalog_verifier(lambda phrase: True)
        self.addCleanup(verify.set_card_vocab, None, None)
        self.addCleanup(verify.set_catalog_verifier, None)

    def test_exact_value_wins(self):
        self.assertEqual(verify.align("leather", [_CAT]), ["leather"])

    def test_trimmed_fragment_expands_to_containing_values(self):
        self.assertEqual(verify.align("Zipper", [_CAT]), ["Zipper closure"])
        self.assertEqual(verify.align("closure", [_CAT]), ["Buckle closure", "Zipper closure"])

    def test_budget_figure_matches_by_number(self):
        self.assertEqual(verify.align("40 dollars", [_CAT]), ["budget around $40.0"])
        self.assertEqual(verify.align("$55", [_CAT]), ["budget around $55.0"])
        self.assertEqual(verify.align("30 dollars", [_CAT]), [])

    def test_reordered_spec_list_aligns(self):
        self.assertEqual(verify.align("5% Spandex, 20% Rayon, 75% Polyester", [_CAT]),
                         ["75% Polyester, 20% Rayon, 5% Spandex"])

    def test_fragment_wrapping_a_value_aligns_to_the_longest_contained_value(self):
        self.assertEqual(verify.align("a Pull-On closure", ["Novelty"]), [])          # no vocab for that category
        self.assertEqual(verify.align("a Zipper closure please", [_CAT]), ["Zipper closure"])
        self.assertEqual(verify.align("something made of leather", [_CAT]), ["leather"])

    def test_merge_split_fragments(self):
        message = "What matters for that is 75% Polyester, 20% Rayon, 5% Spandex; Machine Wash."
        merged = verify.merge_fragments(["75% Polyester", "20% Rayon", "5% Spandex", "Machine Wash"], message, [_CAT])
        self.assertEqual(merged, ["75% Polyester, 20% Rayon, 5% Spandex", "Machine Wash"])
        self.assertEqual(verify.merge_fragments(["leather", "Imported"], "leather; Imported", [_CAT]),
                         ["leather", "Imported"])

    def test_find_category_prefers_the_longest_verbatim_name(self):
        self.assertEqual(verify.find_category("Just browsing for now — I'm after Shoes Loafers & Slip-Ons."), [_CAT])
        self.assertEqual(verify.find_category("For Boots Mid-Calf, cotton is a key requirement I'm after."), ["Boots Mid-Calf"])
        self.assertEqual(verify.find_category("Just browsing for now."), [])

    def test_unaligned_fragment_is_empty(self):
        self.assertEqual(verify.align("waterproof", [_CAT]), [])
        self.assertEqual(verify.align("leather", []), [])

    def test_category_resolution(self):
        self.assertEqual(verify.resolve_category(_CAT), [_CAT])
        self.assertEqual(verify.resolve_category("loafers & slip-ons"), [_CAT])
        self.assertEqual(verify.resolve_category("Shoes"), [_CAT])
        self.assertEqual(verify.resolve_category("Hats"), [])


class SetValuedReplayTests(unittest.TestCase):
    def test_disclose_options_keep_every_candidate_whose_reply_fits(self):
        events = [{"type": "open", "scenario": "browsing", "category": _CAT, "categories": [_CAT]},
                  {"type": "disclose", "ask": "other", "options": [["Zipper closure", "Buckle closure"]]}]
        support = protocol.support(_PRODUCTS, list(_PRODUCTS), events)
        # A and B both answer "leather; <closure>" to `other`; C answers with its spec; D is off-category
        self.assertEqual(set(support), {"A", "B"})
        self.assertEqual(support["A"], {"leather", "Zipper closure"})

    def test_disclose_options_any_order_and_extra_values_allowed(self):
        info = _PRODUCTS["A"]
        base = {"type": "open", "scenario": "browsing", "category": _CAT, "categories": [_CAT]}
        ok, _ = protocol.replay(info, [base, {"type": "disclose", "ask": "other",
                                             "options": [["Zipper closure"], ["leather"]]}])
        self.assertTrue(ok, "two fragments, reversed order")
        ok, _ = protocol.replay(info, [base, {"type": "disclose", "ask": "other", "options": [["leather"]]}])
        self.assertTrue(ok, "one fragment extracted out of a two-value reply")
        ok, _ = protocol.replay(info, [base, {"type": "disclose", "ask": "other",
                                             "options": [["leather"], ["Buckle closure"]]}])
        self.assertFalse(ok)

    def test_option_matching_tolerates_split_and_missing_fragments(self):
        base = {"type": "open", "scenario": "browsing", "category": _CAT, "categories": [_CAT]}
        info = _PRODUCTS["C"]     # replies "75% Polyester, 20% Rayon, 5% Spandex; color: blue" to `other`
        ok, _ = protocol.replay(info, [base, {"type": "disclose", "ask": "other", "options": [
            ["75% Polyester, 20% Rayon, 5% Spandex"], ["Machine Wash"], ["color: blue"]]}])
        self.assertTrue(ok, "three option sets, two predicted values: two matched is enough")
        ok, _ = protocol.replay(info, [base, {"type": "disclose", "ask": "other", "options": [
            ["Machine Wash"], ["Imported"], ["color: blue"]]}])
        self.assertFalse(ok, "only one of the two predicted values is explainable")

    def test_open_options_and_any_scenario(self):
        event = {"type": "open", "scenario": "any", "category": _CAT, "categories": [_CAT], "options": ["leather", "Imported"]}
        ok, disclosed = protocol.replay(_PRODUCTS["A"], [event])
        self.assertTrue(ok)
        self.assertEqual(disclosed, {"leather"})           # matched as a requirement (hard[0])
        ok, disclosed = protocol.replay(_PRODUCTS["B"], [{**event, "options": ["budget around $55.0"]}])
        self.assertTrue(ok)
        self.assertEqual(disclosed, set())                 # matched as the old preference (soft[-1])
        self.assertFalse(protocol.replay(_PRODUCTS["C"], [event])[0])

    def test_open_categories_union(self):
        event = {"type": "open", "scenario": "browsing", "category": "Shoes", "categories": [_CAT, "Boots Mid-Calf"]}
        self.assertEqual(set(protocol.support(_PRODUCTS, list(_PRODUCTS), [event])), {"A", "B", "C", "D"})

    def test_override_options(self):
        events = [{"type": "open", "scenario": "override_open", "category": _CAT, "categories": [_CAT],
                   "value": "budget around $55.0"},
                  {"type": "override", "options": ["leather", "cotton"]}]
        ok, disclosed = protocol.replay(_PRODUCTS["B"], events)
        self.assertTrue(ok)
        self.assertEqual(disclosed, {"leather"})


class EventInferenceTests(unittest.TestCase):
    """End to end through rules.update_state with a mocked LLM reply."""

    def setUp(self):
        verify.set_card_vocab(_vocab, lambda: [_CAT, "Boots Mid-Calf"])
        verify.set_catalog_verifier(lambda phrase: phrase in {"leather", "zipper closure", "imported", "loafers slip ons"})
        self.addCleanup(verify.set_card_vocab, None, None)
        self.addCleanup(verify.set_catalog_verifier, None)

    def _arm(self, replies):
        def fake(message, turn=1, asked=None):
            reply = replies.get(message)
            return None if reply is None else llm_extract.verify_reply(reply, message)
        patcher = mock.patch.object(rules.llm_extract, "extract", fake)
        patcher.start()
        self.addCleanup(patcher.stop)

    def test_llm_turn_becomes_set_valued_events(self):
        opening = f"I'm after {_CAT} and honestly the dealbreaker for me is leather more than anything else."
        reply = "On that front I'd say it fastens with a Zipper pretty much covers what counts for me."
        self._arm({
            opening: {"act": "open", "opening": "requirement", "category": _CAT, "constraints": ["leather"]},
            reply: {"act": "disclose", "constraints": ["Zipper"]},
        })
        state = DialogState(session_id="t", profile={})
        rules.update_state(state, opening, 1)
        state.last_ask = "other"
        rules.update_state(state, reply, 2)
        self.assertEqual([e["type"] for e in state.events], ["open", "disclose"])
        self.assertEqual(state.events[0]["scenario"], "buying")
        self.assertEqual(state.events[0]["options"], ["leather"])
        self.assertEqual(state.opening, "buying")
        self.assertEqual(state.events[1]["options"], [["Zipper closure"]])
        self.assertTrue(protocol.recognised(state.events))
        self.assertEqual(set(protocol.support(_PRODUCTS, list(_PRODUCTS), state.events)), {"A"})

    def test_unaligned_llm_fragment_stays_evidence_only(self):
        reply = "On that front I'd say it should be waterproof."
        self._arm({reply: {"act": "disclose", "constraints": ["waterproof"]}})
        state = DialogState(session_id="t", profile={})
        rules.update_state(state, f"I'm looking for {_CAT}, but I'm still exploring.", 1)
        state.last_ask = "other"
        rules.update_state(state, reply, 2)
        self.assertEqual(state.events[1], {"type": "unknown"})
        self.assertEqual([s.value for s in state.slots], ["waterproof"])
        self.assertFalse(protocol.recognised(state.events))

    def test_llm_labelled_exhausted_and_boundary(self):
        exhausted = "Nope, that's everything I can think of for material."
        boundary = "Honestly whatever you think is best on size."
        self._arm({exhausted: {"act": "exhausted", "attribute": "material"},
                   boundary: {"act": "boundary", "attribute": "size"}})
        state = DialogState(session_id="t", profile={})
        rules.update_state(state, f"I'm looking for {_CAT}, but I'm still exploring.", 1)
        state.last_ask = "material"
        rules.update_state(state, exhausted, 2)
        state.last_ask = "size"
        rules.update_state(state, boundary, 3)
        self.assertEqual(state.events[1], {"type": "exhausted", "ask": "material"})
        self.assertEqual(state.events[2], {"type": "boundary", "ask": "size"})
        self.assertIn("material", state.exhausted)

    def test_rule_path_partial_verification_hands_over_to_llm(self):
        msg = "On that front, here's what counts for me: leather, and something made of rubber"
        self._arm({msg: {"act": "disclose", "constraints": ["leather", "rubber"]}})
        state = DialogState(session_id="t", profile={})
        rules.update_state(state, f"I'm looking for {_CAT}, but I'm still exploring.", 1)
        state.last_ask = "other"
        rules.update_state(state, msg, 2)
        self.assertEqual(state.events[1]["options"], [["leather"], ["Rubber sole"]])

    def test_whole_sentence_slot_never_becomes_an_event_value(self):
        """No colon, no LLM: the rules fall back to splitting the sentence into a slot. That slot
        must not be taken as the opening value (it would align to nothing — or worse, to a card
        value mentioned in passing); the opening is still a plain browsing event."""
        msg = f"So I'm shopping around for {_CAT} at the moment — haven't settled on anything yet."
        state = DialogState(session_id="t", profile={})
        rules.update_state(state, msg, 1)
        self.assertEqual(state.events[0]["type"], "open")
        self.assertEqual(state.events[0]["scenario"], "browsing")
        self.assertEqual(state.events[0]["categories"], [_CAT])

    def test_rule_path_partial_colon_payload_is_set_valued(self):
        """'What matters for that is leather; color: black' — the colon rule only catches 'black';
        the event must not claim the reply was exactly 'black'."""
        verify.set_catalog_verifier(lambda phrase: True)
        state = DialogState(session_id="t", profile={})
        rules.update_state(state, f"I'm looking for {_CAT}, but I'm still exploring.", 1)
        state.last_ask = "other"
        rules.update_state(state, "What matters for that is leather; Zipper closure", 2)
        event = state.events[1]
        self.assertEqual(event["type"], "disclose")
        self.assertIn("options", event)
        self.assertEqual(set(protocol.support(_PRODUCTS, list(_PRODUCTS), state.events)), {"A"})


if __name__ == "__main__":
    unittest.main()
