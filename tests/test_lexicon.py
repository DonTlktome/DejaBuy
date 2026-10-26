from __future__ import annotations

import unittest

from dejabuy.memory.profile_lexicon import TAG_KEYWORDS, profile_soft_terms


class LexiconTest(unittest.TestCase):
    def test_known_tag_returns_keywords(self) -> None:
        terms = profile_soft_terms({"preference_tags": ["durability"]})
        self.assertIn("durable", terms)

    def test_unknown_tag_is_silently_ignored(self) -> None:
        terms = profile_soft_terms({"preference_tags": ["durability", "made_up_tag"]})
        self.assertIn("durable", terms)

    def test_missing_or_empty_profile_returns_empty(self) -> None:
        self.assertEqual(profile_soft_terms(None), [])
        self.assertEqual(profile_soft_terms({}), [])
        self.assertEqual(profile_soft_terms({"preference_tags": []}), [])

    def test_general_shopping_tag_has_no_entry(self) -> None:
        # "general shopping" 是兜底空槽，公开集里没有具体语义，不应在词表里
        self.assertNotIn("general shopping", TAG_KEYWORDS)

    def test_terms_are_deduplicated_across_tags(self) -> None:
        terms = profile_soft_terms({"preference_tags": ["comfort", "comfort", "fit"]})
        self.assertEqual(len(terms), len(set(terms)))

    def test_limit_caps_output_length(self) -> None:
        all_tags = list(TAG_KEYWORDS.keys())
        terms = profile_soft_terms({"preference_tags": all_tags}, limit=3)
        self.assertEqual(len(terms), 3)


if __name__ == "__main__":
    unittest.main()
