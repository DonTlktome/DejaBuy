"""Cross-session memory: time cut-off, gating, write / forget, and zero effect on the official setting."""

import unittest

from dejabuy.memory.semantic import FEATURES, Profile
from dejabuy.memory.episodic import Purchase
from dejabuy.memory.store import MemoryStore

DAY = 86_400_000


def item(asin, store="", coarse="Shirts T-Shirts", dept="", price=None, material=None, color=None, card=()):
    return {"parent_asin": asin, "store": store, "coarse": coarse, "department": dept, "price": price,
            "material": material, "color": color, "card": list(card)}


class StoreTests(unittest.TestCase):
    def setUp(self):
        self.store = MemoryStore(half_life_days=365)
        self.store.record_purchase("u1", "A", 10 * DAY, 5, item("A", store="Hanes", price=10.0, dept="Mens"))
        self.store.record_purchase("u1", "B", 20 * DAY, 2, item("B", store="Nike", price=40.0))
        self.store.record_purchase("u1", "C", 30 * DAY, 5, item("C", store="Hanes", price=12.0))

    def test_as_of_is_a_strict_cut_off(self):
        self.assertEqual([p.asin for p in self.store.purchases("u1", 20 * DAY)], ["A"])
        self.assertEqual(len(self.store.purchases("u1", 31 * DAY)), 3)
        self.assertIsNone(self.store.profile("u1", 10 * DAY))       # nothing strictly before the first purchase
        self.assertIsNone(self.store.profile("nobody", 99 * DAY))

    def test_write_invalidates_cache_and_forget_deletes(self):
        before = self.store.profile("u1", 99 * DAY)
        self.store.record_purchase("u1", "D", 40 * DAY, 5, item("D", store="Puma"))
        after = self.store.profile("u1", 99 * DAY)
        self.assertEqual(after.purchases, before.purchases + 1)
        self.store.forget("u1")
        self.assertIsNone(self.store.profile("u1", 99 * DAY))


class ProfileTests(unittest.TestCase):
    def profile(self):
        purchases = [Purchase("A", 5, 10 * DAY, item("A", store="Hanes", price=10.0, dept="Mens", material="cotton")),
                     Purchase("B", 1, 20 * DAY, item("B", store="Nike", price=40.0)),
                     Purchase("C", 5, 30 * DAY, item("C", store="Hanes", price=12.0, dept="Mens"))]
        return Profile(purchases, as_of=31 * DAY, half_life_days=365)

    def test_features_are_zero_without_signal(self):
        f = dict(zip(FEATURES, self.profile().features(
            {"parent_asin": "Z", "store": "Levi's", "coarse_exact": "Jeans", "price": None})))
        self.assertEqual(f["store_share"], 0.0)
        self.assertEqual(f["store_seen"], 0.0)
        self.assertEqual(f["price_fit"], 0.0)
        self.assertEqual(f["loyal_store"], 0.0)

    def test_store_affinity_and_loyalty(self):
        prof = self.profile()
        self.assertAlmostEqual(prof.loyalty, 0.5)              # C repeats Hanes; B does not
        f = dict(zip(FEATURES, prof.features({"parent_asin": "Z", "store": "HANES", "price": 11.0,
                                              "department": "mens", "material": "cotton"})))
        self.assertEqual(f["store_seen"], 1.0)
        self.assertGreater(f["store_share"], 0.4)          # 2 of 3 purchases, smoothed by +1
        self.assertGreater(f["price_fit"], 0.9)
        self.assertEqual(f["dept_share"] > 0, True)
        self.assertAlmostEqual(f["loyal_store"], 0.5)

    def test_recency_decay(self):
        old = Profile([Purchase("A", 5, 0, item("A", store="Hanes"))], as_of=3 * 365 * DAY, half_life_days=365)
        new = Profile([Purchase("A", 5, 0, item("A", store="Hanes"))], as_of=1 * DAY, half_life_days=365)
        self.assertLess(old.n, new.n)

    def test_prior_is_a_weighted_sum(self):
        prof = self.profile()
        c = {"parent_asin": "Z", "store": "Hanes"}
        weights = {name: 0.0 for name in FEATURES}
        weights["store_seen"] = 2.0
        self.assertAlmostEqual(prof.prior(c, weights), 2.0)


if __name__ == "__main__":
    unittest.main()
