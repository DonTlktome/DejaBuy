"""Semantic memory: a user's stable preferences distilled from their purchase episodes.

The profile is a set of time-decayed attribute distributions (store, department, material,
colour, intent-card values, category) plus a price band. It enters the belief layer only as a
*prior* over candidates — `prior(candidate)` is a learned linear score on per-candidate affinity
features (scripts/train_memory.py) — and is added to the conversation evidence, so evidence
always overrides it: a returning customer shopping in a new style, or for someone else, is
simply out-voted by what they say. Nothing is ever skipped because of memory.

Gating is structural: every feature is zero when the history says nothing about that candidate
(no purchase from that store, no price on record, …), so a user without a usable signal gets
exactly the cold-start ranking. The permutation analysis behind this (bench/memory) found the
signal concentrated in brand-loyal returning customers; features are built so that the prior
is sharp there and silent elsewhere.
"""

from __future__ import annotations

import json
import math
from collections import defaultdict
from pathlib import Path

from dejabuy.memory.episodic import Purchase

FEATURES = (
    "store_share",      # decayed share of past purchases from the candidate's store
    "store_seen",       # bought from that store at least once
    "store_bad",        # share of that store's purchases rated ≤ 2 (a returning customer can be a lapsed one)
    "dept_share",       # share of department-labelled purchases in the candidate's department
    "price_fit",        # closeness of the candidate's price to the user's median paid price (log scale)
    "material_share",
    "color_share",
    "card_share",       # how many of the candidate's intent-card values the user's past products carried
    "category_share",   # share of past purchases in the candidate's coarse category
    "same_item",        # the user bought this very product before
    "loyal_store",      # store_seen × the user's own repeat-store rate: brand loyalty is a habit some users have
)

DEFAULT_WEIGHTS = {
    "store_share": 2.0, "store_seen": 1.0, "store_bad": -1.0, "dept_share": 0.3, "price_fit": 0.3,
    "material_share": 0.2, "color_share": 0.2, "card_share": 0.2, "category_share": 0.0, "same_item": 1.0, "loyal_store": 1.0,
}


def _lower(value) -> str:
    return str(value or "").strip().lower()


class Profile:
    """Decayed attribute distributions of one user as of one moment."""

    def __init__(self, purchases: list[Purchase], as_of: int, half_life_days: float) -> None:
        self.n = 0.0
        self.store: dict[str, float] = defaultdict(float)
        self.store_bad: dict[str, float] = defaultdict(float)
        self.dept: dict[str, float] = defaultdict(float)
        self.material: dict[str, float] = defaultdict(float)
        self.color: dict[str, float] = defaultdict(float)
        self.card: dict[str, float] = defaultdict(float)
        self.category: dict[str, float] = defaultdict(float)
        self.items: set[str] = set()
        prices: list[tuple[float, float]] = []
        dept_total = 0.0
        for p in purchases:
            age_days = max(0.0, (as_of - p.ts) / 86_400_000)
            w = 0.5 ** (age_days / half_life_days) if half_life_days > 0 else 1.0
            item = p.item
            self.n += w
            self.items.add(p.asin)
            store = _lower(item.get("store"))
            if store:
                self.store[store] += w
                if p.rating and p.rating <= 2:
                    self.store_bad[store] += w
            dept = _lower(item.get("department"))
            if dept:
                self.dept[dept] += w
                dept_total += w
            if item.get("material"):
                self.material[item["material"]] += w
            if item.get("color"):
                self.color[item["color"]] += w
            for value in set(item.get("card") or ()):
                self.card[value] += w
            if item.get("coarse"):
                self.category[item["coarse"]] += w
            if item.get("price"):
                prices.append((math.log(float(item["price"])), w))
        self.dept_total = dept_total
        # Repeat-store rate: share of purchases (after the first) from a store bought from before.
        seen_stores: set[str] = set()
        repeats = 0
        for p in sorted(purchases, key=lambda p: p.ts):
            store = _lower(p.item.get("store"))
            if store and store in seen_stores:
                repeats += 1
            if store:
                seen_stores.add(store)
        self.loyalty = repeats / (len(purchases) - 1) if len(purchases) > 1 else 0.0
        self.log_price = _weighted_median(prices) if prices else None
        self.purchases = len(purchases)

    # ------------------------------------------------------------------ features
    def features(self, c: dict) -> list[float]:
        n1 = self.n + 1.0
        store = _lower(c.get("store"))
        s_w = self.store.get(store, 0.0) if store else 0.0
        dept = _lower(c.get("department"))
        price = c.get("price")
        card = list(c.get("card_hard") or ()) + list(c.get("card_soft") or ())
        if not card and c.get("card"):
            card = list(c["card"])
        card_hits = sum(self.card.get(v, 0.0) for v in set(card)) / n1 if card else 0.0
        return [
            s_w / n1,
            1.0 if s_w > 0 else 0.0,
            (self.store_bad.get(store, 0.0) / n1) if store else 0.0,
            (self.dept.get(dept, 0.0) / (self.dept_total + 1.0)) if dept and self.dept_total else 0.0,
            math.exp(-abs(math.log(price) - self.log_price)) if price and self.log_price is not None else 0.0,
            (self.material.get(c.get("material"), 0.0) / n1) if c.get("material") else 0.0,
            (self.color.get(c.get("color"), 0.0) / n1) if c.get("color") else 0.0,
            min(card_hits, 1.0),
            self.category.get(c.get("coarse_exact") or c.get("coarse") or "", 0.0) / n1,
            1.0 if c.get("parent_asin") in self.items else 0.0,
            self.loyalty if s_w > 0 else 0.0,
        ]

    def prior(self, c: dict, weights: dict[str, float]) -> float:
        return sum(weights.get(name, 0.0) * value for name, value in zip(FEATURES, self.features(c)))

    def summary(self, top: int = 3) -> dict:
        """Human-readable view for the trace and the demo's memory panel."""
        def best(table: dict[str, float]) -> list[str]:
            return [k for k, _ in sorted(table.items(), key=lambda kv: -kv[1])[:top]]
        return {
            "purchases": self.purchases,
            "stores": best(self.store),
            "departments": best(self.dept),
            "materials": best(self.material),
            "colors": best(self.color),
            "categories": best(self.category),
            "median_price": round(math.exp(self.log_price), 2) if self.log_price is not None else None,
            "repeat_store_rate": round(self.loyalty, 2),
        }


def _weighted_median(values: list[tuple[float, float]]) -> float:
    values = sorted(values)
    total = sum(w for _, w in values)
    acc = 0.0
    for value, w in values:
        acc += w
        if acc >= total / 2:
            return value
    return values[-1][0]


def load_weights(path: str | Path | None) -> dict[str, float]:
    if path and Path(path).exists():
        try:
            data = json.loads(Path(path).read_text(encoding="utf-8"))
            return {k: float(v) for k, v in data.get("weights", data).items() if k in FEATURES}
        except (OSError, ValueError, json.JSONDecodeError):
            pass
    return dict(DEFAULT_WEIGHTS)
