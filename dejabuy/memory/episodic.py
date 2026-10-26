"""Episodic memory: what this user did, one record per event, in time order.

An episode is a purchase (product, rating, timestamp, the product's attributes) or a finished
conversation with the agent (what was asked, what the user refused, which slate converted).
Episodes are never rewritten; the semantic profile and the procedural habits are derived from
them as of a point in time, so any past state of a user's memory can be reconstructed exactly
(that is what the time-ordered benchmark relies on: memory as of the moment before the target
purchase, nothing later)."""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class Purchase:
    asin: str
    rating: float
    ts: int                     # milliseconds since epoch (Amazon Reviews 2023 convention)
    item: dict                  # compact attributes (memory/items.py)


@dataclass
class SessionEpisode:
    ts: int
    category: str
    asked: list[str] = field(default_factory=list)          # attributes the agent asked, in order
    refused: list[str] = field(default_factory=list)        # attributes the user declined (boundary / exhausted)
    disclosed: list[str] = field(default_factory=list)      # constraint values the user gave
    converted: str | None = None                            # product bought at the end, if any
    turns: int = 0
