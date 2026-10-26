"""Memory store: per-user episodes with timestamps, time decay and deletion.

Write path: `record_purchase` / `record_session` append episodes (the REST service calls them
when a conversation converts or ends). Read path: `profile(user_id, as_of)` derives the semantic
profile from the episodes strictly before `as_of`; `habits(user_id, as_of)` derives the
procedural habits. `forget(user_id)` drops everything about a user.

Persistence is two JSONL files: users (user_id → purchase events) and items (product → compact
attributes). The benchmark ships them under data/memory_bench/; an empty store is valid and
makes every user a cold-start user.
"""

from __future__ import annotations

import gzip
import json
import sys
import threading
from collections import OrderedDict
from pathlib import Path

from dejabuy.memory.episodic import Purchase, SessionEpisode
from dejabuy.memory.procedural import Habits
from dejabuy.memory.semantic import Profile


_RUNTIME_FIELDS = ("parent_asin", "store", "coarse", "department", "price", "material", "color")


def _compact(rec: dict, keep_card: bool) -> dict:
    out = {k: (sys.intern(v) if isinstance(v, str) else v) for k, v in rec.items() if k in _RUNTIME_FIELDS}
    if keep_card and rec.get("card"):
        out["card"] = rec["card"]
    return out


def _open(path: Path):
    return gzip.open(path, "rt", encoding="utf-8") if path.suffix == ".gz" else path.open(encoding="utf-8")


class MemoryStore:
    def __init__(self, half_life_days: float = 365.0, cache_size: int = 4096) -> None:
        self.half_life_days = half_life_days
        self.events: dict[str, list[tuple[str, float, int]]] = {}   # user_id -> [(asin, rating, ts)]
        self.items: dict[str, dict] = {}
        self.sessions: dict[str, list[SessionEpisode]] = {}
        self._habits: dict[str, Habits] = {}
        self._cache: OrderedDict = OrderedDict()
        self._cache_size = cache_size
        self._lock = threading.Lock()

    # ------------------------------------------------------------------ persistence
    @classmethod
    def load(cls, directory: str | Path, half_life_days: float = 365.0, keep_card: bool = False) -> "MemoryStore":
        """`keep_card=False` drops the intent-card strings of history items (the bulk of the file):
        the learned prior gives them zero weight, and without them the store is a third the size."""
        store = cls(half_life_days=half_life_days)
        directory = Path(directory)
        users = next((p for p in (directory / "users.jsonl.gz", directory / "users.jsonl") if p.exists()), None)
        items = next((p for p in (directory / "items.jsonl.gz", directory / "items.jsonl") if p.exists()), None)
        if items:
            with _open(items) as handle:
                for line in handle:
                    rec = json.loads(line)
                    store.items[rec["parent_asin"]] = _compact(rec, keep_card)
        if users:
            with _open(users) as handle:
                for line in handle:
                    rec = json.loads(line)
                    store.events[rec["user_id"]] = [(e[0], float(e[1] or 0), int(e[2])) for e in rec["events"]]
        return store

    # ------------------------------------------------------------------ write path
    def record_purchase(self, user_id: str, asin: str, ts: int, rating: float = 0.0, item: dict | None = None) -> None:
        with self._lock:
            self.events.setdefault(user_id, []).append((asin, float(rating), int(ts)))
            if item is not None:
                self.items.setdefault(asin, item)
            self._invalidate(user_id)

    def record_session(self, user_id: str, episode: SessionEpisode) -> None:
        with self._lock:
            self.sessions.setdefault(user_id, []).append(episode)
            self._invalidate(user_id)

    def record_answer(self, user_id: str, attribute: str, answered: bool) -> None:
        with self._lock:
            self._habits.setdefault(user_id, Habits()).record(attribute, answered)

    def habits(self, user_id: str) -> Habits:
        return self._habits.get(user_id) or Habits()

    def forget(self, user_id: str) -> None:
        with self._lock:
            self.events.pop(user_id, None)
            self.sessions.pop(user_id, None)
            self._habits.pop(user_id, None)
            self._invalidate(user_id)

    def _invalidate(self, user_id: str) -> None:
        for key in [k for k in self._cache if k[0] == user_id]:
            self._cache.pop(key, None)

    # ------------------------------------------------------------------ read path
    def purchases(self, user_id: str, as_of: int | None = None) -> list[Purchase]:
        out = []
        for asin, rating, ts in self.events.get(user_id, ()):
            if as_of is not None and ts >= as_of:
                continue
            item = self.items.get(asin)
            if item is not None:
                out.append(Purchase(asin, rating, ts, item))
        return out

    def session_episodes(self, user_id: str, as_of: int | None = None) -> list[SessionEpisode]:
        return [e for e in self.sessions.get(user_id, ()) if as_of is None or e.ts < as_of]

    def profile(self, user_id: str, as_of: int | None = None) -> Profile | None:
        key = (user_id, as_of)
        with self._lock:
            if key in self._cache:
                self._cache.move_to_end(key)
                return self._cache[key]
        purchases = self.purchases(user_id, as_of)
        profile = Profile(purchases, as_of if as_of is not None else max((p.ts for p in purchases), default=0) + 1,
                          self.half_life_days) if purchases else None
        with self._lock:
            self._cache[key] = profile
            if len(self._cache) > self._cache_size:
                self._cache.popitem(last=False)
        return profile

    def __len__(self) -> int:
        return len(self.events)
