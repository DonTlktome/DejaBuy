"""Utility of a session outcome, as the benchmark scores it.

score = 0.5 * hit + 0.3 * (1 / rank) + 0.2 * clip((11 - turns) / 10, 0, 1)

The planner compares actions by the expected value of this function over the remaining
turns. Kept in one place so the product-mode and benchmark-mode planners share it.
"""

from __future__ import annotations

MAX_TURNS = 10
MISS_TURN_VALUE = 11


def efficiency(turns_to_convert: int | None) -> float:
    turn = MISS_TURN_VALUE if turns_to_convert is None else turns_to_convert
    return min(max((MISS_TURN_VALUE - turn) / 10.0, 0.0), 1.0)


def session_score(hit: bool, rank: int | None, turns_to_convert: int | None) -> float:
    rr = 1.0 / rank if (hit and rank) else 0.0
    return 0.5 * float(hit) + 0.3 * rr + 0.2 * efficiency(turns_to_convert if hit else None)
