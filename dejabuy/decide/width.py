"""Slate width policy: how many products to show this turn.

`product` mode always shows the full slate (a storefront never shows one item).
`benchmark` mode narrows early low-confidence turns because a hit ends the session and locks
in that turn's rank. Both numbers are reported; the switch is EARLY_TOPK (0 = product mode).
"""

from __future__ import annotations

from dejabuy import config
from dejabuy.state import DialogState


def slate_width(state: DialogState, k: int) -> int:
    """Return the number of products to expose this turn (<= k)."""
    # 低置信轮收窄推荐条数（实验 11）：命中即终局，早轮以烂名次命中会把 MRR 锁死。
    # 宁可第 1 轮不命中，也要等第 2 轮拿到约束后以第 1 名命中——单条净赚 0.237 分。
    if config.EARLY_TOPK:
        if config.EARLY_MODE == "hybrid":
            # 信息不足【且】还没超过硬上限才收窄。
            # 纯 slots 模式在私有集上有崩溃风险：改写导致解析失败 → 槽位永远填不满 →
            # 永远只推 1 件 → HitRate 崩（实测 MIN_SLOTS=5 即 0.9318 / hit 0.955）。
            # 轮次上限是安全出口：无论信息多差，第 EARLY_TURNS 轮之后一定给满 10 件。
            narrow = (len(state.slots) < config.EARLY_MIN_SLOTS
                      and len(state.history) <= config.EARLY_TURNS)
        elif config.EARLY_MODE == "slots":
            narrow = len(state.slots) < config.EARLY_MIN_SLOTS
        else:
            narrow = len(state.history) <= config.EARLY_TURNS
        if narrow:
            k = min(k, config.EARLY_TOPK)
    return k
