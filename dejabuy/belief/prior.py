"""Purchase priors used by the evidence scorer.

Targets in this task are drawn from real purchase records, so they are systematically the
popular, fully-listed products: review volume and the presence of a price are strong priors
that have nothing to do with the words in the conversation. Weights live in `config`.
"""

from __future__ import annotations

import math

from dejabuy import config


def popularity(candidate: dict) -> float:
    """log-compressed review volume in [0, ~1]."""
    return math.log10(1 + candidate.get("rating_number", 0)) / 5.0


def has_price(candidate: dict) -> float:
    return 1.0 if candidate.get("price") is not None else 0.0


def prior_scale(evidence_count: int, turn_index: int) -> float:
    """How much the priors should weigh given how much evidence the conversation holds."""
    scale = 1.0 / (1.0 + config.PRIOR_DECAY * evidence_count) if config.PRIOR_DECAY else 1.0
    if config.EARLY_PRIOR_BOOST != 1.0 and turn_index <= config.EARLY_TURNS:
        scale *= config.EARLY_PRIOR_BOOST
    return scale
