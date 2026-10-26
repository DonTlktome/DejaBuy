"""Belief layer orchestration: candidates in, ordered slate (and beliefs) out.

`rank()` keeps the historical signature used by the agent and the benches. `beliefs()` turns
the same evidence scores into a probability distribution over candidates, which the decision
layer and the trace consume.
"""

from __future__ import annotations

import math

from dejabuy import config
from dejabuy.belief import evidence, rerank_llm
from dejabuy.decide.width import slate_width
from dejabuy.state import DialogState


def order(state: DialogState, candidates: list[dict]) -> list[dict]:
    """Full evidence ordering of the candidate pool (no width policy applied)."""
    if not candidates:
        return []
    score = evidence.make_scorer(state, candidates)
    return sorted(candidates, key=score, reverse=True)


def beliefs(state: DialogState, candidates: list[dict], temperature: float = 1.0) -> list[tuple[dict, float]]:
    """Softmax of the evidence scores: P(target = candidate | conversation)."""
    if not candidates:
        return []
    score = evidence.make_scorer(state, candidates)
    scored = [(c, score(c)) for c in candidates]
    top = max(s for _, s in scored)
    weights = [math.exp((s - top) / temperature) for _, s in scored]
    total = sum(weights) or 1.0
    return sorted(((c, w / total) for (c, _), w in zip(scored, weights)), key=lambda x: x[1], reverse=True)


def strict_beliefs(state: DialogState, members: list[dict], temperature: float | None = None) -> list[tuple[str, float]]:
    """Belief over the strict support set: softmax of the evidence score at `temperature`.

    Every member already explains the whole transcript, so what separates them is the
    purchase prior (review volume, listed price), the mirror-card consistency and the
    BM25 rank for those that the retrieval pool also returned."""
    if not members:
        return []
    tau = config.BELIEF_TEMPERATURE if temperature is None else temperature
    # The BM25 rank is a property of the retrieval pool, not of the support set: inside the
    # support set it is noise (and its pool normalisation breaks when few members carry a
    # rank), so the belief is built from the evidence score with that component neutralised.
    neutral = [{**c, "bm25_rank": config.CANDIDATE_POOL} for c in members]
    score = evidence.make_scorer(state, neutral)
    scored = [(c["parent_asin"], score(c)) for c in neutral]
    top = max(s for _, s in scored)
    weights = [(pid, math.exp((s - top) / tau)) for pid, s in scored]
    total = sum(w for _, w in weights) or 1.0
    return sorted(((pid, w / total) for pid, w in weights), key=lambda x: x[1], reverse=True)


def rank(state: DialogState, candidates: list[dict], k: int = 10) -> list[dict]:
    """Ordered slate for this turn: evidence ordering, width policy, optional LLM re-rank."""
    if not candidates:
        return []
    ordered = order(state, candidates)
    k = slate_width(state, k)
    if config.USE_LLM:
        ordered = rerank_llm.llm_rerank(state, ordered[: config.LLM_RERANK_POOL]) + ordered[config.LLM_RERANK_POOL:]
    return ordered[:k]
