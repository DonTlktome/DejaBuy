"""Evaluator entry point (the organiser's harness imports `starter.agent.Agent`).

AGENT_IMPL=baseline selects the reference weak-BM25 baseline for the 0.107 sanity check.
"""

from __future__ import annotations

import os

if os.environ.get("AGENT_IMPL", "dejabuy") == "baseline":
    from dejabuy.baselines.weak_bm25 import Agent  # noqa: F401
else:
    from dejabuy.agent import DejaBuyAgent as Agent  # noqa: F401
