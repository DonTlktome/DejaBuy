"""Dense route (bge-small-en-v1.5, INT8 ONNX) on vs off, under the unmodified evaluator.

    python3 -m bench.dense_eval                         # L0 + rule paraphrases L2–L4, rules only
    python3 -m bench.dense_eval --levels L3 L4 --out golden/dense.json

The dense route only adds candidates to the retrieval pool (non-displacing union: BM25 + phrase
recall stay in place, dense-only hits are appended), so it can matter only when the agent ranks
from that pool (the loose regime) — the strict regime ranks the intent-card support set, which is
built from the whole category. Besides the official metrics this reports, per retrieve() call,
whether the target was in the pool at all ("pool recall") and how often only the dense tail had it.

The LLM extraction layer is switched off (LLM_PARSE=0) so the comparison is deterministic and
runs without a key; memory is off (public profiles carry no user id anyway).
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

os.environ.setdefault("LLM_PARSE", "0")
os.environ.setdefault("MEMORY", "0")
os.environ.setdefault("DEJABUY_TRACE", "0")

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from bench.paraphrase_rules import ParaphrasingAgent, load_jsonl  # noqa: E402
from dejabuy import config  # noqa: E402
from dejabuy.agent import DejaBuyAgent as Agent  # noqa: E402
from evaluator.local_evaluator import catalog_index, evaluate  # noqa: E402


class PoolProbe:
    """Wraps the agent: knows the current target (sessions run in sample order) and records,
    for every retrieve() call, whether the target is in the pool and whether only the dense
    tail contributed it."""

    def __init__(self, inner, samples: list[dict]) -> None:
        self.inner = inner
        self.samples = samples
        self.index = -1
        self.calls: list[dict] = []
        retriever = self._agent().retriever
        original = retriever.retrieve

        def retrieve(state, message, k=100):
            target = self.target
            dense = retriever.dense
            if dense is None:
                pool = original(state, message, k)
                base_hit = any(c["parent_asin"] == target for c in pool)
                self.calls.append({"i": self.index, "hit": base_hit, "dense_only": False})
                return pool
            retriever.dense = None
            try:
                base = original(state, message, k)
            finally:
                retriever.dense = dense
            pool = original(state, message, k)
            base_hit = any(c["parent_asin"] == target for c in base)
            hit = any(c["parent_asin"] == target for c in pool)
            self.calls.append({"i": self.index, "hit": hit, "dense_only": hit and not base_hit,
                               "added": len(pool) - len(base)})
            return pool

        retriever.retrieve = retrieve

    def _agent(self):
        return getattr(self.inner, "inner", self.inner)

    @property
    def target(self) -> str:
        return str(self.samples[self.index]["ground_truth"]["parent_asin"])

    def reset(self, session_id, profile):
        self.index += 1
        self.inner.reset(session_id, profile)

    def respond(self, session_id, message, turn, top_k):
        return self.inner.respond(session_id, message, turn, top_k)


def run(level: str, dense: bool, samples, index, catalog: str) -> dict:
    config.USE_DENSE = dense
    agent = Agent(catalog)
    if dense and agent.retriever.dense is None:
        raise SystemExit("dense route requested but not available (embeddings / onnx model missing)")
    probe = PoolProbe(ParaphrasingAgent(agent, level), samples)
    started = time.perf_counter()
    result = evaluate(probe, samples, *index)
    seconds = time.perf_counter() - started
    calls = probe.calls
    first = {}
    for call in calls:
        first.setdefault(call["i"], call)
    sessions_dense_only = len({c["i"] for c in calls if c["dense_only"]})
    added = [c["added"] for c in calls if "added" in c]
    return {
        "level": level,
        "dense": dense,
        "score": result["recommended_technical_score"],
        "hit": result["hit_rate_at_10"],
        "mrr": result["mrr"],
        "mttc": result["mttc"],
        "pool_recall_turn1": round(sum(c["hit"] for c in first.values()) / max(1, len(first)), 4),
        "pool_recall_all_calls": round(sum(c["hit"] for c in calls) / max(1, len(calls)), 4),
        "sessions_target_only_from_dense": sessions_dense_only,
        "mean_dense_added": round(sum(added) / len(added), 1) if added else 0,
        "per_session_rr": [s.get("reciprocal_rank", s.get("rr")) for s in result.get("sessions", [])],
        "seconds": round(seconds, 1),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--catalog", default="data/catalog.jsonl")
    parser.add_argument("--dataset", default="data/public_set.jsonl")
    parser.add_argument("--levels", nargs="+", default=["L0", "L2", "L3", "L4"])
    parser.add_argument("--out", default="")
    args = parser.parse_args()

    samples = load_jsonl(args.dataset)
    index = catalog_index(args.catalog)
    rows = []
    print("%-4s %-6s %8s %6s %6s %6s %9s %9s %9s %7s" % (
        "set", "dense", "score", "hit", "mrr", "turns", "pool@t1", "pool@all", "dense-only", "added"))
    for level in args.levels:
        for dense in (False, True):
            row = run(level, dense, samples, index, args.catalog)
            rows.append(row)
            print("%-4s %-6s %8.4f %6.3f %6.3f %6.2f %9.3f %9.3f %9d %7.1f" % (
                level, "on" if dense else "off", row["score"], row["hit"], row["mrr"], row["mttc"],
                row["pool_recall_turn1"], row["pool_recall_all_calls"],
                row["sessions_target_only_from_dense"], row["mean_dense_added"]), flush=True)
    if args.out:
        for row in rows:
            row.pop("per_session_rr", None)
        Path(args.out).write_text(json.dumps({"dataset": args.dataset, "rows": rows}, indent=1))
        print("→", args.out)


if __name__ == "__main__":
    main()
