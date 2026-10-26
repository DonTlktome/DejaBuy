"""Procedural-memory benchmark: a persona simulator where some shoppers never answer open questions.

    python3 -m bench.memory.persona --n 400

The reference simulator has no user-level habits (its one refusal, in the boundary scenario,
hits whatever is asked first), so procedural memory cannot be measured on it. This wrapper adds
one realistic, persistent habit and nothing else: a `closed` shopper declines every open-ended
question (`other`) with the simulator's own boundary sentence, and answers concrete attributes
exactly like the reference simulator. Half the users are `closed`, half `plain` (no habit, the
control: procedural memory must not cost them anything).

Protocol per user: a warm-up conversation about a different product (another category, browsing),
then the measured conversation about the real target. Two agents:

  session-only  habits learned inside a conversation only (two declines of the same attribute)
  procedural    the same, plus what the warm-up taught is kept for the next conversation

Semantic memory is switched off for both, so the difference is procedural memory alone. This is
a synthetic persona by construction: it shows the mechanism and what it is worth when the habit
exists; it is not evidence about how many real shoppers have it.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from dejabuy import config  # noqa: E402
from dejabuy.agent import DejaBuyAgent  # noqa: E402
from evaluator.local_evaluator import (  # noqa: E402
    catalog_index, coarse_category, customer_reply, initial_message, materialize_hidden_fields,
    normalize_recommendations)

MAX_TURNS, TOP_K = 10, 10


def persona_of(user_id: str) -> str:
    return "closed" if int(hashlib.sha256(f"persona:{user_id}".encode()).hexdigest()[:8], 16) % 2 else "plain"


def run_session(agent, sample: dict, products: dict, categories: dict, catalog_ids: set, persona: str) -> dict:
    """The evaluator's loop (evaluator/local_evaluator.py: evaluate) with a persona-aware reply."""
    session_id = f"{sample['sample_id']}_{random.random():.9f}"
    agent.reset(session_id, sample["user_profile"])
    target = sample["ground_truth"]["parent_asin"]
    card, behavior = materialize_hidden_fields(sample, products)
    sample = {**sample, "intent_card": card, "behavior": behavior}
    disclosed: set[str] = set()
    boundary_used = False
    override_applied = sample["scenario_type"] != "intent_override"
    message = initial_message(sample, coarse_category(categories.get(target, [])), disclosed)
    for turn in range(1, MAX_TURNS + 1):
        response = agent.respond(session_id, message, turn, TOP_K)
        ranked = normalize_recommendations(response.get("recommendations"), catalog_ids)
        if override_applied and target in ranked:
            return {"hit": True, "turn": turn, "rr": 1.0 / (ranked.index(target) + 1)}
        if turn == MAX_TURNS:
            break
        override = behavior.get("override") or {}
        ask = response.get("ask_attribute")
        if not override_applied and turn + 1 == int(override.get("turn", 3)):
            override_applied = True
            if override.get("new_value"):
                disclosed.add(str(override["new_value"]))
            message = str(override.get("message"))
        elif persona == "closed" and ask == "other":
            message = "I don't have a preference for other; please use your judgment."
        else:
            message, boundary_used = customer_reply(sample, ask, disclosed, boundary_used)
    return {"hit": False, "turn": None, "rr": 0.0}


def summarise(rows: list[dict]) -> dict:
    n = len(rows)
    hits = [r for r in rows if r["hit"]]
    mttc = sum(r["turn"] for r in hits) / len(hits) if hits else None
    hit, mrr = len(hits) / n, sum(r["rr"] for r in rows) / n
    eff = max(0.0, min(1.0, (11 - mttc) / 10)) if mttc else 0.0
    return {"n": n, "score": round(0.5 * hit + 0.3 * mrr + 0.2 * eff, 4), "hit": round(hit, 4),
            "mrr": round(mrr, 4), "mttc": round(mttc, 3) if mttc else None}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--n", type=int, default=400)
    parser.add_argument("--catalog", default=str(ROOT / "data" / "catalog.jsonl"))
    args = parser.parse_args()

    config.MEMORY_SCALE = 0.0          # isolate procedural memory: no semantic prior in either arm
    samples = [json.loads(line) for line in (ROOT / "data" / "memory_bench" / "test.jsonl").open(encoding="utf-8")][: args.n]
    catalog_ids, categories, products = catalog_index(args.catalog)
    by_category = defaultdict(list)
    for asin, cats in categories.items():
        by_category[coarse_category(cats)].append(asin)
    category_names = sorted(by_category)

    results = {}
    for arm in ("session-only", "procedural"):
        agent = DejaBuyAgent(args.catalog)
        rows = defaultdict(list)
        for s in samples:
            uid = s["user_profile"]["user_id"]
            persona = persona_of(uid)
            rng = random.Random(uid)
            target_cat = coarse_category(categories.get(s["ground_truth"]["parent_asin"], []))
            other_cat = rng.choice([c for c in category_names if c != target_cat and len(by_category[c]) >= 20])
            warm = {"sample_id": f"warm_{s['sample_id']}", "scenario_type": "browsing",
                    "ground_truth": {"parent_asin": rng.choice(by_category[other_cat])},
                    "user_profile": s["user_profile"]}
            run_session(agent, warm, products, categories, catalog_ids, persona)
            if arm == "session-only" and agent.memory is not None:
                agent.memory.forget(uid)          # nothing carries over between conversations
            rows[persona].append(run_session(agent, s, products, categories, catalog_ids, persona))
        results[arm] = {p: summarise(r) for p, r in sorted(rows.items())}
        print(arm, results[arm], flush=True)
    out = ROOT / "golden" / "memory_procedural.json"
    out.write_text(json.dumps({"n": len(samples), "results": results}, indent=1), encoding="utf-8")
    print(f"→ {out}")


if __name__ == "__main__":
    main()
