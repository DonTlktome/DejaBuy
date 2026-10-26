"""Conversation-level memory benchmark: the official simulator protocol on real returning customers.

    python3 -m bench.memory.run --split test --conditions cold memory shuffled
    python3 -m bench.memory.run --split test_loyal

Every session is a full conversation under the unmodified evaluator (evaluator/local_evaluator.py):
the simulated shopper's hidden intent card is built from the target product, the scenario comes
from the benchmark sample. Only what the agent is told about the user changes:

  cold      user_profile without a user id → no memory (the official setting)
  memory    the user's own purchase history strictly before the target purchase
  shuffled  another test user's history (with that user's cut-off) — the permutation control:
            if memory helped only because "any history" shifts the ranking, this would match it

Reports the official score, HitRate@10, MRR, mean turns to conversion, first-turn hit rate,
overall and per stratum, and paired bootstrap 95% intervals for memory − cold and
memory − shuffled on per-session reciprocal rank and turns.
"""

from __future__ import annotations

import argparse
import json
import random
import sys
import time
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from evaluator.local_evaluator import catalog_index, evaluate  # noqa: E402
from dejabuy.agent import DejaBuyAgent as Agent  # noqa: E402

MAX_TURNS = 10


def load(split: str, n: int) -> list[dict]:
    rows = [json.loads(line) for line in (ROOT / "data" / "memory_bench" / f"{split}.jsonl").open(encoding="utf-8")]
    return rows[:n] if n else rows


def condition_samples(samples: list[dict], condition: str) -> list[dict]:
    if condition == "memory":
        return samples
    if condition == "cold":
        return [{**s, "user_profile": {}} for s in samples]
    rng = random.Random(11)
    order = list(range(len(samples)))
    rng.shuffle(order)
    # derangement: nobody keeps their own history
    for i in range(len(order)):
        if order[i] == i:
            j = (i + 1) % len(order)
            order[i], order[j] = order[j], order[i]
    return [{**s, "user_profile": samples[order[i]]["user_profile"]} for i, s in enumerate(samples)]


def session_score(row: dict) -> float:
    turns = row["first_hit_turn"] or (MAX_TURNS + 1)
    return 0.5 * row["hit"] + 0.3 * row["reciprocal_rank"] + 0.2 * max(0.0, (11 - turns) / 10)


def summarise(rows: list[dict]) -> dict:
    n = len(rows)
    hits = [r for r in rows if r["hit"]]
    mttc = sum(r["first_hit_turn"] for r in hits) / len(hits) if hits else None
    hit_rate = len(hits) / n
    mrr = sum(r["reciprocal_rank"] for r in rows) / n
    eff = max(0.0, min(1.0, (11 - mttc) / 10)) if mttc is not None else 0.0
    return {"n": n, "score": round(0.5 * hit_rate + 0.3 * mrr + 0.2 * eff, 4), "hit": round(hit_rate, 4),
            "mrr": round(mrr, 4), "mttc": round(mttc, 3) if mttc is not None else None,
            "first_turn": round(sum(1 for r in rows if r["first_hit_turn"] == 1) / n, 4)}


def paired_ci(a: list[float], b: list[float], reps: int = 2000) -> tuple[float, float, float]:
    rng = random.Random(5)
    diffs = [x - y for x, y in zip(a, b)]
    n = len(diffs)
    boots = sorted(sum(diffs[rng.randrange(n)] for _ in range(n)) / n for _ in range(reps))
    return sum(diffs) / n, boots[int(0.025 * reps)], boots[int(0.975 * reps)]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--split", default="test")
    parser.add_argument("--n", type=int, default=0)
    parser.add_argument("--conditions", nargs="+", default=["cold", "memory", "shuffled"])
    parser.add_argument("--catalog", default=str(ROOT / "data" / "catalog.jsonl"))
    parser.add_argument("--out", default="")
    args = parser.parse_args()

    samples = load(args.split, args.n)
    catalog_ids, categories, products = catalog_index(args.catalog)
    agent = Agent(args.catalog)
    results: dict[str, list[dict]] = {}
    for condition in args.conditions:
        t0 = time.time()
        run = condition_samples(samples, condition)
        res = evaluate(agent, run, catalog_ids, categories, products)
        rows = res["sessions"]
        for row, s in zip(rows, samples):
            row["stratum"] = s.get("stratum", "other")
            row["score"] = session_score(row)
        results[condition] = rows
        print(f"{condition:9s} {summarise(rows)}  ({time.time() - t0:.0f}s)", flush=True)

    report: dict = {"split": args.split, "n": len(samples), "conditions": {}, "strata": {}, "paired": {}}
    for condition, rows in results.items():
        report["conditions"][condition] = summarise(rows)
        groups = defaultdict(list)
        for r in rows:
            groups[r["stratum"]].append(r)
        report["strata"][condition] = {g: summarise(rs) for g, rs in sorted(groups.items())}
    if "memory" in results:
        for base in ("cold", "shuffled"):
            if base not in results:
                continue
            mem, ref = results["memory"], results[base]
            report["paired"][f"memory-{base}"] = {
                metric: [round(v, 4) for v in paired_ci([key(r) for r in mem], [key(r) for r in ref])]
                for metric, key in (("rr", lambda r: r["reciprocal_rank"]),
                                    ("turns", lambda r: r["first_hit_turn"] or MAX_TURNS + 1),
                                    ("session_score", lambda r: r["score"]))
            }

    print(f"\n{args.split}: {len(samples)} sessions")
    for condition, groups in report["strata"].items():
        for g, v in groups.items():
            print(f"  {condition:9s} {g:14s} {v}")
    for name, metrics in report["paired"].items():
        print(f"  {name}: " + "  ".join(f"{m} {v[0]:+.4f} [{v[1]:+.4f}, {v[2]:+.4f}]" for m, v in metrics.items()))
    out = Path(args.out or ROOT / "golden" / f"memory_{args.split}.json")
    out.write_text(json.dumps(report, indent=1), encoding="utf-8")
    print(f"→ {out}")


if __name__ == "__main__":
    main()
