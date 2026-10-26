"""Held-out regression sets: the official simulator on targets nobody tuned on.

    python3 -m bench.synthetic --build            # data/synthetic/{uniform,purchase}.jsonl (1,200 each)
    python3 -m bench.synthetic                    # evaluate both, unmodified evaluator

The public set's 200 targets are real purchases and skew popular; every constant in the agent
was chosen on them. Two fresh target sets check that nothing was fitted to those 200:

  uniform   catalogue products drawn uniformly (most have few reviews and no price: the hard tail)
  purchase  products drawn in proportion to how often they appear as a real purchase in the
            Amazon 2023 sequences (data/sequences.jsonl), excluding the public targets — the same
            distribution the public set was drawn from, new draws

Scenarios follow the public mix (buying 40 / browsing 40 / intent-override 15 / boundary 5).
"""

from __future__ import annotations

import argparse
import json
import random
import sys
import time
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

SCENARIOS = ["buying"] * 40 + ["browsing"] * 40 + ["intent_override"] * 15 + ["boundary"] * 5
OUT = ROOT / "data" / "synthetic"


def build(n: int, seed: int) -> None:
    rng = random.Random(seed)
    catalog = [json.loads(line)["parent_asin"] for line in (ROOT / "data" / "catalog.jsonl").open(encoding="utf-8")]
    public = {json.loads(line)["ground_truth"]["parent_asin"]
              for line in (ROOT / "data" / "public_set.jsonl").open(encoding="utf-8")}
    in_catalog = set(catalog)
    counts: Counter = Counter()
    with (ROOT / "data" / "sequences.jsonl").open(encoding="utf-8") as handle:
        for line in handle:
            for asin, _r, _t in json.loads(line)["events"]:
                if asin in in_catalog and asin not in public:
                    counts[asin] += 1
    pool, weights = zip(*counts.items())
    sets = {
        "uniform": rng.sample([a for a in catalog if a not in public], n),
        "purchase": list(dict.fromkeys(rng.choices(pool, weights=weights, k=n * 10)))[:n],
    }
    OUT.mkdir(parents=True, exist_ok=True)
    for name, targets in sets.items():
        with (OUT / f"{name}.jsonl").open("w", encoding="utf-8") as handle:
            for i, asin in enumerate(targets):
                handle.write(json.dumps({"sample_id": f"{name}_{i:04d}", "scenario_type": rng.choice(SCENARIOS),
                                         "ground_truth": {"parent_asin": asin}, "user_profile": {}}) + "\n")
        print(f"{name}: {len(targets)} targets → {OUT / (name + '.jsonl')}")


def evaluate_sets(names: list[str]) -> None:
    from evaluator.local_evaluator import catalog_index, evaluate
    from dejabuy.agent import DejaBuyAgent
    catalog_ids, categories, products = catalog_index(ROOT / "data" / "catalog.jsonl")
    agent = DejaBuyAgent(str(ROOT / "data" / "catalog.jsonl"))
    report = {}
    for name in names:
        samples = [json.loads(line) for line in (OUT / f"{name}.jsonl").open(encoding="utf-8")]
        t0 = time.time()
        res = evaluate(agent, samples, catalog_ids, categories, products)
        report[name] = {k: res[k] for k in ("sample_count", "hit_rate_at_10", "mrr", "mttc", "recommended_technical_score")}
        report[name]["scenario_metrics"] = res.get("scenario_metrics")
        print(f"{name}: {report[name]['recommended_technical_score']:.4f}  hit {res['hit_rate_at_10']:.3f}  "
              f"mrr {res['mrr']:.3f}  mttc {res['mttc']:.2f}  ({time.time() - t0:.0f}s)", flush=True)
    (ROOT / "golden" / "synthetic.json").write_text(json.dumps(report, indent=1), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--build", action="store_true")
    parser.add_argument("--n", type=int, default=1200)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--sets", nargs="+", default=["uniform", "purchase"])
    args = parser.parse_args()
    if args.build:
        build(args.n, args.seed)
    else:
        evaluate_sets(args.sets)


if __name__ == "__main__":
    main()
