"""Ablation matrix: one switch, one number, on the official 200-session public set.

    python3 -m bench.ablations                 # → golden/ablations.json and a table

Each row runs the unmodified evaluator in a fresh process with one environment variable changed
from the default configuration (dejabuy/config.py). Robustness-only components (the LLM
extraction layer, alignment) do nothing on the template-exact public set by construction; their
ablations are reported on the paraphrase sets (bench/paraphrase_llm.py with LLM_PARSE=0) and
memory on the memory benchmark (bench/memory), not here.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

ROWS = [
    ("default", {}, "full system"),
    ("planner_fixed", {"PLANNER": "fixed"}, "no environment model / expected-utility planner (week-1 policy)"),
    ("width_product", {"WIDTH_MODE": "product"}, "always show the 10 most likely (no show-vs-defer)"),
    ("belief_t0.5", {"BELIEF_TEMPERATURE": "0.5"}, "sharper belief over the support set"),
    ("belief_t2", {"BELIEF_TEMPERATURE": "2.0"}, "flatter belief over the support set"),
    ("no_popularity", {"POP_WEIGHT": "0"}, "no review-volume prior"),
    ("no_has_price", {"HAS_PRICE_WEIGHT": "0"}, "no listed-price prior"),
    ("no_mirror", {"MIRROR_BONUS": "0"}, "no intent-card consistency bonus"),
    ("no_phrase_recall", {"PHRASE_RECALL": "0"}, "no per-constraint phrase recall route"),
    ("plan_top_50", {"PLAN_TOP": "50"}, "lookahead pruned to the top 50 candidates"),
]


def run(env_overrides: dict) -> dict:
    env = {**os.environ, **env_overrides, "DEJABUY_TRACE": "0"}
    out = ROOT / "results_ablation.json"
    subprocess.run([sys.executable, "-m", "evaluator.local_evaluator", "--output", str(out)],
                   cwd=ROOT, env=env, check=True, capture_output=True)
    r = json.loads(out.read_text(encoding="utf-8"))
    out.unlink()
    return {k: r[k] for k in ("hit_rate_at_10", "mrr", "mttc", "recommended_technical_score")}


LOOSE_ROWS = [
    ("default", {}, "full system"),
    ("no_mirror", {"MIRROR_BONUS": "0"}, "no intent-card consistency bonus"),
    ("no_phrase_recall", {"PHRASE_RECALL": "0"}, "no per-constraint phrase recall route"),
    ("no_popularity", {"POP_WEIGHT": "0"}, "no review-volume prior"),
    ("no_alignment", {"LLM_PARSE": "0"}, "(rules only — the LLM layer is off in every row here)"),
]


def run_paraphrase(env_overrides: dict, level: str) -> dict:
    """The rule paraphrase set at one level (no key, so the rule path): the loose regime is where
    retrieval-side components (mirror bonus, phrase recall) do their work."""
    env = {**os.environ, **env_overrides, "DEJABUY_TRACE": "0", "LLM_PARSE": "0"}
    out = subprocess.run([sys.executable, "bench/paraphrase_rules.py", "--levels", level], cwd=ROOT, env=env,
                         check=True, capture_output=True, text=True).stdout
    line = next(l for l in out.splitlines() if l.startswith(level + " "))
    score, hit, mrr, mttc = (float(x) for x in line.split()[1:5])
    return {"hit_rate_at_10": hit, "mrr": mrr, "mttc": mttc, "recommended_technical_score": score}


def main() -> None:
    if len(sys.argv) > 2 and sys.argv[1] == "--paraphrase":
        level = sys.argv[2]
        report = {}
        print(f"rule paraphrase set {level} (loose regime), rules only")
        print("%-18s %8s %8s %8s %8s  %s" % ("row", "score", "hit", "mrr", "turns", "what changes"))
        for name, overrides, what in LOOSE_ROWS[:-1]:
            r = run_paraphrase(overrides, level)
            report[name] = {**r, "env": overrides, "what": what}
            print("%-18s %8.4f %8.3f %8.3f %8.2f  %s" % (name, r["recommended_technical_score"], r["hit_rate_at_10"],
                                                       r["mrr"], r["mttc"], what), flush=True)
        (ROOT / "golden" / f"ablations_{level}.json").write_text(json.dumps(report, indent=1), encoding="utf-8")
        return
    report = {}
    print("%-18s %8s %8s %8s %8s  %s" % ("row", "score", "hit", "mrr", "turns", "what changes"))
    for name, overrides, what in ROWS:
        t0 = time.time()
        r = run(overrides)
        report[name] = {**r, "env": overrides, "what": what, "seconds": round(time.time() - t0)}
        print("%-18s %8.4f %8.3f %8.3f %8.2f  %s" % (name, r["recommended_technical_score"], r["hit_rate_at_10"],
                                                   r["mrr"], r["mttc"], what), flush=True)
    (ROOT / "golden" / "ablations.json").write_text(json.dumps(report, indent=1), encoding="utf-8")


if __name__ == "__main__":
    main()
