"""Run the official evaluator and print the headline numbers.

    python -m scripts.run_eval                 # -> results.json
    AGENT_IMPL=baseline python -m scripts.run_eval
"""

from __future__ import annotations

import json
import subprocess
import sys

if __name__ == "__main__":
    subprocess.run([sys.executable, "-m", "evaluator.local_evaluator", "--output", "results.json"], check=True)
    r = json.load(open("results.json", encoding="utf-8"))
    print(json.dumps({k: r[k] for k in ("hit_rate_at_10", "mrr", "mttc", "recommended_technical_score")}, indent=1))
