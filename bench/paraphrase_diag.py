"""Where does a paraphrase level lose score? Per-session comparison against the golden L0 run:
the rewritten utterances the agent saw, the protocol events it inferred, the regime per turn,
and the outcome. Prints only the sessions whose reciprocal rank dropped.

    python3 bench/paraphrase_diag.py --level P1            # LLM paraphrase set (cached)
    python3 bench/paraphrase_diag.py --level L2 --rules    # rule paraphrase set
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from evaluator.local_evaluator import catalog_index, evaluate, load_jsonl  # noqa: E402
from bench.paraphrase_llm import ParaphraseSet  # noqa: E402
from bench.paraphrase_rules import paraphrase as rule_paraphrase  # noqa: E402
from dejabuy.agent import DejaBuyAgent as Agent  # noqa: E402


class DiagAgent:
    def __init__(self, inner, rewrite) -> None:
        self.inner = inner
        self.rewrite = rewrite
        self.log: dict[str, list[dict]] = {}
        self.order: list[str] = []          # the evaluator names sessions by uuid, in sample order

    def reset(self, session_id: str, user_profile: dict) -> None:
        self.inner.reset(session_id, user_profile)
        self.log[session_id] = []
        self.order.append(session_id)

    def respond(self, session_id: str, user_message: str, turn: int, top_k: int) -> dict:
        rewritten = self.rewrite(user_message)
        out = self.inner.respond(session_id, rewritten, turn, top_k)
        state = self.inner.sessions[session_id]
        self.log[session_id].append({
            "turn": turn, "original": user_message, "seen": rewritten,
            "event": state.events[-1] if state.events else None,
            "slots": [(s.attribute, s.value, s.hard) for s in state.slots],
            "ask": out["ask_attribute"], "top1": out["recommendations"][0]["parent_asin"] if out["recommendations"] else None,
        })
        return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--level", default="P1")
    parser.add_argument("--rules", action="store_true", help="use the rule paraphrases (L1–L4)")
    parser.add_argument("--catalog", default="data/catalog.jsonl")
    parser.add_argument("--dataset", default="data/public_set.jsonl")
    parser.add_argument("--golden", default="golden/results_public200.json")
    parser.add_argument("--max", type=int, default=30)
    args = parser.parse_args()

    rewrite = (lambda m: rule_paraphrase(m, args.level)) if args.rules else ParaphraseSet(args.level).get
    samples = load_jsonl(args.dataset)
    catalog_ids, categories, products = catalog_index(args.catalog)
    agent = DiagAgent(Agent(args.catalog), rewrite)
    result = evaluate(agent, samples, catalog_ids, categories, products)
    golden = {s["sample_id"]: s for s in json.load(open(args.golden, encoding="utf-8"))["sessions"]}
    targets = {s["sample_id"]: (s.get("ground_truth") or {}).get("parent_asin") for s in samples}

    print(f"{args.level}: score {result['recommended_technical_score']:.4f}  hit {result['hit_rate_at_10']:.3f}  "
          f"mrr {result['mrr']:.3f}  mttc {result['mttc']:.2f}\n")
    shown = 0
    by_sample = {sample["sample_id"]: agent.log[agent.order[i]] for i, sample in enumerate(samples) if i < len(agent.order)}
    for session in result["sessions"]:
        sid = session["sample_id"]
        gold = golden.get(sid, {})
        if session["reciprocal_rank"] >= gold.get("reciprocal_rank", 0) and session["hit"] == gold.get("hit", True):
            continue
        shown += 1
        if shown > args.max:
            break
        print(f"--- {sid} ({session['scenario_type']}): rr {gold.get('reciprocal_rank')} → {session['reciprocal_rank']}, "
              f"hit turn {gold.get('first_hit_turn')} → {session.get('first_hit_turn')}  target {targets.get(sid)}")
        for row in by_sample.get(sid, []):
            event = row["event"] or {}
            kind = event.get("type")
            detail = {k: v for k, v in event.items() if k not in ("type", "categories")}
            print(f"  t{row['turn']} ask→{row['ask']:<8} [{kind}] {json.dumps(detail, ensure_ascii=False)[:120]}")
            print(f"      orig: {row['original'][:110]}")
            print(f"      seen: {row['seen'][:110]}")
    print(f"\n{shown} sessions lost score (showing up to {args.max}).")


if __name__ == "__main__":
    main()
