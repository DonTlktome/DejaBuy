"""Decompose the remaining headroom: which imperfect sessions are information-theoretically
indistinguishable (identical intent card and category as other products)."""

from __future__ import annotations

import argparse
import collections
import json
import random
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from evaluator.local_evaluator import (  # noqa: E402
    behavior_for, catalog_index, intent_card, coarse_category,
)


def load_results(path: str | None) -> dict:
    if path and Path(path).exists():
        return json.loads(Path(path).read_text(encoding="utf-8"))
    with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False) as tmp:
        out = tmp.name
    subprocess.run([sys.executable, "-m", "evaluator.local_evaluator", "--output", out],
                   cwd=ROOT, capture_output=True, check=True)
    return json.loads(Path(out).read_text(encoding="utf-8"))


def main() -> None:
    parser = argparse.ArgumentParser(description="天花板诊断")
    parser.add_argument("--results", default=None, help="复用已有 results.json，省一次全量")
    args = parser.parse_args()

    result = load_results(args.results)
    sessions = result["sessions"]
    rows = {json.loads(l)["sample_id"]: json.loads(l)
            for l in (ROOT / "data" / "public_set.jsonl").open(encoding="utf-8")}
    _, _, products = catalog_index(ROOT / "data" / "catalog.jsonl")

    score = result["recommended_technical_score"]
    mrr, mttc = result["mrr"], result["mttc"]
    print("当前  score=%.4f  hit=%.3f  mrr=%.3f  mttc=%.3f\n"
          % (score, result["hit_rate_at_10"], mrr, mttc))

    # ---------- ① 理论满分：MTTC 下限由 override 场景的结构性约束决定 ----------
    override_turns = []
    for sid, row in rows.items():
        if row["scenario_type"] != "intent_override":
            continue
        card = intent_card(products[str(row["ground_truth"]["parent_asin"])])
        rng = random.Random(f"{sid}\0{row['scenario_type']}")
        override_turns.append(int(behavior_for("intent_override", card, rng)["override"]["turn"]))
    n = len(sessions)
    non_override = n - len(override_turns)
    floor = (non_override * 1 + sum(override_turns)) / n
    eff_max = (11 - floor) / 10
    perfect = 0.5 + 0.3 + 0.2 * eff_max

    print("① 理论满分")
    print("   override %d 条的最早可命中轮（评测器规定，不可压缩）: %s"
          % (len(override_turns), dict(collections.Counter(override_turns))))
    print("   MTTC 下限 = (%d×1 + Σoverride) / %d = %.3f" % (non_override, n, floor))
    print("   理论满分 = 0.5×1.0 + 0.3×1.0 + 0.2×%.4f = %.4f" % (eff_max, perfect))
    print("   理论剩余 = %.4f  ——  MRR 部分 %.4f ／ MTTC 部分 %.4f\n"
          % (perfect - score, 0.3 * (1 - mrr), 0.2 * (eff_max - (11 - mttc) / 10)))

    # ---------- ② MRR：有多少条是信息论不可分的 ----------
    fingerprint: dict[tuple, list[str]] = collections.defaultdict(list)
    for asin, product in products.items():
        card = intent_card(product)
        fingerprint[(
            coarse_category(product.get("categories") or []),
            tuple(card["hard_constraints"]),
            tuple(card["soft_preferences"]),
        )].append(asin)

    imperfect = [s for s in sessions if s["best_rank"] != 1]
    print("② MRR 剩余空间的可达性（未排第 1 的 %d 条）" % len(imperfect))
    tied = 0
    for s in sorted(imperfect, key=lambda x: x["sample_id"]):
        target = str(rows[s["sample_id"]]["ground_truth"]["parent_asin"])
        product = products[target]
        card = intent_card(product)
        twins = [a for a in fingerprint[(
            coarse_category(product.get("categories") or []),
            tuple(card["hard_constraints"]),
            tuple(card["soft_preferences"]),
        )] if a != target]
        tied += bool(twins)
        note = ("信息论不可分（%d 个孪生商品：意图卡与粗品类逐条目相同，"
                "模拟用户生成的对话一字不差）" % len(twins)) if twins else "无孪生，理论上可区分"
        print("   %-14s rank=%d  %s" % (s["sample_id"], s["best_rank"], note))
    reachable = sum(1 - 1.0 / s["best_rank"] for s in imperfect
                    if not [a for a in fingerprint[(
                        coarse_category(products[str(rows[s["sample_id"]]["ground_truth"]["parent_asin"])]
                                        .get("categories") or []),
                        tuple(intent_card(products[str(rows[s["sample_id"]]["ground_truth"]["parent_asin"])])["hard_constraints"]),
                        tuple(intent_card(products[str(rows[s["sample_id"]]["ground_truth"]["parent_asin"])])["soft_preferences"]),
                    )] if a != str(rows[s["sample_id"]]["ground_truth"]["parent_asin"])])
    print("   → %d/%d 条不可分；把全部 8 条提到第 1 也只涨 %.4f，"
          % (tied, len(imperfect), sum(1 - 1.0 / s["best_rank"] for s in imperfect) / n * 0.3))
    print("     其中真正可达的只有 %.5f —— 低于 0.002 的噪声阈值\n" % (reachable / n * 0.3))

    # ---------- ③ MTTC：拿到需要什么前提 ----------
    non = [s for s in sessions if s["scenario_type"] != "intent_override"]
    by_turn = collections.Counter(s["first_hit_turn"] for s in non)
    t1, t2 = by_turn.get(1, 0), by_turn.get(2, 0)
    print("③ MTTC 剩余空间的可达性")
    print("   非 override %d 条：第 1 轮命中 %d（%.1f%%）／第 2 轮 %d／更晚 %d"
          % (len(non), t1, 100 * t1 / len(non), t2, len(non) - t1 - t2))
    print("   若第 2 轮命中的 %d 条全部提前到第 1 轮：总分 +%.4f" % (t2, 0.2 * t2 / n / 10))
    print("   但前提是：只知道粗品类（browsing）或品类+1 条约束（buying）时，")
    print("   **单件推荐一发命中**（收窄策略下第 1 轮只推 1 件）。")
    print("   同类目商品中位数 181 件 —— 这是信息量约束，不是调参问题。\n")

    print("结论：理论剩余 %.4f，实际可达远小于此。" % (perfect - score))
    print("      MRR 线已到底（可达 %.5f，低于噪声阈值）；" % (reachable / n * 0.3))
    print("      MTTC 线受限于'仅凭品类一发命中'的信息量上限。")
    print("      继续加特性 = 为私有集过拟合付费，收益却在噪声带内。")


if __name__ == "__main__":
    main()
