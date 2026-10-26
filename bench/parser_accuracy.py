#!/usr/bin/env python3
"""Score the understanding layer on its own (event type and verbatim value recall) so parsing
defects are visible without the end-to-end number."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from evaluator.local_evaluator import intent_card  # noqa: E402
from bench.paraphrase_rules import paraphrase  # noqa: E402
from dejabuy.understand import rules as parser  # noqa: E402
from dejabuy.understand.normalize import normalize  # noqa: E402
from dejabuy.state import DialogState  # noqa: E402


def build_cases(catalog_path: str, n: int) -> list[tuple[str, list[str]]]:
    """(原始消息, [嵌入的约束原文]) 对。三种载荷消息，模板逐字镜像评测器。"""
    cases: list[tuple[str, list[str]]] = []
    with open(catalog_path, encoding="utf-8") as handle:
        for line in handle:
            if len(cases) >= n * 3:
                break
            try:
                product = json.loads(line)
            except json.JSONDecodeError:
                continue
            card = intent_card(product)
            hard, soft = card["hard_constraints"], card["soft_preferences"]
            if len(hard) < 2 or not soft:
                continue  # 只取信息完整的卡，避免退化样本稀释统计
            # ① 吐约束（最常见载荷：2 条 "; " 拼接）
            cases.append((f"For that, what matters is: {hard[0]}; {hard[1]}.", [hard[0], hard[1]]))
            # ② buying 开场（1 条）
            cases.append((f"I'm looking for {card['target_category'][:40]}. "
                          f"A key requirement is: {soft[0]}.", [soft[0]]))
            # ③ 改需求（1 条）
            cases.append((f"Actually, ignore my earlier preference. What I need is: {hard[0]}.", [hard[0]]))
    return cases


def judge(state: DialogState, expected: list[str]) -> tuple[int, int, int]:
    """返回 (逐字命中数, 部分命中数, 垃圾槽位数)。"""
    exact = partial = 0
    matched_slots: set[int] = set()
    for exp in expected:
        norm_exp = normalize(exp)
        hit = False
        for i, slot in enumerate(state.slots):
            if slot.value == exp.strip().rstrip("."):
                exact += 1
                matched_slots.add(i)
                hit = True
                break
        if hit:
            continue
        for i, slot in enumerate(state.slots):
            norm_slot = normalize(slot.value)
            if norm_slot and norm_exp and (norm_slot in norm_exp or norm_exp in norm_slot):
                partial += 1
                matched_slots.add(i)
                break
    garbage = len(state.slots) - len(matched_slots)
    return exact, partial, garbage


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--catalog", default="data/catalog.jsonl")
    ap.add_argument("--n", type=int, default=200, help="采样商品数（每件 3 条消息）")
    ap.add_argument("--levels", nargs="+", default=["L0", "L1", "L2", "L3", "L4"])
    args = ap.parse_args()

    cases = build_cases(args.catalog, args.n)
    from dejabuy import config
    from dejabuy.belief.retrieve import Retriever
    parser.set_catalog_verifier(Retriever(args.catalog).phrase_exists)  # 与真实管线一致
    mode = "规则+LLM 三层" if config.LLM_PARSE else "规则-only"
    print(f"解析器内在准确率 · {len(cases)} 条消息 · {mode}\n")
    print("%-4s %14s %14s %12s" % ("档位", "recall_exact", "recall_partial", "garbage/msg"))
    print("-" * 50)
    for level in args.levels:
        total_expected = exact = partial = garbage = 0
        parser._llm_parse_failures = 0  # 每档重置熔断
        for message, expected in cases:
            state = DialogState(session_id="t", profile={})
            parser.update_state(state, paraphrase(message, level), 1)
            e, p, g = judge(state, expected)
            exact += e
            partial += p
            garbage += g
            total_expected += len(expected)
        print("%-4s %13.1f%% %13.1f%% %12.2f" % (
            level, 100 * exact / total_expected,
            100 * (exact + partial) / total_expected, garbage / len(cases)))
    print("\n口径：exact=提取值与嵌入约束逐字相同（指纹完整）；partial=归一化后互为子串；")
    print("      garbage=不对应任何嵌入约束的槽位（污染排序的噪声）。")


if __name__ == "__main__":
    main()
