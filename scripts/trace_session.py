#!/usr/bin/env python3
"""Print one session's conversation, state and slate turn by turn."""

from __future__ import annotations

import argparse
import sys
import uuid
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from evaluator.local_evaluator import (  # noqa: E402
    MAX_TURNS,
    TOP_K,
    catalog_index,
    coarse_category,
    customer_reply,
    initial_message,
    load_jsonl,
    materialize_hidden_fields,
    normalize_recommendations,
)
from dejabuy.agent import DejaBuyAgent as Agent  # noqa: E402


def title_of(products: dict, asin: str, width: int = 62) -> str:
    text = str((products.get(asin) or {}).get("title") or "?")
    return text[:width] + ("…" if len(text) > width else "")


def trace(agent, sample, catalog_ids, categories, products) -> None:
    target = str(sample["ground_truth"]["parent_asin"])
    card, behavior = materialize_hidden_fields(sample, products)
    effective = {**sample, "intent_card": card, "behavior": behavior}

    print("=" * 78)
    print(f"会话 {sample['sample_id']}  场景={sample['scenario_type']}  难度={sample.get('difficulty_bucket')}")
    print(f"🎯 隐藏答案: {target}  {title_of(products, target)}")
    print(f"👤 画像: {sample['user_profile'].get('summary')}")
    print(f"📋 模拟用户的底牌 hard={card['hard_constraints']}")
    print(f"            soft={card['soft_preferences']}")
    if behavior.get("override"):
        print(f"🔄 第 {behavior['override']['turn']} 轮会变卦 → {behavior['override']['new_value'][:60]}")
    print("=" * 78)

    session_id = f"trace_{uuid.uuid4().hex}"
    agent.reset(session_id, sample["user_profile"])
    disclosed: set[str] = set()
    boundary_used = False
    override_applied = sample["scenario_type"] != "intent_override"
    user_message = initial_message(effective, coarse_category(categories.get(target, [])), disclosed)
    hit_turn = None
    best_rank = None

    for turn in range(1, MAX_TURNS + 1):
        print(f"\n── 第 {turn} 轮 ──")
        print(f"👤 用户: {user_message}")
        response = agent.respond(session_id, user_message, turn, TOP_K)
        ranked = normalize_recommendations(response.get("recommendations"), catalog_ids)
        print(f"🤖 Agent: {response.get('message')}   [提问字段: {response.get('ask_attribute')}]")
        if ranked:
            for index, asin in enumerate(ranked, start=1):
                mark = "  ✅" if asin == target else "    "
                print(f"{mark} {index:>2}. {asin}  {title_of(products, asin)}")
        else:
            print("     (本轮没有给推荐)")

        if override_applied and target in ranked:
            best_rank = ranked.index(target) + 1
            hit_turn = turn
            print(f"\n🎉 命中！第 {turn} 轮，排名第 {best_rank}，本会话贡献 RR={1 / best_rank:.3f}")
            break
        if turn == MAX_TURNS:
            break

        override = effective.get("behavior", {}).get("override") or {}
        if not override_applied and turn + 1 == int(override.get("turn", 3)):
            override_applied = True
            new_value = str(override.get("new_value", ""))
            if new_value:
                disclosed.add(new_value)
            user_message = str(override.get("message", "Actually, please ignore my earlier preference."))
        else:
            user_message, boundary_used = customer_reply(
                effective, response.get("ask_attribute"), disclosed, boundary_used
            )

    if hit_turn is None:
        print(f"\n❌ 10 轮用尽仍未命中（MTTC 记 11，RR 记 0）")


def main() -> None:
    parser = argparse.ArgumentParser(description="逐轮回放一条评测会话")
    parser.add_argument("--catalog", default=str(ROOT / "data/catalog.jsonl"))
    parser.add_argument("--dataset", default=str(ROOT / "data/public_set.jsonl"))
    parser.add_argument("--id", help="指定 sample_id，例如 public_0007")
    parser.add_argument("--scenario", help="buying / browsing / intent_override / boundary")
    parser.add_argument("-n", "--limit", type=int, default=1, help="回放几条")
    parser.add_argument("--search", help="不跑会话，直接用一句话查目录看看返回什么")
    args = parser.parse_args()

    print("加载目录中（约 58MB，十几秒）…", flush=True)
    catalog_ids, categories, products = catalog_index(args.catalog)
    agent = Agent(args.catalog)
    print(f"目录就绪：{len(catalog_ids)} 个商品\n", flush=True)

    if args.search:
        session_id = "manual"
        agent.reset(session_id, {})
        response = agent.respond(session_id, args.search, 1, TOP_K)
        print(f"🔍 查询: {args.search}")
        for index, item in enumerate(response["recommendations"], start=1):
            asin = item["parent_asin"]
            print(f"  {index:>2}. {asin}  {title_of(products, asin)}")
        return

    samples = load_jsonl(args.dataset)
    if args.id:
        chosen = [s for s in samples if s["sample_id"] == args.id]
    elif args.scenario:
        chosen = [s for s in samples if s["scenario_type"] == args.scenario][: args.limit]
    else:
        chosen = samples[: args.limit]
    if not chosen:
        print("没找到符合条件的会话")
        return
    for sample in chosen:
        trace(agent, sample, catalog_ids, categories, products)


if __name__ == "__main__":
    main()
