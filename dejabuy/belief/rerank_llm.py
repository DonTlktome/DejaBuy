"""Optional LLM list-wise re-ranking of the head of the belief ordering (USE_LLM=1).

Measured at parity with the evidence scorer when given the same verified evidence, and
~100x slower, so it ships disabled; kept as a reproducible ablation arm.
"""

from __future__ import annotations

from dejabuy import config
from dejabuy.llm import client as llm_client
from dejabuy.state import DialogState

_RERANK_SYSTEM_EVIDENCE = (
    "You are ranking products for a shopper. You are given the shopper's exact stated "
    "requirements, and for each candidate, which of those requirements were VERIFIED to appear "
    "verbatim in that product's catalog text. A verified match is hard evidence — trust it over "
    "how the title reads. Rank by (1) most requirements verified, then (2) plausibility. "
    'Reply with json only: {"ranking": [best_index, next_index, ...]} using the given indices.'
)

_RERANK_SYSTEM = (
    "You are a product ranking expert for a clothing/shoes/jewelry catalog. "
    "Given the shopper's stated constraints and a numbered candidate list, rank candidates "
    "by how well they satisfy ALL constraints (category, material, color, budget, features). "
    'Reply with json only: {"ranking": [best_index, next_index, ...]} using the given indices.'
)



def llm_rerank(state: DialogState, pool: list[dict]) -> list[dict]:
    """LLM listwise 精排（增强路径）：只重排规则序头部；任何失败原样返回规则序。"""
    if len(pool) < 3:
        return pool
    if config.LLM_PROMPT == "evidence":
        system, user = _evidence_prompt(state, pool)
    else:
        system = _RERANK_SYSTEM
        lines = [f"Shopper constraints: {state.distilled or state.category or 'unknown'}", "Candidates:"]
        for index, c in enumerate(pool, start=1):
            price = f" ${c['price']}" if c.get("price") is not None else ""
            lines.append(f"{index}. {c['title'][:90]}{price}")
        user = "\n".join(lines)
    reply = llm_client.chat_json(system, user, max_tokens=150)
    if not reply or not isinstance(reply.get("ranking"), list):
        return pool
    order: list[int] = []
    for value in reply["ranking"]:
        if isinstance(value, int) and 1 <= value <= len(pool) and value not in order:
            order.append(value)
    reranked = [pool[i - 1] for i in order]
    reranked += [c for i, c in enumerate(pool, start=1) if i not in order]
    return reranked


def _evidence_prompt(state: DialogState, pool: list[dict]) -> tuple[str, str]:
    """C-T9 对照臂：把规则打分器看得见的"命中证据"原样交给 LLM。

    实验 4b 的负收益有两个未分离的原因：① LLM 只拿到标题、看不到"哪条约束在哪件商品
    详情里逐字出现"这个指纹证据 ② 喂给它的上下文是蒸馏后的拼接串。
    本函数把 ① 补齐（并顺带修掉 ②：给完整约束原文而非蒸馏串），
    从而判断负收益到底是信息缺失造成的，还是 LLM 在这个任务上本就帮不上忙。
    """
    slots = [s for s in state.slots if s.terms]
    lines = []
    if state.category:
        lines.append(f"Product category: {state.category}")
    lines.append("Shopper's stated requirements:")
    for i, slot in enumerate(slots, start=1):
        kind = "must-have" if slot.hard else "preference"
        lines.append(f"  R{i} ({kind}): {slot.value[:120]}")
    lines.append("")
    lines.append("Candidates (VERIFIED = that requirement appears verbatim in the product's catalog text):")
    for index, c in enumerate(pool, start=1):
        text = c.get("norm_text", "")
        hits = [f"R{i}" for i, slot in enumerate(slots, start=1)
                if any(t in text for t in slot.terms)]
        price = f" ${c['price']}" if c.get("price") is not None else ""
        verified = f"VERIFIED {','.join(hits)}" if hits else "VERIFIED none"
        lines.append(f"{index}. [{verified}] {c['title'][:80]}{price}")
    return _RERANK_SYSTEM_EVIDENCE, "\n".join(lines)

