"""Evidence scoring: how well each candidate explains the conversation so far.

Score components, in order of weight: verbatim constraint hits (hard > soft, longer fragment
> shorter), exact price, category match, purchase priors (review volume, has-price), mirror-card
consistency, normalised BM25 rank. The purchase priors answer a different question from the
rest — not "which product matches this sentence" but "which product a real shopper is more
likely to have bought" — and decay as evidence accumulates (PRIOR_DECAY).
"""

from __future__ import annotations

import math

from dejabuy import config
from dejabuy.state import DialogState
from dejabuy.understand.normalize import normalize


def make_scorer(state: DialogState, candidates: list[dict]):
    """Build the per-candidate scoring closure for this turn's state."""
    if not candidates:
        return []
    # BM25 名次分按【BM25 子池】归一化（实验 22，与短语召回路捆绑）：
    # 追加候选（短语路/稠密路，bm25_rank ≥ CANDIDATE_POOL 哨兵）不在名次体系里，记 0。
    bm25_pool = max(sum(1 for c in candidates
                        if isinstance(c.get("bm25_rank"), int)
                        and c["bm25_rank"] < config.CANDIDATE_POOL), 1)

    # 每槽一组归一化 terms（A 产出；空列表 = 该槽不参与文本匹配，budget 槽即如此）。
    # 计权取命中的最长 term、同槽不重复计分（SPEC §5 消费约定）
    slot_matchers = [(slot.terms, 2.0 if slot.hard else 0.75)
                     for slot in state.slots if slot.terms]

    # 长约束分片兜底已迁入 A 的 constraint_terms（Slot.terms 接口，FRAGMENT_WEIGHT 开关随迁）：
    # C 实测接口路径与此处旧的独立加分通道等价（±0.004 噪声带），故通道收掉、接口独占。

    norm_category = normalize(state.category) if state.category else ""

    # 先验衰减系数：证据越多，先验越该让位给证据（第 1 轮无证据时先验全权重）
    evidence = len(slot_matchers)
    prior_scale = 1.0 / (1.0 + config.PRIOR_DECAY * evidence) if config.PRIOR_DECAY else 1.0
    if config.EARLY_PRIOR_BOOST != 1.0 and len(state.history) <= config.EARLY_TURNS:
        prior_scale *= config.EARLY_PRIOR_BOOST

    memory = state.memory
    memory_weights = state.memory_weights

    def score(c: dict) -> float:
        text = c.get("norm_text", "")
        s = 0.0
        for terms, base in slot_matchers:
            hit_len = max((len(t) for t in terms if t in text), default=0)
            if hit_len:
                s += base * (1.0 + min(hit_len, 60) / 30.0)   # 长逐字片段 → 最高 3 倍
        # 开场句的品类 = 评测器用目标商品 categories 尾部生成，精确命中是强判别信号
        if norm_category and c.get("coarse_cat") == norm_category:
            s += 2.5
        # 精确价格命中（budget around $X 里的 X 就是目标商品的标价）强加分
        if state.budget is not None and c.get("price") is not None:
            if abs(c["price"] - state.budget) < 0.005:
                s += 4.0
            elif state.budget * 0.9 <= c["price"] <= state.budget * 1.1:
                s += 0.5
        # 先验轴（回答"哪件更可能是真人买的那一件"，而非"哪件更匹配这句话"）
        # 热度先验：log 压缩后归一化到 0..1（rating_number 跨度 0~10 万，线性会淹没约束信号）
        if config.POP_WEIGHT:
            s += prior_scale * config.POP_WEIGHT * (math.log10(1 + c.get("rating_number", 0)) / 5.0)
        # has_price 先验：真实卖出去的商品才有价格数据（目标 89.0% vs 全目录 20.8%，且与热度独立）
        if config.HAS_PRICE_WEIGHT and c.get("price") is not None:
            s += prior_scale * config.HAS_PRICE_WEIGHT
        # features 条数先验：目标商品的详情页更完整（中位数 8 条 vs 全目录 5 条）
        if config.FEATURE_COUNT_WEIGHT:
            s += config.FEATURE_COUNT_WEIGHT * min(c.get("feature_count", 0), 12) / 12.0
        # 意图卡镜像一致性（实验 22）：槽位值命中候选自身的镜像卡条目 → 每槽 +MIRROR_BONUS。
        # 意图卡由候选元数据确定性生成、77.6% 全局唯一——这是"该候选会生成这段对话"的直接证据
        if config.MIRROR_BONUS:
            card = c.get("card_norm") or ()
            for terms, _base in slot_matchers:
                if any(t in entry for entry in card for t in terms):
                    s += config.MIRROR_BONUS
        # 跨会话记忆先验（memory/semantic.py）：学到的线性亲和分，与热度先验同一轴，证据可压过
        if memory is not None:
            s += config.MEMORY_SCALE * memory.prior(c, memory_weights)
        # BM25 名次分（子池归一化）：追加候选不在名次体系里，记 0
        rank_val = c.get("bm25_rank")
        if isinstance(rank_val, int) and rank_val < config.CANDIDATE_POOL:
            s += 1.0 - rank_val / bm25_pool
        return s
    return score
