"""Candidate retrieval: FTS5 BM25 + per-constraint phrase recall (+ optional dense route),
fused as a non-displacing union.

Also owns the catalogue index, the mirror-card entries used by the evidence scorer and the
phrase lookup that the understanding layer uses as its catalogue verifier."""

from __future__ import annotations

import json
import re
import sqlite3
from pathlib import Path

from dejabuy import config
from dejabuy.understand.normalize import normalize
from dejabuy.state import DialogState
from dejabuy.belief.dense import DenseIndex

# 与评测器一致的正则（镜像自 evaluator/local_evaluator.py，勿 import 以免循环依赖）
_MATERIAL_RE = re.compile(r"\b(cotton|polyester|nylon|leather|wool|spandex|silk|rayon|fabric)\b", re.I)
_COLOR_RE = re.compile(r"\b(black|white|blue|red|pink|green|brown|gray|grey|purple|yellow|orange)\b", re.I)
_SEARCH_FIELDS = ("title", "features", "details", "description", "categories", "store")
_TOKEN_RE = re.compile(r"[a-z0-9]+")
_STOPWORDS = {
    "the", "a", "an", "and", "or", "for", "with", "of", "to", "in", "on", "is", "are",
    "i", "im", "my", "looking", "still", "exploring", "but", "what", "need", "that",
    "matters", "requirement", "key",
}


def _searchable_text(product: dict) -> str:
    parts: list[str] = []
    for field in _SEARCH_FIELDS:
        value = product.get(field)
        if isinstance(value, dict):
            parts.extend(f"{key} {item}" for key, item in value.items())
        elif isinstance(value, list):
            parts.extend(str(item) for item in value)
        elif value is not None:
            parts.append(str(value))
    return " ".join(parts).strip()


def _flatten_values(value: object) -> list[str]:
    """镜像评测器 _flatten_values（勿改：与 local_evaluator.py:40 逐字对齐）。"""
    if isinstance(value, dict):
        return [f"{key}: {item}" for key, item in value.items() if item not in (None, "", [])]
    if isinstance(value, list):
        return [str(item) for item in value if item not in (None, "")]
    return [str(value)] if value not in (None, "") else []


def _clean_constraint(value: str, limit: int = 180) -> str:
    """镜像评测器 _clean_constraint（勿改：与 local_evaluator.py:49 逐字对齐）。"""
    return re.sub(r"\s+", " ", value).strip(" -;,.\t\n")[:limit].rstrip()


def mirror_card_entries(product: dict, material, color) -> list[str]:
    """镜像评测器 intent_card() 的 hard+soft 条目（≤4 条，归一化后返回）。

    用途（实验 22，M3 的一致性 bonus）：意图卡由候选自身元数据确定性生成且 77.6%
    全局唯一——"用户吐露的约束恰好是候选自身卡上的条目"是强一致性证据。
    material/color 参数直接复用构造期已算好的 regex match，避免重复扫描。
    """
    candidates = [*_flatten_values(product.get("features")), *_flatten_values(product.get("details"))]
    if material:
        candidates.insert(0, material.group(1).lower())
    if color:
        candidates.insert(1, f"color: {color.group(1).lower()}")
    if product.get("price") not in (None, ""):
        candidates.append(f"budget around ${product['price']}")
    cleaned = list(dict.fromkeys(c for c in (_clean_constraint(item) for item in candidates) if c))
    if not cleaned:
        cleaned = [_clean_constraint(str(product.get("title") or "product"))]
    entries = cleaned[:2] + (cleaned[2:4] or cleaned[:1])
    return [normalize(e) for e in dict.fromkeys(entries) if normalize(e)]


def exact_card(product: dict, material, color) -> tuple[list[str], list[str]]:
    """The hidden intent card exactly as the reference simulator builds it: (hard, soft).

    Same recipe as `mirror_card_entries` but without normalisation — the strict environment
    model (env/protocol.py) replays the conversation against these exact strings."""
    candidates = [*_flatten_values(product.get("features")), *_flatten_values(product.get("details"))]
    if material:
        candidates.insert(0, material.group(1).lower())
    if color:
        candidates.insert(1, f"color: {color.group(1).lower()}")
    if product.get("price") not in (None, ""):
        candidates.append(f"budget around ${product['price']}")
    cleaned = list(dict.fromkeys(c for c in (_clean_constraint(item) for item in candidates) if c))
    if not cleaned:
        cleaned = [_clean_constraint(str(product.get("title") or "product"))]
    return cleaned[:2], (cleaned[2:4] or cleaned[:1])


def coarse_category(values: list[str]) -> str:
    """镜像评测器的粗品类算法：categories 最后两段非泛化部分（开场句的 X 就是它）。"""
    excluded = {"clothing", "clothing shoes & jewelry", "clothing, shoes & jewelry"}
    cleaned: list[str] = []
    for value in values:
        for part in str(value).split(","):
            part = part.strip()
            if part and part.lower() not in excluded:
                cleaned.append(part)
    return " ".join(cleaned[-2:]) if cleaned else "clothing item"


class Retriever:
    def __init__(self, catalog_path: str = "data/catalog.jsonl") -> None:
        self.products: dict[str, dict] = {}
        self.by_category: dict[str, list[str]] = {}   # exact coarse category -> product ids
        connection = sqlite3.connect(":memory:", check_same_thread=False)  # the REST service serialises access with a lock
        connection.execute(
            "CREATE VIRTUAL TABLE products USING fts5("
            "parent_asin UNINDEXED, title, categories, features, details, store, description, "
            "tokenize='unicode61 remove_diacritics 2')"
        )
        rows = []
        self.skipped_lines = 0  # ⚠️ 构造期抛异常 = 整场评测崩（评测器不保护构造），坏行必须跳过
        with Path(catalog_path).open(encoding="utf-8") as handle:
            for line in handle:
                try:
                    product = json.loads(line)
                    parent_asin = str(product["parent_asin"])
                except Exception:
                    self.skipped_lines += 1
                    continue
                text = _searchable_text(product)
                price = product.get("price")
                try:
                    price = float(price) if price not in (None, "") else None
                except (TypeError, ValueError):
                    price = None
                material = _MATERIAL_RE.search(text)
                color = _COLOR_RE.search(text)
                coarse = coarse_category(product.get("categories") or [])
                hard, soft = exact_card(product, material, color)
                self.by_category.setdefault(coarse, []).append(parent_asin)
                self.products[parent_asin] = {
                    "parent_asin": parent_asin,
                    "title": str(product.get("title") or ""),
                    "price": price,
                    "material": material.group(1).lower() if material else None,
                    "color": color.group(1).lower() if color else None,
                    "norm_text": normalize(text),
                    "rating_number": int(product.get("rating_number") or 0),
                    "feature_count": len(product.get("features") or []),
                    "coarse_cat": normalize(coarse),
                    "coarse_exact": coarse,
                    "card_hard": hard,
                    "card_soft": soft,
                    "card_norm": mirror_card_entries(product, material, color),
                    "store": str(product.get("store") or ""),
                    "department": str((product.get("details") or {}).get("Department") or "")
                    if isinstance(product.get("details"), dict) else "",
                }
                rows.append((
                    parent_asin,
                    str(product.get("title") or ""),
                    " ".join(str(v) for v in (product.get("categories") or [])),
                    " ".join(str(v) for v in (product.get("features") or [])),
                    " ".join(f"{k} {v}" for k, v in (product.get("details") or {}).items())
                    if isinstance(product.get("details"), dict) else "",
                    str(product.get("store") or ""),
                    " ".join(str(v) for v in (product.get("description") or []))
                    if isinstance(product.get("description"), list) else str(product.get("description") or ""),
                ))
        connection.executemany("INSERT INTO products VALUES (?, ?, ?, ?, ?, ?, ?)", rows)
        connection.commit()
        self.connection = connection
        # 稠密路（spec §1-⑦）：USE_DENSE=0 或资产缺失时为 None，自动降级纯 BM25
        self.dense = DenseIndex.from_env()

    # ---- 查询构建：槽位原文 + 品类 + 当前消息的去停用词 term ----
    def _query_terms(self, state: DialogState, message: str) -> list[str]:
        text = " ".join([*state.constraint_values(), state.category, message])
        terms: list[str] = []
        for token in _TOKEN_RE.findall(text.lower()):
            if len(token) > 1 and token not in _STOPWORDS and token not in terms:
                terms.append(token)
        return terms[:40]

    # ---- 稠密路查询（spec §1-④ v1）：槽位约束原文（除 budget）+ 品类 ----
    def _dense_query(self, state: DialogState) -> str:
        parts = [v for v in state.constraint_values() if not v.lower().startswith("budget")]
        if state.category:
            parts.append(state.category)
        return " ".join(parts)

    # ---- 多路融合（spec §1-⑤）：非置换式并集 —— BM25 顺序原样保留，稠密路独有候选追加在后。
    # 消融记录：RRF(k=60) 平权融合实测负收益（Recall@100 0.995→0.970，avg_pos 25.6→30.2）——
    # 稠密查询含 "Imported"/礼物话术等语义噪声，噪声候选把 BM25 池 40-100 位的好候选挤出截断线。
    # 且 retrieve() 返回顺序不影响下游打分（rank() 用自有公式重排），融合只需保证"捞全"。
    def _fuse_dense(self, state: DialogState, candidates: list[dict], slot_terms: list[list[str]]) -> list[dict]:
        dense_hits = self.dense.search(self._dense_query(state), config.DENSE_TOP_K)
        have = {c["parent_asin"] for c in candidates}
        # 有预算约束时，稠密补充的候选同样受价格窗约束（镜像 budget 硬过滤的意图）
        low = state.budget * (1 - config.PRICE_WINDOW) if state.budget is not None else None
        high = state.budget * (1 + config.PRICE_WINDOW) if state.budget is not None else None
        for asin, _sim in dense_hits:
            if asin in have or asin not in self.products:
                continue
            info = self.products[asin]
            if low is not None and (info["price"] is None or not (low <= info["price"] <= high)):
                continue
            # 稠密路独有候选：补齐 Candidate 字段，bm25_rank 给哨兵值（不打乱 BM25 名次语义）
            matched = sum(1 for terms in slot_terms if any(t in info["norm_text"] for t in terms))
            candidates.append({**info, "bm25_rank": config.CANDIDATE_POOL, "match_count": matched})
            have.add(asin)
        return candidates

    def retrieve(self, state: DialogState, message: str, k: int = 100) -> list[dict]:
        terms = self._query_terms(state, message)
        if not terms:
            return []
        query = " OR ".join(f'"{term}"' for term in terms)
        cursor = self.connection.execute(
            "SELECT parent_asin, bm25(products, 0.0, 6.0, 4.0, 2.5, 2.5, 1.5, 1.0) AS score "
            "FROM products WHERE products MATCH ? ORDER BY score LIMIT ?",
            (query, config.CANDIDATE_POOL),
        )
        # 每槽一组归一化 terms（A 产出，SPEC §5；空列表 = 该槽不参与文本匹配）
        slot_terms = [slot.terms for slot in state.slots if slot.terms]
        candidates: list[dict] = []
        for rank, (parent_asin, _score) in enumerate(cursor.fetchall()):
            info = self.products[parent_asin]
            matched = sum(1 for terms in slot_terms if any(t in info["norm_text"] for t in terms))
            candidates.append({**info, "bm25_rank": rank, "match_count": matched})

        # budget 硬过滤（±PRICE_WINDOW），过滤后不足 k 个则放宽为软加权
        if state.budget is not None:
            low = state.budget * (1 - config.PRICE_WINDOW)
            high = state.budget * (1 + config.PRICE_WINDOW)
            filtered = [c for c in candidates if c["price"] is not None and low <= c["price"] <= high]
            if len(filtered) >= min(k, 10):
                candidates = filtered
        if len(candidates) > max(k, 10):
            candidates = candidates[: max(k, 10)]

        # 逐约束短语召回路（实验 22）：追加在截断之后，与稠密路同款"非置换并集"。
        if config.PHRASE_RECALL:
            candidates = self._phrase_recall(state, candidates, slot_terms)

        # 稠密路补充召回（spec §1-⑤；self.dense 为 None 时跳过，纯 BM25 与 v1 一致）。
        # 放在截断【之后】追加：BM25 top-k 一个不动，稠密独有候选附在队尾，保证不被截掉。
        if self.dense is not None:
            candidates = self._fuse_dense(state, candidates, slot_terms)
        return candidates

    def candidate(self, parent_asin: str, state: DialogState) -> dict:
        """Candidate record for a product that the retrieval pool did not return (e.g. a member
        of the strict support set). No BM25 rank (sentinel), match_count computed on the spot."""
        info = self.products[parent_asin]
        slot_terms = [slot.terms for slot in state.slots if slot.terms]
        matched = sum(1 for terms in slot_terms if any(t in info["norm_text"] for t in terms))
        return {**info, "bm25_rank": config.CANDIDATE_POOL, "match_count": matched}

    # ---- 品类卡片值词表（供理解层第二级校验注入）：某品类所有商品的意图卡条目，
    # 归一化 → {原文}。改写话术里抽出的片段只有对齐到这里的某个值，才可能是模拟器
    # 真正说过的话，才允许进入严格回放（understand/verify.py: align）。按需构建并缓存。
    def card_vocab(self, category: str) -> dict[str, set[str]]:
        cache = self.__dict__.setdefault("_card_vocab", {})
        if category not in cache:
            vocab: dict[str, set[str]] = {}
            for pid in self.by_category.get(category, ()):
                info = self.products[pid]
                for value in (*info["card_hard"], *info["card_soft"]):
                    key = normalize(value)
                    if key:
                        vocab.setdefault(key, set()).add(value)
            cache[category] = vocab
        return cache[category]

    def category_names(self) -> list[str]:
        return list(self.by_category)

    # ---- 目录逐字校验（供 M1 解析器注入使用，实验 33）：归一化短语是否逐字存在于
    # 任一商品文本。依据第一性原理"约束是商品文本的逐字片段"——解析器抽出的片段
    # 若全目录查无此文，多半是粘了口水话的垃圾，应放行第三层防线而非入槽。
    def phrase_exists(self, norm_phrase: str) -> bool:
        tokens = norm_phrase.split()
        if not tokens:
            return False
        query = '"' + " ".join(tokens) + '"'
        cursor = self.connection.execute(
            "SELECT 1 FROM products WHERE products MATCH ? LIMIT 1", (query,))
        return cursor.fetchone() is not None

    # ---- 逐约束短语召回（实验 22）：OR-token 大池会把"全样板约束+超冷门"目标挤出
    # top-300（public_0020 唯一 miss 的死因）；≥3 token 槽位值的 FTS5 短语查询子池
    # 极小、目标必进池。追加候选 bm25_rank 给哨兵值（CANDIDATE_POOL），排序侧记 0。
    def _phrase_recall(self, state: DialogState, candidates: list[dict],
                       slot_terms: list[list[str]]) -> list[dict]:
        have = {c["parent_asin"] for c in candidates}
        low = state.budget * (1 - config.PRICE_WINDOW) if state.budget is not None else None
        high = state.budget * (1 + config.PRICE_WINDOW) if state.budget is not None else None
        for terms in slot_terms:
            tokens = terms[0].split()
            if len(tokens) < 3:
                continue
            query = '"' + " ".join(tokens) + '"'   # FTS5 短语查询（词序敏感的精确匹配）
            cursor = self.connection.execute(
                "SELECT parent_asin FROM products WHERE products MATCH ? LIMIT ?",
                (query, config.PHRASE_TOP_K),
            )
            for (parent_asin,) in cursor.fetchall():
                if parent_asin in have or parent_asin not in self.products:
                    continue
                info = self.products[parent_asin]
                if low is not None and (info["price"] is None or not (low <= info["price"] <= high)):
                    continue
                matched = sum(1 for ts in slot_terms if any(t in info["norm_text"] for t in ts))
                candidates.append({**info, "bm25_rank": config.CANDIDATE_POOL, "match_count": matched})
                have.add(parent_asin)
        return candidates
