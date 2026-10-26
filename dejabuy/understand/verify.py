"""Catalogue verification for extracted fragments.

First principle of the understanding layer: a genuine constraint is a verbatim fragment of
some product's catalogue text. Anything the rules or the LLM pull out of an utterance is
checked here before it may become evidence. Two tiers:

* `in_catalog(fragment)` — the fragment occurs verbatim in at least one product's text. Enough
  for the fragment to become *evidence* (a slot the ranker scores).
* `align(fragment, categories)` — the fragment is, or is contained in, one of the intent-card
  values the reference simulator can actually utter for products of those categories. Only an
  aligned fragment may become a *protocol event* for the strict environment model: the event
  then carries the set of card values it could stand for, and a candidate is consistent when
  the reply it would have given is in that set. A fragment that aligns to nothing stays
  evidence-only, so it can never prune the true target out of the strict replay.

The verifiers are injected by the agent (they are the retriever's FTS5 phrase lookup and its
per-category card vocabulary) so this module has no dependency on the index.
"""

from __future__ import annotations

import re

from dejabuy.understand.normalize import normalize

_JUNK_WORDS = {
    "it", "one", "something", "anything", "that", "this", "them", "those",
    "please", "thanks", "sure", "okay", "ok", "yeah", "yes", "no", "well",
    "hmm", "sorry", "really", "just", "still", "maybe", "actually",
}
_NUMBER_RE = re.compile(r"(\d+(?:\.\d+)?)")
_BUDGET_WORDS = re.compile(r"\$|\bdollars?\b|\bbucks?\b|\bbudget\b|\busd\b", re.I)
_MAX_OPTIONS = 400       # a fragment contained in more card values than this is not a constraint


_catalog_check = None    # Callable[[str], bool] | None，由 agent 注入（retriever.phrase_exists）
_card_vocab = None       # Callable[[str], dict[str, set[str]]] | None：品类 → {归一化卡片值: {原文}}
_category_names = None   # Callable[[], list[str]] | None：全部精确粗品类名


def set_catalog_verifier(check) -> None:
    """注入"归一化短语是否逐字存在于任一商品文本"的校验器（M2 的 FTS 短语查询）。

    第一性原理：约束是商品文本的逐字片段。规则抽出的片段若全目录查无此文，
    多半粘了口水话——不算抽取成功，应放行第三层防线。未注入时一律视为通过（安全降级）。"""
    global _catalog_check
    _catalog_check = check


def set_card_vocab(vocab, categories=None) -> None:
    """注入品类卡片值词表（retriever.card_vocab）与品类名列表（retriever.category_names）。"""
    global _card_vocab, _category_names
    _card_vocab = vocab
    _category_names = categories


def in_catalog(fragment: str) -> bool:
    if _catalog_check is None:
        return True
    try:
        return bool(_catalog_check(normalize(fragment)))
    except Exception:
        return True  # 校验器故障不得改变解析行为


def is_useful(fragment: str) -> bool:
    """过滤掉进了槽位也匹配不到任何商品、反而可能误伤排序的碎片。"""
    cleaned = fragment.strip(" .,;:!?\"'—–")
    if len(cleaned) < 3:
        return False
    words = [w for w in re.findall(r"[a-z0-9%]+", cleaned.lower()) if w]
    if not words:
        return False
    # 全是口水词 → 丢弃
    return any(w not in _JUNK_WORDS for w in words)


# ------------------------------------------------------------------ second tier: alignment
def align(fragment: str, categories: list[str]) -> list[str]:
    """Card values (exact strings) of the given categories that the fragment could stand for.

    Exact normalised equality wins and returns just that value; otherwise every card value
    that contains the fragment as a whole-word phrase; a budget figure matches the card's
    `budget around $X` by number. Empty when the vocabulary is not injected (no alignment
    possible) or nothing matches."""
    if _card_vocab is None or not categories:
        return []
    norm = normalize(fragment)
    if not norm:
        return []
    try:
        vocabs = [_card_vocab(category) or {} for category in categories]
    except Exception:
        return []
    exact: set[str] = set()
    for vocab in vocabs:
        exact.update(vocab.get(norm, ()))
    if exact:
        return sorted(exact)
    found: set[str] = set()
    number = _NUMBER_RE.search(fragment)
    if number and _BUDGET_WORDS.search(fragment):
        target = float(number.group(1))
        for vocab in vocabs:
            for key, values in vocab.items():
                if key.startswith("budget around"):
                    for value in values:
                        figure = re.search(r"\$(\d+(?:\.\d+)?)", value)
                        if figure and abs(float(figure.group(1)) - target) < 1e-6:
                            found.add(value)
        return sorted(found)
    needle = f" {norm} "
    for vocab in vocabs:
        for key, values in vocab.items():
            if needle in f" {key} ":
                found.update(values)
                if len(found) > _MAX_OPTIONS:
                    return []
    if found:
        return sorted(found)
    # A reordered list ("5% Spandex, 20% Rayon, 75% Polyester"): every part contained, any order.
    parts = [normalize(p) for p in re.split(r",|\band\b|;", fragment)]
    parts = [p for p in parts if len(p) >= 3]
    if len(parts) >= 2:
        for vocab in vocabs:
            for key, values in vocab.items():
                padded = f" {key} "
                if all(f" {part} " in padded for part in parts):
                    found.update(values)
                    if len(found) > _MAX_OPTIONS:
                        return []
    if found:
        return sorted(found)
    # The fragment wraps a card value in filler ("a Pull-On closure", "something made of cotton"):
    # the longest card values contained in the fragment.
    padded = f" {norm} "
    best_len = 0
    for vocab in vocabs:
        for key, values in vocab.items():
            if len(key) >= 4 and f" {key} " in padded:
                if len(key) > best_len:
                    found, best_len = set(values), len(key)
                elif len(key) == best_len:
                    found.update(values)
    return sorted(found)


def merge_fragments(fragments: list[str], message: str, categories: list[str]) -> list[str]:
    """Re-join fragments an extractor split apart when the message span from the first to the
    last of a run is itself (normalised) a card value: ["Polyester", "Cotton", "Spandex"] copied
    out of "Polyester,Cotton,Spandex; Machine Wash" becomes the one value again."""
    if _card_vocab is None or len(fragments) < 2 or not categories:
        return fragments
    try:
        vocabs = [_card_vocab(category) or {} for category in categories]
    except Exception:
        return fragments
    keys = set()
    for vocab in vocabs:
        keys.update(vocab)
    lowered = message.lower()
    merged: list[str] = []
    i = 0
    while i < len(fragments):
        best_j, best_span = i, None
        start = lowered.find(fragments[i].lower())
        if start >= 0:
            for j in range(len(fragments) - 1, i, -1):
                end_at = lowered.find(fragments[j].lower(), start)
                if end_at < 0:
                    continue
                span = message[start:end_at + len(fragments[j])]
                if normalize(span) in keys:
                    best_j, best_span = j, span
                    break
        if best_span is not None:
            merged.append(best_span)
            i = best_j + 1
        else:
            merged.append(fragments[i])
            i += 1
    return merged


def find_category(message: str) -> list[str]:
    """Category names that occur verbatim (normalised) in the message — the longest ones. The
    simulator's category phrase is a catalogue name and survives paraphrase, so this beats
    any lead-in regex ("browsing for now — I'm after X" must not yield "now")."""
    if _category_names is None or not message:
        return []
    try:
        names = list(_category_names())
    except Exception:
        return []
    padded = f" {normalize(message)} "
    best: list[str] = []
    best_len = 0
    for name in names:
        key = normalize(name)
        if len(key) < 3 or f" {key} " not in padded:
            continue
        if len(key) > best_len:
            best, best_len = [name], len(key)
        elif len(key) == best_len:
            best.append(name)
    return best


def resolve_category(text: str) -> list[str]:
    """Exact coarse categories an utterance's category phrase could mean: the exact name when
    it is one, else every category name containing it (or contained in it) as a whole-word
    phrase. Empty when the names are not injected or nothing matches."""
    if _category_names is None or not text:
        return []
    try:
        names = list(_category_names())
    except Exception:
        return []
    if text in names:
        return [text]
    norm = normalize(text)
    if not norm:
        return []
    hits = [name for name in names if normalize(name) == norm]
    if hits:
        return hits
    needle = f" {norm} "
    hits = [name for name in names if needle in f" {normalize(name)} " or f" {normalize(name)} " in needle]
    return hits if len(hits) <= 20 else []
