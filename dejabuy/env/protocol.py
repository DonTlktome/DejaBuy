"""Strict environment model: replay the conversation against each candidate's hidden intent
card exactly as the reference simulator would have generated it.

The simulator is deterministic given the target product: its card is (hard[:2], soft[:2]) built
from the product's own fields, and every reply to a question is a pure function of the card,
the set of already-disclosed values and the attribute asked. So for any candidate product we
can compute the reply the shopper *would* have given at every turn and compare it with what
was actually said. A candidate whose replayed transcript matches is *consistent*; the set of
consistent products in the opening category is the support set the decision layer plans over.

No evaluation labels are used anywhere here: the card is reconstructed from catalogue fields
only (see `belief/retrieve.py: exact_card`), and the dialogue events come from the
understanding layer.
"""

from __future__ import annotations

import re

ATTRIBUTES = ("category", "material", "color", "size", "style", "brand", "budget", "feature", "use_case", "other")
EXHAUSTED = "<none>"           # reply key when the shopper has nothing left for that attribute
OVERRIDE_FALLBACK_OLD = "I prefer a different style."
OVERRIDE_FALLBACK_NEW = "Please prioritize the target requirements."

_MATERIALS = ("cotton", "polyester", "nylon", "leather", "wool", "spandex", "silk", "rayon", "fabric")
_BUDGET_RE = re.compile(r"(?:\$|<=|under)\s*\d")


def classify_constraint(value: str) -> str:
    """Mirror of the simulator's attribute bucket for one constraint string."""
    lowered = value.lower()
    if "budget" in lowered or _BUDGET_RE.search(lowered):
        return "budget"
    if any(material in lowered for material in _MATERIALS):
        return "material"
    if any(word in lowered for word in ("color", "black", "white", "blue", "red", "pink", "green")):
        return "color"
    if any(word in lowered for word in ("size", "sizing", "width", "wide", "narrow")):
        return "size"
    if any(word in lowered for word in ("department", "style", "fit", "sleeve", "neck")):
        return "style"
    if any(word in lowered for word in ("hiking", "running", "gym", "winter", "outdoor", "work")):
        return "use_case"
    return "feature"


def card_values(info: dict) -> list[str]:
    """hard + soft constraints in simulator order (duplicates possible when soft falls back to hard[:1])."""
    return [*info["card_hard"], *info["card_soft"]]


def predicted_reply(info: dict, disclosed: set[str], ask: str | None) -> list[str]:
    """Values the shopper would disclose for `ask` given what is already disclosed (≤ 2)."""
    attribute = ask if ask in ATTRIBUTES else "other"
    matches = [v for v in card_values(info)
               if v not in disclosed and (attribute == "other" or classify_constraint(v) == attribute)]
    return matches[:2]


def reply_key(info: dict, disclosed: set[str], ask: str | None) -> str:
    """The visible reply string class for partitioning candidates by what they would answer."""
    matches = predicted_reply(info, disclosed, ask)
    return "; ".join(matches) if matches else EXHAUSTED


def override_values(info: dict) -> tuple[str, str]:
    """(old_value, new_value) the simulator uses for an intent-override session on this product."""
    hard, soft = info["card_hard"], info["card_soft"]
    old = soft[-1] if soft else OVERRIDE_FALLBACK_OLD
    new = hard[0] if hard else OVERRIDE_FALLBACK_NEW
    return old, new


def _accepts(event: dict, key: str, actual: str) -> bool:
    """Exact (`key`) or set-valued (`options`) event value against what the simulator would say."""
    options = event.get("options")
    if options is not None:
        return actual in options
    return event.get(key) == actual


def _match_options(matches: list[str], options: list[list[str]]) -> bool:
    """min(len(options), len(matches)) option sets are claimed by distinct predicted values, in
    any order. Extra predicted values are allowed (the extraction may have missed one) and so
    are extra option sets (the extraction may have split one value into several); the strict
    model only ever asks that what was recognised is explainable by the candidate."""
    need = min(len(options), len(matches))
    if need == 0:
        return not options or not matches
    return _assign(matches, options, need, 0, set())


def _assign(matches: list[str], options: list[list[str]], need: int, index: int, used: set[int]) -> bool:
    if need == 0:
        return True
    if index >= len(options) or len(options) - index < need:
        return False
    for m, value in enumerate(matches):
        if m not in used and value in options[index]:
            used.add(m)
            if _assign(matches, options, need - 1, index + 1, used):
                return True
            used.discard(m)
    return _assign(matches, options, need, index + 1, used)


def replay(info: dict, events: list[dict]) -> tuple[bool, set[str]]:
    """Replay the recognised events against one candidate.

    Returns (consistent, disclosed): `disclosed` is the simulator's disclosed-set for this
    candidate after the transcript, needed to predict its next reply.

    Events carry either the simulator's exact words (`value` / `body`, template and rule
    paths) or `options` — the card values an LLM-extracted fragment could stand for
    (`understand/canonical.py`). The latter is a weaker likelihood: it keeps every candidate
    whose reply is in the set, never pruning the true target over a trimmed fragment."""
    disclosed: set[str] = set()
    for event in events:
        kind = event.get("type")
        if kind == "open":
            categories = event.get("categories") or [event.get("category")]
            if info.get("coarse_exact") not in categories:
                return False, disclosed
            scenario = event.get("scenario")
            hard = info["card_hard"]
            if scenario == "buying":
                if not hard or not _accepts(event, "value", hard[0]):
                    return False, disclosed
                disclosed.add(hard[0])
            elif scenario == "override_open":
                old, _new = override_values(info)
                if not _accepts(event, "value", old):
                    return False, disclosed
            elif scenario == "any":            # requirement or preference, the extraction could not tell
                old, _new = override_values(info)
                if hard and _accepts(event, "value", hard[0]):
                    disclosed.add(hard[0])
                elif not _accepts(event, "value", old):
                    return False, disclosed
            # browsing (and the boundary scenario, which opens the same way) constrains nothing more
        elif kind == "disclose":
            matches = predicted_reply(info, disclosed, event.get("ask"))
            if not matches:
                return False, disclosed
            options = event.get("options")
            if options is not None:
                if event.get("complete") and len(options) != len(matches):
                    return False, disclosed       # the payload was lifted whole: same number of values
                if not _match_options(matches, options):
                    return False, disclosed
            elif "; ".join(matches) != event.get("body"):
                return False, disclosed
            disclosed.update(matches)
        elif kind == "exhausted":
            if predicted_reply(info, disclosed, event.get("ask")):
                return False, disclosed
        elif kind == "override":
            _old, new = override_values(info)
            if event.get("options") is not None or event.get("value") is not None:
                if not _accepts(event, "value", new):
                    return False, disclosed
                disclosed.add(new)
        elif kind in ("boundary", "null"):
            continue
        else:                       # unknown utterance: the strict model cannot vouch for anything
            return False, disclosed
    return True, disclosed


def recognised(events: list[dict]) -> bool:
    """True when every utterance so far was a recognised protocol event and the opening is known."""
    return bool(events) and events[0].get("type") == "open" and all(e.get("type") != "unknown" for e in events)


def support(products: dict[str, dict], candidate_ids: list[str], events: list[dict]) -> dict[str, set[str]]:
    """Consistent candidates among `candidate_ids` -> their simulator disclosed-set."""
    out: dict[str, set[str]] = {}
    for pid in candidate_ids:
        info = products.get(pid)
        if info is None:
            continue
        ok, disclosed = replay(info, events)
        if ok:
            out[pid] = disclosed
    return out
