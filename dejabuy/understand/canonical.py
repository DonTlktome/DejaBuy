"""Typed protocol events from unfamiliar phrasing.

The strict environment model needs the conversation as *events* (open / disclose / exhausted /
boundary / override / null). The template fast path produces them directly; when the shopper
does not use the reference phrasing, the rule salvage (and, behind it, the LLM extraction)
still recovers the category, the speech act and the constraint fragments — this module turns
that into the same typed events, with a two-tier guarantee:

* fragments the *rules* lifted out of a colon payload and verified verbatim in the catalogue
  are taken as the simulator's exact words (`value` / `body`, exact match on replay);
* fragments the *LLM* copied out of free speech are aligned to the intent-card vocabulary of
  the opening categories (`verify.align`) and the event carries the resulting `options` — a
  candidate is consistent when the reply it would have given is one of them. Set-valued, so
  a trimmed or partial fragment ("Zipper" for "Zipper closure", "40 dollars" for
  "budget around $40") narrows the support set without pruning the true target.

A fragment that aligns to nothing cannot be a simulator reply the strict model can vouch
for; such turns stay `unknown`, the fragment remains soft evidence for the ranker, and the
decision layer falls back to the loose regime for the rest of the session.
"""

from __future__ import annotations

import re

from dejabuy.env.protocol import ATTRIBUTES
from dejabuy.state import DialogState
from dejabuy.understand import verify

_ATTR_RE = re.compile(r"\b(" + "|".join(ATTRIBUTES) + r")\b", re.I)
_EXHAUSTED = re.compile(
    r"\b(?:nothing (?:else|more)|no (?:other|additional|further)|don't have (?:any )?(?:more|other|additional)|"
    r"that's (?:all|it))\b.{0,40}?\b(?P<attr>" + "|".join(ATTRIBUTES) + r")\b", re.I | re.S)
_BOUNDARY = re.compile(
    r"\b(?:no (?:strong )?(?:feelings|preference|opinion)|don't (?:really )?(?:mind|care)|up to you|"
    r"leave (?:that|it|this)(?: one)? to you|your (?:call|judgment|judgement))\b", re.I | re.S)
_NULL_PENALTY = re.compile(r"\b(?:none of (?:those|these)|not (?:quite )?right|ask me about one)\b", re.I)
_BUYING_HINT = re.compile(r"\b(?:need|require|must|essential|dealbreaker|key requirement)\b", re.I)


def snapshot(state: DialogState) -> dict:
    return {"category": state.category, "values": list(state.constraint_values()),
            "scenario": state.scenario, "exhausted": set(state.exhausted)}


def speech_act(state: DialogState, message: str, turn: int) -> dict | None:
    """Replies that carry no constraint payload: exhausted / boundary / null-penalty.

    Handled before the salvage so their wording is never mistaken for a constraint."""
    if turn <= 1:
        return None
    if _NULL_PENALTY.search(message) and not _EXHAUSTED.search(message):
        return {"type": "null"}
    match = _EXHAUSTED.search(message)
    if match:
        return _exhausted(state, match.group("attr").lower())
    if _BOUNDARY.search(message):
        found = _ATTR_RE.search(message)
        attr = found.group(1).lower() if found else (state.last_ask or "other")
        return _boundary(state, attr)
    return None


def _exhausted(state: DialogState, attr: str) -> dict:
    state.exhausted.add(attr)
    if attr == "other":
        state.all_disclosed = True
    return {"type": "exhausted", "ask": attr}


def _boundary(state: DialogState, attr: str) -> dict:
    state.scenario = "boundary"
    return {"type": "boundary", "ask": attr}


def _tail_value(message: str, category: str) -> str | None:
    """Text after the category mention when it is itself a catalogue phrase — the shape of
    the override opening (`… {category}. {old value}`)."""
    idx = message.lower().find(category.lower())
    if idx < 0:
        return None
    tail = message[idx + len(category):].strip().lstrip(".,;:!?—– ").strip()
    tail = re.sub(r"^(?:and|the|with)\s+", "", tail, flags=re.I).rstrip(" .")
    if len(tail) >= 3 and verify.in_catalog(tail):
        return tail
    return None


def resolve_categories(state: DialogState, message: str | None = None) -> list[str]:
    """Exact coarse categories the session is about; cached on the state.

    At the opening the category phrase found verbatim in the message wins (`verify.find_category`:
    it is a catalogue name and survives any paraphrase); the salvaged / LLM phrase is used only
    when nothing is found verbatim."""
    if state.categories:
        return state.categories
    if message is not None:
        found = verify.find_category(message)
        if found:
            if state.category not in found:
                state.category = found[0]
            state.categories = found
            return found
    if state.category:
        state.categories = verify.resolve_category(state.category)
    return state.categories


def _options(values: list[str], state: DialogState, message: str) -> list[list[str]] | None:
    """Card-value option sets for this turn's fragments (split fragments re-joined first); None
    when any fragment aligns to nothing."""
    categories = resolve_categories(state)
    values = verify.merge_fragments(values, message, categories)
    out: list[list[str]] = []
    for value in values:
        aligned = verify.align(value, categories)
        if not aligned:
            return None
        out.append(aligned)
    return out


def infer_event(state: DialogState, message: str, turn: int, before: dict, llm: dict | None = None,
                complete: bool = False) -> dict:
    """Typed event for an utterance the templates did not recognise, after the salvage ran.

    Every value here — whether the rule salvage or the LLM lifted it out — is aligned to the
    category's card vocabulary and the event carries the option set (`understand/verify.py`);
    only the template fast path (rules.py) emits exact values. `llm` is the verified extraction
    when the third layer answered this turn, None when the rules alone changed the state;
    `complete` marks a colon payload the rules lifted whole (every fragment catalogue-verified):
    the simulator's reply is then exactly these values, and the replay may demand a one-to-one
    match instead of "explainable"."""
    act = llm.get("act") if llm else None

    # Speech acts the LLM labelled (the regex pass above runs first and handles the rest).
    if act == "null" and turn > 1:
        return {"type": "null"}
    if act == "exhausted" and turn > 1:
        return _exhausted(state, llm.get("attribute") or state.last_ask or "other")
    if act == "boundary" and turn > 1:
        return _boundary(state, llm.get("attribute") or state.last_ask or "other")

    # Values added this turn. A slot that is (nearly) the whole utterance is the rules' last-resort
    # "split the sentence" fallback, not a constraint: it must never become an event value — with
    # containment alignment it could otherwise pick up any card value mentioned in passing.
    limit = len(message.strip()) - 8
    new_values = [v for v in state.constraint_values() if v not in before["values"] and len(v.strip()) < limit]

    if state.scenario == "override" and before["scenario"] != "override":
        hard_new = [s.value for s in state.slots if s.hard and s.value in new_values] or new_values
        if not hard_new:
            return {"type": "override", "value": None}
        options = _options(hard_new[:1], state, message)
        if options is None:
            return {"type": "unknown"}
        return {"type": "override", "options": options[0]}

    opening = turn == 1 or (not before["category"] and state.category)
    if opening:
        categories = resolve_categories(state, message)
        if not categories:
            return {"type": "unknown"}
        value = None if llm else _tail_value(message, state.category)
        value = value or (new_values[0] if new_values else None)
        base = {"type": "open", "category": state.category, "categories": categories}
        if value is None:
            if llm and llm.get("opening") == "preference":
                return {"type": "unknown"}      # a preference was stated but nothing usable copied out
            state.opening = "browsing"
            return {**base, "scenario": "browsing"}
        options = _options([value], state, message)
        if options is None:
            return {"type": "unknown"}
        hint = llm.get("opening") if llm else None
        if hint == "requirement":
            scenario = "buying"
        elif hint == "preference":
            scenario = "override_open"
        elif _BUYING_HINT.search(message):
            scenario = "buying"
        else:
            scenario = "any" if llm else "override_open"
        if scenario != "override_open":
            for slot in state.slots:
                if slot.value == value:
                    slot.hard = True
        # `any` (requirement or preference, undecided) is scored as a buying opening: hits count
        # from turn one, so shown candidates are refuted — the conservative reading.
        state.opening = "override_open" if scenario == "override_open" else "buying"
        return {**base, "scenario": scenario, "options": options[0]}

    if new_values and turn > 1:
        options = _options(new_values, state, message)
        if options is None:
            return {"type": "unknown"}
        event = {"type": "disclose", "ask": state.last_ask, "options": options}
        if complete and llm is None:
            event["complete"] = True
        return event
    return {"type": "unknown"}
