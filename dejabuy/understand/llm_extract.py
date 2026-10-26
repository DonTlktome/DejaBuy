"""Third understanding layer: LLM structured extraction with verbatim verification.

Only reached when the template fast path and the rule salvage both fail. The model labels
the speech act of the turn and may only *copy* fragments out of the message — every value
is checked to be a contiguous substring of the utterance (after normalisation) and discarded
otherwise. Whether a verified fragment may enter the strict environment model is decided one
layer up (`canonical.py`: catalogue alignment); here nothing is trusted beyond "it was said".
Two consecutive failures trip a breaker for the rest of the process so an offline run behaves
exactly like a rules-only run.
"""

from __future__ import annotations

import json

from dejabuy import config
from dejabuy.env.protocol import ATTRIBUTES
from dejabuy.llm import client as llm_client
from dejabuy.understand.normalize import normalize
from dejabuy.understand.verify import is_useful

ACTS = ("open", "disclose", "exhausted", "boundary", "override", "null", "other")
OPENINGS = ("requirement", "preference", "exploring")

_PARSE_SYSTEM = (
    "You label one turn of a shopper talking to a shopping assistant and copy out the product "
    "constraints it contains. Reply with JSON only:\n"
    '{"act": "open"|"disclose"|"exhausted"|"boundary"|"override"|"null"|"other", '
    '"opening": "requirement"|"preference"|"exploring"|null, '
    '"category": string|null, "attribute": string|null, "override": boolean, '
    '"constraints": [string, ...]}\n'
    "act: open = the shopper states what product type they are looking for (first turn); "
    "disclose = answers the assistant's question with one or more concrete constraints; "
    "exhausted = says they have nothing more to add about the asked attribute; "
    "boundary = says they have no preference and leave it to the assistant; "
    "override = drops an earlier preference and states a new requirement; "
    "null = says the shown options are wrong and asks to be questioned about one attribute; "
    "other = none of these.\n"
    "opening (only when act=open): requirement = a must-have is stated; preference = a "
    "mere preference or remark is stated; exploring = still browsing, nothing stated.\n"
    "category: the product type being sought, copied verbatim from the message, else null. "
    "attribute: for exhausted/boundary, the attribute talked about, one of "
    + ", ".join(a for a in ATTRIBUTES if a != "category") + ", else null.\n"
    "constraints: every concrete product constraint in the message (materials, specs, "
    "features, colours, sizes, budget figures). Each MUST be an exact contiguous substring "
    "copied character-for-character from the message, trimmed to the attribute phrase itself "
    "(no 'made of', 'I want', 'something like'); never rephrase, translate or merge; at most "
    "4; empty list if none. override: true only if the shopper discards an earlier preference."
)


_failures = 0  # consecutive failures; >= 2 trips the breaker for the process lifetime


def extract(message: str, turn: int = 1, asked: str | None = None) -> dict | None:
    """Return the labelled, verbatim-verified extraction for one utterance, or None when the
    layer is disabled / tripped / failed.

    {"act", "opening", "category", "attribute", "override", "constraints"} — `constraints`
    and `category` are substrings of the message; `attribute` is a protocol attribute."""
    global _failures
    if not config.LLM_PARSE or _failures >= 2:
        return None
    user = json.dumps({"turn": turn, "assistant_asked_about": asked, "message": message}, ensure_ascii=False)
    reply = llm_client.chat_json(_PARSE_SYSTEM, user, max_tokens=240, timeout=config.LLM_PARSE_TIMEOUT)
    if reply is None:
        _failures += 1
        return None
    _failures = 0
    return verify_reply(reply, message)


def verify_reply(reply: dict, message: str) -> dict:
    """Keep only what can be checked: acts from the enum, fragments verbatim in the message."""
    norm_message = normalize(message)
    act = reply.get("act")
    opening = reply.get("opening")
    attribute = reply.get("attribute")
    result = {
        "act": act if act in ACTS else "other",
        "opening": opening if opening in OPENINGS else None,
        "attribute": attribute.lower() if isinstance(attribute, str) and attribute.lower() in ATTRIBUTES else None,
        "override": reply.get("override") is True or act == "override",
        "category": None,
        "constraints": [],
    }
    category = reply.get("category")
    if isinstance(category, str):
        category = category.strip(" .,;:!?\"'—–")
        if is_useful(category) and normalize(category) in norm_message:
            result["category"] = category
    constraints = reply.get("constraints")
    if isinstance(constraints, list):
        for item in constraints[:4]:
            if not isinstance(item, str):
                continue
            fragment = item.strip().strip(" .,;:!?\"'—–")
            if not is_useful(fragment) or normalize(fragment) not in norm_message:
                continue
            if fragment not in result["constraints"]:
                result["constraints"].append(fragment)
    return result


def reset_breaker() -> None:
    global _failures
    _failures = 0
