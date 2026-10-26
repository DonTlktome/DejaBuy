"""Loose environment model: a candidate is consistent with the conversation when every
disclosed constraint occurs verbatim in its catalogue text. Used as the fallback likelihood
when the strict protocol replay (env/protocol.py, week 2) cannot recognise the transcript.
"""

from __future__ import annotations

from dejabuy.state import DialogState


def consistent(candidate: dict, state: DialogState) -> bool:
    text = candidate.get("norm_text", "")
    return all(any(t in text for t in slot.terms) for slot in state.slots if slot.terms)
