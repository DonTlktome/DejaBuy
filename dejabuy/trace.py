"""Per-turn decision trace: what the agent believed, considered and did.

Enabled with DEJABUY_TRACE=1 (JSON lines to DEJABUY_TRACE_PATH, default trace.jsonl). Each
record carries the top beliefs, the chosen question, the slate width and the memory signals,
so any session can be replayed and debugged without re-running the evaluator.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

_ENABLED = os.environ.get("DEJABUY_TRACE", "0") == "1"
_PATH = Path(os.environ.get("DEJABUY_TRACE_PATH", "trace.jsonl"))


def enabled() -> bool:
    return _ENABLED


def record(**fields) -> None:
    if not _ENABLED:
        return
    with _PATH.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(fields, ensure_ascii=False) + "\n")
