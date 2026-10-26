"""Per-turn latency and peak memory under the official simulator.

    python3 -m bench.latency                     # public set, default configuration
    python3 -m bench.latency --set synthetic/uniform

Wraps the agent so every respond() call is timed (wall clock, single process, no LLM key so the
extraction layer is idle — the rule path the public set exercises), then reports P50 / P95 / P99 /
max per turn, split by regime, plus construction time and peak resident memory.
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


class Timed:
    def __init__(self, inner) -> None:
        self.inner = inner
        self.samples: list[tuple[float, str]] = []

    def reset(self, session_id, profile):
        self.inner.reset(session_id, profile)

    def respond(self, session_id, message, turn, top_k):
        t0 = time.perf_counter()
        out = self.inner.respond(session_id, message, turn, top_k)
        ms = (time.perf_counter() - t0) * 1000
        regime = (self.inner.debug.get(session_id) or {}).get("regime", "?")
        self.samples.append((ms, regime))
        return out


def peak_rss_mb() -> float:
    """Peak resident memory of this process, in MB (Linux/macOS via resource, Windows via psapi)."""
    try:
        import resource
        usage = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        return usage / 1024 if sys.platform != "darwin" else usage / (1024 * 1024)
    except ImportError:  # Windows
        import ctypes
        from ctypes import wintypes

        class Counters(ctypes.Structure):
            _fields_ = [("cb", wintypes.DWORD), ("PageFaultCount", wintypes.DWORD),
                        ("PeakWorkingSetSize", ctypes.c_size_t), ("WorkingSetSize", ctypes.c_size_t),
                        ("QuotaPeakPagedPoolUsage", ctypes.c_size_t), ("QuotaPagedPoolUsage", ctypes.c_size_t),
                        ("QuotaPeakNonPagedPoolUsage", ctypes.c_size_t), ("QuotaNonPagedPoolUsage", ctypes.c_size_t),
                        ("PagefileUsage", ctypes.c_size_t), ("PeakPagefileUsage", ctypes.c_size_t)]

        kernel32, psapi = ctypes.windll.kernel32, ctypes.windll.psapi
        kernel32.GetCurrentProcess.restype = wintypes.HANDLE
        psapi.GetProcessMemoryInfo.argtypes = [wintypes.HANDLE, ctypes.POINTER(Counters), wintypes.DWORD]
        psapi.GetProcessMemoryInfo.restype = wintypes.BOOL
        counters = Counters()
        counters.cb = ctypes.sizeof(Counters)
        if not psapi.GetProcessMemoryInfo(kernel32.GetCurrentProcess(), ctypes.byref(counters), counters.cb):
            return float("nan")
        return counters.PeakWorkingSetSize / (1024 * 1024)


def pct(values: list[float], q: float) -> float:
    values = sorted(values)
    return values[min(len(values) - 1, int(q * len(values)))]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--set", default="public_set", help="public_set or synthetic/uniform, synthetic/purchase")
    parser.add_argument("--level", default="L0", help="rule paraphrase level (bench/paraphrase_rules.py); L0 = as is")
    parser.add_argument("--tag", default="", help="suffix for the golden file, e.g. dense")
    args = parser.parse_args()
    from evaluator.local_evaluator import catalog_index, evaluate
    from dejabuy.agent import DejaBuyAgent

    t0 = time.time()
    agent = DejaBuyAgent(str(ROOT / "data" / "catalog.jsonl"))
    agent.debug_enabled = True
    build_s = time.time() - t0
    rss_after_build = peak_rss_mb()
    catalog_ids, categories, products = catalog_index(ROOT / "data" / "catalog.jsonl")
    samples = [json.loads(line) for line in (ROOT / "data" / f"{args.set}.jsonl").open(encoding="utf-8")]
    inner = agent
    if args.level != "L0":
        from bench.paraphrase_rules import ParaphrasingAgent
        inner = ParaphrasingAgent(agent, args.level)
        inner.debug = agent.debug
    timed = Timed(inner)
    evaluate(timed, samples, catalog_ids, categories, products)
    ms = [m for m, _ in timed.samples]
    import platform
    report = {
        "machine": f"{platform.system()} {platform.machine()}, {__import__('os').cpu_count()} cpu, python {platform.python_version()}",
        "set": args.set, "level": args.level, "dense": agent.retriever.dense is not None,
        "memory_store": agent.memory is not None,
        "turns": len(ms), "build_seconds": round(build_s, 1),
        "rss_mb_after_build": round(rss_after_build), "peak_rss_mb": round(peak_rss_mb()),
        "all": {"p50": round(pct(ms, .5), 1), "p95": round(pct(ms, .95), 1), "p99": round(pct(ms, .99), 1),
                "max": round(max(ms), 1), "mean": round(statistics.mean(ms), 1)},
    }
    for regime in ("strict", "loose"):
        sub = [m for m, r in timed.samples if r == regime]
        if sub:
            report[regime] = {"turns": len(sub), "p50": round(pct(sub, .5), 1), "p95": round(pct(sub, .95), 1)}
    print(json.dumps(report, indent=1))
    name = args.set.replace("/", "_") + ("" if args.level == "L0" else "_" + args.level) + (f"_{args.tag}" if args.tag else "")
    (ROOT / "golden" / f"latency_{name}.json").write_text(json.dumps(report, indent=1), encoding="utf-8")


if __name__ == "__main__":
    main()
