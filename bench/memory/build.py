"""Build the cross-session memory benchmark from real Amazon purchase sequences.

    python3 -m bench.memory.build          # → data/memory_bench/{train,dev,test}.jsonl, users.jsonl.gz, items.jsonl.gz

Inputs (see scripts/scan_sequences.py and scripts/filter_meta.py):
  data/sequences.jsonl        per-user time-ordered interactions touching the catalogue
  data/history_meta.jsonl.gz  compact attributes of the out-of-catalogue products in those histories
  data/catalog.jsonl          the official 50k-product catalogue

One session per user. The target is the user's last catalogue product that has at least one
earlier interaction; the memory is every interaction strictly before it (time-ordered split,
nothing from the target's own timestamp or later). Users are split train / dev / test by a hash
of the user id, so no user appears in two splits. Popular products would dominate a naive
sample (one T-shirt is the last purchase of 180k users), so each target product appears at most
CAP times per split. The conversation scenario is assigned from the same hash with the public
set's mix (buying 40% / browsing 40% / intent-override 15% / boundary 5%).

Each sample carries a `stratum` for reporting: `same_store` (the user bought from the target's
store before), `same_category` (bought in the target's category, other store), `other`.
Brand-loyal returning customers are only ~6% of a natural sample, so a separate `test_loyal`
split (test-bucket users only, disjoint from `test`) collects `same_store` sessions until it has
--max-loyal of them; it is reported on its own, never mixed into the natural-mix numbers.
"""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from dejabuy.memory.items import catalogue_item  # noqa: E402

SCENARIOS = (("buying", 40), ("browsing", 40), ("intent_override", 15), ("boundary", 5))
SPLITS = (("train", 70), ("dev", 10), ("test", 20))


def _bucket(user_id: str, salt: str) -> int:
    return int(hashlib.sha256(f"{salt}:{user_id}".encode()).hexdigest()[:8], 16) % 100


def _pick(user_id: str, salt: str, table) -> str:
    b, acc = _bucket(user_id, salt), 0
    for name, share in table:
        acc += share
        if b < acc:
            return name
    return table[-1][0]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default=str(ROOT / "data"))
    parser.add_argument("--out", default=str(ROOT / "data" / "memory_bench"))
    parser.add_argument("--cap", type=int, default=3, help="max sessions per target product per split")
    parser.add_argument("--max-train", type=int, default=40000)
    parser.add_argument("--max-dev", type=int, default=1000)
    parser.add_argument("--max-test", type=int, default=2000)
    parser.add_argument("--max-loyal", type=int, default=500)
    args = parser.parse_args()
    data, out = Path(args.data), Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    limits = {"train": args.max_train, "dev": args.max_dev, "test": args.max_test, "test_loyal": args.max_loyal}

    catalog: dict[str, dict] = {}
    with (data / "catalog.jsonl").open(encoding="utf-8") as handle:
        for line in handle:
            product = json.loads(line)
            catalog[str(product["parent_asin"])] = catalogue_item(product)
    print(f"catalogue: {len(catalog):,}")

    light: dict[str, tuple[str, str]] = {a: (c["store"].lower(), c["coarse"]) for a, c in catalog.items()}
    with gzip.open(data / "history_meta.jsonl.gz", "rt", encoding="utf-8") as handle:
        for line in handle:
            rec = json.loads(line)
            light.setdefault(rec["parent_asin"], ((rec.get("store") or "").lower(), rec.get("coarse") or ""))

    def stratum(target: str, history: list) -> str:
        store, coarse = light[target]
        hist = [light[e[0]] for e in history if e[0] in light]
        if store and any(h[0] == store for h in hist):
            return "same_store"
        if any(h[1] == coarse for h in hist):
            return "same_category"
        return "other"

    # Deterministic order independent of file order: sort users by hash, then fill splits.
    users = []
    with (data / "sequences.jsonl").open(encoding="utf-8") as handle:
        for line in handle:
            row = json.loads(line)
            users.append((hashlib.sha256(row["user_id"].encode()).hexdigest(), line))
    users.sort()
    print(f"users with a catalogue purchase after history: {len(users):,}")

    names = [name for name, _ in SPLITS] + ["test_loyal"]
    per_item: dict[str, Counter] = {name: Counter() for name in names}
    samples: dict[str, list] = {name: [] for name in names}
    kept_users: dict[str, list] = {}
    needed_items: set[str] = set()
    for _h, line in users:
        row = json.loads(line)
        uid, events = row["user_id"], row["events"]
        split = _pick(uid, "split", SPLITS)
        if all(len(samples[s]) >= limits[s] for s in names):
            break
        t = max(i for i, e in enumerate(events) if e[0] in catalog)
        if t == 0:
            continue
        target, ts = events[t][0], int(events[t][2])
        history = [e for e in events[:t] if int(e[2]) < ts]
        if not history:
            continue
        group = stratum(target, history)
        if len(samples[split]) >= limits[split]:
            if split != "test" or group != "same_store" or len(samples["test_loyal"]) >= limits["test_loyal"]:
                continue
            split = "test_loyal"
        if per_item[split][target] >= args.cap:
            continue
        per_item[split][target] += 1
        samples[split].append({
            "sample_id": f"mem_{split}_{len(samples[split]):05d}",
            "scenario_type": _pick(uid, "scenario", SCENARIOS),
            "ground_truth": {"parent_asin": target},
            "user_profile": {"user_id": uid, "as_of": ts},
            "history_len": len(history),
            "stratum": group,
        })
        kept_users[uid] = history
        needed_items.update(e[0] for e in history)

    # Item attributes for every product any kept history references.
    items: dict[str, dict] = {a: catalog[a] for a in needed_items if a in catalog}
    with gzip.open(data / "history_meta.jsonl.gz", "rt", encoding="utf-8") as handle:
        for line in handle:
            rec = json.loads(line)
            if rec["parent_asin"] in needed_items and rec["parent_asin"] not in items:
                items[rec["parent_asin"]] = {k: rec.get(k) for k in (
                    "parent_asin", "title", "store", "coarse", "department", "price", "rating_number",
                    "material", "color", "card")}
    missing = len(needed_items) - len(items)

    for split, rows in samples.items():
        with (out / f"{split}.jsonl").open("w", encoding="utf-8") as handle:
            for s in rows:
                handle.write(json.dumps(s) + "\n")
        mix = Counter(s["stratum"] for s in rows)
        scen = Counter(s["scenario_type"] for s in rows)
        print(f"{split}: {len(rows):,} sessions, {len(per_item[split]):,} distinct targets · "
              f"strata {dict(mix)} · scenarios {dict(scen)}")

    with gzip.open(out / "users.jsonl.gz", "wt", encoding="utf-8") as handle:
        for uid, history in kept_users.items():
            handle.write(json.dumps({"user_id": uid, "events": history}) + "\n")
    with gzip.open(out / "items.jsonl.gz", "wt", encoding="utf-8") as handle:
        for rec in items.values():
            handle.write(json.dumps(rec, ensure_ascii=False) + "\n")
    print(f"users: {len(kept_users):,} · history items: {len(items):,} (missing attributes: {missing:,}) → {out}")


if __name__ == "__main__":
    main()
