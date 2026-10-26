"""Scan the Amazon Reviews 2023 rating-only file for purchase sequences that touch our catalogue.

    python scripts/scan_sequences.py data/Clothing_Shoes_and_Jewelry.csv.gz

Input: the 0-core rating-only file (user_id, parent_asin, rating, timestamp; .csv or .csv.gz).
Two streaming passes, so memory stays small even though the file has ~65M rows:

  pass 1  users with at least one interaction on a catalogue product
  pass 2  the full, time-ordered interaction sequence of those users (in- and out-of-catalogue)

Writes
  data/sequences.jsonl      one line per user whose sequence has a catalogue product preceded by
                            at least one earlier interaction: {"user_id", "events": [[asin, rating, ts], ...]}
  data/history_asins.txt    out-of-catalogue products that appear in those histories (what the
                            metadata file would have to be filtered down to, if we need it)
and prints the counts that decide whether the item metadata file is needed at all.
"""

from __future__ import annotations

import argparse
import csv
import gzip
import io
import json
import sys
import time
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def open_rows(path: Path):
    raw = gzip.open(path, "rb") if path.suffix == ".gz" else path.open("rb")
    reader = csv.reader(io.TextIOWrapper(raw, encoding="utf-8", newline=""))
    header = next(reader)
    index = {name: i for i, name in enumerate(header)}
    for need in ("user_id", "parent_asin", "rating", "timestamp"):
        if need not in index:
            sys.exit(f"column {need!r} missing; header is {header}")
    u, a, r, t = index["user_id"], index["parent_asin"], index["rating"], index["timestamp"]
    for row in reader:
        yield row[u], row[a], row[r], row[t]


def catalogue_ids(path: Path) -> set[str]:
    ids: set[str] = set()
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            try:
                ids.add(str(json.loads(line)["parent_asin"]))
            except (json.JSONDecodeError, KeyError):
                continue
    return ids


def progress(n: int, start: float, label: str) -> None:
    if n % 5_000_000 == 0:
        print(f"  {label}: {n / 1e6:.0f}M rows, {time.time() - start:.0f}s", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("ratings", help="Clothing_Shoes_and_Jewelry.csv.gz (0-core rating-only)")
    parser.add_argument("--catalog", default=str(ROOT / "data" / "catalog.jsonl"))
    parser.add_argument("--out", default=str(ROOT / "data"))
    args = parser.parse_args()

    ratings, out = Path(args.ratings), Path(args.out)
    catalog = catalogue_ids(Path(args.catalog))
    print(f"catalogue: {len(catalog):,} products")

    start = time.time()
    users: set[str] = set()
    rows = 0
    for rows, (user, asin, _r, _t) in enumerate(open_rows(ratings), 1):
        if asin in catalog:
            users.add(user)
        progress(rows, start, "pass 1")
    print(f"pass 1: {rows:,} rows, {len(users):,} users touch the catalogue ({time.time() - start:.0f}s)")

    start = time.time()
    sequences: dict[str, list] = {user: [] for user in users}
    for n, (user, asin, rating, ts) in enumerate(open_rows(ratings), 1):
        events = sequences.get(user)
        if events is not None:
            events.append((asin, float(rating or 0), int(ts or 0)))
        progress(n, start, "pass 2")
    print(f"pass 2: done ({time.time() - start:.0f}s)")

    # A usable session: a catalogue product (the target) with at least one earlier interaction (the memory).
    prior_counts: Counter = Counter()       # users by number of interactions before their last catalogue product
    prior_in_catalog: Counter = Counter()   # same, counting only catalogue products in the history
    history_asins: set[str] = set()
    kept = 0
    out.mkdir(parents=True, exist_ok=True)
    with (out / "sequences.jsonl").open("w", encoding="utf-8") as handle:
        for user, events in sequences.items():
            events.sort(key=lambda e: e[2])
            last_target = max((i for i, e in enumerate(events) if e[0] in catalog), default=-1)
            if last_target <= 0:
                continue
            history = events[:last_target]
            prior_counts[min(len(history), 10)] += 1
            prior_in_catalog[min(sum(1 for e in history if e[0] in catalog), 10)] += 1
            history_asins.update(e[0] for e in history if e[0] not in catalog)
            handle.write(json.dumps({"user_id": user, "events": events}) + "\n")
            kept += 1
    (out / "history_asins.txt").write_text("\n".join(sorted(history_asins)) + "\n", encoding="utf-8")

    def at_least(counter: Counter, k: int) -> int:
        return sum(v for key, v in counter.items() if key >= k)

    print(f"\nusers with a catalogue product preceded by ≥1 interaction: {kept:,}")
    for k in (1, 2, 3, 5):
        print(f"  ≥{k} prior interactions: {at_least(prior_counts, k):,} users"
              f"   (≥{k} prior *catalogue* products: {at_least(prior_in_catalog, k):,})")
    print(f"out-of-catalogue products in those histories: {len(history_asins):,} → {out / 'history_asins.txt'}")
    print(f"sequences → {out / 'sequences.jsonl'}")


if __name__ == "__main__":
    main()
