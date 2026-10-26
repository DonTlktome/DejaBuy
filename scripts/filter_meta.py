"""Cut the Amazon Reviews 2023 item-metadata file down to the products our users bought before.

    python scripts/filter_meta.py data/meta_Clothing_Shoes_and_Jewelry.jsonl.gz

Streams the metadata file once (it is ~18 GB uncompressed; nothing is loaded whole) and keeps
only the products listed in data/history_asins.txt (written by scan_sequences.py). For each one
it stores a compact record: title, store, coarse category, price, popularity, the evaluator-style
intent-card values (the exact vocabulary the simulator speaks in), the material / colour
mentions and the first features. Output: data/history_meta.jsonl.gz.
"""

from __future__ import annotations

import argparse
import gzip
import json
import re
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from dejabuy.belief.retrieve import (  # noqa: E402
    _COLOR_RE, _MATERIAL_RE, _searchable_text, coarse_category, exact_card,
)

_ASIN = re.compile(r'"parent_asin":\s*"([^"]+)"')


def compact(product: dict) -> dict:
    text = _searchable_text(product)
    material, color = _MATERIAL_RE.search(text), _COLOR_RE.search(text)
    hard, soft = exact_card(product, material, color)
    price = product.get("price")
    try:
        price = float(price) if price not in (None, "", "None") else None
    except (TypeError, ValueError):
        price = None
    details = product.get("details") if isinstance(product.get("details"), dict) else {}
    return {
        "parent_asin": str(product["parent_asin"]),
        "title": str(product.get("title") or "")[:100],
        "store": str(product.get("store") or ""),
        "coarse": coarse_category(product.get("categories") or []),
        "department": str(details.get("Department") or ""),
        "price": price,
        "rating_number": int(product.get("rating_number") or 0),
        "average_rating": product.get("average_rating"),
        "material": material.group(1).lower() if material else None,
        "color": color.group(1).lower() if color else None,
        "card": [*hard, *soft],
        "features": [str(f)[:80] for f in (product.get("features") or [])[:4]],
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("meta", help="meta_Clothing_Shoes_and_Jewelry.jsonl(.gz)")
    parser.add_argument("--asins", default=str(ROOT / "data" / "history_asins.txt"))
    parser.add_argument("--out", default=str(ROOT / "data" / "history_meta.jsonl.gz"))
    args = parser.parse_args()

    wanted = {line.strip() for line in Path(args.asins).read_text(encoding="utf-8").splitlines() if line.strip()}
    print(f"wanted: {len(wanted):,} products")
    path = Path(args.meta)
    source = gzip.open(path, "rt", encoding="utf-8") if path.suffix == ".gz" else path.open(encoding="utf-8")
    start, kept, seen = time.time(), 0, 0
    with source, gzip.open(args.out, "wt", encoding="utf-8") as out:
        for seen, line in enumerate(source, 1):
            if seen % 500_000 == 0:
                print(f"  {seen / 1e6:.1f}M items scanned, {kept:,} kept, {time.time() - start:.0f}s", flush=True)
            match = _ASIN.search(line)
            if not match or match.group(1) not in wanted:
                continue
            try:
                out.write(json.dumps(compact(json.loads(line)), ensure_ascii=False) + "\n")
                kept += 1
            except (json.JSONDecodeError, KeyError):
                continue
    print(f"done: {seen:,} items scanned, {kept:,} / {len(wanted):,} kept ({time.time() - start:.0f}s) → {args.out}")


if __name__ == "__main__":
    main()
