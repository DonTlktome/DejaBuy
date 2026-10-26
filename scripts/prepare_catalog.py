"""Verify the catalogue archive's SHA-256 and extract data/catalog.jsonl (run once)."""

from __future__ import annotations

import argparse
import gzip
import hashlib
import shutil
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def expected_catalog_checksum(checksum_file: Path) -> str | None:
    if not checksum_file.exists():
        return None
    for line in checksum_file.read_text(encoding="utf-8").splitlines():
        parts = line.split()
        if len(parts) >= 2 and parts[-1].lstrip("*") == "catalog.jsonl.gz":
            return parts[0].lower()
    return None


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description="Verify and extract catalog.jsonl.gz")
    parser.add_argument("--force", action="store_true", help="replace an existing extracted catalog")
    args = parser.parse_args()

    archive = ROOT / "data" / "catalog.jsonl.gz"
    destination = ROOT / "data" / "catalog.jsonl"
    checksum_file = ROOT / "data" / "SHA256SUMS"
    if not archive.exists():
        raise SystemExit(f"Missing catalog archive: {archive}")
    if destination.exists() and not args.force:
        print(f"Catalog already exists: {destination}")
        return

    expected = expected_catalog_checksum(checksum_file)
    if expected:
        actual = sha256(archive)
        if actual != expected:
            raise SystemExit(f"Checksum mismatch for {archive.name}: expected {expected}, got {actual}")
        print("Checksum verified.")

    destination.parent.mkdir(parents=True, exist_ok=True)
    with gzip.open(archive, "rb") as source, destination.open("wb") as target:
        shutil.copyfileobj(source, target)
    print(f"Catalog extracted to {destination}")


if __name__ == "__main__":
    main()
