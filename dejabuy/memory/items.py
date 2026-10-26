"""Compact product attributes shared by the memory store, the benchmark and the retriever.

The same record shape describes a catalogue product and an out-of-catalogue product from a
user's purchase history (scripts/filter_meta.py writes the latter), so a user profile can be
compared against any candidate attribute by attribute."""

from __future__ import annotations

from dejabuy.belief.retrieve import _COLOR_RE, _MATERIAL_RE, _searchable_text, coarse_category, exact_card

FIELDS = ("parent_asin", "title", "store", "coarse", "department", "price", "rating_number",
          "material", "color", "card")


def _price(value) -> float | None:
    try:
        return float(value) if value not in (None, "", "None") else None
    except (TypeError, ValueError):
        return None


def catalogue_item(product: dict) -> dict:
    """Raw catalogue / metadata product → compact attribute record."""
    text = _searchable_text(product)
    material, color = _MATERIAL_RE.search(text), _COLOR_RE.search(text)
    hard, soft = exact_card(product, material, color)
    details = product.get("details") if isinstance(product.get("details"), dict) else {}
    return {
        "parent_asin": str(product["parent_asin"]),
        "title": str(product.get("title") or "")[:100],
        "store": str(product.get("store") or ""),
        "coarse": coarse_category(product.get("categories") or []),
        "department": str(details.get("Department") or ""),
        "price": _price(product.get("price")),
        "rating_number": int(product.get("rating_number") or 0),
        "material": material.group(1).lower() if material else None,
        "color": color.group(1).lower() if color else None,
        "card": [*hard, *soft],
    }
