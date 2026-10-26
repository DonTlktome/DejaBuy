"""Soft-preference lexicon for the organiser-provided aggregate profile.

Kept for the cold-start experiments: the profile's `preference_tags` is a closed nine-value
set; a permutation test showed the aggregate profile carries no target information, so this
lexicon is not on the scoring path. Real personalisation comes from purchase history
(memory/episodic.py, memory/semantic.py)."""

from __future__ import annotations

TAG_KEYWORDS: dict[str, tuple[str, ...]] = {
    "fit": ("fit", "true to size", "relaxed fit", "slim fit", "snug"),
    "material": ("material", "fabric", "cotton", "leather"),
    "comfort": ("comfortable", "comfort", "soft", "cushioned", "cozy"),
    "style": ("stylish", "fashion", "trendy", "classic", "casual"),
    "durability": ("durable", "sturdy", "heavy duty", "long lasting"),
    "performance": ("breathable", "moisture wicking", "quick dry", "athletic"),
    "warmth": ("warm", "insulated", "thermal", "fleece"),
    "weather": ("waterproof", "water resistant", "windproof"),
    # "general shopping" 及私有集可能出现的未知 tag 不在表里 → profile_soft_terms 静默返回 []
}


def profile_soft_terms(profile: dict | None, limit: int = 8) -> list[str]:
    """user_profile → 去重的软偏好关键词列表（供检索查询扩展 / 排序软加权）。"""
    tags = (profile or {}).get("preference_tags") or []
    seen: list[str] = []
    for tag in tags:
        for keyword in TAG_KEYWORDS.get(str(tag), ()):
            if keyword not in seen:
                seen.append(keyword)
    return seen[:limit]
