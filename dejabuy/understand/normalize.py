"""Text normalisation shared by the understanding and belief layers.

`normalize()` lower-cases and strips punctuation so verbatim fragments can be matched as
substrings; `constraint_terms()` turns one disclosed value into the ordered list of search
terms the evidence scorer consumes."""

from __future__ import annotations

import re

from dejabuy import config

_NON_ALNUM_RE = re.compile(r"[^a-z0-9]+")


def normalize(text: str) -> str:
    """小写 + 非字母数字折叠为单空格。约束原文和商品全文都过这个，子串匹配才稳。"""
    return _NON_ALNUM_RE.sub(" ", text.lower()).strip()


def constraint_terms(value: str) -> list[str]:
    """一条约束原文 → 归一化检索词列表（判别力降序）。空列表 = 本槽不参与文本匹配。"""
    full = normalize(value)
    if not full or full.startswith("budget"):
        return []
    # 评测器合成的 "color: {c}" 是虚构键（商品文本里通常没有 "color {c}" 连写），
    # 只留颜色词本身；details 原生的 "Color: Blue" 同样适用。逐字对齐打样实现，
    # 冒号前带空格等其他形态不剥（它们的全串在商品文本里逐字存在，剥了反而丢判别力）
    if value.lower().startswith("color:"):
        stripped = normalize(value.split(":", 1)[1])
        terms = [stripped] if stripped else []
    else:
        terms = [full]
    # 逗号分片变体（C-T8，实验 16 系；FRAGMENT_WEIGHT=0 时关闭 = 消融口径，
    # 权重数值本身不再使用——消费端 max-len 计权，整串命中时分片自然让位）
    if terms and config.FRAGMENT_WEIGHT:
        parts = [p.strip() for p in value.split(",")]
        if len(parts) >= 2:
            for part in parts:
                piece = normalize(part)
                if len(piece) >= 5 and piece not in terms:
                    terms.append(piece)
    return terms
