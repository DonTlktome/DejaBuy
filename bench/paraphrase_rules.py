"""Rule-based paraphrase stress test (four levels) against the unmodified evaluator.

L1 rephrases the sentence templates and keeps constraint values verbatim; L2 rewrites short
values; L3 also reorders long spec strings; L4 is colon-free natural speech. These are
deterministic rewrites meant to isolate failure points layer by layer, not a prediction of a
held-out set; the LLM-generated paraphrase set (bench/paraphrase_llm.py) is the external check."""

from __future__ import annotations

import argparse
import json
import random
import re
import statistics
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from evaluator.local_evaluator import catalog_index, evaluate, load_jsonl  # noqa: E402
from dejabuy.agent import DejaBuyAgent as Agent  # noqa: E402

# ---------------------------------------------------------------- 句式改写（L1+）
# 每条 = (识别原模板的正则, 改写模板)。约束原文用 \g<name> 原样搬运，逐字不动。
_PHRASING = [
    # 开场 · browsing
    (re.compile(r"^I'm looking for (?P<cat>.+), but I'm still exploring\.$"),
     "So I'm shopping around for {cat} at the moment — haven't settled on anything yet."),
    # 开场 · buying
    (re.compile(r"^I'm looking for (?P<cat>.+?)\. A key requirement is: (?P<c>.+)\.$"),
     "I'm after {cat}. The one thing I really need: {c}"),
    # 改需求 · 标准句
    (re.compile(r"^Actually, ignore my earlier preference\. What I need is: (?P<v>.+)\.$"),
     "Hold on — scratch what I said earlier. What I actually want is {v}"),
    # 改需求 · 兜底句
    (re.compile(r"^Actually, please ignore my earlier preference\.$"),
     "Hold on, forget what I told you before."),
    # 吐约束
    (re.compile(r"^For that, what matters is: (?P<body>.+)\.$"),
     "On that front, here's what counts for me: {body}"),
    # 问干了
    (re.compile(r"^I don't have an additional preference for (?P<a>\w+)\.$"),
     "Nothing else springs to mind on {a}, sorry."),
    # boundary 挡第一问
    (re.compile(r"^I don't have a preference for (?P<a>\w+); please use your judgment\.$"),
     "No strong feelings about {a} — I'll leave that one to you."),
    # null 惩罚句
    (re.compile(r"^Those options are not quite right yet\. Ask me about one specific attribute\.$"),
     "Hmm, none of those feel right. Could you ask me about one thing in particular?"),
    # 开场 · override（放最后：模式最宽，避免抢在前面几条之前匹配）
    (re.compile(r"^I'm looking for (?P<cat>.+?)\. (?P<rest>.+)$"),
     "I'm after {cat}. {rest}"),
]

# ---------------------------------------------------------------- L4：无冒号自然改写
# 与 L1 同为"约束原文逐字保留"，但去掉所有冒号、把载荷埋进句子中段——专打第二层防线
# （salvage）赖以生存的冒号载荷规则，隔离测量第三层防线（LLM_PARSE 逐字片段抽取）。
# 正则与 _PHRASING 逐条对应，只换模板。
_PHRASING_L4 = [
    (_PHRASING[0][0],
     "So I've been shopping around for {cat} lately, nothing set in stone yet."),
    (_PHRASING[1][0],
     "I'm after {cat} and honestly the dealbreaker for me is {c} more than anything else."),
    (_PHRASING[2][0],
     "Hold on — scratch what I said earlier, {v} is the thing that actually matters to me."),
    (_PHRASING[3][0],
     "Hold on, forget what I told you before."),
    (_PHRASING[4][0],
     "On that front I'd say {body} pretty much covers what counts for me."),
    (_PHRASING[5][0],
     "Nothing else springs to mind on {a}, sorry."),
    (_PHRASING[6][0],
     "No strong feelings about {a} — I'll leave that one to you."),
    (_PHRASING[7][0],
     "Hmm, none of those feel right. Could you ask me about one thing in particular?"),
    (_PHRASING[8][0],
     "I'm after {cat} and {rest}"),
]

# ---------------------------------------------------------------- 约束值改写（L2/L3）
_SPEC_RE = re.compile(r"\d")          # 含数字 = 规格串（成分表/尺寸），LLM 改写时通常原样保留
_SHORT_LIMIT = 25

_VALUE_REWRITES = [
    (re.compile(r"^(?P<x>.+) closure$", re.I), "it fastens with a {x}"),
    (re.compile(r"^Imported$", re.I), "it's an imported one"),
    (re.compile(r"^Machine Wash$", re.I), "I want to be able to machine wash it"),
    (re.compile(r"^(?P<x>cotton|polyester|nylon|leather|wool|silk|rayon|spandex|fabric)$", re.I),
     "something made of {x}"),
    (re.compile(r"^color:\s*(?P<x>\w+)$", re.I), "the colour should be {x}"),
    (re.compile(r"^budget around \$(?P<x>[\d.]+)$", re.I), "I'd like to stay around {x} dollars"),
]


def _rewrite_value(value: str, aggressive: bool) -> str:
    """把一条约束原文换成自然说法。逐字片段被破坏 = ranker 的指纹信号失效。"""
    for pattern, template in _VALUE_REWRITES:
        match = pattern.match(value.strip())
        if match:
            return template.format(**match.groupdict())
    if not aggressive:
        # L2：长规格串（含数字）保留原文，模拟 LLM 通常不动成分表的行为
        if len(value) > _SHORT_LIMIT or _SPEC_RE.search(value):
            return value
    # L3：规格串也重组——逗号分隔项打乱顺序并改连接词
    parts = [p.strip() for p in value.split(",") if p.strip()]
    if len(parts) > 1:
        return "with " + ", ".join(reversed(parts))
    return f"something like {value}"


def _rewrite_body(body: str, aggressive: bool) -> str:
    """吐约束句的载荷是 '; ' 拼接的多条约束。"""
    return ", and ".join(_rewrite_value(p, aggressive) for p in body.split("; "))


def _core(value: str, level: str) -> str:
    """The attribute phrase a careful extractor would copy out of the rewritten value: the
    verbatim value at L1/L4, the kernel of the natural rewrite at L2/L3 ("something made of
    cotton" → "cotton", "it fastens with a Zipper" → "Zipper", reordered spec lists as is)."""
    if level not in ("L2", "L3"):
        return value.rstrip(".")
    aggressive = level == "L3"
    rewritten = _rewrite_value(value.rstrip("."), aggressive)
    for prefix in ("it fastens with a ", "it's an ", "I want to be able to ", "something made of ",
                   "the colour should be ", "I'd like to stay around ", "with ", "something like "):
        if rewritten.startswith(prefix):
            kernel = rewritten[len(prefix):]
            return kernel[:-len(" one")] if prefix == "it's an " and kernel.endswith(" one") else (
                kernel[:-len(" it")] if prefix == "I want to be able to " and kernel.endswith(" it") else kernel)
    return rewritten


def oracle(message: str, level: str) -> dict | None:
    """What a perfect LLM extraction of the paraphrased message would return (same schema as
    `llm_extract.verify_reply`). None for unrecognised templates. Used by `--oracle` to measure
    the pipeline ceiling behind the extraction layer without an API key."""
    text = message.strip()
    for index, (pattern, _template) in enumerate(_PHRASING):
        match = pattern.match(text)
        if not match:
            continue
        f = match.groupdict()
        base = {"act": "other", "opening": None, "category": None, "attribute": None,
                "override": False, "constraints": []}
        if index == 0:
            return {**base, "act": "open", "opening": "exploring", "category": f["cat"]}
        if index == 1:
            return {**base, "act": "open", "opening": "requirement", "category": f["cat"],
                    "constraints": [_core(f["c"], level)]}
        if index == 2:
            return {**base, "act": "override", "override": True, "constraints": [_core(f["v"], level)]}
        if index == 3:
            return {**base, "act": "override", "override": True}
        if index == 4:
            return {**base, "act": "disclose",
                    "constraints": [_core(p, level) for p in f["body"].rstrip(".").split("; ")]}
        if index == 5:
            return {**base, "act": "exhausted", "attribute": f["a"]}
        if index == 6:
            return {**base, "act": "boundary", "attribute": f["a"]}
        if index == 7:
            return {**base, "act": "null"}
        if index == 8:
            return {**base, "act": "open", "opening": "preference", "category": f["cat"],
                    "constraints": [_core(f["rest"], level)]}
    return None


def paraphrase(message: str, level: str) -> str:
    if level == "L0":
        return message
    text = message.strip()
    for pattern, template in (_PHRASING_L4 if level == "L4" else _PHRASING):
        match = pattern.match(text)
        if not match:
            continue
        fields = dict(match.groupdict())
        if level in ("L2", "L3"):
            aggressive = level == "L3"
            if "body" in fields:
                fields["body"] = _rewrite_body(fields["body"], aggressive)
            for key in ("c", "v", "rest"):
                if key in fields and fields[key]:
                    fields[key] = _rewrite_value(fields[key].rstrip("."), aggressive)
        return template.format(**fields)
    return text  # 未识别的句式原样透传


class ParaphrasingAgent:
    """包装真 Agent：只改它看到的 user_message，其余全部透传。

    `oracle=True` 时同时把"完美抽取"结果登记到 `ORACLE`，供被替换的 LLM 层查表——
    测量的是抽取层之后整条流水线（对齐 → 集合值事件 → 严格回放 → 规划）的上限。"""

    def __init__(self, inner, level: str, oracle: bool = False) -> None:
        self.inner = inner
        self.level = level
        self.oracle = oracle

    def reset(self, session_id: str, user_profile: dict) -> None:
        self.inner.reset(session_id, user_profile)

    def respond(self, session_id: str, user_message: str, turn: int, top_k: int) -> dict:
        rewritten = paraphrase(user_message, self.level)
        if self.oracle:
            ORACLE[rewritten] = oracle(user_message, self.level)
        return self.inner.respond(session_id, rewritten, turn, top_k)


ORACLE: dict[str, dict | None] = {}


def install_oracle() -> None:
    """Replace the LLM extraction with the oracle table (verbatim-verified like a real reply)."""
    from dejabuy.understand import llm_extract

    def extract(message: str, turn: int = 1, asked: str | None = None) -> dict | None:
        reply = ORACLE.get(message)
        return None if reply is None else llm_extract.verify_reply(reply, message)

    llm_extract.extract = extract


def stratified(rows: list[dict], n: int, seed: int) -> list[dict]:
    if n <= 0 or n >= len(rows):
        return rows
    buckets: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        buckets[row["scenario_type"]].append(row)
    picked: list[dict] = []
    for name, items in sorted(buckets.items()):
        take = max(1, round(n * len(items) / len(rows)))
        picked += random.Random(f"{seed}:{name}").sample(items, min(take, len(items)))
    picked.sort(key=lambda r: r["sample_id"])
    return picked[:n]


def main() -> None:
    parser = argparse.ArgumentParser(description="改写压力测试")
    parser.add_argument("--catalog", default="data/catalog.jsonl")
    parser.add_argument("--dataset", default="data/public_set.jsonl")
    parser.add_argument("--levels", nargs="+", default=["L0", "L1", "L2", "L3"])
    parser.add_argument("--n", type=int, default=0, help="抽样条数，0 = 全量 200")
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--show", action="store_true", help="打印各档改写样例后退出")
    parser.add_argument("--oracle", action="store_true",
                        help="用完美抽取替代 LLM 层：测抽取层之后的流水线上限（无需 key）")
    args = parser.parse_args()
    if args.oracle:
        install_oracle()

    if args.show:
        examples = [
            "I'm looking for Tees & Blouses Tunics, but I'm still exploring.",
            "I'm looking for Accessories Belts. A key requirement is: leather.",
            "For that, what matters is: polyester; 75% Polyester, 20% Rayon, 5% Spandex.",
            "Actually, ignore my earlier preference. What I need is: 100% Leather.",
            "I don't have a preference for other; please use your judgment.",
        ]
        for text in examples:
            print(f"\n原句  {text}")
            for level in ("L1", "L2", "L3", "L4"):
                print(f"  {level}  {paraphrase(text, level)}")
        return

    samples = stratified(load_jsonl(args.dataset), args.n, args.seed)
    catalog_ids, categories, products = catalog_index(args.catalog)
    print(f"改写压力测试 · {len(samples)} 条会话 · evaluator 未做任何修改"
          + ("（--oracle：LLM 层替换为完美抽取）" if args.oracle else "") + "\n")

    base = None
    print("%-5s %10s %10s %8s %8s   %s" % ("档位", "Score", "HitRate", "MRR", "MTTC", "相对 L0"))
    print("-" * 74)
    for level in args.levels:
        agent = ParaphrasingAgent(Agent(args.catalog), level, oracle=args.oracle)
        result = evaluate(agent, samples, catalog_ids, categories, products)
        score = result["recommended_technical_score"]
        if base is None:
            base = score
        delta = "" if level == args.levels[0] else "%+.4f  (%.0f%%)" % (score - base, 100 * score / base)
        print("%-5s %10.4f %10.3f %8.3f %8.2f   %s" % (
            level, score, result["hit_rate_at_10"], result["mrr"], result["mttc"], delta))

    print("\n档位含义：L1 只改句式（约束原文逐字保留）／ L2 再改短约束 ／ L3 连长规格串也重组")
    print("        ／ L4 无冒号自然改写（约束逐字保留但载荷埋进句中——测 LLM_PARSE 第三层防线）")
    print("注：机械改写，非真 LLM 改写。价值在于分层隔离故障点，不在于精确预测私有集分数。")


if __name__ == "__main__":
    main()
