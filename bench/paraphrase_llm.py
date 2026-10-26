"""LLM-generated paraphrase set (three levels) against the unmodified evaluator.

The rule rewrites in `paraphrase_rules.py` isolate failure points; this is the external check:
a model rewrites every utterance the simulated shopper produces, the agent only ever sees
the rewrite, the evaluator is untouched.

    P1  句式    sentence rephrased, every constraint value and the category kept verbatim
    P2  短约束  natural phrasing; short attribute values may be reworded, specification
                strings (anything with digits) and the category kept verbatim
    P3  口语    casual spoken English, no colons or lists; anything may be reworded except
                numbers / percentages and the category

Every rewrite is cached in `bench/paraphrases/<level>.jsonl` (source → rewrite), so the set is
fixed once generated and the benchmark replays without a key. P1 rewrites are verified (the
protected spans must survive); a rewrite that loses them is retried once at temperature 0 and
otherwise replaced by the rule rewrite of the same level, flagged `fallback`.

    python3 bench/paraphrase_llm.py --prefill            # paraphrase the L0 transcript in parallel
    python3 bench/paraphrase_llm.py                      # P1–P3, LLM extraction on
    LLM_PARSE=0 python3 bench/paraphrase_llm.py          # same set, rules-only parsing (ablation)
"""

from __future__ import annotations

import argparse
import json
import sys
import threading
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from evaluator.local_evaluator import catalog_index, evaluate, load_jsonl  # noqa: E402
from bench.paraphrase_rules import _PHRASING, paraphrase as rule_paraphrase, stratified  # noqa: E402
from dejabuy import config  # noqa: E402
from dejabuy.agent import DejaBuyAgent as Agent  # noqa: E402
from dejabuy.llm import client as llm_client  # noqa: E402
from dejabuy.understand.normalize import normalize  # noqa: E402

LEVELS = ("P1", "P2", "P3")
_RULE_EQUIVALENT = {"P1": "L1", "P2": "L2", "P3": "L4"}
CACHE_DIR = ROOT / "bench" / "paraphrases"

_SYSTEM = (
    "You rewrite one line that a shopper says to a shopping assistant, the way a different "
    "person would say exactly the same thing. Keep every piece of information: the product "
    "type, each requirement or preference, and what the line is doing (stating what they are "
    "looking for, answering a question, saying they have nothing more to add, saying they have "
    "no preference, changing their mind, or rejecting the shown options). Never add "
    "information, never drop any, never answer as the assistant. One or two sentences. "
    'Reply with JSON only: {"rewrite": string}.'
)
_LEVEL_RULES = {
    "P1": "Change the sentence structure and wording freely, but copy these phrases exactly, "
          "character for character, including capitalisation and punctuation: {protected}. "
          "Keep the product type phrase exactly as written.",
    "P2": "Say it naturally. Short attribute phrases may be reworded into everyday language "
          "(e.g. a material, a colour, a closure type, a care instruction), but copy these "
          "exactly, character for character: {protected}. Keep the product type phrase exactly "
          "as written.",
    "P3": "Casual spoken English, like a chat message: no colons, no lists, contractions are "
          "fine. Everything may be reworded, except keep every number and percentage and copy "
          "the product type phrase exactly as written: {protected}.",
}
_STRICT_SUFFIX = " Your previous rewrite lost one of the required phrases; this time keep them verbatim."


def _fields(message: str) -> dict | None:
    text = message.strip()
    for index, (pattern, _template) in enumerate(_PHRASING):
        match = pattern.match(text)
        if match:
            return {"index": index, **match.groupdict()}
    return None


def _protected(fields: dict, level: str) -> list[str]:
    """Spans that must survive the rewrite at this level."""
    values: list[str] = []
    if fields.get("body"):
        values += [p for p in fields["body"].rstrip(".").split("; ") if p]
    for key in ("c", "v", "rest"):
        if fields.get(key):
            values.append(fields[key].rstrip("."))
    if level == "P2":
        values = [v for v in values if any(ch.isdigit() for ch in v)]
    if level == "P3":
        values = []
    if fields.get("cat"):
        values.append(fields["cat"])
    return values


def _survives(rewrite: str, protected: list[str]) -> bool:
    norm = normalize(rewrite)
    return all(normalize(p) in norm for p in protected)


class ParaphraseSet:
    def __init__(self, level: str) -> None:
        self.level = level
        self.path = CACHE_DIR / f"{level}.jsonl"
        self.cache: dict[str, dict] = {}
        self.new = 0
        self.missing = 0          # utterances served by the rule rewrite because no key was set
        self._lock = threading.Lock()
        if self.path.exists():
            for line in self.path.read_text(encoding="utf-8").splitlines():
                try:
                    row = json.loads(line)
                    self.cache[row["src"]] = row
                except (json.JSONDecodeError, KeyError):
                    continue

    def get(self, message: str) -> str:
        row = self.cache.get(message)
        if row is None and not config.LLM_API_KEY:
            # no key: rule rewrite of the same level, never cached (so a later keyed run fills it)
            self.missing += 1
            return rule_paraphrase(message, _RULE_EQUIVALENT[self.level])
        if row is None:
            row = self._generate(message)
            with self._lock:
                if message not in self.cache:
                    self.cache[message] = row
                    self.path.parent.mkdir(parents=True, exist_ok=True)
                    with self.path.open("a", encoding="utf-8") as handle:
                        handle.write(json.dumps(row, ensure_ascii=False) + "\n")
                    self.new += 1
        return row["dst"]

    def _generate(self, message: str) -> dict:
        fields = _fields(message)
        if fields is None:
            return {"src": message, "dst": message, "fallback": True, "reason": "unrecognised template"}
        protected = _protected(fields, self.level)
        instruction = _LEVEL_RULES[self.level].format(protected=json.dumps(protected))
        for attempt, temperature in enumerate((0.7, 0.0)):
            user = json.dumps({"line": message, "instructions": instruction + (_STRICT_SUFFIX if attempt else "")},
                              ensure_ascii=False)
            reply = llm_client.chat_json(_SYSTEM, user, max_tokens=200, timeout=config.LLM_TIMEOUT,
                                         temperature=temperature, cache=False)   # 本类自带缓存
            rewrite = (reply or {}).get("rewrite")
            if isinstance(rewrite, str) and rewrite.strip() and _survives(rewrite, protected):
                return {"src": message, "dst": rewrite.strip(), "model": config.LLM_MODEL,
                        "fallback": False, "attempt": attempt}
        return {"src": message, "dst": rule_paraphrase(message, _RULE_EQUIVALENT[self.level]),
                "fallback": True, "reason": "no key / protected span lost"}


class LLMParaphrasingAgent:
    def __init__(self, inner, paraphrases: ParaphraseSet) -> None:
        self.inner = inner
        self.paraphrases = paraphrases

    def reset(self, session_id: str, user_profile: dict) -> None:
        self.inner.reset(session_id, user_profile)

    def respond(self, session_id: str, user_message: str, turn: int, top_k: int) -> dict:
        return self.inner.respond(session_id, self.paraphrases.get(user_message), turn, top_k)


class RecordingAgent:
    def __init__(self, inner) -> None:
        self.inner = inner
        self.seen: list[str] = []

    def reset(self, session_id: str, user_profile: dict) -> None:
        self.inner.reset(session_id, user_profile)

    def respond(self, session_id: str, user_message: str, turn: int, top_k: int) -> dict:
        self.seen.append(user_message)
        return self.inner.respond(session_id, user_message, turn, top_k)


def prefill(samples, catalog_ids, categories, products, catalog: str, levels: list[str], workers: int) -> None:
    """Paraphrase every utterance of an L0 run in parallel, so the P-level runs mostly hit the
    cache (openings are identical at every level; later turns depend on the agent's questions)."""
    recorder = RecordingAgent(Agent(catalog))
    evaluate(recorder, samples, catalog_ids, categories, products)
    utterances = list(dict.fromkeys(recorder.seen))
    print(f"L0 transcript: {len(recorder.seen)} turns, {len(utterances)} distinct utterances")
    for level in levels:
        paraphrases = ParaphraseSet(level)
        todo = [u for u in utterances if u not in paraphrases.cache]
        with ThreadPoolExecutor(max_workers=workers) as pool:
            list(pool.map(paraphrases.get, todo))
        fallbacks = sum(1 for u in utterances if paraphrases.cache[u].get("fallback"))
        print(f"{level}: {len(todo)} generated, {fallbacks} fallbacks, cache {paraphrases.path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="LLM 改写集压力测试")
    parser.add_argument("--catalog", default="data/catalog.jsonl")
    parser.add_argument("--dataset", default="data/public_set.jsonl")
    parser.add_argument("--levels", nargs="+", default=list(LEVELS))
    parser.add_argument("--n", type=int, default=0, help="抽样条数，0 = 全量 200")
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--prefill", action="store_true", help="先并行改写 L0 全部话术再退出")
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--show", type=int, default=0, help="打印每档前 N 条缓存改写后退出")
    args = parser.parse_args()

    if args.show:
        for level in args.levels:
            rows = list(ParaphraseSet(level).cache.values())[:args.show]
            print(f"\n== {level} ==")
            for row in rows:
                flag = "  [fallback]" if row.get("fallback") else ""
                print(f"原句  {row['src']}\n改写  {row['dst']}{flag}\n")
        return

    samples = stratified(load_jsonl(args.dataset), args.n, args.seed)
    catalog_ids, categories, products = catalog_index(args.catalog)
    if args.prefill:
        if not config.LLM_API_KEY:
            sys.exit("DEEPSEEK_API_KEY 未设置（.env），无法生成改写集")
        prefill(samples, catalog_ids, categories, products, args.catalog, args.levels, args.workers)
        return

    print(f"LLM 改写集压力测试 · {len(samples)} 条会话 · evaluator 未做任何修改 · "
          f"LLM_PARSE={'1' if config.LLM_PARSE else '0'} · model={config.LLM_MODEL}\n")
    print("%-5s %10s %10s %8s %8s %8s %9s %7s   %s" % ("档位", "Score", "HitRate", "MRR", "MTTC", "新改写", "fallback", "缺缓存", "备注"))
    print("-" * 94)
    incomplete = []
    for level in args.levels:
        paraphrases = ParaphraseSet(level)
        before = llm_client.call_counts()
        agent = LLMParaphrasingAgent(Agent(args.catalog), paraphrases)
        result = evaluate(agent, samples, catalog_ids, categories, products)
        after = llm_client.call_counts()
        fallbacks = sum(1 for row in paraphrases.cache.values() if row.get("fallback"))
        note = f"API {after['api'] - before['api']} / cache {after['cache'] - before['cache']}"
        print("%-5s %10.4f %10.3f %8.3f %8.2f %8d %9d %7d   %s" % (
            level, result["recommended_technical_score"], result["hit_rate_at_10"], result["mrr"],
            result["mttc"], paraphrases.new, fallbacks, paraphrases.missing, note))
        if paraphrases.missing:
            incomplete.append(level)
    if incomplete:
        print(f"\n⚠ {', '.join(incomplete)} 有话术不在改写缓存里（agent 的提问变了，模拟器回了新句子），无 key 时用规则改写顶替，"
              f"分数不作数：填好 DEEPSEEK_API_KEY 重跑一次即可补齐缓存。")
    print("\n档位：P1 只改句式（约束值逐字保留）／ P2 短约束改口语（规格串保留）／ P3 自然口语（无冒号）")
    print("改写缓存在 bench/paraphrases/，生成一次后离线可复现；fallback = 改写丢了受保护片段，用规则改写顶替。")


if __name__ == "__main__":
    main()
