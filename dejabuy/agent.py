"""DejaBuy agent: the only entry point (`reset` / `respond`).

One turn = understand -> remember -> retrieve -> believe & rank -> decide -> surface.
Invariants: `respond()` never raises, always returns a slate, never returns a null question.
"""

from __future__ import annotations

import time
from pathlib import Path

from dejabuy import config, trace
from dejabuy.belief import belief
from dejabuy.belief.retrieve import Retriever
from dejabuy.decide import planner
from dejabuy.env import protocol
from dejabuy.llm import client as llm_client
from dejabuy.memory import session as session_memory
from dejabuy.memory import semantic
from dejabuy.memory.procedural import Habits
from dejabuy.memory.session import distill
from dejabuy.memory.store import MemoryStore
from dejabuy.state import DialogState
from dejabuy.surface.templates import clarify
from dejabuy.understand import canonical, rules, verify


class DejaBuyAgent:
    def __init__(self, catalog_path: str = "data/catalog.jsonl") -> None:
        self.retriever = Retriever(catalog_path)
        self.sessions: dict[str, DialogState] = {}
        self.debug: dict[str, dict] = {}       # last turn's decision record per session (REST / demo)
        self.debug_enabled = False             # the evaluator path never pays for it
        # The understanding layer verifies every extracted fragment against the catalogue
        # (tier 1: verbatim anywhere; tier 2: aligned to a card value of the opening category).
        verify.set_catalog_verifier(self.retriever.phrase_exists)
        verify.set_card_vocab(self.retriever.card_vocab, self.retriever.category_names)
        # Cross-session memory: loaded once; an absent directory is an empty store (all cold-start).
        self.memory = None
        self.memory_weights = semantic.load_weights(config.MEMORY_WEIGHTS)
        if config.MEMORY:
            directory = Path(config.MEMORY_DIR)
            keep_card = bool(self.memory_weights.get("card_share"))
            self.memory = MemoryStore.load(directory, config.MEMORY_HALF_LIFE, keep_card) if directory.exists() \
                else MemoryStore(config.MEMORY_HALF_LIFE)

    # ------------------------------------------------------------------ API
    def reset(self, session_id: str, user_profile: dict) -> None:
        # The evaluator only guards respond(); reset() must never raise either.
        try:
            profile = user_profile if isinstance(user_profile, dict) else {}
            session_memory.clear(session_id)
            state = DialogState(session_id=session_id, profile=profile)
            user_id = profile.get("user_id")
            if self.memory is not None and user_id:
                as_of = profile.get("as_of")
                state.user_id = str(user_id)
                state.memory = self.memory.profile(str(user_id), int(as_of) if as_of is not None else None)
                state.memory_weights = self.memory_weights
                if config.PROCEDURAL:
                    state.habits = self.memory.habits(str(user_id)).copy()
            if config.PROCEDURAL and state.habits is None:
                state.habits = Habits()
            self.sessions[session_id] = state
        except Exception:
            self.sessions[session_id] = DialogState(session_id=str(session_id), profile={})

    def respond(self, session_id: str, user_message: str, turn: int, top_k: int) -> dict:
        state = self.sessions.get(session_id)
        if state is None:
            state = DialogState(session_id=session_id, profile={})
            self.sessions[session_id] = state
        try:
            return self._respond(state, user_message, turn, top_k)
        except Exception:
            # Never go silent: fall back to last turn's slate and settle token usage.
            try:
                usage = llm_client.pop_usage()
            except Exception:
                usage = {"prompt_tokens": 0, "completion_tokens": 0}
            return {
                "message": "Here are the closest matches I found.",
                "ask_attribute": "other",
                "recommendations": [{"parent_asin": pid} for pid in state.last_ranked],
                "usage": usage,
            }

    # ------------------------------------------------------------------ one turn
    def _respond(self, state: DialogState, user_message: str, turn: int, top_k: int) -> dict:
        started = time.perf_counter()
        rules.update_state(state, user_message, turn)            # understand
        self._learn_habit(state, turn)                            # remember (procedural)
        state.distilled = distill(state)                          # remember (session)
        candidates = self.retriever.retrieve(state, user_message, k=100)   # retrieve

        plan = None
        support: dict[str, set[str]] = {}
        if config.PLANNER == "utility" and protocol.recognised(state.events):
            pool_ids = [pid for category in canonical.resolve_categories(state)
                        for pid in self.retriever.by_category.get(category, []) if pid not in state.refuted]
            support = protocol.support(self.retriever.products, pool_ids, state.events)
        beliefs_shown: dict[str, float] = {}
        if support:
            top, ask_attribute, plan = self._strict_turn(state, candidates, support, turn, top_k)
            regime = "strict"
            beliefs_shown = plan.beliefs
        else:
            top = belief.rank(state, candidates, k=top_k)         # believe & order (loose regime)
            ask_attribute = planner.choose_ask(state, candidates)
            regime = "loose"
        state.asked.add(ask_attribute)
        state.last_ask = ask_attribute
        state.last_ranked = [c["parent_asin"] for c in top]
        # A shown candidate either ends the session (hit) or is refuted — but only on turns where
        # the simulator checks hits (an intent-override session is not scored before the override).
        if state.overridden or state.opening != "override_open":
            state.refuted.update(state.last_ranked)
        if trace.enabled():
            trace.record(
                session=state.session_id, turn=turn, regime=regime, scenario=state.scenario,
                category=state.category, slots=[(s.attribute, s.value, s.hard) for s in state.slots],
                support=len(support), ask=ask_attribute, width=len(top), pool=len(candidates),
                utilities=plan.utilities if plan else None, deferred=len(plan.deferred) if plan else 0,
                top=[c["parent_asin"] for c in top], distilled=state.distilled,
                memory=state.memory.summary() if state.memory is not None and turn == 1 else None,
            )
        message = clarify(state, ask_attribute, top)              # surface
        if self.debug_enabled:
            self.debug[state.session_id] = self._debug_record(
                state, turn, regime, support, plan, top, candidates, beliefs_shown, ask_attribute,
                (time.perf_counter() - started) * 1000)
        return {
            "message": message,
            "ask_attribute": ask_attribute,
            "recommendations": [
                {"parent_asin": c["parent_asin"], "score": round(1.0 / (i + 1), 4)}
                for i, c in enumerate(top)
            ],
            "usage": llm_client.pop_usage(),
        }

    def _strict_turn(self, state: DialogState, candidates: list[dict], support: dict[str, set[str]],
                     turn: int, top_k: int) -> tuple[list[dict], str, planner.Plan]:
        """Strict regime: beliefs over the support set, expected-utility question and slate."""
        pool = {c["parent_asin"]: c for c in candidates}
        members = [pool[pid] if pid in pool else self.retriever.candidate(pid, state) for pid in support]
        beliefs = belief.strict_beliefs(state, members)
        plan = planner.plan_strict(state, self.retriever.products, support, beliefs, turn, top_k)
        plan.beliefs = dict(beliefs) if self.debug_enabled else {}
        by_id = {c["parent_asin"]: c for c in members}
        slate = [by_id[pid] for pid in plan.shown]
        # Hedge against a wrong world model: fill the remaining slots with the best non-support
        # candidates (they never displace a shown support member, so the plan's ranks hold).
        if len(slate) < top_k:
            exclude = set(support)
            for c in belief.order(state, candidates):
                if c["parent_asin"] in exclude:
                    continue
                slate.append(c)
                if len(slate) >= top_k:
                    break
        return slate, plan.ask, plan

    def _learn_habit(self, state: DialogState, turn: int) -> None:
        """Did the user answer last turn's question? A boundary reply is a decline; a disclosure or
        an exhausted reply ("nothing more on that") is an answer."""
        if state.habits is None or turn <= 1 or not state.last_ask or not state.events:
            return
        kind = state.events[-1].get("type")
        if kind not in ("boundary", "disclose", "exhausted"):
            return
        answered = kind != "boundary"
        if state.user_id and self.memory is not None:
            self.memory.record_answer(state.user_id, state.last_ask, answered)   # cross-session: every reply
        if answered:
            state.habits.record(state.last_ask, True)
            return
        # Within this conversation a single "use your judgment" is not yet a habit (plenty of shoppers wave
        # off the first question once); a second decline of the same attribute is.
        state.session_declines[state.last_ask] = state.session_declines.get(state.last_ask, 0) + 1
        if state.session_declines[state.last_ask] >= 2:
            state.habits.record(state.last_ask, False)

    # ------------------------------------------------------------------ demo / REST introspection
    def _debug_record(self, state: DialogState, turn: int, regime: str, support: dict, plan, top: list[dict],
                      candidates: list[dict], beliefs_shown: dict[str, float], ask: str, latency_ms: float) -> dict:
        if not beliefs_shown:
            beliefs_shown = {c["parent_asin"]: p for c, p in belief.beliefs(state, candidates)}
        profile = state.memory
        slate = []
        for rank, c in enumerate(top, 1):
            info = self.retriever.products.get(c["parent_asin"], c)
            slate.append({
                "rank": rank, "parent_asin": c["parent_asin"], "title": info.get("title", ""),
                "price": info.get("price"), "store": info.get("store", ""),
                "belief": round(beliefs_shown.get(c["parent_asin"], 0.0), 4),
                "in_support": c["parent_asin"] in support,
                "memory": round(profile.prior(info, state.memory_weights), 3) if profile is not None else None,
            })
        return {
            "turn": turn, "regime": regime, "scenario": state.scenario, "category": state.category,
            "event": state.events[-1] if state.events else None,
            "slots": [{"attribute": s.attribute, "value": s.value, "hard": s.hard} for s in state.slots],
            "support": len(support), "ask": ask,
            "utilities": {k: round(v, 4) for k, v in (plan.utilities.items() if plan else [])},
            "deferred": len(plan.deferred) if plan else 0,
            "slate": slate,
            "memory": profile.summary() if profile is not None else None,
            "latency_ms": round(latency_ms, 1),
        }


Agent = DejaBuyAgent
