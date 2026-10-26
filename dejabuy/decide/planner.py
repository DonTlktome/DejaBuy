"""Decision layer: which attribute to ask and which candidates to show this turn.

Two regimes share one interface:

* **Strict regime** (`plan_strict`) — the environment model recognised the whole transcript,
  so the decision is made over the *support set* (candidates whose replayed transcript matches)
  with a belief distribution over it. Each candidate action is scored by expected utility
  under the benchmark's own scoring function (`decide/utility.py`): asking attribute `a`
  partitions the support set by the reply each candidate would give; a candidate's value if
  deferred is the score it would earn next turn at its rank *inside its reply group*. The
  slate width follows from the same comparison (show now vs. defer), under `WIDTH_MODE`.

* **Loose regime** (`choose_ask`) — the transcript was not fully recognised (paraphrased
  speech, LLM-extracted fragments). The evidence ordering stands and the question policy is
  the week-1 one (`other_first` / `entropy`).

Adaptive lookahead: the rollout only considers the top candidates that hold
`PLAN_MASS` of the belief (at most `PLAN_TOP`); one-step by default. A two-step rollout is
only opened when few candidates remain and the top of the slate is not yet separated
(`LOOKAHEAD=2`), which keeps the planner inside the per-turn latency budget.
"""

from __future__ import annotations

import math
from collections import Counter, defaultdict
from dataclasses import dataclass, field

from dejabuy import config
from dejabuy.decide import utility
from dejabuy.env import protocol
from dejabuy.state import DialogState

_ENTROPY_ATTRS = ("color", "material", "budget")


# ----------------------------------------------------------------------------- loose regime
def _entropy(values: list) -> float:
    counts = Counter(values)
    total = sum(counts.values())
    if total <= 1:
        return 0.0
    return -sum((n / total) * math.log2(n / total) for n in counts.values())


def _entropy_pick(state: DialogState, candidates: list[dict]) -> str:
    covered = {slot.attribute for slot in state.slots}
    best_attr, best_score = "other", 0.0
    for attr in _ENTROPY_ATTRS:
        if attr in state.exhausted or attr in covered:
            continue
        if attr == "budget":
            values = [int(c["price"] // 10) for c in candidates if c.get("price") is not None]
        else:
            values = [c.get(attr) for c in candidates if c.get(attr)]
        coverage = len(values) / max(len(candidates), 1)
        score = _entropy(values) * coverage
        if score > best_score:
            best_attr, best_score = attr, score
    return best_attr


def choose_ask(state: DialogState, candidates: list[dict]) -> str:
    """Loose-regime question policy. Never returns None."""
    if config.ASK_POLICY == "entropy" and not state.all_disclosed:
        return _entropy_pick(state, candidates)
    habits = state.habits
    if habits is not None and habits.answer_prob("other") < 0.5:
        # this user does not answer open questions: ask the most informative concrete attribute instead
        return _entropy_pick(state, candidates)
    return "other"


# ----------------------------------------------------------------------------- strict regime
@dataclass
class Plan:
    ask: str
    shown: list[str]                      # support members to expose this turn, in rank order
    deferred: list[str]                   # support members deliberately held back
    expected_utility: float
    utilities: dict[str, float] = field(default_factory=dict)   # per attribute, for the trace
    info_only: bool = False               # hits do not count this turn (pre-override phase)
    beliefs: dict[str, float] = field(default_factory=dict)      # filled by the agent for introspection


def value_now(turn: int, rank: int) -> float:
    return utility.session_score(True, rank, turn)


def _defer_value(turn: int, group_position: int) -> float:
    """Expected score of a candidate held back this turn.

    `group_position` is its rank among the *deferred* candidates that would give the same
    reply to the chosen question (shown candidates never compete for next-turn ranks: a
    shown candidate either ends the session or is refuted). Two continuation policies are
    considered and the better one taken: show the whole group next turn (rank = position),
    or walk the group one candidate per turn at rank one (found at turn + position)."""
    if turn >= utility.MAX_TURNS:
        return 0.0
    together = value_now(turn + 1, group_position) if group_position <= 10 else (
        config.DEFER_PENALTY * value_now(turn + 2, min(10, group_position - 10))
        if turn + 1 < utility.MAX_TURNS else 0.0)
    one_by_one = value_now(turn + group_position, 1) if turn + group_position <= utility.MAX_TURNS else 0.0
    return max(together, one_by_one)


def _candidate_actions(state: DialogState) -> list[str]:
    actions = [a for a in protocol.ATTRIBUTES if a != "category"]   # the category is already known
    # attributes the shopper said they had nothing more for are worthless to ask again
    return [a for a in actions if a not in state.exhausted] or ["other"]


def plan_strict(state: DialogState, products: dict[str, dict], support: dict[str, set[str]],
                beliefs: list[tuple[str, float]], turn: int, top_k: int) -> Plan:
    """Expected-utility choice of question and slate over the support set.

    `beliefs` is the support set ordered by belief, as (parent_asin, probability)."""
    ordered = [pid for pid, _ in beliefs]
    prob = dict(beliefs)
    # Pre-override phase of an intent-override session: the simulator does not check hits yet,
    # so showing is free and only the information value of the question matters.
    info_only = state.opening == "override_open" and not state.overridden
    product_mode = config.WIDTH_MODE == "product"

    # Adaptive pruning: plan over the candidates that carry most of the mass.
    mass, planned = 0.0, []
    for pid in ordered:
        planned.append(pid)
        mass += prob[pid]
        if mass >= config.PLAN_MASS or len(planned) >= config.PLAN_TOP:
            break
    rest = ordered[len(planned):]

    best: Plan | None = None
    utilities: dict[str, float] = {}
    for ask in _candidate_actions(state):
        groups = {pid: protocol.reply_key(products[pid], support[pid], ask) for pid in planned}
        for pid in rest:                      # unplanned tail: assume it stays undivided
            groups[pid] = "<tail>"
        # Procedural memory: a declined question partitions nothing, so a deferred candidate's next-turn
        # position is then its position among *all* deferred candidates (mixed by P(answer)).
        p_answer = state.habits.answer_prob(ask) if state.habits is not None else 1.0
        deferred_in_group: dict[str, int] = defaultdict(int)
        shown: list[str] = []
        deferred: list[str] = []
        eu = 0.0
        for pid in ordered:
            p = prob[pid]
            position = deferred_in_group[groups[pid]] + 1
            defer = _defer_value(turn, position)
            if p_answer < 1.0:
                defer = p_answer * defer + (1.0 - p_answer) * _defer_value(turn, len(deferred) + 1)
            if info_only:
                deferred.append(pid)
                deferred_in_group[groups[pid]] += 1
                eu += p * defer
                continue
            rank = len(shown) + 1
            now = value_now(turn, rank) if rank <= top_k else 0.0
            show = rank <= top_k and (product_mode or now >= defer or not shown)
            if show:
                shown.append(pid)
                eu += p * now
            else:
                deferred.append(pid)
                deferred_in_group[groups[pid]] += 1
                eu += p * defer
        utilities[ask] = eu
        if best is None or eu > best.expected_utility + 1e-12 or (
                abs(eu - best.expected_utility) <= 1e-12 and ask == "other"):
            best = Plan(ask=ask, shown=shown, deferred=deferred, expected_utility=eu, info_only=info_only)
    assert best is not None
    if info_only:
        # nothing to lose by showing: expose the belief order (storefront behaviour) anyway
        best.shown, best.deferred = ordered[:top_k], ordered[top_k:]
    best.utilities = utilities
    return best
