"""Procedural memory: how to talk to this user.

The habit tracked is whether the user actually answers a given kind of question. Some shoppers
never answer open-ended questions ("anything else that matters?") but happily name a material;
some never give a budget. Every time the agent asks attribute `a`, the reply is logged as answered
(a disclosure, or "nothing more on that") or declined ("no preference, use your judgment"). The
habit is a Beta-Binomial estimate of P(answer | a) per user, shrunk towards a population prior so
one refusal does not silence an attribute for good.

The decision layer uses it as the probability that a question yields information: a declined
question spends a turn and partitions nothing, so its expected utility is a mix of the informed
plan and the uninformed one (decide/planner.py). Within a conversation the same estimate is
updated turn by turn, so a refusal is respected immediately even for a first-time customer.
"""

from __future__ import annotations

from collections import defaultdict

PRIOR_ANSWERED = 3.0     # population prior: most questions get answered
PRIOR_DECLINED = 1.0


class Habits:
    def __init__(self) -> None:
        self.answered: dict[str, float] = defaultdict(float)
        self.declined: dict[str, float] = defaultdict(float)

    def record(self, attribute: str, answered: bool) -> None:
        if answered:
            self.answered[attribute] += 1.0
        else:
            self.declined[attribute] += 1.0

    def answer_prob(self, attribute: str) -> float:
        """1.0 until this user has declined `attribute` at least once (nothing to adjust for)."""
        if not self.declined.get(attribute):
            return 1.0
        a = self.answered.get(attribute, 0.0) + PRIOR_ANSWERED
        d = self.declined.get(attribute, 0.0) + PRIOR_DECLINED
        return a / (a + d)

    def copy(self) -> "Habits":
        other = Habits()
        other.answered.update(self.answered)
        other.declined.update(self.declined)
        return other

    def summary(self) -> dict:
        declined = {k: v for k, v in self.declined.items() if v}
        return {"declines": sorted(declined, key=lambda k: -declined[k]),
                "answer_prob": {k: round(self.answer_prob(k), 2) for k in sorted(set(self.answered) | set(declined))}}

    def __bool__(self) -> bool:
        return bool(self.answered or self.declined)
