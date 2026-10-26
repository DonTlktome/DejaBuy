"""Within-session memory: cross-turn signals and the distilled context line.

Signals (per session): stagnation counter (no new slot and the top-5 barely moved) and the
set of candidates already shown while the simulator was actually checking for a hit. The
distilled line compresses the whole state into one bounded string; its length tracks the
number of live constraints, not the number of turns.

Cross-session memory (episodic / semantic / procedural) lives in the sibling modules and is
wired in week 4-5.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from dejabuy.state import DialogState


@dataclass
class _Memory:
    prev_scenario: str = "unknown"
    prev_slot_count: int = 0
    prev_top: list[str] = field(default_factory=list)
    stagnant_turns: int = 0
    rejected: set[str] = field(default_factory=set)


_STORE: dict[str, _Memory] = {}


def update(state: DialogState) -> None:
    """更新一个 session 的跨轮信号。在 distill() 开头调用，每轮一次。

    Reflection 的 gate（`prev_scenario != "unknown"` 才追加 rejected）：
    evaluator 只有 override_applied=True 才检查命中；override 场景在触发句
    出现前 state.scenario 保持默认值 "unknown"，那几轮"未命中"不代表候选
    里没有真实目标，只是评测器没查——不加这个 gate 会把还没被检验过的候选
    误判成负样本永久拉黑，反而可能砸掉 override 分项。buying/browsing/
    boundary 从第 1 轮起 scenario 就非 unknown，gate 从第 2 轮起自然打开。
    """
    mem = _STORE.setdefault(state.session_id, _Memory())

    if mem.prev_scenario != "unknown":
        mem.rejected.update(state.last_ranked)

    grew = len(state.slots) > mem.prev_slot_count
    overlap = len(set(mem.prev_top[:5]) & set(state.last_ranked[:5])) if mem.prev_top else 0
    scenario_changed = state.scenario != mem.prev_scenario
    mem.stagnant_turns = 0 if (grew or scenario_changed or overlap < 4) else mem.stagnant_turns + 1

    mem.prev_scenario = state.scenario
    mem.prev_slot_count = len(state.slots)
    mem.prev_top = state.last_ranked


def rejected_asins(session_id: str) -> frozenset[str]:
    """上一轮之后、gate 打开期间累计展示但未命中的 parent_asin 集合（原始信号，仅供观测/调试）。

    ⚠️ 不要直接拿这个去过滤候选池——诊断发现（scratchpad/diagnose_reflection.py，
    public_0087 复现）：这个模拟器里"展示过未命中"不等于"用户拒绝"，只是"当时信息不够"。
    排序会随新约束到来而改善，永久拉黑会把新一轮本该翻身的目标一并拉黑（该样本第1轮弱信息
    猜测未命中，第2轮拿到新约束后排序本可把它送回 rank 10，被拉黑后变成 rank 11=永久错过）。
    要用于实际过滤，请用 actionable_rejections()。
    """
    return frozenset(_STORE.get(session_id, _Memory()).rejected)


def actionable_rejections(session_id: str, min_stagnant: int = 2) -> frozenset[str]:
    """只有连续 >= min_stagnant 轮"无新约束 + 候选池不变"时才返回非空集合。

    诊断验证（全量 200 条 A/B 对比，见 team/experiments.md #10）：不加这个门槛时
    19 轮改善 / 1 轮恶化（public_0087 被误伤，见 rejected_asins 文档）；加上停滞门槛后
    该样本在恶化的那一轮 stagnant_turns 必然 <2（连 update() 调用次数都不够），
    不会被误伤——只有排序已经"卡住不变"时才值得强制换血。
    """
    if stagnant_turns(session_id) < min_stagnant:
        return frozenset()
    return rejected_asins(session_id)


def stagnant_turns(session_id: str) -> int:
    """连续多少轮：无新槽位 且 候选池 top-5 与上一轮高度重合（≥4/5）。"""
    return _STORE.get(session_id, _Memory()).stagnant_turns


from dejabuy.memory import profile_lexicon as lexicon  # noqa: E402

signals = __import__(__name__)  # self-reference so `distill` reads like the layered design


def distill(state: DialogState) -> str:
    update(state)

    parts = [f"category: {state.category}" if state.category else ""]
    parts += [f"[{s.attribute}] {s.value}" for s in state.slots]

    tags = state.profile.get("preference_tags") or []
    if tags:
        parts.append("profile tags: " + ", ".join(str(t) for t in tags[:5]))
    soft_terms = lexicon.profile_soft_terms(state.profile)
    if soft_terms:
        parts.append("profile hints: " + ", ".join(soft_terms))

    stagnant = stagnant_turns(state.session_id)
    if stagnant:
        parts.append(f"stagnant_turns: {stagnant}")
    rejected = rejected_asins(state.session_id)
    if rejected:
        parts.append(f"rejected_count: {len(rejected)}")

    return " | ".join(p for p in parts if p)


def clear(session_id: str) -> None:
    """Drop a session's cross-turn signals (called from reset so ids can be reused)."""
    _STORE.pop(session_id, None)
