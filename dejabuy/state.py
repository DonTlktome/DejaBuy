"""Session state shared by every layer.

A `Slot` is one constraint the shopper has disclosed (attribute, verbatim value, hard/soft,
normalised search terms). `DialogState` is everything the agent knows about one session;
it is created in `reset()` and mutated turn by turn."""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class Slot:
    attribute: str        # ask_attribute 枚举之一（budget/material/color/... /feature）
    value: str            # 约束原文（评测器逐字吐出，保留原文用于子串匹配）
    hard: bool            # 硬约束(过滤/强加权) or 软偏好
    turn_added: int
    terms: list[str] = field(default_factory=list)
                          # 归一化检索词（A 产出，判别力降序；空 = 不参与文本匹配）。
                          # B/C 直接消费；匹配取命中的最长 term 计权（SPEC §5）


@dataclass
class DialogState:
    session_id: str
    profile: dict
    slots: list[Slot] = field(default_factory=list)
    asked: set[str] = field(default_factory=set)       # 已问过的属性
    exhausted: set[str] = field(default_factory=set)   # 用户明说"没有更多偏好"的属性
    all_disclosed: bool = False                        # other 已问干（所有约束都拿到了）
    category: str = ""                                 # "I'm looking for X" 的 X
    scenario: str = "unknown"                          # buying/browsing/override/boundary/unknown
    budget: float | None = None                        # "budget around $X" 解析出的 X
    history: list[dict] = field(default_factory=list)
    distilled: str = ""                                # 会话内蒸馏上下文
    last_ranked: list[str] = field(default_factory=list)  # 兜底：上一轮的 top-10
    # 协议事件日志（理解层产出，环境模型消费）。每条 = {"type": open|disclose|exhausted|boundary|
    # override|null|unknown, ...}。出现 unknown 即表示严格回放不可用，决策层退回宽松模式。
    events: list[dict] = field(default_factory=list)
    last_ask: str | None = None                        # 上一轮发出的 ask_attribute（解释本轮回复）
    overridden: bool = False                           # 改需求事件是否已发生（之前的命中不计分）
    opening: str = ""                                  # buying | browsing | override_open | ""
    refuted: set[str] = field(default_factory=set)     # 已展示且未命中的候选（命中计分的轮次才算）
    memory: object | None = None                       # 语义记忆画像（memory/semantic.Profile），冷启动为 None
    memory_weights: dict = field(default_factory=dict)
    user_id: str | None = None                         # 记忆库里的用户（写回程序记忆用），匿名会话为 None
    habits: object | None = None                       # 程序记忆（memory/procedural.Habits），本会话内实时更新
    session_declines: dict = field(default_factory=dict)  # 本会话内各属性被拒答的次数
    categories: list[str] = field(default_factory=list)  # 开场品类对齐到的精确粗品类名（严格回放的候选池；
                                                       # 模板路径恰为 [category]，改写话术下可能是多个）

    def constraint_values(self) -> list[str]:
        return [slot.value for slot in self.slots]

    def erase_preferences(self) -> None:
        """Intent Override：擦掉全部偏好槽位（保留 category）。"""
        self.slots.clear()
        self.budget = None
        self.exhausted.clear()
        self.all_disclosed = False

    def demote_preferences(self) -> None:
        """Intent Override 的温和版：旧约束降为软偏好（保留检索信号），并重开提问。

        依据：override 场景目标商品不变，旧约束仍描述目标；且评测器不会二次吐露
        已 disclosed 的约束，硬擦除等于永久丢失信息（消融实证：擦除版 override
        HitRate 0.667，降权版见 experiments.md）。
        """
        for slot in self.slots:
            slot.hard = False
        self.exhausted.clear()
        self.all_disclosed = False
