"""Understanding layer, first two tiers.

Tier 1 — template fast path: the exact sentence shapes the reference simulator emits
(opening / disclosure / exhausted / boundary / override / null-penalty).
Tier 2 — rule salvage: when no template matches, pull category, override intent and
colon-delimited constraint payloads out of unfamiliar phrasing; every fragment is checked
against the catalogue (`verify.py`).
Tier 3 — LLM extraction (`llm_extract.py`) is called only when both tiers above fail.

Templates below are quoted from the reference simulator so the public set is parsed
byte-for-byte identically to the strict path."""

from __future__ import annotations

import re

from dejabuy import config
from dejabuy.understand.normalize import constraint_terms, normalize
from dejabuy.state import DialogState, Slot
from dejabuy.understand import canonical, llm_extract, verify

# 与评测器 classify_constraint 保持一致的镜像分类器（不 import evaluator，避免循环依赖）
_MATERIALS = ("cotton", "polyester", "nylon", "leather", "wool", "spandex", "silk", "rayon", "fabric")
_BUDGET_RE = re.compile(r"budget around \$([0-9]+(?:\.[0-9]+)?)", re.I)


def classify_constraint(value: str) -> str:
    lowered = value.lower()
    if "budget" in lowered or re.search(r"(?:\$|<=|under)\s*\d", lowered):
        return "budget"
    if any(material in lowered for material in _MATERIALS):
        return "material"
    if any(word in lowered for word in ("color", "black", "white", "blue", "red", "pink", "green")):
        return "color"
    if any(word in lowered for word in ("size", "sizing", "width", "wide", "narrow")):
        return "size"
    if any(word in lowered for word in ("department", "style", "fit", "sleeve", "neck")):
        return "style"
    if any(word in lowered for word in ("hiking", "running", "gym", "winter", "outdoor", "work")):
        return "use_case"
    return "feature"


def _add_constraint(state: DialogState, value: str, turn: int, hard: bool = True) -> None:
    value = value.strip().rstrip(".")
    if not value or value in state.constraint_values():
        return
    attribute = classify_constraint(value)
    if attribute == "budget":
        match = _BUDGET_RE.search(value)
        if match:
            state.budget = float(match.group(1))
    state.slots.append(Slot(
        attribute=attribute, value=value, hard=hard, turn_added=turn,
        terms=constraint_terms(value),
    ))


def _promote_or_add(state: DialogState, value: str, turn: int) -> None:
    """override 点名的约束：若早前已吐露入槽（降权后为软），重新提升为 hard；否则新增硬槽。

    不能直接走 _add_constraint——它遇到重复值直接跳过，会让用户最强的信号停留在软权重
    （bug 实录见 team/A-任务清单.md T1，public_0003 复现）。
    """
    cleaned = value.strip().rstrip(".")
    for slot in state.slots:
        if slot.value == cleaned:
            slot.hard = True
            return
    _add_constraint(state, cleaned, turn, hard=True)


def update_state(state: DialogState, message: str, turn: int) -> None:
    state.history.append({"turn": turn, "user": message})
    msg = message.strip()

    # ---- 改需求（override）：旧槽位降权保留 + 点名约束提升/置入为 hard ----
    # 宽松匹配：同时覆盖标准句、无点名约束的兜底句、以及轻度措辞变体
    if msg.startswith("Actually") and "ignore my earlier preference" in msg:
        state.scenario = "override"
        state.overridden = True
        state.demote_preferences()
        new_value = None
        if "What I need is:" in msg:
            new_value = msg.split("What I need is:", 1)[1].strip().rstrip(".")
            _promote_or_add(state, new_value, turn)
        state.events.append({"type": "override", "value": new_value})
        return

    # ---- 吐约束 ----
    if msg.startswith("For that, what matters is:"):
        body = msg.split("For that, what matters is:", 1)[1].strip()
        if body.endswith("."):
            body = body[:-1]          # the simulator appends exactly one full stop
        for part in body.split("; "):
            _add_constraint(state, part, turn, hard=True)
        state.events.append({"type": "disclose", "ask": state.last_ask, "body": body})
        return

    # ---- 问干了 ----
    match = re.match(r"I don't have an additional preference for (\w+)\.", msg)
    if match:
        attribute = match.group(1)
        state.exhausted.add(attribute)
        if attribute == "other":
            state.all_disclosed = True   # other 匹配任意约束，问干 = 全部拿到
        state.events.append({"type": "exhausted", "ask": attribute})
        return

    # ---- boundary 挡了第一问（该属性之后仍可再问，不标记 exhausted）----
    if re.match(r"I don't have a preference for \w+; please use your judgment\.", msg):
        state.scenario = "boundary"
        state.events.append({"type": "boundary", "ask": state.last_ask})
        return

    # ---- null 惩罚句（不应出现：我们永不发 null）----
    if msg.startswith("Those options are not quite right yet"):
        state.events.append({"type": "null"})
        return

    # ---- 开场三种（只在会话开头：改写后的吐约束句也可能以 "I'm looking for:" 起头，不能当开场）----
    if msg.startswith("I'm looking for ") and (turn == 1 or not state.category):
        rest = msg[len("I'm looking for "):]
        if rest.endswith(", but I'm still exploring."):
            state.category = rest[: -len(", but I'm still exploring.")].strip()
            state.scenario = "browsing"
            state.opening = "browsing"
            state.events.append({"type": "open", "scenario": "browsing", "category": state.category})
            return
        if ". A key requirement is: " in rest:
            category, constraint = rest.split(". A key requirement is: ", 1)
            state.category = category.strip()
            state.scenario = "buying"
            state.opening = "buying"
            value = constraint.strip().rstrip(".")
            _add_constraint(state, constraint, turn, hard=True)
            state.events.append({"type": "open", "scenario": "buying", "category": state.category, "value": value})
            return
        # override 开场："I'm looking for {cat}. {old_value}"（old_value 是软偏好，3-4 轮后会被擦掉）
        if ". " in rest:
            category, old_value = rest.split(". ", 1)
            state.category = category.strip()
            state.opening = "override_open"
            _add_constraint(state, old_value, turn, hard=False)
            state.events.append({"type": "open", "scenario": "override_open", "category": state.category,
                                 "value": old_value.strip()})
        else:
            state.category = rest.strip().rstrip(".")
            state.events.append({"type": "unknown"})
        return

    # ---- 兜底：未知句式走"载荷抽取"（私有集改写场景的主力路径，见下方 _salvage）----
    act = canonical.speech_act(state, msg, turn)          # 问干 / 边界 / null 这类无载荷回复
    if act is not None:
        state.events.append(act)
        return
    before = canonical.snapshot(state)
    reply, complete = _salvage(state, msg, turn)
    # 抽取结果对齐到品类卡片值后转成集合值事件；`complete` = 规则从冒号载荷里捞出了全部片段且
    # 逐条目录校验通过（载荷完整，回放要求候选的回复与之一一对应）；校验不过的轮次记 unknown，
    # 严格回放随即退回宽松模式
    state.events.append(canonical.infer_event(state, msg, turn, before, llm=reply, complete=complete))


# ===========================================================================
# 载荷抽取兜底（改写鲁棒性）
# ===========================================================================
# 动机（实测，bench/paraphrase_rules.py）：官方保留了给模拟用户加自然语言改写的权利。
# 一旦句式变化，上面那批 startswith/精确正则全部落空，整句话会被当成【一条】软约束塞进
# 槽位——而 ranker 的主信号是"约束原文作为连续子串出现在商品全文中"，一整句话永远匹配不上。
#
# 改写压力测试的分层结论（全量 200 条，evaluator 未改）：
#     L0 原样                        0.9620
#     L1 只改句式，约束原文一字不动   0.7792   ← -0.183，占全部损失的 87%
#     L2 再改短约束                  0.7585   ← -0.021
#     L3 连长规格串也重组            0.7523   ← -0.007
# 也就是说：**脆的是句式匹配，不是逐字匹配**。逐字指纹信号本身相当结实。
# 所以兜底的关键不是"理解语义"，而是"把约束片段从陌生句式里捞出来"——纯规则即可，不需要 LLM。
#
# 设计约束：本函数【只在严格模板全部落空时】才会被调用，因此对公开集（模板固定）
# 逐字节零影响。这是它可以放心上线的前提。

# 改需求信号：动词 + 时间指示词，覆盖 ignore/forget/scratch/disregard 等常见改写
_OVERRIDE_HINT = re.compile(
    r"\b(?:ignore|forget|scratch|disregard|never\s+mind)\b.{0,48}?"
    r"\b(?:earlier|before|previous(?:ly)?|already\s+said|told\s+you|first)\b",
    re.I | re.S,
)
# 改需求的"新需求"载荷：actually want / need is / what I want …
_OVERRIDE_PAYLOAD = re.compile(
    r"\b(?:what\s+I\s+(?:actually\s+)?(?:want|need)\s+is|actually\s+(?:want|need)|"
    r"instead\s+I\s+(?:want|need)|I\s+really\s+need)\b[:\s]*(?P<payload>.+)$",
    re.I | re.S,
)
# 粗品类：常见"我在找 X"的说法。停在标点或转折词处，避免把整句吞进来。
_CATEGORY_HINT = re.compile(
    r"\b(?:looking\s+for|shopping\s+(?:around\s+)?for|browsing\s+for|"
    r"in\s+the\s+market\s+for|after|hunting\s+for|want|need)\s+"
    r"(?P<cat>[^.,;:!?]{3,60}?)"
    r"(?=\s*(?:[.,;:!?—–]|$|\bbut\b|\bat\s+the\s+moment\b|\bright\s+now\b|\bthough\b))",
    re.I,
)
# 载荷分隔符：分号是评测器原生的多约束分隔；", and " 是改写后常见的连接
_PAYLOAD_SPLIT = re.compile(r";\s*|\s+and\s+also\s+|,\s+and\s+")

def _candidate_payloads(text: str) -> list[str]:
    """冒号载荷的候选集：首冒号后 + 末冒号后（去重）。

    评测器句式是 'lead-in: payload'，但约束值自身可能带冒号（'Department: womens' 这类
    "Key: value" 形态约占 1/8）——只取末冒号会把值劈碎（badcase 实录见实验 33），
    只取首冒号又赌不起 lead-in 里出现冒号。两个都要，让目录校验选赢家。"""
    payloads: list[str] = []
    for index in dict.fromkeys((text.find(":"), text.rfind(":"))):
        if index != -1:
            payload = text[index + 1:].strip().strip("\"'")
            if payload:
                payloads.append(payload)
    return payloads


def _collect_fragments(payload: str) -> list[str]:
    out: list[str] = []
    for fragment in _PAYLOAD_SPLIT.split(payload):
        fragment = fragment.strip().strip(" .,;:!?\"'—–")
        if verify.is_useful(fragment):
            out.append(fragment)
    return out


def _add_fragments(state: DialogState, payload: str, turn: int, hard: bool) -> int:
    added = 0
    for fragment in _collect_fragments(payload):
        before = len(state.slots)
        _add_constraint(state, fragment, turn, hard=hard)
        added += len(state.slots) - before
    return added


def _salvage(state: DialogState, message: str, turn: int) -> tuple[dict | None, bool]:
    """从陌生句式里尽量捞出：改需求信号 / 粗品类 / 约束片段。

    返回 (LLM 抽取结果或 None, 载荷是否完整)。LLM 应答即表示本轮新增的槽位来自 LLM 片段；
    "完整" = 规则从冒号载荷里捞出的片段逐条通过目录校验（这时片段就是模拟器的原话、一条不少），
    回放可以要求一一对应；其余情况（LLM 抄出、部分校验、整句切分）只能要求"抽到的能被解释"。"""
    # ① 改需求——语义等价于上面的严格分支，只是句式放宽
    if _OVERRIDE_HINT.search(message):
        state.scenario = "override"
        state.overridden = True
        state.demote_preferences()
        promoted = 0
        match = _OVERRIDE_PAYLOAD.search(message)
        if match:
            payload = match.group("payload").strip().strip(" .\"'")
            for fragment in _PAYLOAD_SPLIT.split(payload):
                fragment = fragment.strip().strip(" .,;:!?\"'—–")
                if verify.is_useful(fragment) and verify.in_catalog(fragment):
                    _promote_or_add(state, fragment, turn)
                    promoted += 1
        if not promoted:
            # 改需求信号命中但"新需求"载荷没捞到（改写把它藏进了规则盖不住的句型）
            # → 第三层上。LLM 失败也无妨：降权已完成，语义主体保住了。
            return _llm_extract(state, message, turn), False
        return None, True

    # ② 粗品类——只在还没拿到时抽取（开场轮）。品类精确命中在 ranker 里值 +2.5。
    # 先找目录品类名的逐字出现（改写不会改动品类名）；找不到再用"我在找 X"的句式规则，
    # 且规则捞出的短语必须能对齐到某个品类，否则宁可空着让第三层补。
    if not state.category:
        found = verify.find_category(message)
        if found:
            state.category = found[0]
            state.categories = found
        else:
            match = _CATEGORY_HINT.search(message)
            if match:
                candidate = match.group("cat").strip().strip(" .,;:!?\"'—–")
                if verify.is_useful(candidate) and verify.resolve_category(candidate):
                    state.category = candidate

    # ③ 约束片段——冒号载荷出双候选（首/末冒号），目录逐字校验选赢家（实验 33）。
    # "抽出来的片段全目录查无此文" = 大概率粘了口水话的垃圾，不算成功。
    best_frags: list[str] = []
    best_verified = -1
    for payload in _candidate_payloads(message):
        frags = _collect_fragments(payload)
        verified = sum(1 for f in frags if verify.in_catalog(f))
        if verified > best_verified or (verified == best_verified and len(frags) > len(best_frags)):
            best_frags, best_verified = frags, verified
    if best_verified > 0 and best_verified == len(best_frags):
        for fragment in best_frags:
            _add_constraint(state, fragment, turn, hard=False)
        return None, True
    # ③.5 第三层防线：LLM 逐字片段抽取。触发条件：冒号载荷抽不出"目录里逐字存在"的
    # 片段，或只有一部分存在（另一部分被改写成了自然说法）——包括值内冒号劈碎、
    # 口水话粘连这两类 badcase（实验 33 实录）。实验 16a 定位"脆的是片段抽取不是语义"，
    # LLM 在这里干的就是边界检测：把属性短语从句子里原样抄出来。
    reply = _llm_extract(state, message, turn)
    if reply is not None:
        return reply, False
    # LLM 未启用/失败：冒号片段仍保底入槽（信息只多不少），再不济整句切分
    if best_frags:
        for fragment in best_frags:
            _add_constraint(state, fragment, turn, hard=False)
        return None, False
    _add_fragments(state, message, turn, hard=False)
    return None, False




def _llm_extract(state: DialogState, message: str, turn: int) -> dict | None:
    """Apply the LLM layer's verified fields to the state and return them. A reply means the
    layer answered and owns this message (an empty but confirmed extraction also counts, so no
    junk fragments are added); None = disabled / tripped / failed, the rules continue."""
    reply = llm_extract.extract(message, turn, state.last_ask)
    if reply is None:
        return None
    if reply["override"] and turn > 1:
        state.scenario = "override"
        state.overridden = True
        state.demote_preferences()
    if not state.category and reply["category"] and (turn == 1 or reply["act"] == "open"):
        state.category = reply["category"]
    for fragment in reply["constraints"]:
        if reply["override"] and turn > 1:
            _promote_or_add(state, fragment, turn)
        else:
            _add_constraint(state, fragment, turn, hard=False)
    return reply


# Backwards-compatible aliases used by scripts and tests.
set_catalog_verifier = verify.set_catalog_verifier
