"""Minimal OpenAI-compatible chat client (standard library only).

JSON mode, one retry, token accounting per turn (`pop_usage`), and a full on-disk response
cache: every (model, system, user, temperature, max_tokens) tuple is answered from
`LLM_CACHE_PATH` when it has been seen before, so a run that once used the API replays
byte-for-byte without a key or a network. Without an API key an uncached call returns None
immediately so the offline path never waits on the network.
"""

from __future__ import annotations

import hashlib
import json
import threading
import urllib.error
import urllib.request
from pathlib import Path

from dejabuy import config

_usage = {"prompt_tokens": 0, "completion_tokens": 0}
_cache: dict[str, dict | None] | None = None
_cache_lock = threading.Lock()
_calls = {"api": 0, "cache": 0}


def pop_usage() -> dict:
    """取走并清零本轮累计的 token 用量（agent 每轮上报用）。"""
    global _usage
    current, _usage = _usage, {"prompt_tokens": 0, "completion_tokens": 0}
    return current


def call_counts() -> dict:
    """{"api": 真实请求数, "cache": 缓存命中数}（进程累计，bench 报告用）。"""
    return dict(_calls)


# ------------------------------------------------------------------------------- cache
def _cache_path() -> Path:
    return Path(config.LLM_CACHE_PATH)


def _load_cache() -> dict[str, dict | None]:
    global _cache
    if _cache is None:
        loaded: dict[str, dict | None] = {}
        path = _cache_path()
        if path.exists():
            for line in path.read_text(encoding="utf-8").splitlines():
                try:
                    row = json.loads(line)
                    loaded[row["key"]] = row["reply"]
                except (json.JSONDecodeError, KeyError, TypeError):
                    continue
        _cache = loaded
    return _cache


def _cache_key(model: str, system: str, user: str, temperature: float, max_tokens: int) -> str:
    payload = json.dumps([model, system, user, temperature, max_tokens], ensure_ascii=False)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _remember(key: str, reply: dict | None, note: dict) -> None:
    cache = _load_cache()
    with _cache_lock:
        cache[key] = reply
        path = _cache_path()
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps({"key": key, "reply": reply, **note}, ensure_ascii=False) + "\n")


# -------------------------------------------------------------------------------- call
def chat_json(system: str, user: str, max_tokens: int = 200, timeout: float | None = None,
              temperature: float = 0.0, cache: bool = True) -> dict | None:
    """一次 JSON 模式调用。先查缓存；失败/超时/空内容 → 重试一次 → 仍失败返回 None（上层走规则兜底）。

    成功的回复写入缓存（失败不缓存，下次仍会重试）。temperature=0 是解析层的口径：排序 /
    抽取任务要可复现；改写集生成用更高温度，靠缓存固定结果。"""
    model = config.LLM_MODEL
    key = _cache_key(model, system, user, temperature, max_tokens)
    if cache:
        hit = _load_cache()
        if key in hit:
            _calls["cache"] += 1
            return hit[key]
    if not config.LLM_API_KEY:
        return None
    body: dict = {
        "model": model,
        "messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
        "thinking": {"type": "disabled"},
        "response_format": {"type": "json_object"},
        "temperature": temperature,
        "max_tokens": max_tokens,
    }
    reply = _post(body, config.LLM_TIMEOUT if timeout is None else timeout)
    if reply is not None and cache:
        _remember(key, reply, {"model": model, "system_sha": hashlib.sha256(system.encode()).hexdigest()[:12],
                               "user": user})
    return reply


def _post(body: dict, timeout: float) -> dict | None:
    """两次尝试；若服务端拒绝扩展字段（400），去掉 thinking / response_format 再试。"""
    attempts = [body]
    for _attempt in range(2):
        payload = attempts[-1]
        request = urllib.request.Request(
            f"{config.LLM_BASE_URL}/chat/completions",
            data=json.dumps(payload).encode(),
            headers={"Content-Type": "application/json", "Authorization": f"Bearer {config.LLM_API_KEY}"},
        )
        try:
            _calls["api"] += 1
            with urllib.request.urlopen(request, timeout=timeout) as response:
                data = json.loads(response.read())
            usage = data.get("usage") or {}
            _usage["prompt_tokens"] += int(usage.get("prompt_tokens") or 0)
            _usage["completion_tokens"] += int(usage.get("completion_tokens") or 0)
            content = (data["choices"][0]["message"]["content"] or "").strip()
            if content:
                return _parse_json(content)
        except urllib.error.HTTPError as error:
            if error.code == 400 and ("thinking" in payload or "response_format" in payload):
                slim = {k: v for k, v in payload.items() if k not in ("thinking", "response_format")}
                attempts.append(slim)
                continue
        except (urllib.error.URLError, TimeoutError, KeyError, ValueError, json.JSONDecodeError):
            continue
    return None


def _parse_json(content: str) -> dict | None:
    """JSON 模式偶尔会带 ``` 围栏或前后缀；取第一个 { 到最后一个 } 之间的内容。"""
    try:
        parsed = json.loads(content)
        return parsed if isinstance(parsed, dict) else None
    except json.JSONDecodeError:
        start, end = content.find("{"), content.rfind("}")
        if start < 0 or end <= start:
            return None
        try:
            parsed = json.loads(content[start:end + 1])
        except json.JSONDecodeError:
            return None
        return parsed if isinstance(parsed, dict) else None
