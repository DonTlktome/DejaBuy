"""REST service and live demo (standard library only).

    python3 -m dejabuy.server            # http://127.0.0.1:8000

Endpoints (JSON in, JSON out):
  GET  /api/customers          demo customers: a new shopper plus returning customers from the memory benchmark
  POST /api/session            {customer_id?, memory?}            → {session_id, memory}
  POST /api/turn               {session_id, message}              → {reply, ask, debug}
  POST /api/simulate           {customer_id, memory?}             → a whole conversation with the reference
                                                                     simulator playing the customer, turn by turn
  POST /api/buy                {session_id, parent_asin}          → records the purchase in the customer's memory
  POST /api/forget             {customer_id}                      → deletes everything remembered about them
  GET  /api/health

The agent is the same object the evaluator drives; the service only adds sessions, turn
counters and the per-turn decision record (`agent.debug`) the demo renders.
"""

from __future__ import annotations

import argparse
import json
import threading
import time
import uuid
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

from dejabuy import config
from dejabuy.agent import DejaBuyAgent

ROOT = Path(__file__).resolve().parents[1]
STATIC = Path(__file__).resolve().parent / "static"
TOP_K = 10
MAX_TURNS = 10


class Service:
    def __init__(self, catalog: str, demo_customers: int = 8) -> None:
        self.agent = DejaBuyAgent(catalog)
        self.agent.debug_enabled = True
        self.lock = threading.Lock()
        self.sessions: dict[str, dict] = {}
        self.customers = self._load_customers(demo_customers * 5)
        self.raw = self._raw_products({c["target"] for c in self.customers if c.get("target")}, catalog)
        self._pick_demo_customers(demo_customers)

    # ------------------------------------------------------------------ demo customers
    def _load_customers(self, n: int) -> list[dict]:
        out = [{"customer_id": None, "label": "New customer", "detail": "No history: everything comes from the conversation."}]
        path = ROOT / "data" / "memory_bench" / "test_loyal.jsonl"
        store = self.agent.memory
        if not path.exists() or store is None:
            return out
        seen_categories: set[str] = set()
        for line in path.open(encoding="utf-8"):
            sample = json.loads(line)
            target = sample["ground_truth"]["parent_asin"]
            info = self.agent.retriever.products.get(target)
            if info is None or info["coarse_exact"] in seen_categories:
                continue
            uid, as_of = sample["user_profile"]["user_id"], int(sample["user_profile"]["as_of"])
            profile = store.profile(uid, as_of)
            if profile is None:
                continue
            seen_categories.add(info["coarse_exact"])
            summary = profile.summary()
            stores = ", ".join(s.title() for s in summary["stores"][:2])
            out.append({
                "customer_id": uid, "as_of": as_of, "target": target, "scenario": sample["scenario_type"],
                "category": info["coarse_exact"],
                "label": f"Returning customer {len(out)}",
                "detail": (f"{summary['purchases']} past purchase{'s' if summary['purchases'] != 1 else ''}; "
                           f"often buys {stores}"),
                "memory": summary,
            })
            if len(out) > n:
                break
        return out

    def _pick_demo_customers(self, n: int) -> None:
        """Run every candidate once with and once without memory; keep mostly customers where memory
        changes the outcome, plus a couple where it does not (the demo should show both)."""
        scored = []
        for c in self.customers[1:]:
            with_m = self.simulate(c["customer_id"], True)["outcome"]
            without = self.simulate(c["customer_id"], False)["outcome"]
            def cost(o):
                return (o["turn"] or 11) + (1 - 1 / o["rank"] if o["rank"] else 1)
            c["effect"] = {"with": with_m, "without": without}
            scored.append((cost(without) - cost(with_m), c))
        helped = [c for gain, c in sorted(scored, key=lambda x: -x[0]) if gain > 0]
        neutral = [c for gain, c in scored if gain == 0]
        chosen = helped[: max(n - 2, 1)] + neutral[:2]
        for i, c in enumerate(chosen, 1):
            c["label"] = f"Returning customer {i}"
        self.customers = self.customers[:1] + chosen
        self.agent.sessions.clear()
        self.agent.debug.clear()
        self.sessions.clear()
        if self.agent.memory is not None:          # the trial runs must not leave habits behind
            self.agent.memory._habits.clear()

    def _raw_products(self, wanted: set[str], catalog: str) -> dict[str, dict]:
        raw = {}
        with open(catalog, encoding="utf-8") as handle:
            for line in handle:
                if not wanted:
                    break
                product = json.loads(line)
                if product.get("parent_asin") in wanted:
                    raw[product["parent_asin"]] = product
                    wanted.discard(product["parent_asin"])
        return raw

    def _customer(self, customer_id):
        return next((c for c in self.customers if c["customer_id"] == customer_id), None)

    # ------------------------------------------------------------------ sessions
    def open_session(self, customer_id=None, memory: bool = True) -> dict:
        customer = self._customer(customer_id) if customer_id else None
        session_id = f"s_{uuid.uuid4().hex[:12]}"
        profile = {}
        if customer and memory:
            profile = {"user_id": customer["customer_id"], "as_of": customer["as_of"]}
        with self.lock:
            self.agent.reset(session_id, profile)
            state = self.agent.sessions[session_id]
        self.sessions[session_id] = {"turn": 0, "customer": customer, "memory": memory}
        return {"session_id": session_id,
                "memory": state.memory.summary() if state.memory is not None else None}

    def turn(self, session_id: str, message: str) -> dict:
        session = self.sessions.get(session_id)
        if session is None:
            raise KeyError("unknown session; open one with POST /api/session")
        session["turn"] += 1
        with self.lock:
            out = self.agent.respond(session_id, message, session["turn"], TOP_K)
            debug = self.agent.debug.get(session_id, {})
        return {"turn": session["turn"], "reply": out["message"], "ask": out["ask_attribute"], "debug": debug}

    def buy(self, session_id: str, asin: str) -> dict:
        session = self.sessions.get(session_id)
        if session is None:
            raise KeyError("unknown session")
        customer = session["customer"]
        user_id = customer["customer_id"] if customer else f"guest_{session_id}"
        item = self.agent.memory.items.get(asin) if self.agent.memory else None
        if item is None:
            from dejabuy.memory.items import catalogue_item
            raw = self._raw_products({asin}, config_catalog())
            item = catalogue_item(raw[asin]) if asin in raw else None
        ts = (customer or {}).get("as_of", int(time.time() * 1000))
        self.agent.memory.record_purchase(user_id, asin, ts - 1, 5.0, item)
        profile = self.agent.memory.profile(user_id, ts)
        return {"remembered": True, "user_id": user_id, "memory": profile.summary() if profile else None}

    def forget(self, customer_id: str) -> dict:
        if self.agent.memory is not None and customer_id:
            self.agent.memory.forget(customer_id)
        return {"forgotten": customer_id}

    # ------------------------------------------------------------------ simulated conversation
    def simulate(self, customer_id: str, memory: bool = True) -> dict:
        from evaluator.local_evaluator import (coarse_category, customer_reply, initial_message,
                                               materialize_hidden_fields)
        customer = self._customer(customer_id)
        if customer is None or customer.get("target") not in self.raw:
            raise KeyError("pick one of the returning customers from /api/customers")
        target = customer["target"]
        sample = {"sample_id": f"demo_{customer_id}", "scenario_type": customer["scenario"],
                  "ground_truth": {"parent_asin": target}}
        card, behavior = materialize_hidden_fields(sample, self.raw)
        sample = {**sample, "intent_card": card, "behavior": behavior}
        opened = self.open_session(customer_id, memory)
        session_id = opened["session_id"]
        disclosed: set[str] = set()
        boundary_used = False
        override_applied = sample["scenario_type"] != "intent_override"
        message = initial_message(sample, coarse_category(self.raw[target].get("categories") or []), disclosed)
        turns, outcome = [], {"hit": False, "turn": None, "rank": None}
        for t in range(1, MAX_TURNS + 1):
            step = self.turn(session_id, message)
            ranked = [row["parent_asin"] for row in step["debug"].get("slate", [])]
            turns.append({"user": message, **step})
            if override_applied and target in ranked:
                outcome = {"hit": True, "turn": t, "rank": ranked.index(target) + 1}
                break
            override = behavior.get("override") or {}
            if not override_applied and t + 1 == int(override.get("turn", 3)):
                override_applied = True
                if override.get("new_value"):
                    disclosed.add(str(override["new_value"]))
                message = str(override.get("message"))
            else:
                message, boundary_used = customer_reply(sample, step["ask"], disclosed, boundary_used)
        info = self.agent.retriever.products[target]
        return {"session_id": session_id, "memory": opened["memory"], "scenario": sample["scenario_type"],
                "target": {"parent_asin": target, "title": info["title"], "store": info.get("store"),
                           "price": info.get("price")},
                "outcome": outcome, "turns": turns}


def config_catalog() -> str:
    return str(ROOT / "data" / "catalog.jsonl")


# ---------------------------------------------------------------------- HTTP plumbing
def make_handler(service: Service):
    class Handler(BaseHTTPRequestHandler):
        server_version = "DejaBuy/1.0"

        def log_message(self, fmt, *args):  # quiet
            pass

        def _send(self, status: int, body: bytes, content_type: str) -> None:
            self.send_response(status)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(body)

        def _json(self, status: int, payload) -> None:
            self._send(status, json.dumps(payload, ensure_ascii=False).encode("utf-8"), "application/json; charset=utf-8")

        def _body(self) -> dict:
            length = int(self.headers.get("Content-Length") or 0)
            if not length:
                return {}
            try:
                return json.loads(self.rfile.read(length) or b"{}")
            except json.JSONDecodeError:
                return {}

        def do_GET(self):
            if self.path in ("/", "/index.html"):
                return self._send(200, (STATIC / "index.html").read_bytes(), "text/html; charset=utf-8")
            if self.path == "/api/health":
                return self._json(200, {"ok": True, "memory_users": len(service.agent.memory or [])})
            if self.path == "/api/customers":
                return self._json(200, [{k: v for k, v in c.items() if k not in ("target", "as_of", "effect")}
                                        for c in service.customers])
            self._json(404, {"error": f"no route for GET {self.path}"})

        def do_POST(self):
            body = self._body()
            try:
                if self.path == "/api/session":
                    return self._json(200, service.open_session(body.get("customer_id"), body.get("memory", True)))
                if self.path == "/api/turn":
                    message = str(body.get("message") or "").strip()
                    if not message:
                        return self._json(400, {"error": "message is empty"})
                    return self._json(200, service.turn(str(body.get("session_id")), message))
                if self.path == "/api/simulate":
                    return self._json(200, service.simulate(body.get("customer_id"), body.get("memory", True)))
                if self.path == "/api/buy":
                    return self._json(200, service.buy(str(body.get("session_id")), str(body.get("parent_asin"))))
                if self.path == "/api/forget":
                    return self._json(200, service.forget(str(body.get("customer_id") or "")))
            except KeyError as error:
                return self._json(HTTPStatus.NOT_FOUND, {"error": str(error).strip("'")})
            self._json(404, {"error": f"no route for POST {self.path}"})

    return Handler


def main() -> None:
    parser = argparse.ArgumentParser(description="DejaBuy REST service + demo")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--catalog", default=config_catalog())
    args = parser.parse_args()
    t0 = time.time()
    service = Service(args.catalog)
    print(f"DejaBuy ready in {time.time() - t0:.0f}s · {len(service.customers) - 1} returning customers in the demo"
          f" · http://{args.host}:{args.port}")
    ThreadingHTTPServer((args.host, args.port), make_handler(service)).serve_forever()


if __name__ == "__main__":
    main()
