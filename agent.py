"""Root shim: `from agent import Agent` as the evaluator expects."""

from dejabuy.agent import DejaBuyAgent as Agent  # noqa: F401
