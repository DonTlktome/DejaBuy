"""LLM client: on-disk cache replays without a key; JSON fences are tolerated."""

import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from dejabuy.llm import client


class ClientTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        path = str(Path(self.tmp.name) / "cache.jsonl")
        for patcher in (mock.patch.object(client.config, "LLM_CACHE_PATH", path),
                        mock.patch.object(client.config, "LLM_API_KEY", ""),
                        mock.patch.object(client, "_cache", None)):
            patcher.start()
            self.addCleanup(patcher.stop)

    def test_no_key_and_no_cache_returns_none(self):
        self.assertIsNone(client.chat_json("sys", "user"))

    def test_cache_replays_without_key(self):
        key = client._cache_key(client.config.LLM_MODEL, "sys", "user", 0.0, 200)
        client._remember(key, {"act": "disclose"}, {"user": "user"})
        self.assertEqual(client.chat_json("sys", "user"), {"act": "disclose"})
        # the cache file survives a fresh process (reload from disk)
        client._cache = None
        self.assertEqual(client.chat_json("sys", "user"), {"act": "disclose"})
        self.assertIsNone(client.chat_json("sys", "other user"))

    def test_json_fence_tolerated(self):
        self.assertEqual(client._parse_json('```json\n{"a": 1}\n```'), {"a": 1})
        self.assertEqual(client._parse_json('{"a": 1}'), {"a": 1})
        self.assertIsNone(client._parse_json("[1, 2]"))
        self.assertIsNone(client._parse_json("not json"))


if __name__ == "__main__":
    unittest.main()
