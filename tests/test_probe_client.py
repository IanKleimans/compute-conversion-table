"""Runs the streaming client and a full probe slot against a fake OpenRouter on localhost."""

import datetime as dt
import json
import os
import tempfile
import threading
import unittest
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

from grain import openrouter
from grain.probe import run_slot
from grain.tasks import PUBLIC_SEED, make_task

PROBE_CFG = {
    "tasks_per_slot": 4, "type_cycle": ["T1", "T2", "T1", "T3"],
    "target_input_tokens": {"T1": 1000, "T2": 16000, "T3": 1500},
    "max_tokens": {"T1": 1000, "T2": 1000, "T3": 4000}, "timeout_s": 10, "max_usd_per_run": 1.0,
    "interactive": {"max_ttft_s": 2.0, "min_tokens_per_s": {"M": 1}},
}
MODELS = {"models": [{"code": "M", "openrouter": "openai/gpt-oss-120b", "reasoning": {"effort": "low"},
                      "providers": ["good", "limited"]}]}


class FakeOpenRouter(BaseHTTPRequestHandler):
    answers: dict = {}

    def log_message(self, *args):
        pass

    def do_POST(self):
        body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        provider = body["provider"]["order"][0]
        assert body["provider"]["allow_fallbacks"] is False and body["stream"] is True
        if provider == "limited":
            self.send_response(429)
            self.end_headers()
            self.wfile.write(b'{"error":{"message":"rate limited"}}')
            return
        answer = self.answers.get(body["messages"][0]["content"], "no idea")
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        self.end_headers()
        events = [": OPENROUTER PROCESSING", "",
                  json.dumps({"provider": "Good", "choices": [{"delta": {"reasoning": "thinking"}}]})]
        for word in answer.split(" "):
            events.append(json.dumps({"provider": "Good", "choices": [{"delta": {"content": word + " "}}]}))
        events.append(json.dumps({"choices": [{"delta": {}, "finish_reason": "stop"}],
                                  "usage": {"prompt_tokens": 1000, "completion_tokens": 20, "cost": 0.0011}}))
        for e in events:
            line = e if e.startswith(":") or not e else "data: " + e
            self.wfile.write((line + "\n\n").encode())
            self.wfile.flush()
        self.wfile.write(b"data: [DONE]\n\n")


class ProbeTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        os.environ.pop("HTTPS_PROXY", None)
        os.environ.pop("https_proxy", None)
        os.environ["NO_PROXY"] = "127.0.0.1,localhost"
        cls.server = ThreadingHTTPServer(("127.0.0.1", 0), FakeOpenRouter)
        cls.url = f"http://127.0.0.1:{cls.server.server_address[1]}/api/v1/chat/completions"
        threading.Thread(target=cls.server.serve_forever, daemon=True).start()

    @classmethod
    def tearDownClass(cls):
        cls.server.shutdown()

    def call(self, provider, prompt="hi"):
        return openrouter.chat_stream("key", "openai/gpt-oss-120b", provider, prompt, 100, url=self.url)

    def test_stream_parsed(self):
        FakeOpenRouter.answers = {"hi": "ANSWER: 42"}
        r = self.call("good")
        self.assertIsNone(r["error"])
        self.assertEqual(r["text"].strip(), "ANSWER: 42")
        self.assertEqual(r["out_tokens"], 20)
        self.assertEqual(r["cost_usd"], 0.0011)
        self.assertEqual(r["served_provider"], "Good")
        self.assertIsNotNone(r["ttft_s"])
        self.assertEqual(r["reasoning_chars"], len("thinking"))

    def test_rate_limit_recorded(self):
        r = self.call("limited")
        self.assertEqual(r["http_status"], 429)
        self.assertEqual(r["error"]["code"], 429)

    def test_full_slot(self):
        day, slot = dt.date(2026, 12, 1), 8
        from grain.probe import task_indices
        tasks = [make_task(PUBLIC_SEED, i, PROBE_CFG["type_cycle"], PROBE_CFG["target_input_tokens"], "public")
                 for i in task_indices(day, slot, PROBE_CFG["tasks_per_slot"])]
        FakeOpenRouter.answers = {t.prompt: (f"ANSWER: {t.answer}" if t.kind == "number" else t.answer)
                                  for t in tasks}

        def client(*args):
            return openrouter.chat_stream(*args, url=self.url)

        with tempfile.TemporaryDirectory() as tmp:
            path = run_slot(MODELS, PROBE_CFG, "key", None, day, slot, Path(tmp), client=client,
                            log=lambda *a: None)
            rows = [json.loads(l) for l in path.read_text().splitlines()]
        self.assertEqual(len(rows), 8)
        good = [r for r in rows if r["provider"] == "good"]
        limited = [r for r in rows if r["provider"] == "limited"]
        self.assertTrue(all(r["passed"] for r in good), [r["answer_excerpt"] for r in good])
        self.assertTrue(all(not r["passed"] and r["error"]["code"] == 429 for r in limited))
        self.assertNotIn("prompt", rows[0])


if __name__ == "__main__":
    unittest.main()
