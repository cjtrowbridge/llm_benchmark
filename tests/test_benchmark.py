import json
import threading
import unittest
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

from benchmark import mtp_metrics, report, run_one
from generate_prompts import SIZES, make_prompt


class FakeOllama(BaseHTTPRequestHandler):
    def log_message(self, *_args):
        pass

    def do_POST(self):
        request = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        assert self.path == "/api/generate"
        assert request["stream"] is True
        self.send_response(200)
        self.send_header("Content-Type", "application/x-ndjson")
        self.end_headers()
        for event in (
            {"response": "blue", "done": False},
            {"response": " key", "done": False},
            {"response": "", "done": True, "prompt_eval_count": 100, "prompt_eval_duration": 500_000_000,
             "eval_count": 10, "eval_duration": 1_000_000_000, "draft_n": 20, "draft_n_accepted": 15},
        ):
            self.wfile.write((json.dumps(event) + "\n").encode())
            self.wfile.flush()


class BenchmarkTests(unittest.TestCase):
    def test_prompt_byte_lengths_and_fact(self):
        for size in SIZES:
            prompt = make_prompt(size)
            self.assertEqual(len(prompt.encode("utf-8")), size)
            if size:
                self.assertRegex(prompt, r"The brass key was (blue|pink|gold)\.")
                self.assertIn("What color", prompt)

    def test_stream_metrics_and_report(self):
        server = ThreadingHTTPServer(("127.0.0.1", 0), FakeOllama)
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        try:
            answer, sample = run_one(f"http://127.0.0.1:{server.server_port}", "test-mtp", "Hi", {"parameters": "draft_num_predict 4"})
        finally:
            server.shutdown()
            server.server_close()
            thread.join()
        self.assertEqual(answer, "blue key")
        self.assertEqual(sample["metrics"]["prefill_tps"], 200)
        self.assertEqual(sample["metrics"]["eval_tps"], 10)
        self.assertEqual(sample["metrics"]["mtp_acceptance_rate"], 0.75)
        self.assertIsNotNone(sample["metrics"]["ttft_s"])
        run = {"fqdn": "example.test", "endpoint": "http://example.test", "started_at": "start", "finished_at": "end", "rounds": 1, "models": ["test-mtp"], "prompts": ["100.txt"]}
        markdown = report(run, [{"model": "test-mtp", "prompt": "100.txt", "round": 1, "metrics": sample["metrics"]}])
        self.assertIn("75.00", markdown)
        self.assertIn("200.00", markdown)

    def test_mtp_unknown_without_evidence(self):
        self.assertEqual(mtp_metrics({}, {}, "model"), ("unknown", None))


if __name__ == "__main__":
    unittest.main()
