import json
import tempfile
import threading
import unittest
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

from acceptance import LogReader, parse_log_counters
from benchmark import draft_counters, mtp_metrics, report, run_one
from generate_prompts import SIZES, generate, make_prompt


class FakeOllama(BaseHTTPRequestHandler):
    log_path = None

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
             "eval_count": 10, "eval_duration": 1_000_000_000},
        ):
            if event["done"] and self.log_path:
                with Path(self.log_path).open("a", encoding="utf-8") as log:
                    log.write("slot print_timing: id 0 | task 1 | draft acceptance = 0.75000 (15 accepted / 20 generated)\n")
            self.wfile.write((json.dumps(event) + "\n").encode())
            self.wfile.flush()


class BenchmarkTests(unittest.TestCase):
    def test_prompt_byte_lengths_and_fact(self):
        self.assertEqual(SIZES, (100, 1000, 10000))
        for size in SIZES:
            prompt = make_prompt(size)
            self.assertEqual(len(prompt.encode("utf-8")), size)
            if size:
                self.assertRegex(prompt, r"The brass key was (blue|pink|gold)\.")
                self.assertIn("What color", prompt)
        with tempfile.TemporaryDirectory() as temp:
            generate(Path(temp))
            self.assertEqual({p.name for p in Path(temp).glob("*.txt")}, {"100.txt", "1000.txt", "10000.txt"})

    def test_stream_metrics_and_report(self):
        server = ThreadingHTTPServer(("127.0.0.1", 0), FakeOllama)
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        with tempfile.TemporaryDirectory() as temp:
            log_path = Path(temp) / "server.log"
            log_path.write_text("old unrelated output\n", encoding="utf-8")
            FakeOllama.log_path = log_path
            try:
                answer, sample = run_one(f"http://127.0.0.1:{server.server_port}", "test-mtp", "Hi", {"parameters": "draft_num_predict 4"}, LogReader(str(log_path)))
            finally:
                FakeOllama.log_path = None
                server.shutdown()
                server.server_close()
                thread.join()
        self.assertEqual(answer, "blue key")
        self.assertEqual(sample["metrics"]["prefill_tps"], 200)
        self.assertEqual(sample["metrics"]["eval_tps"], 10)
        self.assertEqual(sample["metrics"]["mtp_acceptance_rate"], 0.75)
        self.assertEqual(sample["metrics"]["acceptance_source"], str(log_path))
        self.assertIsNotNone(sample["metrics"]["ttft_s"])
        run = {"fqdn": "example.test", "endpoint": "http://example.test", "started_at": "start", "finished_at": "end", "rounds": 1, "models": ["test-mtp"], "prompts": ["100.txt"]}
        markdown = report(run, [{"model": "test-mtp", "prompt": "100.txt", "round": 1, "metrics": sample["metrics"]}])
        self.assertIn("75.00", markdown)
        self.assertIn("200.00", markdown)

    def test_mtp_unknown_without_evidence(self):
        self.assertEqual(mtp_metrics({}, {}, "model"), ("unknown", None))
        self.assertEqual(
            mtp_metrics({}, {"parameters": "draft_num_predict 4", "model_info": {"qwen35moe.nextn_predict_layers": 1}}, "unlabeled-model"),
            ("on", None),
        )

    def test_nested_api_counters_and_ambiguous_log(self):
        self.assertEqual(draft_counters({"timings": {"draft_n": 20, "draft_n_accepted": 15}}), (20, 15))
        line = "draft acceptance = 0.75000 (15 accepted / 20 generated)\n"
        self.assertIsNone(parse_log_counters(line + line))


if __name__ == "__main__":
    unittest.main()
