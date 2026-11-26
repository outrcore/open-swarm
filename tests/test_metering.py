from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
import inspect
import sys
from pathlib import Path
import tempfile
import threading
import subprocess
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from metering import PricingSchedule, RunMeter, SpendCapExceeded
from swarm_run import apply_file_blocks, build_shared_repo_prefix, openai_chat, run_swarm


class OpenSwarmMeteringTest(unittest.TestCase):
    def test_cost_and_cache_accounting(self) -> None:
        meter = RunMeter(PricingSchedule(0.055, 0.0125, 0.50), spend_cap_usd=1.0)
        meter.record_usage("coder-a", prompt_tokens=10_000, cached_tokens=9_500, completion_tokens=1_000, gen_s=5.0, ttft_ms=25.0)
        meter.record_usage("coder-b", prompt_tokens=10_000, cached_tokens=9_500, completion_tokens=1_000, gen_s=4.0, ttft_ms=20.0)
        s = meter.summary(wall_s=6.0)
        self.assertEqual(s["prompt_tokens"], 20_000)
        self.assertEqual(s["uncached_input_tokens"], 1_000)
        self.assertEqual(s["cached_input_tokens"], 19_000)
        self.assertEqual(s["cache_hit_pct"], 95.0)
        # Expected: (1000*0.055 + 19000*0.0125 + 2000*0.50) / 1e6 = 0.0012925
        self.assertAlmostEqual(s["cost_usd"], 0.001293, places=6)
        self.assertEqual(s["mean_ttft_ms"], 22.5)

    def test_spend_cap_enforced(self) -> None:
        meter = RunMeter(PricingSchedule(0.055, 0.0125, 0.50), spend_cap_usd=0.001)
        with self.assertRaises(SpendCapExceeded):
            meter.record_usage("coder-a", prompt_tokens=100, cached_tokens=0, completion_tokens=5000, gen_s=1.0)

    def test_apply_file_blocks_confines_paths(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            txt = (
                "```file:pkg/mod.py\ndef add(a, b):\n    return a + b\n```\n"
                "```file:../escape.py\nprint('bad')\n```\n"
            )
            written = apply_file_blocks(root, txt)
            self.assertEqual(written, ["pkg/mod.py"])
            self.assertTrue((root / "pkg" / "mod.py").is_file())
            self.assertFalse((root.parent / "escape.py").exists())
            prefix = build_shared_repo_prefix(root)
            self.assertIn("pkg/mod.py", prefix)

    def test_build_shared_repo_prefix_caps_chars_and_prioritizes_readme(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            (root / "README.md").write_text("# Open Swarm\nCore doc\n", encoding="utf-8")
            (root / "a_large.py").write_text("x = 1\n" * 600, encoding="utf-8")
            (root / "b_small.py").write_text("y = 2\n" * 100, encoding="utf-8")
            prefix = build_shared_repo_prefix(root, max_total_chars=800)
            self.assertIn("README.md", prefix)
            self.assertIn("Core doc", prefix)
            self.assertNotIn("x = 1\n" * 10, prefix)

    def test_openai_chat_parses_sse_and_reasoning(self) -> None:
        class SSEHandler(BaseHTTPRequestHandler):
            def log_message(self, format: str, *args) -> None:
                pass

            def do_POST(self) -> None:
                body = (
                    'data: {"choices":[{"delta":{"reasoning_content":"plan "}}]}\n\n'
                    'data: {"choices":[{"delta":{"content":"```file:a.py\\npass\\n```"}}]}\n\n'
                    'data: {"choices":[{"delta":{},"finish_reason":"stop"}],"usage":{"prompt_tokens":50,"completion_tokens":10,"prompt_tokens_details":{"cached_tokens":40}}}\n\n'
                    'data: [DONE]\n\n'
                ).encode("utf-8")
                self.send_response(200)
                self.send_header("Content-Type", "text/event-stream")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)

        srv = ThreadingHTTPServer(("127.0.0.1", 0), SSEHandler)
        t = threading.Thread(target=srv.serve_forever, daemon=True)
        t.start()
        try:
            res = openai_chat(
                base_url=f"http://127.0.0.1:{srv.server_port}/v1",
                api_key="test",
                model="test-model",
                messages=[{"role": "user", "content": "hi"}],
                stream=True,
            )
            self.assertEqual(res["reasoning_content"], "plan ")
            self.assertIn("```file:a.py", res["content"])
            self.assertEqual(res["prompt_tokens"], 50)
            self.assertEqual(res["cached_tokens"], 40)
            self.assertEqual(res["completion_tokens"], 10)
            self.assertGreater(res["tok_s"], 0.0)
        finally:
            srv.shutdown()
            srv.server_close()

    def test_apply_file_blocks_recovers_trailing_unclosed_fence_safely(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            txt = (
                "```file:ok.py\nprint('closed')\n```\n"
                "```file:../evil.py\nprint('no')\n"
                "```file:partial.py\ndef f():\n    return 42\n"
            )
            written = apply_file_blocks(root, txt)
            self.assertEqual(written, ["ok.py", "partial.py"])
            self.assertFalse((root.parent / "evil.py").exists())
            self.assertIn("return 42", (root / "partial.py").read_text(encoding="utf-8"))

    def test_run_swarm_emits_progress_and_reports_empty_blocks_error(self) -> None:
        self.assertEqual(inspect.signature(run_swarm).parameters["max_tokens"].default, 4096)

        class EmptySSEHandler(BaseHTTPRequestHandler):
            def log_message(self, format: str, *args) -> None:
                pass

            def do_POST(self) -> None:
                body = (
                    'data: {"choices":[{"delta":{"reasoning_content":"too much thinking"}}]}\n\n'
                    'data: {"choices":[{"delta":{},"finish_reason":"length"}],"usage":{"prompt_tokens":20,"completion_tokens":4096,"prompt_tokens_details":{"cached_tokens":0}}}\n\n'
                    'data: [DONE]\n\n'
                ).encode("utf-8")
                self.send_response(200)
                self.send_header("Content-Type", "text/event-stream")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)

        srv = ThreadingHTTPServer(("127.0.0.1", 0), EmptySSEHandler)
        threading.Thread(target=srv.serve_forever, daemon=True).start()
        events: list[dict] = []
        try:
            with tempfile.TemporaryDirectory() as td:
                repo = Path(td)
                subprocess.run(["git", "init"], cwd=str(repo), capture_output=True, check=True)
                subprocess.run(["git", "config", "user.name", "Test"], cwd=str(repo), capture_output=True, check=True)
                subprocess.run(["git", "config", "user.email", "t@example.com"], cwd=str(repo), capture_output=True, check=True)
                (repo / "README.md").write_text("# Test\n", encoding="utf-8")
                subprocess.run(["git", "add", "README.md"], cwd=str(repo), capture_output=True, check=True)
                subprocess.run(["git", "commit", "-m", "init"], cwd=str(repo), capture_output=True, check=True)

                res = run_swarm(
                    repo=repo,
                    tasks=[{"title": "T1", "spec": "make x.py", "acceptance_cmd": "true"}],
                    base_url=f"http://127.0.0.1:{srv.server_port}/v1",
                    model="test-model",
                    on_progress=events.append,
                )
            self.assertEqual(res["passed_count"], 0)
            r0 = res["results"][0]
            self.assertEqual(r0["stage"], "failed")
            self.assertIsNotNone(r0["error"])
            self.assertIn("No ```file:...``` blocks emitted", r0["error"])
            stages = [e.get("stage") for e in events if e.get("kind") == "agent_stage"]
            self.assertIn("worktree", stages)
            self.assertIn("thinking", stages)
            self.assertIn("failed", stages)
        finally:
            srv.shutdown()
            srv.server_close()

    def test_run_swarm_self_heals_failing_test(self) -> None:
        call_count = [0]

        class SelfHealHandler(BaseHTTPRequestHandler):
            def log_message(self, format: str, *args) -> None:
                pass

            def do_POST(self) -> None:
                n = int(self.headers.get("Content-Length", "0") or "0")
                _ = self.rfile.read(n) if n > 0 else b"{}"
                call_count[0] += 1
                if call_count[0] == 1:
                    code = "```file:mod.py\ndef add(a, b):\n    return a - b\n```\n"
                else:
                    code = "```file:mod.py\ndef add(a, b):\n    return a + b\n```\n"
                frames = [
                    {"choices": [{"delta": {"content": code}}]},
                    {
                        "choices": [{"delta": {}, "finish_reason": "stop"}],
                        "usage": {"prompt_tokens": 30, "completion_tokens": 20, "prompt_tokens_details": {"cached_tokens": 10}},
                    },
                ]
                body = ("".join(f"data: {json.dumps(f)}\n\n" for f in frames) + "data: [DONE]\n\n").encode("utf-8")
                self.send_response(200)
                self.send_header("Content-Type", "text/event-stream")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)

        srv = ThreadingHTTPServer(("127.0.0.1", 0), SelfHealHandler)
        threading.Thread(target=srv.serve_forever, daemon=True).start()
        try:
            with tempfile.TemporaryDirectory() as td:
                repo = Path(td)
                subprocess.run(["git", "init"], cwd=str(repo), capture_output=True, check=True)
                subprocess.run(["git", "config", "user.name", "Test"], cwd=str(repo), capture_output=True, check=True)
                subprocess.run(["git", "config", "user.email", "t@example.com"], cwd=str(repo), capture_output=True, check=True)
                (repo / "README.md").write_text("# Test\n", encoding="utf-8")
                subprocess.run(["git", "add", "README.md"], cwd=str(repo), capture_output=True, check=True)
                subprocess.run(["git", "commit", "-m", "init"], cwd=str(repo), capture_output=True, check=True)

                res = run_swarm(
                    repo=repo,
                    tasks=[
                        {
                            "title": "Fix add",
                            "spec": "Create mod.py with add(a, b)",
                            "acceptance_cmd": '/usr/local/bin/python3 -c "import mod; assert mod.add(2, 3) == 5"',
                        }
                    ],
                    base_url=f"http://127.0.0.1:{srv.server_port}/v1",
                    model="test-model",
                    max_repair_turns=2,
                )
            self.assertEqual(res["passed_count"], 1)
            r0 = res["results"][0]
            self.assertTrue(r0["tests_ok"])
            self.assertEqual(r0["repair_attempts"], 1)
            self.assertEqual(r0["written_files"], ["mod.py"])
        finally:
            srv.shutdown()
            srv.server_close()


if __name__ == "__main__":
    unittest.main()
