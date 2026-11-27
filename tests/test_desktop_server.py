#!/usr/bin/env python3
"""Unit tests for open_swarm/desktop_server.py (offline, no GPU required)."""
from __future__ import annotations

from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
import os
import subprocess
import sys
import tempfile
import threading
import time
import unittest
from unittest import mock
import urllib.error
import urllib.request
from pathlib import Path

SWARM_DIR = Path(__file__).resolve().parents[1]
if str(SWARM_DIR) not in sys.path:
    sys.path.insert(0, str(SWARM_DIR))

from continuity_store import compute_file_hashes  # noqa: E402
from desktop_server import DesktopAppState, start_server  # noqa: E402


class _MockOpenAIBackend(BaseHTTPRequestHandler):
    def log_message(self, format: str, *args) -> None:  # noqa: A002
        pass

    def do_POST(self) -> None:  # noqa: N802
        n = int(self.headers.get("Content-Length", "0") or "0")
        req = json.loads((self.rfile.read(n) if n > 0 else b"{}").decode("utf-8"))
        if req.get("stream"):
            frames = [
                {"choices": [{"delta": {"reasoning_content": "Let me think..."}}]},
                {"choices": [{"delta": {"content": "Hello "}}]},
                {"choices": [{"delta": {"content": "world!"}}]},
                {
                    "choices": [{"delta": {}, "finish_reason": "stop"}],
                    "usage": {"prompt_tokens": 25, "completion_tokens": 12, "prompt_tokens_details": {"cached_tokens": 15}},
                },
            ]
            body = "".join(f"data: {json.dumps(f)}\n\n" for f in frames) + "data: [DONE]\n\n"
            raw = body.encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Type", "text/event-stream")
            self.send_header("Content-Length", str(len(raw)))
            self.end_headers()
            self.wfile.write(raw)
        else:
            raw = json.dumps(
                {
                    "choices": [{"message": {"role": "assistant", "reasoning_content": "step1", "content": "answer"}}],
                    "usage": {"prompt_tokens": 20, "completion_tokens": 10, "prompt_tokens_details": {"cached_tokens": 10}},
                }
            ).encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(raw)))
            self.end_headers()
            self.wfile.write(raw)


def _http_json(url: str, method: str = "GET", payload: dict | None = None) -> tuple[int, dict]:
    data = json.dumps(payload).encode("utf-8") if payload is not None else None
    hdrs = {"Content-Type": "application/json"} if data is not None else {}
    req = urllib.request.Request(url, data=data, headers=hdrs, method=method)
    with urllib.request.urlopen(req, timeout=5.0) as resp:
        return int(resp.status), json.loads(resp.read().decode("utf-8"))


class DesktopServerTests(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        self.tmp_dir = Path(self.tmp.name)
        self.repo_dir = self.tmp_dir / "repo"
        self.repo_dir.mkdir()
        subprocess.run(["git", "init"], cwd=str(self.repo_dir), capture_output=True, check=True)
        subprocess.run(["git", "config", "user.name", "Test"], cwd=str(self.repo_dir), capture_output=True, check=True)
        subprocess.run(["git", "config", "user.email", "t@example.com"], cwd=str(self.repo_dir), capture_output=True, check=True)
        (self.repo_dir / "README.md").write_text("# Test\n", encoding="utf-8")
        subprocess.run(["git", "add", "README.md"], cwd=str(self.repo_dir), capture_output=True, check=True)
        subprocess.run(["git", "commit", "-m", "init"], cwd=str(self.repo_dir), capture_output=True, check=True)

        def fake_runner(**kwargs):
            return {
                "run_id": "fake",
                "repo": str(kwargs["repo"]),
                "backend_url": kwargs["base_url"],
                "model": kwargs["model"],
                "agents_count": len(kwargs["tasks"]),
                "passed_count": 1,
                "metering": {"aggregate_tok_s": 88.5, "cache_hit_pct": 91.2, "cost_usd": 0.0012},
                "results": [
                    {
                        "agent_id": "coder-a",
                        "ticket": kwargs["tasks"][0]["title"],
                        "worktree": str(self.tmp_dir / "nonexistent_wt"),
                        "written_files": ["README.md"],
                        "tests_ok": True,
                        "test_output": "OK",
                        "diff": (
                            "diff --git a/hello.txt b/hello.txt\n"
                            "new file mode 100644\n"
                            "index 0000000..e69de29\n"
                            "--- /dev/null\n"
                            "+++ b/hello.txt\n"
                            "@@ -0,0 +1 @@\n"
                            "+open swarm ok\n"
                        ),
                    }
                ],
            }

        self.state = DesktopAppState(runs_dir=self.tmp_dir / "runs", runner_fn=fake_runner)
        self.srv = start_server(host="127.0.0.1", port=0, state=self.state)
        self.port = self.srv.server_address[1]
        self.thread = threading.Thread(target=self.srv.serve_forever, daemon=True)
        self.thread.start()
        self.base = f"http://127.0.0.1:{self.port}"

    def tearDown(self) -> None:
        self.srv.shutdown()
        self.srv.server_close()
        self.tmp.cleanup()

    def test_health_ui_and_config(self) -> None:
        st, health = _http_json(f"{self.base}/health")
        self.assertEqual(st, 200)
        self.assertEqual(health["status"], "ok")

        st_c, cfg = _http_json(f"{self.base}/api/config")
        self.assertEqual(st_c, 200)
        self.assertGreaterEqual(len(cfg["presets"]), 2)
        urls = [p["backend_url"] for p in cfg["presets"]]
        self.assertTrue(any("api.outrcore.com" in u for u in urls))

        # Custom presets.json overrides cleanly
        custom_presets = self.tmp_dir / "custom_presets.json"
        custom_presets.write_text(
            json.dumps([{"id": "custom_test", "label": "Custom", "backend_url": "http://127.0.0.1:9999/v1", "model": "m1"}]),
            encoding="utf-8",
        )
        with mock.patch.dict(os.environ, {"OPEN_SWARM_PRESETS_PATH": str(custom_presets)}):
            st_c2, cfg2 = _http_json(f"{self.base}/api/config")
        self.assertEqual(st_c2, 200)
        self.assertEqual(cfg2["presets"][0]["id"], "custom_test")

        with urllib.request.urlopen(f"{self.base}/", timeout=5.0) as resp:
            html = resp.read().decode("utf-8")
            self.assertEqual(resp.status, 200)
            self.assertIn("Open Swarm", html)

    def test_start_poll_and_apply_diff(self) -> None:
        st, run = _http_json(
            f"{self.base}/api/runs",
            method="POST",
            payload={
                "repo": str(self.repo_dir),
                "backend_url": "https://api.outrcore.com/v1",
                "model": "thinkingmachines/Inkling-2-Mini",
                "spend_cap_usd": 0.25,
                "tasks": [{"title": "Add hello.txt", "spec": "write hello", "acceptance_cmd": "true"}],
            },
        )
        self.assertEqual(st, 202)
        rid = run["run_id"]

        for _ in range(40):
            _, cur = _http_json(f"{self.base}/api/runs/{rid}")
            if cur["status"] == "completed":
                break
            time.sleep(0.05)
        self.assertEqual(cur["status"], "completed")
        self.assertEqual(cur["passed_count"], 1)

        st_a, ap = _http_json(f"{self.base}/api/runs/{rid}/apply", method="POST", payload={"agent_id": "coder-a"})
        self.assertEqual(st_a, 200)
        self.assertIn("coder-a", ap["applied_agents"])
        self.assertEqual((self.repo_dir / "hello.txt").read_text(encoding="utf-8").strip(), "open swarm ok")

    def test_chat_and_streaming_chat_with_thinking_and_tok_s(self) -> None:
        mock_srv = ThreadingHTTPServer(("127.0.0.1", 0), _MockOpenAIBackend)
        m_port = mock_srv.server_address[1]
        threading.Thread(target=mock_srv.serve_forever, daemon=True).start()
        try:
            st, nonstream = _http_json(
                f"{self.base}/api/chat",
                method="POST",
                payload={
                    "backend_url": f"http://127.0.0.1:{m_port}/v1",
                    "model": "thinkingmachines/Inkling-2-Mini",
                    "messages": [{"role": "user", "content": "hi"}],
                },
            )
            self.assertEqual(st, 200)
            self.assertEqual(nonstream["reasoning_content"], "step1")
            self.assertIsInstance(nonstream["tok_s"], (int, float))
            self.assertGreater(nonstream["tok_s"], 0.0)

            req = urllib.request.Request(
                f"{self.base}/api/chat/stream",
                data=json.dumps(
                    {
                        "backend_url": f"http://127.0.0.1:{m_port}/v1",
                        "model": "thinkingmachines/Inkling-2-Mini",
                        "messages": [{"role": "user", "content": "hi"}],
                    }
                ).encode("utf-8"),
                headers={"Content-Type": "application/json"},
                method="POST",
            )
            with urllib.request.urlopen(req, timeout=5.0) as resp:
                sse = resp.read().decode("utf-8")
            self.assertIn("event: thinking", sse)
            self.assertIn("Let me think...", sse)
            self.assertIn("event: content", sse)
            self.assertIn("Hello ", sse)
            self.assertIn("event: done", sse)
            self.assertIn('"tok_s":', sse)

            st_t, tel = _http_json(f"{self.base}/api/chat/telemetry")
            self.assertEqual(st_t, 200)
            self.assertGreaterEqual(tel["turns"], 2)
            self.assertIsNotNone(tel["avg_tok_s"])
        finally:
            mock_srv.shutdown()
            mock_srv.server_close()

    def test_self_build_isolated_repo_and_preview_lifecycle(self) -> None:
        self.state.preview.port = 18797
        try:
            st, sb = _http_json(f"{self.base}/api/self-build-repo", method="POST", payload={})
            self.assertEqual(st, 200)
            sb_path = Path(sb["path"])
            self.assertTrue((sb_path / ".git").exists())
            self.assertTrue((sb_path / "desktop_server.py").is_file())
            self.assertTrue((sb_path / "desktop_ui.html").is_file())

            st_p, prev = _http_json(
                f"{self.base}/api/preview/start",
                method="POST",
                payload={"repo": str(sb_path)},
            )
            self.assertEqual(st_p, 200)
            self.assertTrue(prev["running"])
            self.assertTrue(prev["healthy"])

            st_s, stat = _http_json(f"{self.base}/api/preview/status")
            self.assertEqual(st_s, 200)
            self.assertTrue(stat["healthy"])
        finally:
            _http_json(f"{self.base}/api/preview/stop", method="POST", payload={})

    def test_fleet_continuity_review_context_guard_and_tournament(self) -> None:
        st_cfg, cfg = _http_json(f"{self.base}/api/config")
        self.assertEqual(st_cfg, 200)
        self.assertIn("lanes", cfg)
        self.assertIn("coder", cfg["lanes"])

        st_fl, fleet = _http_json(f"{self.base}/api/fleet")
        self.assertEqual(st_fl, 200)
        self.assertIn("counts", fleet)
        self.assertIn("projects", fleet)

        st_cp, cp = _http_json(
            f"{self.base}/api/continuity/checkpoint",
            method="POST",
            payload={
                "repo": str(self.repo_dir),
                "role": "orchestrator",
                "summary": "Initial checkpoint",
                "next_step": "Run swarm",
            },
        )
        self.assertEqual(st_cp, 200)
        self.assertEqual(cp["revision"], 1)

        st_msg, msg = _http_json(
            f"{self.base}/api/continuity/messages",
            method="POST",
            payload={
                "repo": str(self.repo_dir),
                "from_role": "orchestrator",
                "to_role": "coder-a",
                "body": "Keep diff minimal",
            },
        )
        self.assertEqual(st_msg, 200)
        self.assertEqual(msg["to_role"], "coder-a")

        st_cs, cstat = _http_json(
            f"{self.base}/api/continuity/status?repo={urllib.request.quote(str(self.repo_dir), safe='')}"
        )
        self.assertEqual(st_cs, 200)
        self.assertEqual(len(cstat["checkpoints"]), 1)
        self.assertEqual(len(cstat["messages"]), 1)

        # Mock OpenAI endpoint returning valid review JSON & blind tournament judge JSON
        class _CriticAndJudgeHandler(BaseHTTPRequestHandler):
            def log_message(self, format: str, *args) -> None:
                pass

            def do_POST(self) -> None:
                n = int(self.headers.get("Content-Length", "0") or "0")
                req = json.loads((self.rfile.read(n) if n > 0 else b"{}").decode("utf-8"))
                sys_prompt = ((req.get("messages") or [{}])[0].get("content") or "").lower()
                if "judge" in sys_prompt or "candidate" in sys_prompt:
                    reply = json.dumps(
                        {
                            "winner": "Candidate A",
                            "rationale": "Cleaner minimal diff",
                            "rankings": [
                                {"label": "Candidate A", "score": 9, "strengths": "Concise", "objection": "None"},
                                {"label": "Candidate B", "score": 7, "strengths": "Safe", "objection": "Verbose"},
                            ],
                        }
                    )
                else:
                    reply = json.dumps(
                        {
                            "verdict": "approve",
                            "summary": "Looks solid",
                            "findings": [],
                        }
                    )
                raw = json.dumps(
                    {
                        "choices": [{"message": {"role": "assistant", "content": reply}}],
                        "usage": {"prompt_tokens": 30, "completion_tokens": 20},
                    }
                ).encode("utf-8")
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(raw)))
                self.end_headers()
                self.wfile.write(raw)

        mock_srv = ThreadingHTTPServer(("127.0.0.1", 0), _CriticAndJudgeHandler)
        m_port = mock_srv.server_address[1]
        threading.Thread(target=mock_srv.serve_forever, daemon=True).start()
        try:
            # 1) Tournament mode run end-to-end
            def tourney_runner(**kwargs):
                return {
                    "run_id": "t-fake",
                    "repo": str(kwargs["repo"]),
                    "backend_url": kwargs["base_url"],
                    "model": kwargs["model"],
                    "agents_count": len(kwargs["tasks"]),
                    "passed_count": 2,
                    "metering": {"aggregate_tok_s": 55.0, "cache_hit_pct": 80.0, "cost_usd": 0.001},
                    "results": [
                        {
                            "agent_id": "cand-a",
                            "ticket": "T-001",
                            "title": "Build a.txt",
                            "model": "thinkingmachines/Inkling-2-Mini",
                            "worktree": str(self.tmp_dir / "wt-a"),
                            "written_files": ["a.txt"],
                            "base_file_hashes": compute_file_hashes(self.repo_dir, ["a.txt"]),
                            "tests_ok": True,
                            "test_output": "OK",
                            "diff": (
                                "diff --git a/a.txt b/a.txt\n"
                                "new file mode 100644\n"
                                "index 0000000..ce01362\n"
                                "--- /dev/null\n"
                                "+++ b/a.txt\n"
                                "@@ -0,0 +1 @@\n"
                                "+hello\n"
                            ),
                        },
                        {
                            "agent_id": "cand-b",
                            "ticket": "T-001",
                            "title": "Build a.txt",
                            "model": "thinkingmachines/Inkling-2-Mini",
                            "worktree": str(self.tmp_dir / "wt-b"),
                            "written_files": ["a.txt"],
                            "tests_ok": True,
                            "test_output": "OK",
                            "diff": "diff --git a/a.txt b/a.txt\n--- /dev/null\n+++ b/a.txt\n@@ -0,0 +1 @@\n+hi\n",
                        },
                        {
                            "agent_id": "cand-c",
                            "ticket": "T-001",
                            "title": "Build a.txt",
                            "model": "thinkingmachines/Inkling-2-Mini",
                            "worktree": str(self.tmp_dir / "wt-c"),
                            "written_files": [],
                            "tests_ok": False,
                            "test_output": "FAIL",
                            "diff": "",
                        },
                    ],
                }

            orig_runner = self.state.runner_fn
            self.state.runner_fn = tourney_runner
            try:
                st_tr, tr = _http_json(
                    f"{self.base}/api/runs",
                    method="POST",
                    payload={
                        "repo": str(self.repo_dir),
                        "backend_url": f"http://127.0.0.1:{m_port}/v1",
                        "model": "thinkingmachines/Inkling-2-Mini",
                        "mode": "tournament",
                        "tasks": [{"title": "Build a.txt", "spec": "write a.txt", "acceptance_cmd": "true"}],
                    },
                )
                self.assertEqual(st_tr, 202)
                rid = tr["run_id"]
                final = tr
                for _ in range(50):
                    _, final = _http_json(f"{self.base}/api/runs/{rid}")
                    if final["status"] == "completed":
                        break
                    time.sleep(0.05)
                self.assertEqual(final["status"], "completed")
                self.assertEqual(final["mode"], "tournament")
                self.assertEqual(final["tournament"]["winner_agent_id"], "cand-a")
                self.assertEqual(len(final["tournament"]["disqualified"]), 1)

                # 2) Critic review route
                st_rv, rv = _http_json(
                    f"{self.base}/api/runs/{rid}/review",
                    method="POST",
                    payload={
                        "agent_id": "cand-a",
                        "reviewer_model": "gpt-6-sol",
                        "reviewer_base_url": f"http://127.0.0.1:{m_port}/v1",
                    },
                )
                self.assertEqual(st_rv, 200)
                self.assertEqual(rv["verdict"], "approve")
                self.assertTrue(rv["eligibility"]["cross_vendor"])

                # 3) Context guard conflict (409) when target modified externally, then force=True succeeds
                (self.repo_dir / "a.txt").write_text("external edit\n", encoding="utf-8")
                with self.assertRaises(urllib.error.HTTPError) as ctx:
                    _http_json(
                        f"{self.base}/api/runs/{rid}/apply",
                        method="POST",
                        payload={"agent_id": "cand-a", "force": False},
                    )
                self.assertEqual(ctx.exception.code, 409)
                ctx.exception.close()

                # Remove untracked file or keep force=True test cleanly: git apply needs non-colliding file or revert
                (self.repo_dir / "a.txt").unlink()
                # Simulate conflict via README hash while applying non-colliding a.txt diff:
                rec_obj = self.state.get_run(rid)
                rec_obj["results"][0]["base_file_hashes"]["README.md"] = "deadbeef"
                self.state._save_run(rec_obj)
                with self.assertRaises(urllib.error.HTTPError) as ctx2:
                    _http_json(
                        f"{self.base}/api/runs/{rid}/apply",
                        method="POST",
                        payload={"agent_id": "cand-a", "force": False},
                    )
                self.assertEqual(ctx2.exception.code, 409)
                ctx2.exception.close()

                st_ap, ap = _http_json(
                    f"{self.base}/api/runs/{rid}/apply",
                    method="POST",
                    payload={"agent_id": "cand-a", "force": True},
                )
                self.assertEqual(st_ap, 200)
                self.assertEqual((self.repo_dir / "a.txt").read_text(encoding="utf-8").strip(), "hello")
            finally:
                self.state.runner_fn = orig_runner
        finally:
            mock_srv.shutdown()
            mock_srv.server_close()

    def test_goal_decomposition_plan_and_apply_all(self) -> None:
        class _PlannerHandler(BaseHTTPRequestHandler):
            def log_message(self, format: str, *args) -> None:
                pass

            def do_POST(self) -> None:
                plan = [
                    {"title": "Add calc", "spec": "write pkg/calc.py", "acceptance_cmd": "true", "lane": "coder", "owned_files": ["pkg/calc.py"]},
                    {"title": "Add note", "spec": "write pkg/note.txt", "acceptance_cmd": "true", "lane": "mechanical", "owned_files": ["pkg/note.txt"]},
                ]
                raw = json.dumps(
                    {
                        "choices": [{"message": {"role": "assistant", "content": json.dumps(plan)}}],
                        "usage": {"prompt_tokens": 15, "completion_tokens": 25},
                    }
                ).encode("utf-8")
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(raw)))
                self.end_headers()
                self.wfile.write(raw)

        mock_srv = ThreadingHTTPServer(("127.0.0.1", 0), _PlannerHandler)
        m_port = mock_srv.server_address[1]
        threading.Thread(target=mock_srv.serve_forever, daemon=True).start()
        try:
            st_p, planned = _http_json(
                f"{self.base}/api/plan",
                method="POST",
                payload={
                    "repo": str(self.repo_dir),
                    "goal": "Add calc and note modules",
                    "backend_url": f"http://127.0.0.1:{m_port}/v1",
                    "model": "test-model",
                },
            )
            self.assertEqual(st_p, 200)
            self.assertEqual(len(planned["tasks"]), 2)

            def two_agent_runner(**kwargs):
                return {
                    "run_id": "goal-fake",
                    "repo": str(kwargs["repo"]),
                    "backend_url": kwargs["base_url"],
                    "model": kwargs["model"],
                    "agents_count": len(kwargs["tasks"]),
                    "passed_count": 2,
                    "metering": {"aggregate_tok_s": 60.0, "cache_hit_pct": 90.0, "cost_usd": 0.001},
                    "results": [
                        {
                            "agent_id": "coder-a",
                            "ticket": "T-001",
                            "title": kwargs["tasks"][0]["title"],
                            "worktree": str(self.tmp_dir / "wta"),
                            "written_files": ["calc.txt"],
                            "tests_ok": True,
                            "test_output": "OK",
                            "diff": "diff --git a/calc.txt b/calc.txt\nnew file mode 100644\nindex 0000000..ce01362\n--- /dev/null\n+++ b/calc.txt\n@@ -0,0 +1 @@\n+hello\n",
                        },
                        {
                            "agent_id": "coder-b",
                            "ticket": "T-002",
                            "title": kwargs["tasks"][1]["title"],
                            "worktree": str(self.tmp_dir / "wtb"),
                            "written_files": ["note.txt"],
                            "tests_ok": True,
                            "test_output": "OK",
                            "diff": "diff --git a/note.txt b/note.txt\nnew file mode 100644\nindex 0000000..ce01362\n--- /dev/null\n+++ b/note.txt\n@@ -0,0 +1 @@\n+hello\n",
                        },
                    ],
                }

            orig_runner = self.state.runner_fn
            self.state.runner_fn = two_agent_runner
            try:
                st_g, run_g = _http_json(
                    f"{self.base}/api/runs",
                    method="POST",
                    payload={
                        "repo": str(self.repo_dir),
                        "backend_url": f"http://127.0.0.1:{m_port}/v1",
                        "model": "test-model",
                        "goal": "Add calc and note files",
                    },
                )
                self.assertEqual(st_g, 202)
                rid = run_g["run_id"]
                final = run_g
                for _ in range(50):
                    _, final = _http_json(f"{self.base}/api/runs/{rid}")
                    if final["status"] == "completed":
                        break
                    time.sleep(0.05)
                self.assertEqual(final["status"], "completed")
                self.assertEqual(len(final["tasks"]), 2)

                st_all, ap_all = _http_json(
                    f"{self.base}/api/runs/{rid}/apply-all",
                    method="POST",
                    payload={},
                )
                self.assertEqual(st_all, 200)
                self.assertEqual(sorted(ap_all["applied"]), ["coder-a", "coder-b"])
                self.assertTrue((self.repo_dir / "calc.txt").is_file())
                self.assertTrue((self.repo_dir / "note.txt").is_file())
            finally:
                self.state.runner_fn = orig_runner
        finally:
            mock_srv.shutdown()
            mock_srv.server_close()

    def test_research_mode_run_sparkline_and_apply_best(self) -> None:
        (self.repo_dir / "kernel.py").write_text("def score_val():\n    return 10\n", encoding="utf-8")
        (self.repo_dir / "eval_bench.py").write_text(
            "from kernel import score_val\nprint(f'score: {score_val()}')\n",
            encoding="utf-8",
        )
        (self.repo_dir / "test_kernel.py").write_text(
            "import time, unittest\nfrom kernel import score_val\nclass T(unittest.TestCase):\n    def test_pos(self):\n        time.sleep(0.12)\n        self.assertGreater(score_val(), 0)\nif __name__ == '__main__':\n    unittest.main()\n",
            encoding="utf-8",
        )
        subprocess.run(["git", "add", "-A"], cwd=str(self.repo_dir), check=True, capture_output=True)
        subprocess.run(["git", "commit", "-m", "Add kernel benchmark"], cwd=str(self.repo_dir), check=True, capture_output=True)

        class _ResearchMockHandler(BaseHTTPRequestHandler):
            def log_message(self, format: str, *args) -> None:
                pass

            def do_POST(self) -> None:
                n = int(self.headers.get("Content-Length", "0") or "0")
                _ = self.rfile.read(n)
                raw = json.dumps(
                    {
                        "choices": [
                            {
                                "message": {
                                    "role": "assistant",
                                    "content": (
                                        "Optimize score_val\n"
                                        "```file:kernel.py\n"
                                        "def score_val():\n"
                                        "    return 42\n"
                                        "```\n"
                                    ),
                                }
                            }
                        ],
                        "usage": {"prompt_tokens": 20, "cached_tokens": 10, "completion_tokens": 18},
                    }
                ).encode("utf-8")
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(raw)))
                self.end_headers()
                self.wfile.write(raw)

        mock_srv = ThreadingHTTPServer(("127.0.0.1", 0), _ResearchMockHandler)
        m_port = mock_srv.server_address[1]
        threading.Thread(target=mock_srv.serve_forever, daemon=True).start()
        try:
            st_r, rec = _http_json(
                f"{self.base}/api/runs",
                method="POST",
                payload={
                    "repo": str(self.repo_dir),
                    "backend_url": f"http://127.0.0.1:{m_port}/v1",
                    "model": "test-model",
                    "mode": "research",
                    "goal": "Maximize kernel score_val",
                    "research_config": {
                        "eval_cmd": f"{sys.executable} eval_bench.py",
                        "health_cmd": f"{sys.executable} -m unittest test_kernel.py",
                        "editable_paths": ["kernel.py"],
                        "readonly_paths": ["eval_bench.py", "test_kernel.py"],
                        "direction": "max",
                        "islands": 2,
                        "rounds": 1,
                        "use_tool_loop": False,
                    },
                },
            )
            self.assertEqual(st_r, 202)
            rid = rec["run_id"]
            final = rec
            for _ in range(80):
                _, final = _http_json(f"{self.base}/api/runs/{rid}")
                if final["status"] in ("completed", "error"):
                    break
                time.sleep(0.05)
            self.assertEqual(final["status"], "completed", final.get("error"))
            self.assertEqual(final["mode"], "research")
            self.assertEqual(final["baseline_score"], 10.0)
            self.assertEqual(final["best_score"], 42.0)
            self.assertEqual(len(final["experiments"]), 2)
            self.assertLess(final["experiments"][0]["eval_s"], 0.10)
            self.assertGreaterEqual(final["experiments"][0]["health_s"], 0.10)
            self.assertIn("Live Score Sparkline", urllib.request.urlopen(f"{self.base}/", timeout=5.0).read().decode("utf-8"))

            st_ap, ap = _http_json(
                f"{self.base}/api/runs/{rid}/apply",
                method="POST",
                payload={"agent_id": final["best_island"]},
            )
            self.assertEqual(st_ap, 200)
            self.assertIn("42", (self.repo_dir / "kernel.py").read_text(encoding="utf-8"))
        finally:
            mock_srv.shutdown()
            mock_srv.server_close()

    def test_devices_api_pair_list_and_remote_execution(self) -> None:
        from swarm_node import NodeStateStore, SwarmNodeServer
        from swarm_run import run_swarm

        class _RemoteExecOpenAIHandler(BaseHTTPRequestHandler):
            def log_message(self, format: str, *args) -> None:
                pass

            def do_POST(self) -> None:
                n = int(self.headers.get("Content-Length", "0") or "0")
                _ = self.rfile.read(n)
                raw = json.dumps(
                    {
                        "choices": [
                            {
                                "message": {
                                    "role": "assistant",
                                    "content": (
                                        "```file:cuda_op.py\n"
                                        "def fast_op(x):\n"
                                        "    return x * 2\n"
                                        "```\n"
                                        "```file:tests/test_cuda_op.py\n"
                                        "import unittest\n"
                                        "from cuda_op import fast_op\n"
                                        "class CudaOpTest(unittest.TestCase):\n"
                                        "    def test_fast_op(self):\n"
                                        "        self.assertEqual(fast_op(21), 42)\n"
                                        "if __name__ == '__main__':\n"
                                        "    unittest.main()\n"
                                        "```\n"
                                    ),
                                }
                            }
                        ],
                        "usage": {"prompt_tokens": 30, "cached_tokens": 15, "completion_tokens": 40},
                    }
                ).encode("utf-8")
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(raw)))
                self.end_headers()
                self.wfile.write(raw)

        node_state_file = self.tmp_dir / "remote_node_state.json"
        store = NodeStateStore(node_state_file)
        pin = store.generate_pair_code()
        node_srv = SwarmNodeServer(
            host="127.0.0.1",
            port=0,
            state_path=node_state_file,
            node_name="Mock DGX Spark",
        )
        node_port = node_srv.start()

        mock_oai = ThreadingHTTPServer(("127.0.0.1", 0), _RemoteExecOpenAIHandler)
        m_port = mock_oai.server_address[1]
        threading.Thread(target=mock_oai.serve_forever, daemon=True).start()
        orig_runner = self.state.runner_fn
        self.state.runner_fn = run_swarm

        try:
            st_pair, paired = _http_json(
                f"{self.base}/api/devices/pair",
                method="POST",
                payload={
                    "host": "127.0.0.1",
                    "port": node_port,
                    "pair_code": pin,
                    "name": "DGX Spark Test",
                },
            )
            self.assertEqual(st_pair, 200)
            self.assertEqual(paired["name"], "DGX Spark Test")
            self.assertNotIn("token", paired)

            st_list, devs = _http_json(f"{self.base}/api/devices")
            self.assertEqual(st_list, 200)
            self.assertGreaterEqual(len(devs["devices"]), 2)
            remote_entry = next(d for d in devs["devices"] if not d.get("is_local"))
            self.assertTrue(remote_entry["online"])
            self.assertNotIn("token", remote_entry)

            # Force remote execution directly via target_device pointing to paired node
            st_run, rec = _http_json(
                f"{self.base}/api/runs",
                method="POST",
                payload={
                    "repo": str(self.repo_dir),
                    "backend_url": f"http://127.0.0.1:{m_port}/v1",
                    "model": "test-model",
                    "tasks": [
                        {
                            "id": "MESH-1",
                            "title": "Add cuda_op helper",
                            "spec": "Implement fast_op in cuda_op.py",
                            "acceptance_cmd": "python3 -m unittest tests/test_cuda_op.py",
                            "target_device": remote_entry["device_id"],
                        }
                    ],
                },
            )
            self.assertEqual(st_run, 202)
            rid = rec["run_id"]
            final = rec
            for _ in range(80):
                _, final = _http_json(f"{self.base}/api/runs/{rid}")
                if final["status"] in ("completed", "error"):
                    break
                time.sleep(0.05)
            self.assertEqual(final["status"], "completed", final.get("error"))
            self.assertEqual(final["passed_count"], 1)
            r0 = final["results"][0]
            self.assertEqual(r0["device_id"], remote_entry["device_id"])
            self.assertEqual(r0["device_name"], "DGX Spark Test")

            st_ap, ap = _http_json(
                f"{self.base}/api/runs/{rid}/apply",
                method="POST",
                payload={"agent_id": r0["agent_id"]},
            )
            self.assertEqual(st_ap, 200)
            self.assertIn("fast_op", (self.repo_dir / "cuda_op.py").read_text(encoding="utf-8"))
        finally:
            self.state.runner_fn = orig_runner
            node_srv.stop()
            mock_oai.shutdown()
            mock_oai.server_close()

    def test_v040_csrf_board_proxy_cancel_and_merge_gate(self) -> None:
        # 1) HTML injects open-swarm-csrf meta tag & /api/config returns v0.4.0 + token
        html = urllib.request.urlopen(f"{self.base}/", timeout=5.0).read().decode("utf-8")
        self.assertIn('name="open-swarm-csrf"', html)
        st_cfg, cfg = _http_json(f"{self.base}/api/config")
        self.assertEqual(st_cfg, 200)
        self.assertEqual(cfg["version"], "0.4.0")
        csrf_tok = cfg["csrf_token"]

        # 2) Bad Host, bad Origin, and browser Origin without X-OpenSwarm-CSRF are blocked (403)
        req_bad_host = urllib.request.Request(f"{self.base}/health", headers={"Host": "evil.example:18790"})
        with self.assertRaises(urllib.error.HTTPError) as h_ctx:
            urllib.request.urlopen(req_bad_host, timeout=5.0)
        self.assertEqual(h_ctx.exception.code, 403)
        h_ctx.exception.close()

        req_bad_orig = urllib.request.Request(
            f"{self.base}/api/board/post",
            data=b"{}",
            headers={"Content-Type": "application/json", "Origin": "https://evil.example"},
            method="POST",
        )
        with self.assertRaises(urllib.error.HTTPError) as o_ctx:
            urllib.request.urlopen(req_bad_orig, timeout=5.0)
        self.assertEqual(o_ctx.exception.code, 403)
        o_ctx.exception.close()

        req_no_csrf = urllib.request.Request(
            f"{self.base}/api/board/post",
            data=b'{"run_id":"r1","from_agent":"coder-a","kind":"FINDING","body":"hi"}',
            headers={"Content-Type": "application/json", "Origin": self.base},
            method="POST",
        )
        with self.assertRaises(urllib.error.HTTPError) as c_ctx:
            urllib.request.urlopen(req_no_csrf, timeout=5.0)
        self.assertEqual(c_ctx.exception.code, 403)
        c_ctx.exception.close()

        # 3) Valid CSRF + Board claim, conflict 409, decision, and SSE stream
        req_ok = urllib.request.Request(
            f"{self.base}/api/board/post",
            data=json.dumps({"run_id": "r_board", "from_agent": "coder-a", "kind": "NEED_DECISION", "body": "SQLite or JSON?"}).encode("utf-8"),
            headers={"Content-Type": "application/json", "Origin": self.base, "X-OpenSwarm-CSRF": csrf_tok},
            method="POST",
        )
        with urllib.request.urlopen(req_ok, timeout=5.0) as r:
            need_msg = json.loads(r.read().decode("utf-8"))
        self.assertEqual(need_msg["kind"], "NEED_DECISION")

        st_cl, cl1 = _http_json(
            f"{self.base}/api/board/claim",
            method="POST",
            payload={"run_id": "r_board", "rel_path": "pkg/api.py", "owner_agent": "coder-a"},
        )
        self.assertEqual(st_cl, 200)
        self.assertTrue(cl1["claimed"])

        with self.assertRaises(urllib.error.HTTPError) as cl_ctx:
            _http_json(
                f"{self.base}/api/board/claim",
                method="POST",
                payload={"run_id": "r_board", "rel_path": "pkg/api.py", "owner_agent": "coder-b"},
            )
        self.assertEqual(cl_ctx.exception.code, 409)
        cl_ctx.exception.close()

        st_res, res_d = _http_json(
            f"{self.base}/api/board/resolve",
            method="POST",
            payload={"seq": need_msg["seq"], "resolution": "Use SQLite", "resolver": "matt"},
        )
        self.assertEqual(st_res, 200)
        self.assertTrue(res_d["resolved"])

        sse_raw = urllib.request.urlopen(f"{self.base}/api/board/stream?run_id=r_board&once=1", timeout=5.0).read().decode("utf-8")
        self.assertIn("event: message", sse_raw)
        self.assertIn("Use SQLite", sse_raw)

        # 4) Scoped Coordinator LLM Proxy forwards upstream key without leaking to worker
        seen_auth = {}

        class _ProxyUpstreamHandler(BaseHTTPRequestHandler):
            def log_message(self, format: str, *args) -> None:
                pass

            def do_POST(self) -> None:
                seen_auth["auth"] = self.headers.get("Authorization")
                n = int(self.headers.get("Content-Length", "0") or "0")
                _ = self.rfile.read(n)
                raw = json.dumps({"choices": [{"message": {"role": "assistant", "content": "proxied"}}]}).encode("utf-8")
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(raw)))
                self.end_headers()
                self.wfile.write(raw)

        up_srv = ThreadingHTTPServer(("127.0.0.1", 0), _ProxyUpstreamHandler)
        up_port = up_srv.server_address[1]
        threading.Thread(target=up_srv.serve_forever, daemon=True).start()
        try:
            ptok = self.state.issue_proxy_token(
                run_id="r_proxy",
                base_url=f"http://127.0.0.1:{up_port}/v1",
                api_key="sk_secret_upstream_12345",
            )
            p_req = urllib.request.Request(
                f"{self.base}/api/llm/proxy/v1/chat/completions",
                data=b'{"model":"m","messages":[]}',
                headers={"Content-Type": "application/json", "Authorization": f"Bearer {ptok}"},
                method="POST",
            )
            with urllib.request.urlopen(p_req, timeout=5.0) as r:
                pres = json.loads(r.read().decode("utf-8"))
            self.assertEqual(pres["choices"][0]["message"]["content"], "proxied")
            self.assertEqual(seen_auth["auth"], "Bearer sk_secret_upstream_12345")
            self.state.revoke_run_proxy_tokens("r_proxy")
        finally:
            up_srv.shutdown()
            up_srv.server_close()

        # 5) Post-merge integration gate blocks failing combined tests without modifying repo
        rec_pass = {
            "run_id": "gate-run-1",
            "repo": str(self.repo_dir),
            "status": "completed",
            "tasks": [{"title": "T1", "acceptance_cmd": "true"}],
            "results": [
                {
                    "agent_id": "coder-a",
                    "tests_ok": True,
                    "written_files": ["gated.txt"],
                    "diff": "diff --git a/gated.txt b/gated.txt\nnew file mode 100644\nindex 0000000..ce01362\n--- /dev/null\n+++ b/gated.txt\n@@ -0,0 +1 @@\n+hello\n",
                }
            ],
            "applied_agents": [],
        }
        self.state._save_run(rec_pass)
        with self.assertRaises(urllib.error.HTTPError) as g_ctx:
            _http_json(
                f"{self.base}/api/runs/gate-run-1/apply-all",
                method="POST",
                payload={"verify_cmd": "python3 -c 'import sys; sys.exit(1)'"},
            )
        self.assertEqual(g_ctx.exception.code, 400)
        g_ctx.exception.close()
        self.assertFalse((self.repo_dir / "gated.txt").exists())

        st_ok_g, ap_g = _http_json(
            f"{self.base}/api/runs/gate-run-1/apply-all",
            method="POST",
            payload={"verify_cmd": "test -f gated.txt"},
        )
        self.assertEqual(st_ok_g, 200)
        self.assertTrue((self.repo_dir / "gated.txt").is_file())
        self.assertTrue(ap_g["integration_gate"]["ok"])

        # 6) Cancel active run endpoint
        st_cxl, cxl = _http_json(f"{self.base}/api/runs/gate-run-1/cancel", method="POST", payload={})
        self.assertEqual(st_cxl, 200)
        self.assertTrue(cxl["ok"])


if __name__ == "__main__":
    unittest.main()
