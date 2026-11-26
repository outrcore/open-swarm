"""Tests for M2 Karpathy Autoresearch Swarm Engine (`open_swarm/research_loop.py`)."""
from __future__ import annotations

from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import threading
import time
import types
import unittest

SWARM_DIR = Path(__file__).resolve().parent.parent
if str(SWARM_DIR) not in sys.path:
    sys.path.insert(0, str(SWARM_DIR))

from continuity_store import ContinuityStore  # noqa: E402
from research_loop import ResearchConfig, run_research_swarm  # noqa: E402


class _MockOpenAIServer:
    """Local mock OpenAI-compatible /v1/chat/completions HTTP server."""

    def __init__(self, responder) -> None:
        self.responder = responder
        self.requests: list[dict] = []
        self._lock = threading.Lock()
        parent = self

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, *args) -> None:
                pass

            def do_POST(self) -> None:
                if self.path != "/v1/chat/completions":
                    self.send_response(404)
                    self.end_headers()
                    return
                n = int(self.headers.get("Content-Length") or 0)
                body = json.loads(self.rfile.read(n).decode("utf-8"))
                with parent._lock:
                    idx = len(parent.requests)
                    parent.requests.append(body)
                reply = parent.responder(idx, body)
                msg = reply if isinstance(reply, dict) else {"role": "assistant", "content": reply}
                finish_reason = "tool_calls" if msg.get("tool_calls") else "stop"
                payload = json.dumps(
                    {
                        "id": f"chatcmpl-mock-{idx}",
                        "object": "chat.completion",
                        "model": body.get("model", "mock-model"),
                        "choices": [{"index": 0, "message": msg, "finish_reason": finish_reason}],
                        "usage": {
                            "prompt_tokens": 100,
                            "completion_tokens": 40,
                            "prompt_tokens_details": {"cached_tokens": 60},
                        },
                    }
                ).encode("utf-8")
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(payload)))
                self.end_headers()
                self.wfile.write(payload)

        self.srv = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        self.port = self.srv.socket.getsockname()[1]
        self.url = f"http://127.0.0.1:{self.port}/v1"
        self.thread = threading.Thread(target=self.srv.serve_forever, daemon=True)

    def __enter__(self) -> _MockOpenAIServer:
        self.thread.start()
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.srv.shutdown()
        self.srv.server_close()
        self.thread.join(timeout=5)


def _git(cwd: Path, *args: str) -> subprocess.CompletedProcess:
    return subprocess.run(
        ["git", *args],
        cwd=str(cwd),
        capture_output=True,
        text=True,
        check=True,
    )


def _init_sample_repo(root: Path) -> Path:
    repo = root / "repo"
    repo.mkdir(parents=True, exist_ok=True)
    _git(repo, "init")
    _git(repo, "config", "user.name", "Test User")
    _git(repo, "config", "user.email", "test@example.com")
    (repo / "kernel.py").write_text(
        "def compute(x: int) -> int:\n    return x + 1\n",
        encoding="utf-8",
    )
    (repo / "eval_bench.py").write_text(
        "import kernel\n"
        "val = kernel.compute(5)\n"
        "print(f'score={val:.2f}')\n",
        encoding="utf-8",
    )
    (repo / "health_check.py").write_text(
        "import kernel\n"
        "assert kernel.compute(1) >= 2, 'invalid compute'\n"
        "print('HEALTH_OK')\n",
        encoding="utf-8",
    )
    _git(repo, "add", "-A")
    _git(repo, "commit", "-m", "initial commit")
    return repo


class TestResearchLoop(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = Path(tempfile.mkdtemp(prefix="research_loop_test_"))
        self.repo = _init_sample_repo(self.tmp)

    def tearDown(self) -> None:
        shutil.rmtree(self.tmp, ignore_errors=True)

    def test_monotonic_ratchet_and_clean_diff_apply(self) -> None:
        """Improving edit is kept & committed; regressive edit is reverted; output diff applies cleanly."""
        responses = [
            # Round 1: improves score from 6.0 to 15.0
            "Boost compute multiplier\n```file:kernel.py\ndef compute(x: int) -> int:\n    return x * 3\n```\n",
            # Round 2: regresses score from 15.0 to 7.0 -> must be reverted
            "Try smaller offset\n```file:kernel.py\ndef compute(x: int) -> int:\n    return x + 2\n```\n",
        ]

        def responder(idx: int, _body: dict) -> str:
            return responses[idx]

        cfg = ResearchConfig(
            goal="Maximize kernel.compute(5)",
            eval_cmd=f"{sys.executable} eval_bench.py",
            health_cmd=f"{sys.executable} health_check.py",
            editable_paths=["kernel.py"],
            readonly_paths=["eval_bench.py", "health_check.py"],
            islands=1,
            rounds=2,
            use_tool_loop=False,
        )
        events: list[dict] = []
        with _MockOpenAIServer(responder) as srv:
            out = run_research_swarm(
                repo=self.repo,
                config=cfg,
                base_url=srv.url,
                model="mock-model",
                keep_worktrees=True,
                on_progress=events.append,
            )

        self.assertEqual(out["mode"], "research")
        self.assertEqual(out["baseline_score"], 6.0)
        self.assertEqual(out["best_score"], 15.0)
        self.assertEqual(out["best_island"], "island-a")
        self.assertEqual(out["passed_count"], 1)
        self.assertEqual(out["agents_count"], 1)

        exps = out["experiments"]
        self.assertEqual(len(exps), 2)
        self.assertEqual(exps[0]["status"], "keep")
        self.assertEqual(exps[0]["score"], 15.0)
        self.assertEqual(exps[1]["status"], "revert")
        self.assertEqual(exps[1]["score"], 7.0)
        self.assertEqual(exps[1]["best_score"], 15.0)

        # Worktree working tree is clean and `.open_swarm/` artifacts are preserved and ignored by git
        wt = Path(out["results"][0]["worktree"])
        self.assertEqual(_git(wt, "status", "--porcelain").stdout.strip(), "")
        self.assertEqual((wt / "kernel.py").read_text(encoding="utf-8"), "def compute(x: int) -> int:\n    return x * 3\n")
        self.assertTrue((wt / ".open_swarm" / "program.md").exists())
        self.assertTrue((wt / ".open_swarm" / "results.tsv").exists())
        self.assertTrue((wt / ".open_swarm" / "findings.md").exists())

        # Diff against base_commit has no .open_swarm leakage and applies cleanly with git apply --check
        diff_txt = out["results"][0]["diff"]
        self.assertIn("kernel.py", diff_txt)
        self.assertNotIn(".open_swarm", diff_txt)
        patch_file = self.tmp / "candidate.patch"
        patch_file.write_text(diff_txt, encoding="utf-8")
        chk = subprocess.run(
            ["git", "apply", "--check", str(patch_file)],
            cwd=str(self.repo),
            capture_output=True,
            text=True,
        )
        self.assertEqual(chk.returncode, 0, chk.stderr)
        self.assertTrue(any(e.get("kind") == "research_step" for e in events))

    def test_health_cmd_isolation_reverts_broken_high_score(self) -> None:
        """An edit that prints a higher eval score by breaking health_cmd fails health and reverts."""
        # health_check checks compute(1) >= 2; returning 1000 only when x==5 and 0 for x==1 breaks health_cmd!
        bad_but_high_eval = (
            "Shortcut for x==5\n"
            "```file:kernel.py\n"
            "def compute(x: int) -> int:\n"
            "    return 1000 if x == 5 else 0\n"
            "```\n"
        )
        cfg = ResearchConfig(
            goal="Maximize compute(5) while passing health_check",
            eval_cmd=f"{sys.executable} eval_bench.py",
            health_cmd=f"{sys.executable} health_check.py",
            editable_paths=["kernel.py"],
            readonly_paths=["eval_bench.py", "health_check.py"],
            islands=1,
            rounds=1,
            use_tool_loop=False,
        )
        with _MockOpenAIServer(lambda _i, _b: bad_but_high_eval) as srv:
            out = run_research_swarm(
                repo=self.repo,
                config=cfg,
                base_url=srv.url,
                model="mock-model",
                keep_worktrees=False,
            )

        self.assertEqual(out["best_score"], 6.0)
        self.assertEqual(out["experiments"][0]["status"], "health_failed")
        self.assertIsNone(out["experiments"][0]["score"])
        self.assertEqual(out["results"][0]["diff"], "")
        self.assertFalse(out["results"][0]["tests_ok"])

    def test_eval_s_strict_timing_isolation(self) -> None:
        """eval_s excludes health_cmd sleep (0.15s) and LLM sleep (0.15s)."""
        (self.repo / "slow_health.py").write_text(
            "import time\ntime.sleep(0.15)\nprint('OK')\n",
            encoding="utf-8",
        )
        _git(self.repo, "add", "slow_health.py")
        _git(self.repo, "commit", "-m", "add slow_health")

        def slow_llm(_idx: int, _body: dict) -> str:
            time.sleep(0.15)
            return "Improve compute\n```file:kernel.py\ndef compute(x: int) -> int:\n    return x * 2\n```\n"

        cfg = ResearchConfig(
            goal="Verify isolated timing",
            eval_cmd=f"{sys.executable} eval_bench.py",
            health_cmd=f"{sys.executable} slow_health.py",
            editable_paths=["kernel.py"],
            readonly_paths=["eval_bench.py", "slow_health.py"],
            islands=1,
            rounds=1,
            use_tool_loop=False,
        )
        with _MockOpenAIServer(slow_llm) as srv:
            out = run_research_swarm(
                repo=self.repo,
                config=cfg,
                base_url=srv.url,
                model="mock-model",
                keep_worktrees=False,
            )

        step = out["experiments"][0]
        self.assertEqual(step["status"], "keep")
        self.assertLess(step["eval_s"], 0.12)
        self.assertGreaterEqual(step["health_s"], 0.10)
        self.assertGreaterEqual(step["llm_s"], 0.10)

    def test_readonly_paths_tamper_detection_reverts(self) -> None:
        """Modifying a readonly eval script triggers status='tamper_reverted' and reverts."""
        tamper_reply = (
            "Tamper eval script\n"
            "```file:eval_bench.py\n"
            "print('score=9999.0')\n"
            "```\n"
        )
        cfg = ResearchConfig(
            goal="Try tampering",
            eval_cmd=f"{sys.executable} eval_bench.py",
            editable_paths=["kernel.py"],
            readonly_paths=["eval_bench.py"],
            islands=1,
            rounds=1,
            use_tool_loop=False,
        )
        with _MockOpenAIServer(lambda _i, _b: tamper_reply) as srv:
            out = run_research_swarm(
                repo=self.repo,
                config=cfg,
                base_url=srv.url,
                model="mock-model",
                keep_worktrees=True,
            )

        self.assertEqual(out["best_score"], 6.0)
        self.assertEqual(out["experiments"][0]["status"], "tamper_reverted")
        wt = Path(out["results"][0]["worktree"])
        self.assertIn("kernel.compute(5)", (wt / "eval_bench.py").read_text(encoding="utf-8"))

    def test_two_island_shepherd_cross_pollination(self) -> None:
        """Island A's winning score/finding is injected into Island B's prompt on round 2+."""
        lock = threading.Lock()
        round2_prompts_for_b: list[str] = []

        def responder(_idx: int, body: dict) -> str:
            user_msg = body["messages"][-1]["content"]
            is_island_a = "(island-a)" in user_msg
            is_round_1 = "Research Round 1/" in user_msg
            if not is_island_a and not is_round_1:
                with lock:
                    round2_prompts_for_b.append(user_msg)
            if is_island_a and is_round_1:
                return "Island A breakthrough multiplier x4\n```file:kernel.py\ndef compute(x: int) -> int:\n    return x * 4\n```\n"
            if not is_island_a and is_round_1:
                return "Island B small tweak +2\n```file:kernel.py\ndef compute(x: int) -> int:\n    return x + 2\n```\n"
            if not is_island_a and not is_round_1:
                return "Island B adopts multiplier x5\n```file:kernel.py\ndef compute(x: int) -> int:\n    return x * 5\n```\n"
            return "Island A hold\n```file:kernel.py\ndef compute(x: int) -> int:\n    return x * 4\n```\n"

        db_path = self.tmp / "continuity.db"
        store = ContinuityStore(db_path)
        cfg = ResearchConfig(
            goal="Maximize compute(5)",
            eval_cmd=f"{sys.executable} eval_bench.py",
            editable_paths=["kernel.py"],
            readonly_paths=["eval_bench.py"],
            islands=2,
            rounds=2,
            cross_pollinate_every=1,
            use_tool_loop=False,
        )
        with _MockOpenAIServer(responder) as srv:
            out = run_research_swarm(
                repo=self.repo,
                config=cfg,
                base_url=srv.url,
                model="mock-model",
                keep_worktrees=False,
                continuity_store=store,
            )

        self.assertEqual(len(round2_prompts_for_b), 1)
        prompt_b = round2_prompts_for_b[0]
        self.assertIn("Shepherd Cross-Pollination from island-a", prompt_b)
        self.assertIn("20.0", prompt_b)
        self.assertIn("Island A breakthrough multiplier x4", prompt_b)
        self.assertIn("x * 4", prompt_b)
        self.assertEqual(out["best_score"], 25.0)
        self.assertEqual(out["best_island"], "island-b")
        msgs = store.read_messages(str(self.repo.resolve()), to_role="island-b")
        self.assertTrue(any("Shepherd R2" in m["body"] for m in msgs))

    def test_min_direction_ratchet(self) -> None:
        """direction='min' keeps strictly lower scores and reverts higher scores."""
        responses = [
            "Lower to 3.0\n```file:kernel.py\ndef compute(x: int) -> int:\n    return x - 2\n```\n",
            "Regress to 4.0\n```file:kernel.py\ndef compute(x: int) -> int:\n    return x - 1\n```\n",
        ]
        cfg = ResearchConfig(
            goal="Minimize compute(5)",
            eval_cmd=f"{sys.executable} eval_bench.py",
            direction="min",
            editable_paths=["kernel.py"],
            readonly_paths=["eval_bench.py"],
            islands=1,
            rounds=2,
            use_tool_loop=False,
        )
        with _MockOpenAIServer(lambda idx, _b: responses[idx]) as srv:
            out = run_research_swarm(
                repo=self.repo,
                config=cfg,
                base_url=srv.url,
                model="mock-model",
                keep_worktrees=False,
            )
        self.assertEqual(out["baseline_score"], 6.0)
        self.assertEqual(out["best_score"], 3.0)
        self.assertEqual(out["experiments"][0]["status"], "keep")
        self.assertEqual(out["experiments"][1]["status"], "revert")

    def test_use_tool_loop_with_agent_loop_and_fallback(self) -> None:
        """Supports agent_loop.run_agent_loop when present and falls back cleanly when absent."""
        # Case 1: stub agent_loop module present
        stub_mod = types.ModuleType("agent_loop")
        calls: list[str] = []

        def fake_run_agent_loop(*, worktree: Path, agent_id: str, **_kwargs):
            calls.append(agent_id)
            (worktree / "kernel.py").write_text(
                "def compute(x: int) -> int:\n    return x * 6\n",
                encoding="utf-8",
            )
            return {
                "content": "Applied tool edit x * 6",
                "written_files": ["kernel.py"],
                "prompt_tokens": 50,
                "cached_tokens": 20,
                "completion_tokens": 15,
            }

        stub_mod.run_agent_loop = fake_run_agent_loop  # type: ignore[attr-defined]
        prev_mod = sys.modules.get("agent_loop")
        sys.modules["agent_loop"] = stub_mod
        try:
            cfg = ResearchConfig(
                goal="Use tool loop",
                eval_cmd=f"{sys.executable} eval_bench.py",
                editable_paths=["kernel.py"],
                readonly_paths=["eval_bench.py"],
                islands=1,
                rounds=1,
                use_tool_loop=True,
            )
            out = run_research_swarm(
                repo=self.repo,
                config=cfg,
                base_url="http://127.0.0.1:1/v1",
                model="mock-model",
                keep_worktrees=False,
            )
            self.assertEqual(calls, ["island-a"])
            self.assertEqual(out["best_score"], 30.0)
            self.assertEqual(out["experiments"][0]["status"], "keep")
        finally:
            if prev_mod is not None:
                sys.modules["agent_loop"] = prev_mod
            else:
                sys.modules.pop("agent_loop", None)

        # Case 2: real agent_loop.run_agent_loop if present, else fallback
        if (SWARM_DIR / "agent_loop.py").exists():
            tool_script = [
                {
                    "role": "assistant",
                    "content": "",
                    "tool_calls": [
                        {
                            "id": "call_1",
                            "type": "function",
                            "function": {
                                "name": "edit_file",
                                "arguments": json.dumps(
                                    {
                                        "path": "kernel.py",
                                        "old_text": "return x + 1",
                                        "new_text": "return x * 7",
                                    }
                                ),
                            },
                        }
                    ],
                },
                {
                    "role": "assistant",
                    "content": "",
                    "tool_calls": [
                        {
                            "id": "call_2",
                            "type": "function",
                            "function": {
                                "name": "finish",
                                "arguments": json.dumps({"summary": "Edited kernel.py to return x * 7"}),
                            },
                        }
                    ],
                },
            ]
            with _MockOpenAIServer(lambda idx, _b: tool_script[idx]) as srv:
                out2 = run_research_swarm(
                    repo=self.repo,
                    config=ResearchConfig(
                        goal="Real agent_loop integration",
                        eval_cmd=f"{sys.executable} eval_bench.py",
                        health_cmd=f"{sys.executable} health_check.py",
                        editable_paths=["kernel.py"],
                        readonly_paths=["eval_bench.py", "health_check.py"],
                        islands=1,
                        rounds=1,
                        use_tool_loop=True,
                    ),
                    base_url=srv.url,
                    model="mock-model",
                    keep_worktrees=False,
                )
                self.assertEqual(out2["best_score"], 35.0)
                self.assertEqual(out2["experiments"][0]["status"], "keep")
                self.assertEqual(out2["metering"]["calls"], 2)
        else:
            with _MockOpenAIServer(
                lambda _i, _b: "Fallback edit\n```file:kernel.py\ndef compute(x: int) -> int:\n    return x * 2\n```\n"
            ) as srv:
                out2 = run_research_swarm(
                    repo=self.repo,
                    config=ResearchConfig(
                        goal="Fallback when agent_loop missing",
                        eval_cmd=f"{sys.executable} eval_bench.py",
                        editable_paths=["kernel.py"],
                        islands=1,
                        rounds=1,
                        use_tool_loop=True,
                    ),
                    base_url=srv.url,
                    model="mock-model",
                    keep_worktrees=False,
                )
                self.assertEqual(out2["best_score"], 10.0)


if __name__ == "__main__":
    unittest.main(verbosity=2)
