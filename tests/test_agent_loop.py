#!/usr/bin/env python3
"""Tests for open_swarm/agent_loop.py (WorktreeToolkit + run_agent_loop against local mock OpenAI server)."""
from __future__ import annotations

from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import inspect
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import threading
import time
import unittest
from unittest import mock

SWARM_DIR = Path(__file__).resolve().parents[1]
if str(SWARM_DIR) not in sys.path:
    sys.path.insert(0, str(SWARM_DIR))

from agent_loop import (  # noqa: E402
    TOOLS_SCHEMA,
    WorktreeToolkit,
    _FALLBACK_TOOL_INSTRUCTIONS,
    _KNOWN_TOOL_NAMES,
    run_agent_loop,
    scrub_secrets,
)
from continuity_store import ContinuityStore  # noqa: E402
from metering import PricingSchedule, RunMeter, SpendCapExceeded  # noqa: E402
from swarm_run import run_swarm  # noqa: E402


def _sse_response(
    *,
    content: str = "",
    tool_calls: list[dict] | None = None,
    model: str = "mock-model",
    prompt_tokens: int = 120,
    cached_tokens: int = 80,
    completion_tokens: int = 25,
) -> bytes:
    chunks: list[dict] = []
    if tool_calls:
        for idx, tc in enumerate(tool_calls):
            fn = tc.get("function") or {}
            args_str = fn.get("arguments")
            if not isinstance(args_str, str):
                args_str = json.dumps(args_str or {})
            mid = max(1, len(args_str) // 2)
            chunks.append(
                {
                    "id": "chatcmpl-mock",
                    "object": "chat.completion.chunk",
                    "created": 1700000000,
                    "model": model,
                    "choices": [
                        {
                            "index": 0,
                            "delta": {
                                "role": "assistant",
                                "tool_calls": [
                                    {
                                        "index": idx,
                                        "id": tc.get("id", f"call_{idx+1}"),
                                        "type": "function",
                                        "function": {"name": fn["name"], "arguments": args_str[:mid]},
                                    }
                                ],
                            },
                            "finish_reason": None,
                        }
                    ],
                }
            )
            if args_str[mid:]:
                chunks.append(
                    {
                        "id": "chatcmpl-mock",
                        "object": "chat.completion.chunk",
                        "created": 1700000000,
                        "model": model,
                        "choices": [
                            {
                                "index": 0,
                                "delta": {
                                    "tool_calls": [
                                        {
                                            "index": idx,
                                            "function": {"arguments": args_str[mid:]},
                                        }
                                    ]
                                },
                                "finish_reason": None,
                            }
                        ],
                    }
                )
    else:
        chunks.append(
            {
                "id": "chatcmpl-mock",
                "object": "chat.completion.chunk",
                "created": 1700000000,
                "model": model,
                "choices": [
                    {
                        "index": 0,
                        "delta": {"role": "assistant", "content": content},
                        "finish_reason": None,
                    }
                ],
            }
        )

    chunks.append(
        {
            "id": "chatcmpl-mock",
            "object": "chat.completion.chunk",
            "created": 1700000000,
            "model": model,
            "choices": [
                {
                    "index": 0,
                    "delta": {},
                    "finish_reason": "tool_calls" if tool_calls else "stop",
                }
            ],
            "usage": {
                "prompt_tokens": prompt_tokens,
                "completion_tokens": completion_tokens,
                "total_tokens": prompt_tokens + completion_tokens,
                "prompt_tokens_details": {"cached_tokens": cached_tokens},
            },
            "timings": {"prompt_ms": 42.0, "gen_ms": 100.0},
        }
    )
    return ("".join(f"data: {json.dumps(c)}\n\n" for c in chunks) + "data: [DONE]\n\n").encode("utf-8")


class MockOpenAIServer:
    """Thread-safe local mock OpenAI `/v1/chat/completions` server supporting SSE, JSON, and HTTP errors."""

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
                    parent.requests.append(body)
                    idx = len(parent.requests) - 1
                resp_spec = parent.responder(idx, body)
                status = int(resp_spec.get("status", 200))
                if status != 200:
                    err_bytes = json.dumps(resp_spec.get("json") or {"error": "bad request"}).encode("utf-8")
                    self.send_response(status)
                    self.send_header("Content-Type", "application/json")
                    self.send_header("Content-Length", str(len(err_bytes)))
                    self.end_headers()
                    self.wfile.write(err_bytes)
                    return

                if "json" in resp_spec:
                    payload = json.dumps(resp_spec["json"]).encode("utf-8")
                    ctype = "application/json"
                else:
                    payload = _sse_response(
                        content=resp_spec.get("content", ""),
                        tool_calls=resp_spec.get("tool_calls"),
                        model=body.get("model", "mock-model"),
                        prompt_tokens=resp_spec.get("prompt_tokens", 120),
                        cached_tokens=resp_spec.get("cached_tokens", 80),
                        completion_tokens=resp_spec.get("completion_tokens", 25),
                    )
                    ctype = "text/event-stream"

                self.send_response(200)
                self.send_header("Content-Type", ctype)
                self.send_header("Content-Length", str(len(payload)))
                self.end_headers()
                self.wfile.write(payload)

        self.srv = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        self.port = self.srv.socket.getsockname()[1]
        self.url = f"http://127.0.0.1:{self.port}/v1"
        self.thread = threading.Thread(target=self.srv.serve_forever, daemon=True)

    def __enter__(self) -> "MockOpenAIServer":
        self.thread.start()
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.srv.shutdown()
        self.srv.server_close()
        self.thread.join(timeout=5)


class WorktreeToolkitTests(unittest.TestCase):
    def test_path_traversal_git_and_symlink_escape_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            wt = root / "wt"
            wt.mkdir()
            (wt / ".git").mkdir()
            (wt / ".git" / "config").write_text("[core]\n\tbare = false\n", encoding="utf-8")
            outside = root / "outside_secret.txt"
            outside.write_text("SECRET", encoding="utf-8")
            outside_dir = root / "outside_dir"
            outside_dir.mkdir()
            (wt / "escape_file_link").symlink_to(outside)
            (wt / "escape_dir_link").symlink_to(outside_dir)

            tk = WorktreeToolkit(wt)

            for bad_path in ("../outside_secret.txt", "sub/../../outside_secret.txt", ".git/config", "/etc/passwd"):
                r_read = tk.read_file(bad_path)
                self.assertFalse(r_read["ok"], bad_path)
                r_edit = tk.edit_file(bad_path, "a", "b")
                self.assertFalse(r_edit["ok"], bad_path)
                r_write = tk.write_file(bad_path, "pwn")
                self.assertFalse(r_write["ok"], bad_path)

            self.assertFalse(tk.read_file("escape_file_link")["ok"])
            self.assertFalse(tk.write_file("escape_dir_link/pwn.txt", "pwn")["ok"])
            self.assertFalse((outside_dir / "pwn.txt").exists())
            self.assertEqual(outside.read_text(encoding="utf-8"), "SECRET")
            self.assertIn("bare = false", (wt / ".git" / "config").read_text(encoding="utf-8"))

    def test_readonly_paths_and_write_note(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            wt = Path(tmp)
            (wt / ".open_swarm").mkdir()
            (wt / ".open_swarm" / "program.md").write_text("Immutable program spec\n", encoding="utf-8")
            (wt / "docs").mkdir()
            (wt / "docs" / "spec.md").write_text("Immutable doc\n", encoding="utf-8")

            tk = WorktreeToolkit(
                wt,
                readonly_paths=(".open_swarm/program.md", "docs/"),
                agent_id="coder-a",
            )
            e_res = tk.edit_file(".open_swarm/program.md", "Immutable", "Changed")
            self.assertEqual(e_res, {"ok": False, "error": "Read-only path: .open_swarm/program.md"})
            w_res = tk.write_file("docs/spec.md", "Overwritten")
            self.assertEqual(w_res, {"ok": False, "error": "Read-only path: docs/spec.md"})
            w_new = tk.write_file("docs/new.md", "Blocked")
            self.assertEqual(w_new, {"ok": False, "error": "Read-only path: docs/new.md"})

            self.assertEqual((wt / ".open_swarm" / "program.md").read_text(encoding="utf-8"), "Immutable program spec\n")
            self.assertEqual((wt / "docs" / "spec.md").read_text(encoding="utf-8"), "Immutable doc\n")
            self.assertFalse((wt / "docs" / "new.md").exists())
            self.assertEqual(
                tk.readonly_violations,
                [".open_swarm/program.md", "docs/spec.md", "docs/new.md"],
            )

            # write_note must succeed even with .open_swarm/program.md readonly
            note_res = tk.write_note("Discovery", "Found root cause in mod.py")
            self.assertTrue(note_res["ok"])
            self.assertEqual(note_res["path"], ".open_swarm/notes/coder-a.md")
            note_file = wt / ".open_swarm" / "notes" / "coder-a.md"
            self.assertTrue(note_file.exists())
            note_text = note_file.read_text(encoding="utf-8")
            self.assertIn("## ", note_text)
            self.assertIn("— Discovery", note_text)
            self.assertIn("Found root cause in mod.py", note_text)

    def test_pycache_purge_and_dontwritebytecode_on_rapid_edits(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            wt = Path(tmp)
            mod_py = wt / "mod.py"
            test_py = wt / "test_mod.py"
            mod_py.write_text("def val():\n    return 1\n", encoding="utf-8")
            test_py.write_text(
                "import unittest\n"
                "import mod\n\n"
                "class T(unittest.TestCase):\n"
                "    def test_val(self):\n"
                "        self.assertEqual(mod.val(), 1)\n",
                encoding="utf-8",
            )

            tk = WorktreeToolkit(wt)
            t0 = time.monotonic()
            # Pre-seed stale __pycache__ to prove run_command purges it before importing
            r1 = tk.run_command(f"{sys.executable} -c 'import mod' && {sys.executable} -m unittest test_mod.py")
            self.assertTrue(r1["ok"], r1)

            # Rapid same-length edits within <0.2s (same filesystem mtime second)
            e1 = tk.edit_file("mod.py", "return 1", "return 2")
            e2 = tk.edit_file("test_mod.py", "self.assertEqual(mod.val(), 1)", "self.assertEqual(mod.val(), 2)")
            self.assertTrue(e1["ok"] and e2["ok"])
            r2 = tk.run_command(
                f"{sys.executable} -c 'import os; assert os.environ.get(\"PYTHONDONTWRITEBYTECODE\") == \"1\"' "
                f"&& {sys.executable} -m unittest test_mod.py"
            )
            elapsed = time.monotonic() - t0
            self.assertTrue(r2["ok"], r2)
            self.assertFalse((wt / "__pycache__").exists())
            # Also verify rapid test execution sanity
            self.assertLess(elapsed, 5.0)

            # Specifically verify two unittest runs back-to-back within <0.2s if cached vs edited
            e3 = tk.edit_file("mod.py", "return 2", "return 3")
            e4 = tk.edit_file("test_mod.py", "self.assertEqual(mod.val(), 2)", "self.assertEqual(mod.val(), 3)")
            self.assertTrue(e3["ok"] and e4["ok"])
            r3 = tk.run_command(f"{sys.executable} -m unittest test_mod.py")
            self.assertTrue(r3["ok"], r3)
            self.assertFalse((wt / "__pycache__").exists())

    def test_read_search_edit_and_command_bounds(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            wt = Path(tmp)
            # Large file > 12,000 chars for read_file bound
            (wt / "big.txt").write_text("\n".join(f"line_{i}_" + ("x" * 180) for i in range(1, 200)), encoding="utf-8")
            # Search targets + skipped dirs
            (wt / ".git").mkdir()
            (wt / ".git" / "secret.py").write_text("NEEDLE_SKIP\n", encoding="utf-8")
            (wt / "__pycache__").mkdir()
            (wt / "__pycache__" / "cached.py").write_text("NEEDLE_SKIP\n", encoding="utf-8")
            (wt / "many.py").write_text("\n".join(f"NEEDLE_{i} = {i}" for i in range(60)), encoding="utf-8")
            (wt / "dup.py").write_text("x = 1\nx = 1\n", encoding="utf-8")

            tk = WorktreeToolkit(wt)
            rd = tk.read_file("big.txt", start_line=1, end_line=200)
            self.assertTrue(rd["ok"])
            self.assertLessEqual(len(rd["content"]), 12_000)
            self.assertTrue(rd["content"].startswith("1: line_1_"))

            sr = tk.search_code("NEEDLE", glob="*.py")
            self.assertTrue(sr["ok"])
            self.assertEqual(len(sr["matches"]), 40)
            self.assertTrue(sr["truncated"])
            self.assertTrue(all(m["path"] == "many.py" for m in sr["matches"]))

            # edit_file error conditions
            self.assertFalse(tk.edit_file("dup.py", "", "y = 2")["ok"])
            self.assertFalse(tk.edit_file("dup.py", "x = 1", "x = 2")["ok"])
            self.assertFalse(tk.edit_file("dup.py", "missing", "x = 2")["ok"])
            self.assertFalse(tk.edit_file("nope.py", "a", "b")["ok"])

            # run_command stdout/stderr <= 6,000 chars combined
            cmd_res = tk.run_command(
                f"{sys.executable} -c \"import sys; print('O'*5000); print('E'*5000, file=sys.stderr)\""
            )
            self.assertTrue(cmd_res["ok"])
            self.assertLessEqual(len(cmd_res["stdout"]) + len(cmd_res["stderr"]), 6_000)


class AgentLoopE2ETests(unittest.TestCase):
    def test_1_multiturn_native_tool_calls(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            wt = Path(tmp)
            (wt / "calc.py").write_text("def add(a, b):\n    return a - b\n", encoding="utf-8")
            (wt / "test_calc.py").write_text(
                "import unittest\n"
                "from calc import add\n\n"
                "class TestCalc(unittest.TestCase):\n"
                "    def test_add(self):\n"
                "        self.assertEqual(add(2, 3), 5)\n",
                encoding="utf-8",
            )

            script = [
                {
                    "tool_calls": [
                        {"id": "tc_1", "type": "function", "function": {"name": "read_file", "arguments": {"path": "calc.py"}}}
                    ]
                },
                {
                    "tool_calls": [
                        {
                            "id": "tc_2",
                            "type": "function",
                            "function": {
                                "name": "edit_file",
                                "arguments": {"path": "calc.py", "old_text": "return a - b", "new_text": "return a + b"},
                            },
                        }
                    ]
                },
                {
                    "tool_calls": [
                        {
                            "id": "tc_3",
                            "type": "function",
                            "function": {"name": "run_command", "arguments": {"cmd": f"{sys.executable} -m unittest test_calc.py -v"}},
                        }
                    ]
                },
                {
                    "tool_calls": [
                        {
                            "id": "tc_4",
                            "type": "function",
                            "function": {"name": "finish", "arguments": {"summary": "Fixed add() and verified tests pass."}},
                        }
                    ]
                },
            ]

            progress_events: list[dict] = []
            meter = RunMeter(pricing=PricingSchedule(), spend_cap_usd=1.0)

            with MockOpenAIServer(lambda idx, _body: script[idx]) as srv:
                res = run_agent_loop(
                    worktree=wt,
                    system_prompt="SYS_PREFIX",
                    user_prompt="Fix add() in calc.py.",
                    base_url=srv.url,
                    api_key="test-key",
                    model="mock-coder",
                    agent_id="coder-a",
                    meter=meter,
                    on_progress=progress_events.append,
                )

                self.assertEqual(len(srv.requests), 4)
                self.assertEqual(srv.requests[0]["tools"], TOOLS_SCHEMA)
                # Verify tool output from read_file was delivered in role="tool"
                self.assertEqual(srv.requests[1]["messages"][-1]["role"], "tool")
                read_payload = json.loads(srv.requests[1]["messages"][-1]["content"])
                self.assertTrue(read_payload["ok"])
                self.assertIn("1: def add(a, b):", read_payload["content"])
                # Verify run_command passed
                cmd_payload = json.loads(srv.requests[3]["messages"][-1]["content"])
                self.assertTrue(cmd_payload["ok"])
                self.assertEqual(cmd_payload["returncode"], 0)

            self.assertTrue(res["ok"])
            self.assertEqual(res["turns"], 4)
            self.assertEqual(res["tool_calls"], 4)
            self.assertEqual(res["written_files"], ["calc.py"])
            self.assertEqual(res["summary"], "Fixed add() and verified tests pass.")
            self.assertEqual(res["prompt_tokens"], 480)
            self.assertEqual(res["cached_tokens"], 320)
            self.assertEqual(res["completion_tokens"], 100)
            self.assertIsNone(res["error"])
            self.assertIn("return a + b", (wt / "calc.py").read_text(encoding="utf-8"))
            self.assertTrue(any(ev.get("kind") == "agent_stage" and ev.get("stage") == "running" for ev in progress_events))
            self.assertEqual(meter.agents["coder-a"].calls, 4)

    def test_2_fenced_tool_fallback_and_http_400_tools_rejection(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            wt = Path(tmp)
            (wt / "app.py").write_text("VALUE = 1\n", encoding="utf-8")

            def responder(idx: int, body: dict) -> dict:
                if "tools" in body:
                    return {"status": 400, "json": {"error": "tools parameter not supported"}}
                # First successful call without tools: fenced ```tool block
                if len([r for r in srv.requests if "tools" not in r]) == 1:
                    return {
                        "json": {
                            "choices": [
                                {
                                    "message": {
                                        "role": "assistant",
                                        "content": (
                                            "I will update app.py.\n"
                                            "```tool\n"
                                            '{"name": "edit_file", "arguments": {"path": "app.py", "old_text": "VALUE = 1", "new_text": "VALUE = 42"}}\n'
                                            "```"
                                        ),
                                    }
                                }
                            ],
                            "usage": {"prompt_tokens": 50, "completion_tokens": 20},
                        }
                    }
                # Second successful call: fenced ```json top-level {"name": ..., "arguments": ...}
                return {
                    "content": (
                        "Done.\n"
                        "```json\n"
                        '{"name": "finish", "arguments": {"summary": "Updated VALUE to 42 via fenced fallback."}}\n'
                        "```"
                    )
                }

            with MockOpenAIServer(responder) as srv:
                res = run_agent_loop(
                    worktree=wt,
                    system_prompt="SYS_FIXED_PREFIX",
                    user_prompt="Set VALUE to 42.",
                    base_url=srv.url,
                    api_key="test-key",
                    model="mock-fallback",
                    agent_id="coder-fb",
                )
                # Request 0 had tools -> 400; Request 1 retried without tools; Request 2 had no tools
                self.assertEqual(len(srv.requests), 3)
                self.assertIn("tools", srv.requests[0])
                self.assertNotIn("tools", srv.requests[1])
                self.assertNotIn("tools", srv.requests[2])
                # System prefix must remain untouched across fallback!
                self.assertEqual(srv.requests[1]["messages"][0]["content"], "SYS_FIXED_PREFIX")
                self.assertEqual(srv.requests[2]["messages"][0]["content"], "SYS_FIXED_PREFIX")

            self.assertTrue(res["ok"])
            self.assertEqual(res["turns"], 2)
            self.assertEqual(res["tool_calls"], 2)
            self.assertEqual(res["written_files"], ["app.py"])
            self.assertEqual((wt / "app.py").read_text(encoding="utf-8"), "VALUE = 42\n")

    def test_3_readonly_paths_rejected_in_tool_calls_and_legacy_file_blocks(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            wt = Path(tmp)
            (wt / ".open_swarm").mkdir()
            (wt / ".open_swarm" / "program.md").write_text("DO NOT TOUCH\n", encoding="utf-8")

            # Part A: native tool_calls attempting write_file + edit_file on readonly path, then finish
            script_tools = [
                {
                    "tool_calls": [
                        {
                            "id": "c1",
                            "type": "function",
                            "function": {"name": "write_file", "arguments": {"path": ".open_swarm/program.md", "content": "hacked"}},
                        },
                        {
                            "id": "c2",
                            "type": "function",
                            "function": {"name": "write_file", "arguments": {"path": "ok_tool.py", "content": "OK = True"}},
                        },
                    ]
                },
                {
                    "tool_calls": [
                        {"id": "c3", "type": "function", "function": {"name": "finish", "arguments": {"summary": "Handled readonly."}}}
                    ]
                },
            ]
            with MockOpenAIServer(lambda idx, _body: script_tools[idx]) as srv:
                res_a = run_agent_loop(
                    worktree=wt,
                    system_prompt="SYS",
                    user_prompt="Task A",
                    base_url=srv.url,
                    api_key="k",
                    model="m",
                    readonly_paths=(".open_swarm/program.md",),
                )
                self.assertIn("Read-only path: .open_swarm/program.md", srv.requests[1]["messages"][-2]["content"])
            self.assertTrue(res_a["ok"])
            self.assertEqual(res_a["written_files"], ["ok_tool.py"])
            self.assertEqual(res_a["readonly_violations"], [".open_swarm/program.md"])
            self.assertEqual((wt / ".open_swarm" / "program.md").read_text(encoding="utf-8"), "DO NOT TOUCH\n")

            # Part B: legacy ```file:rel/path``` blocks attempting readonly + allowed file
            script_legacy = [
                {
                    "content": (
                        "```file:.open_swarm/program.md\n"
                        "OVERWRITTEN\n"
                        "```\n"
                        "```file:ok_legacy.py\n"
                        "LEGACY = 1\n"
                        "```\n"
                    )
                },
                {
                    "tool_calls": [
                        {"id": "c4", "type": "function", "function": {"name": "finish", "arguments": {"summary": "Acked rejection."}}}
                    ]
                },
            ]
            with MockOpenAIServer(lambda idx, _body: script_legacy[idx]) as srv:
                res_b = run_agent_loop(
                    worktree=wt,
                    system_prompt="SYS",
                    user_prompt="Task B",
                    base_url=srv.url,
                    api_key="k",
                    model="m",
                    readonly_paths=(".open_swarm/program.md",),
                )
                # Agent receives rejection error note in next turn's user message
                self.assertIn("Read-only path: .open_swarm/program.md", srv.requests[1]["messages"][-1]["content"])
            self.assertTrue(res_b["ok"])
            self.assertEqual(res_b["written_files"], ["ok_legacy.py"])
            self.assertEqual(res_b["readonly_violations"], [".open_swarm/program.md"])
            self.assertEqual((wt / ".open_swarm" / "program.md").read_text(encoding="utf-8"), "DO NOT TOUCH\n")
            self.assertEqual((wt / "ok_legacy.py").read_text(encoding="utf-8"), "LEGACY = 1\n")

    def test_4_write_note_via_agent_loop(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            wt = Path(tmp)
            (wt / ".open_swarm").mkdir()
            (wt / ".open_swarm" / "program.md").write_text("Pinned spec\n", encoding="utf-8")

            script = [
                {
                    "tool_calls": [
                        {
                            "id": "n1",
                            "type": "function",
                            "function": {
                                "name": "write_note",
                                "arguments": {"title": "Architecture", "body": "- Uses stdlib only"},
                            },
                        }
                    ]
                },
                {
                    "tool_calls": [
                        {"id": "n2", "type": "function", "function": {"name": "finish", "arguments": {"summary": "Wrote note."}}}
                    ]
                },
            ]
            with MockOpenAIServer(lambda idx, _body: script[idx]) as srv:
                res = run_agent_loop(
                    worktree=wt,
                    system_prompt="SYS",
                    user_prompt="Record note.",
                    base_url=srv.url,
                    api_key="k",
                    model="m",
                    agent_id="architect-1",
                    readonly_paths=(".open_swarm/program.md",),
                )
            self.assertTrue(res["ok"])
            self.assertEqual(res["notes_path"], ".open_swarm/notes/architect-1.md")
            note_txt = (wt / ".open_swarm" / "notes" / "architect-1.md").read_text(encoding="utf-8")
            self.assertIn("— Architecture", note_txt)
            self.assertIn("- Uses stdlib only", note_txt)

    def test_5_context_compaction_preserves_system_prefix_byte_for_byte(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            wt = Path(tmp)
            (wt / "big_a.py").write_text("\n".join(f"A_{i} = '{'x' * 50}'" for i in range(40)), encoding="utf-8")
            (wt / "big_b.py").write_text("\n".join(f"B_{i} = '{'y' * 50}'" for i in range(40)), encoding="utf-8")
            (wt / "big_c.py").write_text("\n".join(f"C_{i} = '{'z' * 50}'" for i in range(40)), encoding="utf-8")

            sys_prefix = "KV_CACHE_PREFIX_EXACT_BYTES_0123456789\nDo not alter this prefix."
            task_prompt = "Inspect big_a.py, big_b.py, big_c.py and finish."

            script = [
                {"tool_calls": [{"id": "t1", "type": "function", "function": {"name": "read_file", "arguments": {"path": "big_a.py"}}}]},
                {"tool_calls": [{"id": "t2", "type": "function", "function": {"name": "read_file", "arguments": {"path": "big_b.py"}}}]},
                {"tool_calls": [{"id": "t3", "type": "function", "function": {"name": "read_file", "arguments": {"path": "big_c.py"}}}]},
                {"tool_calls": [{"id": "t4", "type": "function", "function": {"name": "finish", "arguments": {"summary": "Inspected all."}}}]},
            ]

            with MockOpenAIServer(lambda idx, _body: script[idx]) as srv:
                res = run_agent_loop(
                    worktree=wt,
                    system_prompt=sys_prefix,
                    user_prompt=task_prompt,
                    base_url=srv.url,
                    api_key="k",
                    model="m",
                    compact_threshold_chars=900,
                )

                self.assertTrue(res["ok"])
                self.assertEqual(len(srv.requests), 4)
                final_msgs = srv.requests[3]["messages"]
                # messages[0] and turn-0 messages[1] preserved byte-for-byte across compaction
                self.assertEqual(final_msgs[0]["content"], sys_prefix)
                self.assertEqual(final_msgs[1]["content"], srv.requests[0]["messages"][1]["content"])
                self.assertIn(task_prompt, final_msgs[1]["content"])
                self.assertIn("[Env] os=", final_msgs[1]["content"])
                # Middle tool output (t1) compacted to concise 1-line stub
                middle_tool_contents = [m["content"] for m in final_msgs[2:-4] if m.get("role") == "tool"]
                self.assertTrue(middle_tool_contents)
                self.assertTrue(
                    all(c == "[Compacted tool output: read_file -> ok]" for c in middle_tool_contents),
                    middle_tool_contents,
                )
                # Last 4 messages remain verbatim (containing full big_b/big_c contents)
                self.assertIn("B_0 =", final_msgs[-3]["content"])
                self.assertIn("C_0 =", final_msgs[-1]["content"])
                # Total chars shrank significantly compared to uncompacted (>6,000 chars of big_a + big_b + big_c)
                total_chars = sum(len(str(m.get("content") or "")) for m in final_msgs)
                uncompacted_estimate = len(sys_prefix) + len(task_prompt) + 3 * (len("1: A_0 = ") + 40 * 60)
                self.assertLess(total_chars, uncompacted_estimate)

    def test_spend_cap_exceeded_propagates(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            wt = Path(tmp)
            meter = RunMeter(
                pricing=PricingSchedule(input_usd_per_m=1000.0, cached_input_usd_per_m=1000.0, output_usd_per_m=1000.0),
                spend_cap_usd=0.0001,
            )
            script = [
                {
                    "tool_calls": [{"id": "c1", "type": "function", "function": {"name": "finish", "arguments": {"summary": "done"}}}],
                    "prompt_tokens": 500,
                    "cached_tokens": 0,
                    "completion_tokens": 500,
                }
            ]
            with MockOpenAIServer(lambda idx, _body: script[idx]) as srv:
                with self.assertRaises(SpendCapExceeded):
                    run_agent_loop(
                        worktree=wt,
                        system_prompt="SYS",
                        user_prompt="Go",
                        base_url=srv.url,
                        api_key="k",
                        model="m",
                        meter=meter,
                    )


class Phase2BSecretScrubberTests(unittest.TestCase):
    def test_scrub_secrets_redacts_all_patterns(self) -> None:
        samples = [
            ("sk-1234567890abcdef1234", "[REDACTED]"),
            ("sk-proj-abcdef1234567890_XYZ", "[REDACTED]"),
            ("sk-live-abcdef1234567890", "[REDACTED]"),
            ("sk-test-abcdef1234567890", "[REDACTED]"),
            ("pk_live_1234567890ab", "[REDACTED]"),
            ("rk_test_1234567890ab", "[REDACTED]"),
            ("oc_live_12345678", "[REDACTED]"),
            ("oc_sess_abcd1234_ef", "[REDACTED]"),
            ("oc_proxy_abcd1234", "[REDACTED]"),
            ("ghp_12345678901234567890abcd", "[REDACTED]"),
            ("gho_12345678901234567890abcd", "[REDACTED]"),
            ("ghu_12345678901234567890abcd", "[REDACTED]"),
            ("ghs_12345678901234567890abcd", "[REDACTED]"),
            ("github_pat_12345678901234567890_ab", "[REDACTED]"),
            ("AKIAIOSFODNN7EXAMPLE", "[REDACTED]"),
            ("Bearer abcdefghijklmnopqrstuvwxyz123456", "Bearer [REDACTED]"),
            (
                "-----BEGIN RSA PRIVATE KEY-----\nMIIEowIBAAKCAQEA1234\n-----END RSA PRIVATE KEY-----",
                "[REDACTED]",
            ),
        ]
        for secret, expected in samples:
            scrubbed = scrub_secrets(f"prefix {secret} suffix")
            self.assertNotIn(secret, scrubbed)
            self.assertEqual(scrubbed, f"prefix {expected} suffix")

    def test_scrubbing_in_run_command_write_note_and_error(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            wt = Path(tmp)
            tk = WorktreeToolkit(wt, agent_id="sec-a", sandbox_mode="off")
            secret_tok = "sk-proj-verysecret1234567890"
            bearer_tok = "Bearer my_super_long_bearer_token_1234567890"
            cmd_res = tk.run_command(
                f"{sys.executable} -c \"import sys; print('{secret_tok}'); print('{bearer_tok}', file=sys.stderr)\""
            )
            self.assertTrue(cmd_res["ok"])
            self.assertNotIn(secret_tok, cmd_res["stdout"])
            self.assertIn("[REDACTED]", cmd_res["stdout"])
            self.assertIn("Bearer [REDACTED]", cmd_res["stderr"])

            note_res = tk.write_note("Key", f"Found {secret_tok}")
            self.assertTrue(note_res["ok"])
            note_disk = (wt / ".open_swarm" / "notes" / "sec-a.md").read_text(encoding="utf-8")
            self.assertNotIn(secret_tok, note_disk)
            self.assertIn("[REDACTED]", note_disk)
            tk.close()

            with mock.patch(
                "agent_loop._openai_chat_turn",
                side_effect=RuntimeError(f"auth failed with {secret_tok}"),
            ):
                loop_res = run_agent_loop(
                    worktree=wt,
                    system_prompt="SYS",
                    user_prompt="Task",
                    base_url="http://127.0.0.1:1/v1",
                    api_key="k",
                    model="m",
                )
            self.assertFalse(loop_res["ok"])
            self.assertNotIn(secret_tok, loop_res["error"])
            self.assertIn("[REDACTED]", loop_res["error"])


class Phase2BSandboxTests(unittest.TestCase):
    def test_sandbox_off_reports_unsandboxed(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tk = WorktreeToolkit(Path(tmp), sandbox_mode="off")
            res = tk.run_command("echo hello")
            self.assertTrue(res["ok"])
            self.assertEqual(res["sandbox"], "unsandboxed")
            self.assertEqual(res["stdout"].strip(), "hello")

    def test_sandbox_auto_nested_fallback_and_strict_fail_closed(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            wt = Path(tmp)
            with mock.patch("agent_loop.sys.platform", "darwin"), mock.patch(
                "agent_loop.shutil.which", side_effect=lambda b: "/usr/bin/sandbox-exec" if b == "sandbox-exec" else None
            ):
                calls: list[list[str] | str] = []

                def fake_run(cmd, **kwargs):
                    calls.append(cmd)
                    if isinstance(cmd, list) and cmd[0] == "sandbox-exec":
                        return subprocess.CompletedProcess(
                            cmd, 71, stdout="", stderr="sandbox-exec: sandbox_apply: Operation not permitted\n"
                        )
                    return subprocess.CompletedProcess(cmd, 0, stdout="fallback-ok\n", stderr="")

                with mock.patch("agent_loop.subprocess.run", side_effect=fake_run):
                    tk_auto = WorktreeToolkit(wt, sandbox_mode="auto")
                    r_auto = tk_auto.run_command("echo fallback-ok")
                    self.assertTrue(r_auto["ok"])
                    self.assertEqual(r_auto["sandbox"], "unsandboxed-nested-fallback")
                    self.assertEqual(r_auto["stdout"].strip(), "fallback-ok")
                    self.assertEqual(len(calls), 2)

                    calls.clear()
                    tk_strict = WorktreeToolkit(wt, sandbox_mode="strict")
                    r_strict = tk_strict.run_command("echo fail")
                    self.assertFalse(r_strict["ok"])
                    self.assertEqual(r_strict["returncode"], 71)
                    self.assertEqual(r_strict["sandbox"], "seatbelt")
                    self.assertEqual(len(calls), 1)

    def test_linux_bwrap_argv_and_nested_fallback(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            wt = Path(tmp).resolve()
            with mock.patch("agent_loop.sys.platform", "linux"), mock.patch(
                "agent_loop.shutil.which", side_effect=lambda b: "/usr/bin/bwrap" if b == "bwrap" else None
            ):
                seen_argv: list[list[str]] = []

                def fake_bwrap(cmd, **kwargs):
                    if isinstance(cmd, list) and cmd[0] == "bwrap":
                        seen_argv.append(cmd)
                        return subprocess.CompletedProcess(cmd, 0, stdout="bwrap-ok\n", stderr="")
                    return subprocess.CompletedProcess(cmd, 0, stdout="direct\n", stderr="")

                with mock.patch("agent_loop.subprocess.run", side_effect=fake_bwrap):
                    tk = WorktreeToolkit(wt, sandbox_mode="auto")
                    r = tk.run_command("pwd")
                self.assertTrue(r["ok"])
                self.assertEqual(r["sandbox"], "bwrap")
                argv = seen_argv[0]
                self.assertIn("--tmpfs", argv)
                self.assertIn("--bind", argv)
                self.assertIn(str(wt), argv)

    def test_macos_real_seatbelt_if_unrestricted(self) -> None:
        if sys.platform != "darwin" or not shutil.which("sandbox-exec"):
            self.skipTest("macOS sandbox-exec not present")
        probe = subprocess.run(
            ["sandbox-exec", "-p", "(version 1)(allow default)", "/usr/bin/true"],
            capture_output=True,
            text=True,
        )
        if probe.returncode != 0:
            self.skipTest("Already inside nested macOS Seatbelt sandbox")

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp).resolve()
            wt = root / "wt"
            wt.mkdir()
            outside = root / "outside.txt"
            outside.write_text("orig", encoding="utf-8")

            tk = WorktreeToolkit(wt, sandbox_mode="strict")
            r_in = tk.run_command("echo ok > inside.txt && cat inside.txt")
            self.assertTrue(r_in["ok"], r_in)
            self.assertEqual(r_in["sandbox"], "seatbelt")
            self.assertEqual((wt / "inside.txt").read_text(encoding="utf-8").strip(), "ok")

            r_out = tk.run_command(f"echo pwn > {outside}")
            self.assertFalse(r_out["ok"])
            self.assertEqual(outside.read_text(encoding="utf-8"), "orig")

            ssh_dir = Path.home() / ".ssh"
            if ssh_dir.exists():
                r_ssh = tk.run_command(f"ls {ssh_dir}")
                self.assertFalse(r_ssh["ok"])


class Phase2BACIUpgradeTests(unittest.TestCase):
    def test_tools_schema_known_names_and_fallback_instructions(self) -> None:
        expected = {
            "read_file",
            "search_code",
            "list_dir",
            "edit_file",
            "write_file",
            "run_command",
            "start_bg_cmd",
            "poll_bg_cmd",
            "stop_bg_cmd",
            "git_diff",
            "post_board",
            "read_board",
            "claim_file",
            "write_note",
            "finish",
        }
        self.assertEqual(_KNOWN_TOOL_NAMES, expected)
        self.assertEqual({t["function"]["name"] for t in TOOLS_SCHEMA}, expected)
        for tool_name in expected:
            self.assertIn(tool_name, _FALLBACK_TOOL_INSTRUCTIONS)

    def test_list_dir_confinement_and_skip_dirs(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            wt = root / "wt"
            wt.mkdir()
            (wt / ".git").mkdir()
            (wt / ".git" / "HEAD").write_text("ref: refs/heads/main\n", encoding="utf-8")
            (wt / "__pycache__").mkdir()
            (wt / "__pycache__" / "m.pyc").write_bytes(b"00")
            (wt / "src").mkdir()
            (wt / "src" / "a.py").write_text("x = 1\n", encoding="utf-8")
            (wt / "README.md").write_text("hi\n", encoding="utf-8")
            outside = root / "outside"
            outside.mkdir()
            (wt / "bad_link").symlink_to(outside)

            tk = WorktreeToolkit(wt)
            r_root = tk.list_dir()
            self.assertTrue(r_root["ok"])
            names = [e["name"] for e in r_root["entries"]]
            self.assertEqual(names, ["README.md", "src"])

            r_sub = tk.list_dir("src")
            self.assertTrue(r_sub["ok"])
            self.assertEqual([e["path"] for e in r_sub["entries"]], ["src/a.py"])

            r_lim = tk.list_dir("", max_entries=1)
            self.assertTrue(r_lim["ok"])
            self.assertEqual(len(r_lim["entries"]), 1)
            self.assertTrue(r_lim["truncated"])

            self.assertFalse(tk.list_dir("..")["ok"])
            self.assertFalse(tk.list_dir(".git")["ok"])
            self.assertFalse(tk.list_dir("bad_link")["ok"])

    def test_git_diff_full_and_stat_only(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            wt = Path(tmp)
            subprocess.run(["git", "init"], cwd=str(wt), capture_output=True, check=True)
            subprocess.run(["git", "config", "user.name", "Test"], cwd=str(wt), capture_output=True, check=True)
            subprocess.run(["git", "config", "user.email", "t@example.com"], cwd=str(wt), capture_output=True, check=True)
            (wt / "app.py").write_text("VAL = 1\n", encoding="utf-8")
            subprocess.run(["git", "add", "app.py"], cwd=str(wt), capture_output=True, check=True)
            subprocess.run(["git", "commit", "-m", "init"], cwd=str(wt), capture_output=True, check=True)

            tk = WorktreeToolkit(wt)
            tk.edit_file("app.py", "VAL = 1", "VAL = 2")

            full = tk.git_diff(stat_only=False)
            self.assertTrue(full["ok"])
            self.assertIn("-VAL = 1", full["diff"])
            self.assertIn("+VAL = 2", full["diff"])

            stat = tk.git_diff(stat_only=True)
            self.assertTrue(stat["ok"])
            self.assertTrue(stat["stat_only"])
            self.assertIn("app.py", stat["diff"])

    def test_edit_file_closest_match_and_inline_diagnostics(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            wt = Path(tmp)
            lines = [f"line_{i} = {i}" for i in range(1, 15)]
            lines[6] = "def compute(x, y):"
            lines[7] = "    return x + y"
            (wt / "mod.py").write_text("\n".join(lines) + "\n", encoding="utf-8")

            tk = WorktreeToolkit(wt)
            miss = tk.edit_file("mod.py", "def compute(x, y):\n  return x + y", "pass")
            self.assertFalse(miss["ok"])
            self.assertIn("Closest match", miss["error"])
            self.assertIn("7: def compute(x, y):", miss["closest_match"])
            self.assertIn("8:     return x + y", miss["closest_match"])

            # Valid python write -> empty diagnostics and NO __pycache__ left behind
            ok_py = tk.write_file("clean.py", "def f(a):\n    return a * 2\n")
            self.assertTrue(ok_py["ok"])
            self.assertEqual(ok_py["diagnostics"], [])
            self.assertFalse((wt / "__pycache__").exists())

            # Invalid python edit -> diagnostics populated
            bad_py = tk.edit_file("clean.py", "return a * 2", "return (a *")
            self.assertTrue(bad_py["ok"])
            self.assertTrue(bad_py["diagnostics"])

            # JSON diagnostics
            ok_json = tk.write_file("cfg.json", '{"a": 1}\n')
            self.assertTrue(ok_json["ok"])
            self.assertEqual(ok_json["diagnostics"], [])

            bad_json = tk.write_file("bad.json", '{"a": }\n')
            self.assertTrue(bad_json["ok"])
            self.assertTrue(bad_json["diagnostics"])
            self.assertIn("JSONDecodeError", bad_json["diagnostics"][0])

            # JS diagnostics when node is installed
            if shutil.which("node"):
                bad_js = tk.write_file("bad.js", "function broken( { return 1; }\n")
                self.assertTrue(bad_js["ok"])
                self.assertTrue(bad_js["diagnostics"])

    def test_board_tools_standalone_continuity_and_callback(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            wt = Path(tmp) / "wt"
            wt.mkdir()
            (wt / "a.py").write_text("x = 1\n", encoding="utf-8")

            # Standalone mode
            tk_sa = WorktreeToolkit(wt, readonly_paths=("locked.txt",))
            p_sa = tk_sa.post_board("status", "working")
            self.assertTrue(p_sa["ok"])
            self.assertTrue(p_sa["standalone"])
            r_sa = tk_sa.read_board(0)
            self.assertTrue(r_sa["ok"])
            self.assertEqual(r_sa["events"], [])
            c_sa = tk_sa.claim_file("a.py")
            self.assertTrue(c_sa["ok"])
            self.assertTrue(c_sa["claimed"])
            self.assertFalse(tk_sa.claim_file("../escape.py")["ok"])
            self.assertFalse(tk_sa.claim_file("locked.txt")["ok"])

            # ContinuityStore-backed board
            store = ContinuityStore(Path(tmp) / "cont.db")
            tk_a = WorktreeToolkit(wt, agent_id="coder-a", board_store=store, run_id="run-1")
            tk_b = WorktreeToolkit(wt, agent_id="coder-b", board_store=store, run_id="run-1")
            tk_a.post_board("finding", "API ready", to_agent="coder-b")
            tk_a.claim_file("a.py")
            msgs = tk_b.read_board(since_seq=0)
            self.assertTrue(msgs["ok"])
            self.assertGreaterEqual(len(msgs["events"]), 2)
            self.assertTrue(any("API ready" in e["body"] for e in msgs["events"]))

            # Callback-backed board
            seen_cb: list[tuple[str, dict]] = []
            tk_cb = WorktreeToolkit(
                wt,
                agent_id="coder-c",
                run_id="run-cb",
                board_callback=lambda act, payload: seen_cb.append((act, payload)) or {"ok": True, "ack": act},
            )
            self.assertEqual(tk_cb.post_board("status", "hi")["ack"], "post_board")
            self.assertEqual(tk_cb.read_board(5)["ack"], "read_board")
            self.assertEqual(tk_cb.claim_file("a.py")["ack"], "claim_file")
            self.assertEqual([x[0] for x in seen_cb], ["post_board", "read_board", "claim_file"])

    def test_bg_cmd_lifecycle_scrubbing_and_cleanup_on_close(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            wt = Path(tmp)
            tk = WorktreeToolkit(wt, sandbox_mode="off")
            start = tk.start_bg_cmd(
                f"{sys.executable} -u -c \"print('boot sk-proj-secretbg1234567890'); import time; time.sleep(0.05); print('ready')\"",
                name="worker",
            )
            self.assertTrue(start["ok"], start)
            hid = start["handle_id"]
            collected = ""
            for _ in range(30):
                polled = tk.poll_bg_cmd(hid)
                self.assertTrue(polled["ok"])
                collected += polled["output"]
                if "ready" in collected and not polled["running"]:
                    break
                time.sleep(0.05)
            self.assertIn("boot [REDACTED]", collected)
            self.assertIn("ready", collected)
            self.assertNotIn("sk-proj-secretbg1234567890", collected)

            # Long-running process auto-cleaned by close()
            long_res = tk.start_bg_cmd(f"{sys.executable} -c \"import time; time.sleep(60)\"", name="sleeper")
            self.assertTrue(long_res["ok"])
            pid = long_res["pid"]
            tk.close()
            time.sleep(0.05)
            poll_after = tk.poll_bg_cmd(long_res["handle_id"])
            self.assertFalse(poll_after["running"])
            with self.assertRaises(OSError):
                os.kill(pid, 0)

    def test_turn0_env_header_injected_and_system_prompt_preserved(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            wt = Path(tmp)
            script = [
                {
                    "tool_calls": [
                        {"id": "f1", "type": "function", "function": {"name": "finish", "arguments": {"summary": "done"}}}
                    ]
                }
            ]
            with MockOpenAIServer(lambda idx, _body: script[idx]) as srv:
                res = run_agent_loop(
                    worktree=wt,
                    system_prompt="EXACT_SYSTEM_PREFIX",
                    user_prompt="Do the task.",
                    base_url=srv.url,
                    api_key="k",
                    model="m",
                    device_name="Mac Studio M3 Ultra",
                )
                self.assertTrue(res["ok"])
                msgs = srv.requests[0]["messages"]
                self.assertEqual(msgs[0]["content"], "EXACT_SYSTEM_PREFIX")
                user0 = msgs[1]["content"]
                header = user0.split("\n\nDo the task.")[0]
                self.assertEqual(len(header.splitlines()), 3)
                self.assertIn(sys.platform, header)
                self.assertIn("Mac Studio M3 Ultra", header)


class Phase2BSwarmRunTests(unittest.TestCase):
    def test_run_swarm_defaults_and_wiring(self) -> None:
        sig = inspect.signature(run_swarm)
        self.assertIs(sig.parameters["use_tool_loop"].default, True)
        self.assertIn("board_store", sig.parameters)
        self.assertIn("run_id", sig.parameters)

        with tempfile.TemporaryDirectory() as tmp:
            repo = Path(tmp) / "repo"
            repo.mkdir()
            subprocess.run(["git", "init"], cwd=str(repo), capture_output=True, check=True)
            subprocess.run(["git", "config", "user.name", "Test"], cwd=str(repo), capture_output=True, check=True)
            subprocess.run(["git", "config", "user.email", "t@example.com"], cwd=str(repo), capture_output=True, check=True)
            (repo / "app.py").write_text("X = 1\n", encoding="utf-8")
            subprocess.run(["git", "add", "app.py"], cwd=str(repo), capture_output=True, check=True)
            subprocess.run(["git", "commit", "-m", "init"], cwd=str(repo), capture_output=True, check=True)

            store = ContinuityStore(Path(tmp) / "board.db")
            script = [
                {
                    "tool_calls": [
                        {
                            "id": "b1",
                            "type": "function",
                            "function": {"name": "claim_file", "arguments": {"path": "app.py"}},
                        },
                        {
                            "id": "b2",
                            "type": "function",
                            "function": {"name": "edit_file", "arguments": {"path": "app.py", "old_text": "X = 1", "new_text": "X = 99"}},
                        },
                        {
                            "id": "b3",
                            "type": "function",
                            "function": {"name": "post_board", "arguments": {"kind": "status", "body": "updated X"}},
                        },
                        {
                            "id": "b4",
                            "type": "function",
                            "function": {"name": "finish", "arguments": {"summary": "done"}},
                        },
                    ]
                }
            ]
            with MockOpenAIServer(lambda idx, _body: script[idx]) as srv:
                res = run_swarm(
                    repo=repo,
                    tasks=[
                        {
                            "title": "Update X",
                            "spec": "Set X = 99 in app.py",
                            "acceptance_cmd": f"{sys.executable} -c \"import app; assert app.X == 99\"",
                        }
                    ],
                    base_url=srv.url,
                    model="mock-model",
                    board_store=store,
                    run_id="custom-run-42",
                )
            self.assertEqual(res["run_id"], "custom-run-42")
            self.assertEqual(res["passed_count"], 1)
            self.assertIn("+X = 99", res["results"][0]["diff"])
            bmsgs = store.read_board_messages("custom-run-42")
            leases = store.list_file_leases("custom-run-42")
            self.assertTrue(any(l["rel_path"] == "app.py" for l in leases))
            self.assertTrue(any(m["kind"] == "CLAIM_FILE" and "app.py" in m["body"] for m in bmsgs))
            self.assertTrue(any(m["kind"] == "FINDING" and "updated X" in m["body"] for m in bmsgs))


if __name__ == "__main__":
    unittest.main(verbosity=2)
