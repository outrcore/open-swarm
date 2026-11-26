#!/usr/bin/env python3
"""Tests for open_swarm/agent_loop.py (WorktreeToolkit + run_agent_loop against local mock OpenAI server)."""
from __future__ import annotations

from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
import os
from pathlib import Path
import sys
import tempfile
import threading
import time
import unittest

SWARM_DIR = Path(__file__).resolve().parents[1]
if str(SWARM_DIR) not in sys.path:
    sys.path.insert(0, str(SWARM_DIR))

from agent_loop import TOOLS_SCHEMA, WorktreeToolkit, run_agent_loop  # noqa: E402
from metering import PricingSchedule, RunMeter, SpendCapExceeded  # noqa: E402


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
                # messages[0] and messages[1] preserved byte-for-byte
                self.assertEqual(final_msgs[0]["content"], sys_prefix)
                self.assertEqual(final_msgs[1]["content"], task_prompt)
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


if __name__ == "__main__":
    unittest.main(verbosity=2)
