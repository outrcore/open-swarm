#!/usr/bin/env python3
"""Unified CLI for Open Swarm (`v0.5.0`).

Zero-dependency Python 3 standard library entry point providing:
  - `open-swarm doctor`: environment, sandbox, git, Tailscale & endpoint diagnostics
  - `open-swarm bench`: deterministic offline (or custom endpoint) multi-agent scorecard
  - `open-swarm board`: inspect, post, or resolve Swarm Message Board & file leases
  - `open-swarm devices`: inspect paired devices with live GPU/RAM telemetry
  - `open-swarm run`: run a headless worktree coding swarm
  - `open-swarm serve`: launch the desktop HTTP coordinator server
"""
from __future__ import annotations

import argparse
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
import os
from pathlib import Path
import platform
import shutil
import subprocess
import sys
import tempfile
import threading
import time
from typing import Any
import urllib.request

SWARM_DIR = Path(__file__).resolve().parent
if str(SWARM_DIR) not in sys.path:
    sys.path.insert(0, str(SWARM_DIR))

CLI_VERSION = "0.5.1"


def _default_state_dir() -> Path:
    env_dir = os.environ.get("OPEN_SWARM_STATE_DIR", "").strip()
    if env_dir:
        return Path(env_dir).expanduser().resolve().parent
    return Path.home() / ".open_swarm"


def run_doctor(*, backend_url: str = "", api_key: str = "", port: int = 18790) -> dict[str, Any]:
    """Run local runtime, sandbox, git, Tailscale, desktop-server, and device diagnostics."""
    from device_mesh import DeviceRegistry, discover_tailscale_peers
    from desktop_server import probe_backend

    checks: list[dict[str, Any]] = []

    py_ok = sys.version_info >= (3, 10)
    checks.append(
        {
            "name": "python",
            "ok": py_ok,
            "detail": f"{platform.python_version()} ({sys.executable})",
        }
    )

    git_bin = shutil.which("git")
    git_ver = ""
    if git_bin:
        try:
            gp = subprocess.run([git_bin, "--version"], capture_output=True, text=True, timeout=3, check=False)
            git_ver = (gp.stdout or "").strip()
        except Exception:
            git_ver = git_bin
    checks.append({"name": "git", "ok": bool(git_bin), "detail": git_ver or "missing"})

    if sys.platform == "darwin":
        sb_bin = shutil.which("sandbox-exec") or ("/usr/bin/sandbox-exec" if Path("/usr/bin/sandbox-exec").exists() else "")
        checks.append({"name": "os_sandbox", "ok": bool(sb_bin), "detail": f"macos seatbelt ({sb_bin or 'missing'})"})
    else:
        bw_bin = shutil.which("bwrap") or ""
        checks.append({"name": "os_sandbox", "ok": bool(bw_bin), "detail": f"linux bwrap ({bw_bin or 'not installed; auto fallback'})"})

    tmux_bin = shutil.which("tmux") or ""
    checks.append({"name": "tmux", "ok": bool(tmux_bin), "detail": tmux_bin or "not installed (optional)"})

    ts_info = discover_tailscale_peers()
    checks.append(
        {
            "name": "tailscale",
            "ok": bool(ts_info.get("running")),
            "detail": f"peers={len(ts_info.get('peers') or [])}" if ts_info.get("available") else "not detected",
        }
    )

    state_dir = _default_state_dir()
    reg = DeviceRegistry(state_dir / "devices.json")
    devs = reg.list_devices(probe=False)
    checks.append(
        {
            "name": "device_registry",
            "ok": True,
            "detail": f"{len(devs)} configured ({', '.join(d.get('name') or d.get('device_id', '') for d in devs)})",
        }
    )

    desktop_live = False
    desktop_ver = None
    try:
        with urllib.request.urlopen(f"http://127.0.0.1:{int(port)}/health", timeout=1.5) as resp:
            payload = json.loads(resp.read().decode("utf-8"))
            desktop_live = int(resp.status) == 200
            desktop_ver = payload.get("version")
    except Exception:
        desktop_live = False
    checks.append(
        {
            "name": "desktop_server",
            "ok": desktop_live,
            "detail": f"listening on 127.0.0.1:{port} (v{desktop_ver})" if desktop_live else f"not listening on 127.0.0.1:{port}",
        }
    )

    if backend_url.strip():
        probe = probe_backend(backend_url.strip(), api_key=api_key)
        checks.append(
            {
                "name": "llm_backend",
                "ok": bool(probe.get("ok")),
                "detail": f"{backend_url} ({probe.get('latency_ms')}ms)" if probe.get("ok") else str(probe.get("error") or "unreachable"),
            }
        )

    critical_ok = py_ok and bool(git_bin)
    return {
        "ok": critical_ok,
        "version": CLI_VERSION,
        "platform": f"{platform.system()} {platform.machine()}",
        "checks": checks,
    }


def run_offline_benchmark(*, keep_tmp: bool = False) -> dict[str, Any]:
    """Run an offline end-to-end deterministic multi-agent scorecard (zero external network calls)."""
    from continuity_store import ContinuityStore
    from desktop_server import DesktopAppState
    from orchestrator import review_and_repair_agent
    from swarm_run import run_swarm

    tmp_ctx = tempfile.TemporaryDirectory(prefix="open_swarm_bench_")
    tmp_dir = Path(tmp_ctx.name)
    try:
        repo = tmp_dir / "bench_repo"
        repo.mkdir()
        (repo / "pkg").mkdir()
        (repo / "tests").mkdir()
        (repo / "pkg" / "__init__.py").write_text("", encoding="utf-8")
        (repo / "tests" / "__init__.py").write_text("", encoding="utf-8")
        (repo / "README.md").write_text("# Open Swarm Bench Repo\n", encoding="utf-8")

        _g = ["git", "-c", "core.hooksPath=/dev/null", "-c", "core.fsmonitor=false"]
        subprocess.run([*_g, "init"], cwd=str(repo), capture_output=True, check=True)
        subprocess.run([*_g, "config", "user.name", "Open Swarm"], cwd=str(repo), capture_output=True, check=True)
        subprocess.run([*_g, "config", "user.email", "bench@outrcore.com"], cwd=str(repo), capture_output=True, check=True)
        subprocess.run([*_g, "add", "-A"], cwd=str(repo), capture_output=True, check=True)
        subprocess.run([*_g, "commit", "-m", "bench init"], cwd=str(repo), capture_output=True, check=True)

        turn_counter = {"n": 0}
        lock = threading.Lock()

        class _MockBenchHandler(BaseHTTPRequestHandler):
            def log_message(self, fmt: str, *args: Any) -> None:
                pass

            def do_POST(self) -> None:
                n = int(self.headers.get("Content-Length", "0") or "0")
                req = json.loads((self.rfile.read(n) if n > 0 else b"{}").decode("utf-8"))
                msgs = req.get("messages") or []
                full_txt = "\n".join(str(m.get("content") or "") for m in msgs)
                sys_txt = str((msgs[0] if msgs else {}).get("content") or "").lower()

                with lock:
                    turn_counter["n"] += 1
                    idx = turn_counter["n"]

                if "critic gate" in sys_txt or "senior code reviewer" in sys_txt:
                    if "critic_round_1" in full_txt and "def clamp01" not in full_txt:
                        reply = json.dumps(
                            {
                                "verdict": "request_changes",
                                "summary": "Missing edge clamp helper",
                                "test_quality": "weak",
                                "findings": [{"severity": "medium", "file": "pkg/math_ops.py", "issue": "add clamp01", "fix": "implement clamp01"}],
                            }
                        )
                    else:
                        reply = json.dumps(
                            {
                                "verdict": "approve",
                                "summary": "Verified cleanly",
                                "test_quality": "strong",
                                "findings": [],
                            }
                        )
                    msg_obj: dict[str, Any] = {"role": "assistant", "content": reply}
                elif req.get("tools"):
                    has_tool_resp = any(m.get("role") == "tool" for m in msgs)
                    is_math = "math_ops" in full_txt
                    if not has_tool_resp:
                        if is_math:
                            tcs = [
                                {"id": f"c{idx}a", "type": "function", "function": {"name": "claim_file", "arguments": json.dumps({"path": "pkg/math_ops.py"})}},
                                {"id": f"c{idx}b", "type": "function", "function": {"name": "write_file", "arguments": json.dumps({"path": "pkg/math_ops.py", "content": "def add(a, b):\n    return a + b\n"})}},
                                {"id": f"c{idx}c", "type": "function", "function": {"name": "write_file", "arguments": json.dumps({"path": "tests/test_math_ops.py", "content": "import unittest\nfrom pkg.math_ops import add\nclass T(unittest.TestCase):\n    def test_add(self):\n        self.assertEqual(add(2, 3), 5)\n"})}},
                                {"id": f"c{idx}d", "type": "function", "function": {"name": "post_board", "arguments": json.dumps({"kind": "CONTRACT", "body": "pkg.math_ops.add(a, b) exported"})}},
                            ]
                        else:
                            tcs = [
                                {"id": f"c{idx}a", "type": "function", "function": {"name": "claim_file", "arguments": json.dumps({"path": "pkg/str_ops.py"})}},
                                {"id": f"c{idx}b", "type": "function", "function": {"name": "write_file", "arguments": json.dumps({"path": "pkg/str_ops.py", "content": "def slug(s):\n    return '-'.join(s.lower().split())\n"})}},
                                {"id": f"c{idx}c", "type": "function", "function": {"name": "write_file", "arguments": json.dumps({"path": "tests/test_str_ops.py", "content": "import unittest\nfrom pkg.str_ops import slug\nclass T(unittest.TestCase):\n    def test_slug(self):\n        self.assertEqual(slug('Open Swarm'), 'open-swarm')\n"})}},
                                {"id": f"c{idx}d", "type": "function", "function": {"name": "post_board", "arguments": json.dumps({"kind": "DONE", "body": "pkg.str_ops.slug ready"})}},
                            ]
                        msg_obj = {"role": "assistant", "content": "", "tool_calls": tcs}
                    else:
                        msg_obj = {
                            "role": "assistant",
                            "content": "",
                            "tool_calls": [
                                {"id": f"c{idx}z", "type": "function", "function": {"name": "finish", "arguments": json.dumps({"summary": "Completed ticket"})}}
                            ],
                        }
                else:
                    msg_obj = {"role": "assistant", "content": "ok"}

                raw = json.dumps(
                    {
                        "choices": [{"message": msg_obj}],
                        "usage": {"prompt_tokens": 120, "completion_tokens": 45, "prompt_tokens_details": {"cached_tokens": 90}},
                    }
                ).encode("utf-8")
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(raw)))
                self.end_headers()
                self.wfile.write(raw)

        srv = ThreadingHTTPServer(("127.0.0.1", 0), _MockBenchHandler)
        port = srv.server_address[1]
        threading.Thread(target=srv.serve_forever, daemon=True).start()
        t0 = time.perf_counter()
        try:
            store = ContinuityStore(tmp_dir / "bench.sqlite3")
            run_id = "bench-run-1"
            tasks = [
                {
                    "id": "T1",
                    "phase": "contract",
                    "title": "Math utilities",
                    "spec": "Create pkg/math_ops.py and tests/test_math_ops.py",
                    "acceptance_cmd": f"{sys.executable} -m unittest tests/test_math_ops.py",
                },
                {
                    "id": "T2",
                    "depends_on": ["T1"],
                    "title": "String utilities",
                    "spec": "Create pkg/str_ops.py and tests/test_str_ops.py",
                    "acceptance_cmd": f"{sys.executable} -m unittest tests/test_str_ops.py",
                },
            ]
            rec = run_swarm(
                repo=repo,
                tasks=tasks,
                base_url=f"http://127.0.0.1:{port}/v1",
                model="mock-coder",
                continuity_store=store,
                board_store=store,
                run_id=run_id,
                auto_review=True,
                keep_worktrees=True,
                sandbox_mode="off",
            )

            # Verify critic + 1-round auto-repair recovery flow explicitly
            repair_res = review_and_repair_agent(
                ticket_title="critic_round_1",
                ticket_spec="Ensure clamp01 exists",
                diff="+def add(a,b): return a+b",
                test_output="OK",
                author_model="mock-coder",
                reviewer_base_url=f"http://127.0.0.1:{port}/v1",
                reviewer_model="gpt-reviewer",
                max_repair_rounds=1,
                repair_fn=lambda _f, _r: {"diff": "+def clamp01(x): return max(0.0, min(1.0, x))", "test_output": "OK"},
            )

            # Verify post-merge integration gate on DesktopAppState
            app_state = DesktopAppState(runs_dir=tmp_dir / "app_runs")
            app_state._runs[run_id] = dict(rec, tasks=tasks, status="completed", applied_agents=[])
            app_state._save_run(app_state._runs[run_id])
            gate_out = app_state.apply_all_passed(
                run_id,
                verify_cmd=f"{sys.executable} -m unittest discover -s tests -v",
            )
            board_msgs = store.read_board_messages(run_id)
            leases = store.list_file_leases(run_id)
        finally:
            srv.shutdown()
            srv.server_close()

        wall_s = round(time.perf_counter() - t0, 3)
        total_tool_calls = sum(int(r.get("tool_calls") or 0) for r in rec.get("results") or [])
        return {
            "ok": (
                rec["passed_count"] == 2
                and len(gate_out.get("applied") or []) == 2
                and bool(gate_out.get("integration_gate", {}).get("ok"))
                and repair_res.get("verdict") == "approve"
                and bool(repair_res.get("repair_attempted"))
            ),
            "version": CLI_VERSION,
            "agents_count": rec["agents_count"],
            "passed_count": rec["passed_count"],
            "applied_count": len(gate_out.get("applied") or []),
            "integration_gate_ok": bool(gate_out.get("integration_gate", {}).get("ok")),
            "repair_recovery_ok": repair_res.get("verdict") == "approve" and bool(repair_res.get("repair_attempted")),
            "board_messages": len(board_msgs),
            "active_leases": len(leases),
            "total_tool_calls": total_tool_calls,
            "cache_hit_pct": (rec.get("metering") or {}).get("cache_hit_pct"),
            "wall_s": wall_s,
        }
    finally:
        if not keep_tmp:
            tmp_ctx.cleanup()


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="open-swarm", description="OutrCore Open Swarm CLI")
    ap.add_argument("--version", action="version", version=f"open-swarm {CLI_VERSION}")
    sub = ap.add_subparsers(dest="command")

    p_serve = sub.add_parser("serve", help="Run the Open Swarm Desktop HTTP server")
    p_serve.add_argument("--host", default="127.0.0.1")
    p_serve.add_argument("--port", type=int, default=18790)

    p_doc = sub.add_parser("doctor", help="Check local environment, sandbox, git, Tailscale & desktop status")
    p_doc.add_argument("--backend-url", default="")
    p_doc.add_argument("--api-key", default="")
    p_doc.add_argument("--port", type=int, default=18790)
    p_doc.add_argument("--json", action="store_true", dest="as_json")

    p_bench = sub.add_parser("bench", help="Run deterministic offline Open Swarm benchmark scorecard")
    p_bench.add_argument("--json", action="store_true", dest="as_json")

    p_dev = sub.add_parser("devices", help="List paired compute mesh devices and live telemetry")
    p_dev.add_argument("--no-probe", action="store_true")
    p_dev.add_argument("--json", action="store_true", dest="as_json")

    p_board = sub.add_parser("board", help="Read, post, or resolve Swarm Message Board entries")
    p_board.add_argument("--run-id", required=True)
    p_board.add_argument("--db", default="")
    p_board.add_argument("--post-kind", default="")
    p_board.add_argument("--body", default="")
    p_board.add_argument("--from-agent", default="human")
    p_board.add_argument("--resolve-seq", type=int, default=0)
    p_board.add_argument("--resolution", default="")
    p_board.add_argument("--json", action="store_true", dest="as_json")

    p_run = sub.add_parser("run", help="Run headless swarm tickets against a local repo")
    p_run.add_argument("--repo", required=True)
    p_run.add_argument("--backend-url", required=True)
    p_run.add_argument("--model", default="thinkingmachines/Inkling-2-Mini")
    p_run.add_argument("--api-key", default="outrcore-swarm")
    p_run.add_argument("--spend-cap-usd", type=float, default=1.0)
    p_run.add_argument("--tasks-json", required=True)
    p_run.add_argument("--out", default="")

    args = ap.parse_args(argv)
    cmd = args.command or "serve"

    if cmd == "serve":
        from desktop_server import main as server_main

        srv_argv = []
        if getattr(args, "host", None):
            srv_argv += ["--host", args.host]
        if getattr(args, "port", None):
            srv_argv += ["--port", str(args.port)]
        return server_main(srv_argv)

    if cmd == "doctor":
        rep = run_doctor(backend_url=args.backend_url, api_key=args.api_key, port=args.port)
        if args.as_json:
            print(json.dumps(rep, indent=2))
        else:
            print(f"Open Swarm Doctor v{rep['version']} ({rep['platform']})")
            for c in rep["checks"]:
                mark = "PASS" if c["ok"] else "WARN"
                print(f"  [{mark}] {c['name']:16s} {c['detail']}")
        return 0 if rep["ok"] else 1

    if cmd == "bench":
        res = run_offline_benchmark()
        if args.as_json:
            print(json.dumps(res, indent=2))
        else:
            print(f"Open Swarm Offline Benchmark v{res['version']}")
            print(
                f"  passed={res['passed_count']}/{res['agents_count']} applied={res['applied_count']} "
                f"gate_ok={res['integration_gate_ok']} repair_ok={res['repair_recovery_ok']} "
                f"tool_calls={res['total_tool_calls']} wall={res['wall_s']}s"
            )
        return 0 if res["ok"] else 1

    if cmd == "devices":
        from device_mesh import DeviceRegistry

        reg = DeviceRegistry(_default_state_dir() / "devices.json")
        devs = reg.list_devices(probe=not args.no_probe)
        if args.as_json:
            print(json.dumps({"devices": devs}, indent=2))
        else:
            for d in devs:
                gpu = d.get("gpu_util_pct")
                gpu_s = f"GPU {gpu:.0f}%" if gpu is not None else "CPU"
                mem = d.get("mem_avail_gb")
                mem_s = f"{mem:.1f}G free" if mem is not None else ""
                print(f"  {d.get('device_id'):16s} {'ONLINE' if d.get('online') else 'OFFLINE':7s} {gpu_s:10s} {mem_s:10s} {d.get('name')}")
        return 0

    if cmd == "board":
        from continuity_store import ContinuityStore

        db_path = Path(args.db) if args.db else (_default_state_dir() / "continuity.sqlite3")
        store = ContinuityStore(db_path)
        if args.post_kind:
            out = store.post_board_message(args.run_id, args.from_agent, args.post_kind, args.body)
        elif args.resolve_seq:
            out = store.resolve_board_decision(args.resolve_seq, args.resolution or "approved")
        else:
            out = {
                "run_id": args.run_id,
                "messages": store.read_board_messages(args.run_id),
                "leases": store.list_file_leases(args.run_id),
            }
        if args.as_json:
            print(json.dumps(out, indent=2))
        else:
            print(json.dumps(out, indent=2))
        return 0

    if cmd == "run":
        from swarm_run import run_swarm

        tasks = json.loads(Path(args.tasks_json).read_text(encoding="utf-8"))
        rec = run_swarm(
            repo=Path(args.repo),
            tasks=tasks,
            base_url=args.backend_url,
            model=args.model,
            api_key=args.api_key,
            spend_cap_usd=args.spend_cap_usd,
        )
        txt = json.dumps(rec, indent=2)
        if args.out:
            Path(args.out).write_text(txt + "\n", encoding="utf-8")
        print(txt)
        return 0 if rec.get("passed_count") == rec.get("agents_count") else 1

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
