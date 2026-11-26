#!/usr/bin/env python3
"""Open Swarm Desktop Local Backend (stdlib HTTP server for Open Swarm.app).

Binds strictly to 127.0.0.1 and serves the Open Swarm desktop UI, probes remote
or local OutrCore GPUs, streams live GPU chat (thinking + content + decode tok/s),
manages an isolated git self-build workspace + hot-reloadable staging preview
server (port 18795), launches parallel worktree swarm runs, and applies diffs.
"""
from __future__ import annotations

import argparse
import atexit
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
from typing import Any
import urllib.error
from urllib.parse import parse_qs, urlsplit
import urllib.request

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

from agent_tools import (  # noqa: E402
    build_agent_context_brief,
    decompose_goal_to_tasks,
    detect_file_collisions,
)
from continuity_store import (  # noqa: E402
    ContextConflictError,
    ContinuityStore,
    compute_file_hashes,
    extract_diff_paths,
)
from fleet_scanner import scan_fleet  # noqa: E402
from orchestrator import LANES, review_agent_diff, run_tournament  # noqa: E402
from swarm_run import openai_chat, run_swarm  # noqa: E402

PRESETS: list[dict[str, str]] = [
    {
        "id": "outrcore_cloud",
        "label": "OutrCore Cloud (api.outrcore.com)",
        "backend_url": "https://api.outrcore.com/v1",
        "model": "ornith-ai/Ornith-1.5-35B-A3B",
    },
    {
        "id": "local_openai",
        "label": "Local Server (Ollama / vLLM / k5cuda)",
        "backend_url": "http://127.0.0.1:8000/v1",
        "model": "default",
    },
    {
        "id": "local_router",
        "label": "Local GPU Router",
        "backend_url": "http://127.0.0.1:18720/v1",
        "model": "thinkingmachines/Inkling-2-Mini",
    },
]

RUNS_DIR = Path(os.environ.get("OPEN_SWARM_STATE_DIR") or (Path.home() / ".open_swarm" / "runs"))


def load_presets() -> list[dict[str, str]]:
    """Return user/env presets ahead of clean public default presets."""
    out: list[dict[str, str]] = []
    seen_ids: set[str] = set()
    env_url = os.environ.get("OPEN_SWARM_BACKEND_URL", "").strip()
    if env_url:
        out.append(
            {
                "id": "env_backend",
                "label": "Environment Backend",
                "backend_url": env_url,
                "model": os.environ.get("OPEN_SWARM_MODEL", "").strip() or "default",
            }
        )
        seen_ids.add("env_backend")
    presets_file = Path(
        os.environ.get("OPEN_SWARM_PRESETS_PATH") or (RUNS_DIR.parent / "presets.json")
    )
    if presets_file.is_file():
        try:
            raw = json.loads(presets_file.read_text(encoding="utf-8"))
            if isinstance(raw, list):
                for item in raw:
                    if isinstance(item, dict) and item.get("backend_url"):
                        pid = str(item.get("id") or f"custom_{len(out)}")
                        if pid in seen_ids:
                            continue
                        seen_ids.add(pid)
                        out.append(
                            {
                                "id": pid,
                                "label": str(item.get("label") or pid),
                                "backend_url": str(item["backend_url"]).strip(),
                                "model": str(item.get("model") or "default").strip(),
                            }
                        )
        except Exception:
            pass
    for p in PRESETS:
        if p["id"] not in seen_ids:
            seen_ids.add(p["id"])
            out.append(p)
    return out


def create_or_sync_self_build_repo(self_build_dir: Path | None = None, *, reset: bool = False) -> str:
    """Create or ensure an isolated git repository holding a working copy of Open Swarm."""
    target = (self_build_dir or (RUNS_DIR.parent / "self_build_repo")).resolve()
    if reset and target.exists():
        shutil.rmtree(target, ignore_errors=True)
    if (target / ".git").exists() and (target / "desktop_server.py").is_file():
        return str(target)

    target.mkdir(parents=True, exist_ok=True)
    (target / "tools").mkdir(parents=True, exist_ok=True)
    (target / "tests").mkdir(parents=True, exist_ok=True)
    (target / "macos").mkdir(parents=True, exist_ok=True)

    for fname in (
        "desktop_server.py",
        "desktop_ui.html",
        "swarm_run.py",
        "metering.py",
        "continuity_store.py",
        "fleet_scanner.py",
        "orchestrator.py",
        "agent_tools.py",
    ):
        src = HERE / fname
        if src.is_file():
            shutil.copy2(src, target / fname)
    for rel in (
        "macos/OpenSwarmApp.swift",
        "macos/build_app.sh",
        "tests/test_desktop_server.py",
        "tests/test_metering.py",
        "tests/test_continuity_store.py",
        "tests/test_fleet_scanner.py",
        "tests/test_orchestrator.py",
        "tests/test_agent_tools.py",
    ):
        src = HERE / rel
        if src.is_file():
            shutil.copy2(src, target / rel)

    tools_init = target / "tools" / "__init__.py"
    if not tools_init.exists():
        tools_init.write_text('"""Agent-authored Open Swarm tools."""\n', encoding="utf-8")
    tests_init = target / "tests" / "__init__.py"
    if not tests_init.exists():
        tests_init.write_text("", encoding="utf-8")
    readme = target / "README.md"
    if not readme.exists():
        readme.write_text(
            "# Open Swarm Self-Build Copy\n\nIsolated copy where Open Swarm agents build tools & UI updates safely.\n",
            encoding="utf-8",
        )

    if not (target / ".git").exists():
        subprocess.run(["git", "init"], cwd=str(target), capture_output=True, check=True)
        subprocess.run(["git", "config", "user.name", "Open Swarm"], cwd=str(target), capture_output=True, check=True)
        subprocess.run(["git", "config", "user.email", "swarm@outrcore.com"], cwd=str(target), capture_output=True, check=True)
        subprocess.run(["git", "add", "-A"], cwd=str(target), capture_output=True, check=True)
        subprocess.run(["git", "commit", "-m", "Initial Open Swarm copy"], cwd=str(target), capture_output=True, check=True)
    return str(target)


def _default_repo_path() -> str:
    env_repo = os.environ.get("OPEN_SWARM_DEFAULT_REPO", "").strip()
    if env_repo and Path(env_repo).is_dir() and (Path(env_repo) / ".git").exists():
        return env_repo
    try:
        return create_or_sync_self_build_repo()
    except Exception:
        cand = HERE.parent
        if (cand / ".git").exists():
            return str(cand)
        return str(HERE)


def _base_root_url(backend_url: str) -> str:
    u = backend_url.strip().rstrip("/")
    if u.endswith("/v1"):
        return u[:-3]
    return u


def probe_backend(backend_url: str, api_key: str = "") -> dict[str, Any]:
    """Probe /health and /v1/models on a GPU server without mutating state."""
    root = _base_root_url(backend_url)
    hdrs: dict[str, str] = {"User-Agent": "OutrCore-OpenSwarm/0.1", "Accept": "application/json"}
    if api_key.strip():
        hdrs["Authorization"] = f"Bearer {api_key.strip()}"

    health: dict[str, Any] = {}
    models: list[str] = []
    latency_ms: float | None = None
    error: str | None = None

    t0 = time.perf_counter()
    try:
        req = urllib.request.Request(f"{root}/health", headers=hdrs)
        with urllib.request.urlopen(req, timeout=8.0) as resp:
            health = json.loads(resp.read().decode("utf-8"))
        latency_ms = round((time.perf_counter() - t0) * 1000.0, 1)
    except Exception as exc:
        error = f"health probe failed: {exc}"

    try:
        req_m = urllib.request.Request(f"{root}/v1/models", headers=hdrs)
        with urllib.request.urlopen(req_m, timeout=8.0) as resp:
            data = json.loads(resp.read().decode("utf-8"))
        models = [str(m.get("id")) for m in (data.get("data") or []) if m.get("id")]
    except Exception:
        if health.get("model"):
            models = [str(health["model"])]

    return {
        "ok": error is None and health.get("status", "ok") == "ok",
        "backend_url": backend_url,
        "latency_ms": latency_ms,
        "health": health,
        "models": models,
        "error": error,
    }


def create_sample_sandbox_repo() -> str:
    """Create a disposable initialized git repo under ~/.open_swarm/demo_repo for instant testing."""
    demo_dir = RUNS_DIR.parent / "demo_repo"
    if (demo_dir / ".git").exists():
        return str(demo_dir)
    (demo_dir / "pkg").mkdir(parents=True, exist_ok=True)
    (demo_dir / "tests").mkdir(parents=True, exist_ok=True)
    (demo_dir / "pkg" / "__init__.py").write_text("", encoding="utf-8")
    (demo_dir / "tests" / "__init__.py").write_text("", encoding="utf-8")
    (demo_dir / "README.md").write_text("# Open Swarm Sandbox Repo\n", encoding="utf-8")
    subprocess.run(["git", "init"], cwd=str(demo_dir), capture_output=True, check=True)
    subprocess.run(["git", "config", "user.name", "Open Swarm"], cwd=str(demo_dir), capture_output=True, check=True)
    subprocess.run(["git", "config", "user.email", "swarm@outrcore.com"], cwd=str(demo_dir), capture_output=True, check=True)
    subprocess.run(["git", "add", "-A"], cwd=str(demo_dir), capture_output=True, check=True)
    subprocess.run(["git", "commit", "-m", "Initial sandbox"], cwd=str(demo_dir), capture_output=True, check=True)
    return str(demo_dir)


class PreviewServerManager:
    """Manages an isolated hot-reloadable staging preview server running from the self-build copy."""

    def __init__(self, port: int = 18795, default_repo: Path | None = None) -> None:
        self.port = int(port)
        self.default_repo = default_repo or (RUNS_DIR.parent / "self_build_repo")
        self._lock = threading.Lock()
        self._proc: subprocess.Popen | None = None
        self._repo: str = str(self.default_repo)
        self._last_error: str | None = None
        atexit.register(self.stop)

    def _probe_health(self) -> bool:
        try:
            with urllib.request.urlopen(f"http://127.0.0.1:{self.port}/health", timeout=0.8) as r:
                return int(r.status) == 200
        except Exception:
            return False

    def status(self) -> dict[str, Any]:
        with self._lock:
            running = self._proc is not None and self._proc.poll() is None
            pid = self._proc.pid if running and self._proc else None
            repo = self._repo
            err = self._last_error
        healthy = self._probe_health() if running else False
        return {
            "running": running,
            "healthy": healthy,
            "pid": pid,
            "port": self.port,
            "url": f"http://127.0.0.1:{self.port}/",
            "repo": repo,
            "last_error": err,
        }

    def stop(self) -> dict[str, Any]:
        with self._lock:
            proc = self._proc
            self._proc = None
        if proc is not None and proc.poll() is None:
            proc.terminate()
            try:
                proc.wait(timeout=3.0)
            except subprocess.TimeoutExpired:
                proc.kill()
        return self.status()

    def start(self, repo_path: str | None = None) -> dict[str, Any]:
        self.stop()
        repo = Path(repo_path or create_or_sync_self_build_repo(self.default_repo)).expanduser().resolve()
        srv_script = repo / "desktop_server.py"
        if not srv_script.is_file():
            repo = Path(create_or_sync_self_build_repo(self.default_repo))
            srv_script = repo / "desktop_server.py"
        env = dict(os.environ)
        env["PYTHONUNBUFFERED"] = "1"
        env["OPEN_SWARM_DEFAULT_REPO"] = str(repo)
        proc = subprocess.Popen(
            [sys.executable, str(srv_script), "--host", "127.0.0.1", "--port", str(self.port)],
            cwd=str(repo),
            env=env,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.PIPE,
            text=True,
        )
        with self._lock:
            self._proc = proc
            self._repo = str(repo)
            self._last_error = None

        deadline = time.monotonic() + 4.0
        while time.monotonic() < deadline:
            if proc.poll() is not None:
                err_txt = (proc.stderr.read() if proc.stderr else "")[-400:]
                with self._lock:
                    self._last_error = f"preview server exited ({proc.returncode}): {err_txt}"
                break
            if self._probe_health():
                break
            time.sleep(0.1)
        return self.status()

    def restart(self, repo_path: str | None = None) -> dict[str, Any]:
        return self.start(repo_path=repo_path)


class DesktopAppState:
    """Thread-safe coordinator holding active runs, chat telemetry, and preview server."""

    def __init__(
        self,
        runs_dir: Path | None = None,
        runner_fn=None,
        preview_port: int = 18795,
    ) -> None:
        self.runs_dir = runs_dir or RUNS_DIR
        self.runs_dir.mkdir(parents=True, exist_ok=True)
        self.self_build_dir = self.runs_dir.parent / "self_build_repo"
        self.telemetry_path = self.runs_dir.parent / "chat_telemetry.jsonl"
        self.continuity = ContinuityStore(self.runs_dir.parent / "continuity.sqlite3")
        self.preview = PreviewServerManager(port=preview_port, default_repo=self.self_build_dir)
        self.runner_fn = runner_fn or run_swarm
        self._lock = threading.Lock()
        self._runs: dict[str, dict[str, Any]] = {}
        self._load_saved_runs()

    def _load_saved_runs(self) -> None:
        for p in sorted(self.runs_dir.glob("*.json"))[-30:]:
            try:
                rec = json.loads(p.read_text(encoding="utf-8"))
                rid = str(rec.get("run_id") or "")
                if rid:
                    self._runs[rid] = rec
            except Exception:
                continue

    def _save_run(self, rec: dict[str, Any]) -> None:
        rid = str(rec["run_id"])
        with self._lock:
            self._runs[rid] = rec
        (self.runs_dir / f"{rid}.json").write_text(json.dumps(rec, indent=2) + "\n", encoding="utf-8")

    def record_chat_telemetry(self, item: dict[str, Any]) -> None:
        entry = {"ts": time.time(), **item}
        with self._lock:
            with self.telemetry_path.open("a", encoding="utf-8") as fh:
                fh.write(json.dumps(entry) + "\n")

    def get_chat_telemetry(self, limit: int = 30) -> dict[str, Any]:
        rows: list[dict[str, Any]] = []
        if self.telemetry_path.is_file():
            for line in self.telemetry_path.read_text(encoding="utf-8", errors="replace").splitlines()[-limit:]:
                try:
                    rows.append(json.loads(line))
                except Exception:
                    continue
        tok_rates = [float(r["tok_s"]) for r in rows if r.get("tok_s") is not None]
        ttfts = [float(r["ttft_ms"]) for r in rows if r.get("ttft_ms") is not None]
        return {
            "turns": len(rows),
            "avg_tok_s": round(sum(tok_rates) / len(tok_rates), 1) if tok_rates else None,
            "avg_ttft_ms": round(sum(ttfts) / len(ttfts), 1) if ttfts else None,
            "recent": list(reversed(rows)),
        }

    def list_runs(self) -> list[dict[str, Any]]:
        with self._lock:
            items = list(self._runs.values())
        items.sort(key=lambda r: float(r.get("created_at", 0.0)), reverse=True)
        return items[:30]

    def get_run(self, run_id: str) -> dict[str, Any] | None:
        with self._lock:
            return self._runs.get(run_id)

    def start_run(
        self,
        *,
        repo_path: str,
        backend_url: str,
        model: str,
        api_key: str,
        spend_cap_usd: float,
        tasks: list[dict[str, Any]],
        max_tokens: int = 4096,
        mode: str = "swarm",
        goal: str | None = None,
        max_repair_turns: int = 2,
        auto_review: bool = True,
        judge_model: str | None = None,
        judge_base_url: str | None = None,
    ) -> dict[str, Any]:
        repo = Path(repo_path).expanduser().resolve()
        if not repo.is_dir():
            raise ValueError(f"Repository path does not exist: {repo}")
        if not (repo / ".git").exists():
            raise ValueError(f"Directory is not a git repository (.git missing): {repo}")
        clean_goal = (goal or "").strip()
        if not tasks and not clean_goal:
            raise ValueError("Enter a goal or at least one agent ticket")

        run_mode = "tournament" if str(mode).strip().lower() == "tournament" else "swarm"
        run_id = time.strftime("%Y%m%d-%H%M%S") + f"-{os.getpid()}-{int(time.time()*1000)%1000:03d}"
        now_ts = time.time()
        if not tasks and clean_goal:
            live_specs = [
                {"agent_id": "architect", "id": "PLAN", "title": f"Planning: {clean_goal[:60]}"},
            ]
        elif run_mode == "tournament":
            base_title = str(tasks[0].get("title") or tasks[0].get("id") or "T-001")
            live_specs = [
                {"agent_id": "cand-a", "id": "T-001", "title": f"{base_title} [minimal-patch]"},
                {"agent_id": "cand-b", "id": "T-001", "title": f"{base_title} [defensive-edge-cases]"},
                {"agent_id": "cand-c", "id": "T-001", "title": f"{base_title} [clean-modular]"},
            ]
        else:
            live_specs = tasks
        agents_live: dict[str, dict[str, Any]] = {}
        for idx, t in enumerate(live_specs):
            aid = str(t.get("agent_id") or f"coder-{chr(ord('a') + idx)}")
            ticket_id = str(t.get("id") or f"T-{idx+1:03d}")
            agents_live[aid] = {
                "agent_id": aid,
                "ticket": ticket_id,
                "title": str(t.get("title") or ticket_id),
                "stage": "planning" if (not tasks and clean_goal) else "queued",
                "reasoning_chars": 0,
                "content_chars": 0,
                "snippet": "",
                "updated_at": now_ts,
            }
        initial: dict[str, Any] = {
            "run_id": run_id,
            "mode": run_mode,
            "goal": clean_goal,
            "status": "running",
            "created_at": now_ts,
            "repo": str(repo),
            "backend_url": backend_url,
            "model": model,
            "spend_cap_usd": float(spend_cap_usd),
            "max_tokens": int(max_tokens),
            "max_repair_turns": int(max_repair_turns),
            "auto_review": bool(auto_review),
            "agents_count": len(live_specs),
            "passed_count": 0,
            "tasks": list(tasks),
            "results": [],
            "collisions": {},
            "agents_live": agents_live,
            "events": [
                {
                    "ts": now_ts,
                    "kind": "planning" if (not tasks and clean_goal) else "run_start",
                    "msg": (
                        "Decomposing goal into parallel agent tickets..."
                        if (not tasks and clean_goal)
                        else f"Queued {len(live_specs)} {'tournament candidates' if run_mode == 'tournament' else 'parallel agent(s)'}"
                    ),
                }
            ],
            "prewarm_status": "pending",
            "metering": {},
            "applied_agents": [],
            "error": None,
        }
        self._save_run(initial)
        last_persist = [time.monotonic()]

        def _on_progress(ev: dict[str, Any]) -> None:
            should_persist = False
            snap_to_write: dict[str, Any] | None = None
            with self._lock:
                cur = self._runs.get(run_id)
                if not cur or cur.get("status") != "running":
                    return
                ev_kind = str(ev.get("kind") or "")
                ev_ts = time.time()
                if ev_kind == "prewarm_start":
                    cur["prewarm_status"] = "warming"
                    cur.setdefault("events", []).append(
                        {"ts": ev_ts, "kind": "prewarm_start", "msg": "Warming shared KV prefix cache..."}
                    )
                    should_persist = True
                elif ev_kind == "prewarm_done":
                    ok = bool(ev.get("ok"))
                    cur["prewarm_status"] = "warmed" if ok else "skipped"
                    cur.setdefault("events", []).append(
                        {
                            "ts": ev_ts,
                            "kind": "prewarm_done",
                            "msg": f"KV prefix {'warmed' if ok else 'skipped'} ({ev.get('wall_s', 0)}s)",
                        }
                    )
                    should_persist = True
                elif ev_kind == "agent_stage":
                    aid = str(ev.get("agent_id") or "")
                    live_map = cur.setdefault("agents_live", {})
                    prev = live_map.get(aid) or {}
                    prev_stage = prev.get("stage")
                    new_stage = str(ev.get("stage") or prev_stage or "running")
                    live_map[aid] = {
                        "agent_id": aid,
                        "ticket": str(ev.get("ticket") or prev.get("ticket") or ""),
                        "title": str(ev.get("title") or prev.get("title") or aid),
                        "stage": new_stage,
                        "reasoning_chars": int(ev.get("reasoning_chars") or 0),
                        "content_chars": int(ev.get("content_chars") or 0),
                        "snippet": str(ev.get("snippet") or prev.get("snippet") or ""),
                        "updated_at": ev_ts,
                    }
                    if new_stage != prev_stage:
                        cur.setdefault("events", []).append(
                            {
                                "ts": ev_ts,
                                "kind": "agent_stage",
                                "agent_id": aid,
                                "stage": new_stage,
                                "msg": f"{aid} ({live_map[aid]['ticket']}) → {new_stage}",
                            }
                        )
                        should_persist = True
                    elif (time.monotonic() - last_persist[0]) >= 1.5:
                        should_persist = True
                if len(cur.get("events", [])) > 80:
                    cur["events"] = cur["events"][-80:]
                if should_persist:
                    last_persist[0] = time.monotonic()
                    snap_to_write = dict(cur)
            if snap_to_write is not None:
                (self.runs_dir / f"{run_id}.json").write_text(
                    json.dumps(snap_to_write, indent=2) + "\n", encoding="utf-8"
                )

        def _worker() -> None:
            try:
                resolved_tasks = list(tasks)
                if not resolved_tasks and clean_goal:
                    ctx_brief = build_agent_context_brief(
                        repo, continuity=self.continuity, runs_dir=self.runs_dir
                    )
                    resolved_tasks = decompose_goal_to_tasks(
                        repo=repo,
                        goal=clean_goal,
                        base_url=backend_url,
                        model=model,
                        api_key=api_key or "outrcore-swarm",
                        continuity_brief=ctx_brief,
                        max_agents=1 if run_mode == "tournament" else 3,
                    )
                    with self._lock:
                        cur = self._runs.get(run_id)
                        if cur:
                            cur["tasks"] = resolved_tasks
                            cur["agents_count"] = 3 if run_mode == "tournament" else len(resolved_tasks)
                            plan_specs = (
                                [
                                    {"agent_id": "cand-a", "id": "T-001", "title": f"{resolved_tasks[0].get('title', 'T1')} [minimal-patch]"},
                                    {"agent_id": "cand-b", "id": "T-001", "title": f"{resolved_tasks[0].get('title', 'T1')} [defensive-edge-cases]"},
                                    {"agent_id": "cand-c", "id": "T-001", "title": f"{resolved_tasks[0].get('title', 'T1')} [clean-modular]"},
                                ]
                                if run_mode == "tournament"
                                else resolved_tasks
                            )
                            cur["agents_live"] = {
                                str(t.get("agent_id") or f"coder-{chr(ord('a') + idx)}"): {
                                    "agent_id": str(t.get("agent_id") or f"coder-{chr(ord('a') + idx)}"),
                                    "ticket": str(t.get("id") or f"T-{idx+1:03d}"),
                                    "title": str(t.get("title") or f"T-{idx+1:03d}"),
                                    "stage": "queued",
                                    "reasoning_chars": 0,
                                    "content_chars": 0,
                                    "snippet": "",
                                    "updated_at": time.time(),
                                }
                                for idx, t in enumerate(plan_specs)
                            }
                            cur.setdefault("events", []).append(
                                {"ts": time.time(), "kind": "plan_ready", "msg": f"Planned {len(resolved_tasks)} ticket(s) from goal"}
                            )

                if run_mode == "tournament":
                    rec = run_tournament(
                        repo=repo,
                        ticket=resolved_tasks[0],
                        base_url=backend_url,
                        model=model,
                        judge_base_url=judge_base_url or backend_url,
                        judge_model=judge_model or model,
                        api_key=api_key or "outrcore-swarm",
                        spend_cap_usd=float(spend_cap_usd),
                        max_tokens=int(max_tokens),
                        keep_worktrees=True,
                        on_progress=_on_progress,
                        runner_fn=self.runner_fn,
                    )
                else:
                    call_kwargs: dict[str, Any] = {
                        "repo": repo,
                        "tasks": resolved_tasks,
                        "base_url": backend_url,
                        "model": model,
                        "api_key": api_key or "outrcore-swarm",
                        "spend_cap_usd": float(spend_cap_usd),
                        "max_tokens": int(max_tokens),
                        "keep_worktrees": True,
                        "on_progress": _on_progress,
                    }
                    sig = inspect.signature(self.runner_fn)
                    has_varkw = any(p.kind == inspect.Parameter.VAR_KEYWORD for p in sig.parameters.values())
                    extras = {
                        "max_repair_turns": int(max_repair_turns),
                        "auto_review": bool(auto_review),
                        "continuity_store": self.continuity,
                        "runs_dir": self.runs_dir,
                    }
                    for k, v in extras.items():
                        if has_varkw or k in sig.parameters:
                            call_kwargs[k] = v
                    rec = self.runner_fn(**call_kwargs)
                with self._lock:
                    cur = self._runs.get(run_id, initial)
                    cur_events = list(cur.get("events") or [])
                    cur_prewarm = cur.get("prewarm_status", "done")
                    cur_live = dict(cur.get("agents_live") or {})
                    cur_tasks = list(cur.get("tasks") or resolved_tasks)
                merged = dict(initial)
                merged.update(rec)
                merged["run_id"] = run_id
                merged["mode"] = run_mode
                merged["tasks"] = cur_tasks
                merged["status"] = "completed"
                merged["finished_at"] = time.time()
                merged["prewarm_status"] = cur_prewarm
                if "collisions" not in merged:
                    merged["collisions"] = detect_file_collisions(merged.get("results") or [])
                for r in merged.get("results") or []:
                    aid = str(r.get("agent_id") or "")
                    prev_l = cur_live.get(aid) or {}
                    cur_live[aid] = {
                        "agent_id": aid,
                        "ticket": str(r.get("ticket") or prev_l.get("ticket") or ""),
                        "title": str(r.get("title") or prev_l.get("title") or aid),
                        "stage": str(r.get("stage") or ("passed" if r.get("tests_ok") else "failed")),
                        "reasoning_chars": len(r.get("reasoning_preview") or "") or prev_l.get("reasoning_chars", 0),
                        "content_chars": len(r.get("output_preview") or "") or prev_l.get("content_chars", 0),
                        "snippet": str(r.get("error") or r.get("output_preview") or prev_l.get("snippet") or "")[-140:],
                        "updated_at": merged["finished_at"],
                    }
                merged["agents_live"] = cur_live
                cur_events.append(
                    {
                        "ts": merged["finished_at"],
                        "kind": "run_done",
                        "msg": f"Run completed: {merged.get('passed_count', 0)}/{merged.get('agents_count', 0)} passed",
                    }
                )
                merged["events"] = cur_events[-80:]
                self._save_run(merged)
                try:
                    self.continuity.record_action(
                        project_key=str(repo),
                        role="orchestrator",
                        action=f"run:{run_mode}",
                        status="completed",
                        detail=f"{merged.get('passed_count', 0)}/{merged.get('agents_count', 0)} passed",
                        run_id=run_id,
                    )
                except Exception:
                    pass
            except Exception as exc:
                with self._lock:
                    cur = self._runs.get(run_id, initial)
                    cur_events = list(cur.get("events") or [])
                    cur_prewarm = cur.get("prewarm_status", "error")
                    cur_live = dict(cur.get("agents_live") or {})
                failed = dict(initial)
                failed["status"] = "error"
                failed["error"] = f"{type(exc).__name__}: {exc}"
                failed["finished_at"] = time.time()
                failed["prewarm_status"] = cur_prewarm
                failed["agents_live"] = cur_live
                cur_events.append({"ts": failed["finished_at"], "kind": "run_error", "msg": failed["error"]})
                failed["events"] = cur_events[-80:]
                self._save_run(failed)
                try:
                    self.continuity.record_action(
                        project_key=str(repo),
                        role="orchestrator",
                        action=f"run:{run_mode}",
                        status="failed",
                        detail=failed["error"][:400],
                        run_id=run_id,
                    )
                except Exception:
                    pass

        t = threading.Thread(target=_worker, daemon=True)
        t.start()
        return initial

    def apply_agent_diff(self, run_id: str, agent_id: str, *, force: bool = False) -> dict[str, Any]:
        rec = self.get_run(run_id)
        if not rec:
            raise KeyError(f"Unknown run {run_id}")
        repo = Path(rec["repo"])
        target = next((r for r in rec.get("results", []) if r.get("agent_id") == agent_id), None)
        if not target:
            raise KeyError(f"Unknown agent {agent_id} in run {run_id}")
        diff_txt = str(target.get("diff") or "")
        if not diff_txt.strip():
            raise ValueError(f"Agent {agent_id} produced an empty diff")

        base_hashes = target.get("base_file_hashes") or {}
        if base_hashes and not force:
            guard = self.continuity.verify_context_guard(repo, base_hashes)
            if not guard["ok"]:
                raise ContextConflictError(
                    f"Context guard conflict on {len(guard['conflicts'])} file(s): "
                    + ", ".join(c["path"] for c in guard["conflicts"]),
                    conflicts=guard["conflicts"],
                    detail=guard,
                )

        with tempfile.NamedTemporaryFile("w", suffix=".patch", delete=False, encoding="utf-8") as tf:
            tf.write(diff_txt)
            patch_path = tf.name
        try:
            chk = subprocess.run(
                ["git", "apply", "--check", patch_path],
                cwd=str(repo),
                capture_output=True,
                text=True,
            )
            if chk.returncode != 0:
                raise RuntimeError(f"git apply --check failed: {(chk.stderr or chk.stdout).strip()}")
            ap = subprocess.run(
                ["git", "apply", patch_path],
                cwd=str(repo),
                capture_output=True,
                text=True,
            )
            if ap.returncode != 0:
                raise RuntimeError(f"git apply failed: {(ap.stderr or ap.stdout).strip()}")
        finally:
            try:
                os.unlink(patch_path)
            except OSError:
                pass

        applied = list(rec.get("applied_agents") or [])
        if agent_id not in applied:
            applied.append(agent_id)
        rec["applied_agents"] = applied
        self._save_run(rec)

        written = list(target.get("written_files") or extract_diff_paths(diff_txt))
        post_hashes = compute_file_hashes(repo, written)
        try:
            self.continuity.record_action(
                project_key=str(repo),
                role=agent_id,
                action="apply_diff",
                status="completed",
                detail=f"Applied {len(written)} file(s) from {run_id}" + (" (forced)" if force else ""),
                run_id=run_id,
                evidence=[{"kind": "file", "path": p} for p in written],
                repo=repo,
            )
            self.continuity.publish_checkpoint(
                project_key=str(repo),
                role=agent_id,
                summary=f"Applied {agent_id} diff from {run_id}: {', '.join(written)}",
                next_step="Run integration tests or hot-reload preview",
                actor="open-swarm",
                file_hashes=post_hashes,
            )
        except Exception:
            pass
        return {"ok": True, "run_id": run_id, "agent_id": agent_id, "applied_agents": applied, "files": written}

    def review_agent(
        self,
        run_id: str,
        agent_id: str,
        *,
        reviewer_model: str | None = None,
        reviewer_base_url: str | None = None,
        api_key: str | None = None,
        allow_same_vendor_fresh: bool = True,
    ) -> dict[str, Any]:
        rec = self.get_run(run_id)
        if not rec:
            raise KeyError(f"Unknown run {run_id}")
        target = next((r for r in rec.get("results", []) if r.get("agent_id") == agent_id), None)
        if not target:
            raise KeyError(f"Unknown agent {agent_id} in run {run_id}")

        tasks_list = rec.get("tasks") or []
        matching_task = next(
            (t for t in tasks_list if t.get("agent_id") == agent_id or t.get("title") == target.get("title")),
            tasks_list[0] if tasks_list else {},
        )
        verdict = review_agent_diff(
            ticket_title=str(target.get("title") or matching_task.get("title") or target.get("ticket") or agent_id),
            ticket_spec=str(matching_task.get("spec") or ""),
            diff=str(target.get("diff") or ""),
            test_output=str(target.get("test_output") or ""),
            author_model=str(target.get("model") or rec.get("model") or ""),
            reviewer_base_url=reviewer_base_url or str(rec.get("backend_url") or ""),
            reviewer_model=reviewer_model or str(rec.get("model") or ""),
            api_key=api_key or "outrcore-swarm",
            allow_same_vendor_fresh=allow_same_vendor_fresh,
        )
        target["review"] = verdict
        self._save_run(rec)
        try:
            self.continuity.record_action(
                project_key=str(rec.get("repo") or ""),
                role="reviewer",
                action=f"review:{agent_id}",
                status="completed",
                detail=f"verdict={verdict.get('verdict')} findings={len(verdict.get('findings') or [])}",
                run_id=run_id,
            )
        except Exception:
            pass
        return verdict

    def apply_all_passed(self, run_id: str, *, force: bool = False) -> dict[str, Any]:
        rec = self.get_run(run_id)
        if not rec:
            raise KeyError(f"Unknown run {run_id}")
        results = rec.get("results") or []
        collisions = detect_file_collisions(results)
        colliding_agents: set[str] = set()
        for aids in collisions.values():
            colliding_agents.update(aids)
        already = set(rec.get("applied_agents") or [])
        applied: list[str] = []
        skipped: list[dict[str, str]] = []
        for r in results:
            aid = str(r.get("agent_id") or "")
            if aid in already:
                continue
            if not r.get("tests_ok") or not str(r.get("diff") or "").strip():
                skipped.append({"agent_id": aid, "reason": "not passing or empty diff"})
                continue
            if aid in colliding_agents and not force:
                skipped.append({"agent_id": aid, "reason": "file write collision detected"})
                continue
            try:
                self.apply_agent_diff(run_id, aid, force=force)
                applied.append(aid)
            except Exception as exc:
                skipped.append({"agent_id": aid, "reason": str(exc)})
        return {"ok": True, "run_id": run_id, "applied": applied, "skipped": skipped, "collisions": collisions}

    def cleanup_worktrees(self, run_id: str) -> dict[str, Any]:
        rec = self.get_run(run_id)
        if not rec:
            raise KeyError(f"Unknown run {run_id}")
        repo = Path(rec["repo"])
        removed: list[str] = []
        for r in rec.get("results", []):
            wt = Path(str(r.get("worktree") or ""))
            if wt.exists():
                subprocess.run(
                    ["git", "worktree", "remove", "--force", str(wt)],
                    cwd=str(repo),
                    capture_output=True,
                    check=False,
                )
                removed.append(str(wt))
        subprocess.run(["git", "worktree", "prune"], cwd=str(repo), capture_output=True, check=False)
        rec["worktrees_cleaned"] = True
        self._save_run(rec)
        return {"ok": True, "removed": removed}


def _compute_decode_tok_s(comp_tokens: int, t0: float, first_tok_t: float | None, now_t: float) -> tuple[float, float]:
    """Return accurate (decode_tok_s, overall_tok_s) without single-chunk inflation."""
    overall_s = max(0.01, now_t - t0)
    overall_tok_s = round(comp_tokens / overall_s, 1)
    if first_tok_t is None or comp_tokens <= 1:
        return overall_tok_s, overall_tok_s
    decode_s = max(0.01, now_t - first_tok_t)
    return round((comp_tokens - 1) / decode_s, 1), overall_tok_s


def make_handler(state: DesktopAppState, html_path: Path | None = None):
    ui_file = html_path or (HERE / "desktop_ui.html")

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, format: str, *args: Any) -> None:  # noqa: A002
            pass

        def _send_json(self, code: int, payload: Any) -> None:
            raw = json.dumps(payload).encode("utf-8")
            self.send_response(code)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Content-Length", str(len(raw)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(raw)

        def _read_json(self) -> dict[str, Any]:
            n = int(self.headers.get("Content-Length", "0") or "0")
            raw = self.rfile.read(n) if n > 0 else b"{}"
            return json.loads(raw.decode("utf-8") or "{}")

        def do_GET(self) -> None:  # noqa: N802
            path = self.path.split("?", 1)[0]
            if path == "/" or path == "/index.html":
                if not ui_file.is_file():
                    self._send_json(404, {"error": "desktop_ui.html missing"})
                    return
                body = ui_file.read_bytes()
                self.send_response(200)
                self.send_header("Content-Type", "text/html; charset=utf-8")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)
                return
            if path == "/health":
                self._send_json(200, {"status": "ok", "app": "open-swarm-desktop"})
                return
            if path == "/api/config":
                self._send_json(
                    200,
                    {
                        "presets": load_presets(),
                        "lanes": LANES,
                        "default_repo": _default_repo_path(),
                        "self_build_repo": str(state.self_build_dir),
                        "demo_repo": str(state.runs_dir.parent / "demo_repo"),
                        "preview": state.preview.status(),
                    },
                )
                return
            if path == "/api/fleet":
                try:
                    self._send_json(200, scan_fleet(open_swarm_runs_dir=state.runs_dir))
                except Exception as exc:
                    self._send_json(500, {"error": str(exc)})
                return
            if path == "/api/continuity/status":
                qs = parse_qs(urlsplit(self.path).query)
                repo_key = str((qs.get("repo") or [_default_repo_path()])[0])
                try:
                    resolved = str(Path(repo_key).expanduser().resolve())
                except Exception:
                    resolved = repo_key
                self._send_json(200, state.continuity.project_status(resolved))
                return
            if path == "/api/continuity/messages":
                qs = parse_qs(urlsplit(self.path).query)
                repo_key = str((qs.get("repo") or [_default_repo_path()])[0])
                try:
                    resolved = str(Path(repo_key).expanduser().resolve())
                except Exception:
                    resolved = repo_key
                to_role = (qs.get("role") or [None])[0]
                after_id = int((qs.get("after_id") or ["0"])[0] or 0)
                self._send_json(
                    200,
                    {
                        "project_key": resolved,
                        "messages": state.continuity.read_messages(resolved, to_role=to_role, after_id=after_id),
                    },
                )
                return
            if path == "/api/preview/status":
                self._send_json(200, state.preview.status())
                return
            if path == "/api/chat/telemetry":
                self._send_json(200, state.get_chat_telemetry())
                return
            if path == "/api/pick-folder":
                try:
                    proc = subprocess.run(
                        [
                            "osascript",
                            "-e",
                            'POSIX path of (choose folder with prompt "Select Git Repository for Open Swarm")',
                        ],
                        capture_output=True,
                        text=True,
                        timeout=60,
                    )
                    if proc.returncode != 0:
                        self._send_json(200, {"cancelled": True})
                        return
                    chosen = proc.stdout.strip().rstrip("/") or "/"
                    self._send_json(200, {"cancelled": False, "path": chosen})
                except Exception as exc:
                    self._send_json(500, {"error": str(exc)})
                return
            if path == "/api/runs":
                self._send_json(200, {"runs": state.list_runs()})
                return
            if path.startswith("/api/runs/"):
                run_id = path[len("/api/runs/") :].strip("/")
                rec = state.get_run(run_id)
                if not rec:
                    self._send_json(404, {"error": f"run {run_id} not found"})
                    return
                self._send_json(200, rec)
                return
            self._send_json(404, {"error": f"unknown route {path}"})

        def do_POST(self) -> None:  # noqa: N802
            path = self.path.split("?", 1)[0]
            try:
                body = self._read_json()
            except Exception as exc:
                self._send_json(400, {"error": f"invalid JSON: {exc}"})
                return

            if path == "/api/probe":
                url = str(body.get("backend_url") or PRESETS[0]["backend_url"]).strip()
                key = str(body.get("api_key") or "").strip()
                self._send_json(200, probe_backend(url, key))
                return

            if path == "/api/self-build-repo":
                try:
                    p = create_or_sync_self_build_repo(state.self_build_dir, reset=bool(body.get("reset", False)))
                    self._send_json(200, {"ok": True, "path": p})
                except Exception as exc:
                    self._send_json(500, {"error": str(exc)})
                return

            if path == "/api/preview/start":
                try:
                    st = state.preview.start(repo_path=body.get("repo"))
                    self._send_json(200, st)
                except Exception as exc:
                    self._send_json(500, {"error": str(exc)})
                return

            if path == "/api/preview/restart":
                try:
                    st = state.preview.restart(repo_path=body.get("repo"))
                    self._send_json(200, st)
                except Exception as exc:
                    self._send_json(500, {"error": str(exc)})
                return

            if path == "/api/preview/stop":
                self._send_json(200, state.preview.stop())
                return

            if path == "/api/demo-repo":
                try:
                    p = create_sample_sandbox_repo()
                    self._send_json(200, {"ok": True, "path": p})
                except Exception as exc:
                    self._send_json(500, {"error": str(exc)})
                return

            if path == "/api/chat":
                url = str(body.get("backend_url") or PRESETS[0]["backend_url"]).strip()
                model = str(body.get("model") or PRESETS[0]["model"]).strip()
                api_key = str(body.get("api_key") or "outrcore-swarm").strip()
                messages = body.get("messages") or []
                max_tokens = int(body.get("max_tokens") or 512)
                try:
                    t0 = time.perf_counter()
                    res = openai_chat(
                        base_url=url,
                        api_key=api_key,
                        model=model,
                        messages=messages,
                        max_tokens=max_tokens,
                        temperature=float(body.get("temperature") or 0.0),
                    )
                    res["client_wall_ms"] = round((time.perf_counter() - t0) * 1000.0, 1)
                    if res.get("tok_s") is None:
                        res["tok_s"] = round(
                            int(res.get("completion_tokens") or 0)
                            / max(float(res.get("gen_s") or res["client_wall_ms"] / 1000.0), 0.01),
                            2,
                        )
                    state.record_chat_telemetry(
                        {
                            "model": model,
                            "prompt_tokens": res.get("prompt_tokens", 0),
                            "cached_tokens": res.get("cached_tokens", 0),
                            "completion_tokens": res.get("completion_tokens", 0),
                            "ttft_ms": res.get("ttft_ms"),
                            "tok_s": res.get("tok_s"),
                            "elapsed_ms": res["client_wall_ms"],
                        }
                    )
                    self._send_json(200, res)
                except Exception as exc:
                    self._send_json(502, {"error": str(exc)})
                return

            if path == "/api/chat/stream":
                url = str(body.get("backend_url") or PRESETS[0]["backend_url"]).strip()
                model = str(body.get("model") or PRESETS[0]["model"]).strip()
                api_key = str(body.get("api_key") or "outrcore-swarm").strip()
                messages = body.get("messages") or []
                max_tokens = int(body.get("max_tokens") or 768)
                target = url.rstrip("/") + "/chat/completions"
                if not target.endswith("/v1/chat/completions") and "/v1/" not in target:
                    target = url.rstrip("/") + "/v1/chat/completions"

                payload = {
                    "model": model,
                    "messages": messages,
                    "max_tokens": max_tokens,
                    "temperature": float(body.get("temperature") or 0.0),
                    "stream": True,
                    "stream_options": {"include_usage": True},
                }
                req = urllib.request.Request(
                    target,
                    data=json.dumps(payload).encode("utf-8"),
                    headers={
                        "Content-Type": "application/json",
                        "Accept": "text/event-stream",
                        "Authorization": f"Bearer {api_key}",
                        "User-Agent": "OutrCore-OpenSwarm/0.1",
                    },
                    method="POST",
                )

                self.send_response(200)
                self.send_header("Content-Type", "text/event-stream; charset=utf-8")
                self.send_header("Cache-Control", "no-cache")
                self.send_header("X-Accel-Buffering", "no")
                self.end_headers()

                def _emit(event_name: str, data_obj: dict[str, Any]) -> None:
                    frame = f"event: {event_name}\ndata: {json.dumps(data_obj)}\n\n".encode("utf-8")
                    self.wfile.write(frame)
                    self.wfile.flush()

                t0 = time.perf_counter()
                first_tok_t: float | None = None
                est_tokens = 0
                deltas_seen = 0
                usage: dict[str, Any] = {}
                try:
                    with urllib.request.urlopen(req, timeout=300.0) as resp:
                        for raw_line in resp:
                            line = raw_line.decode("utf-8", errors="replace").strip()
                            if not line or line.startswith(":") or not line.startswith("data:"):
                                continue
                            chunk_str = line[5:].strip()
                            if chunk_str == "[DONE]":
                                break
                            evt = json.loads(chunk_str)
                            if evt.get("usage"):
                                usage = evt["usage"]
                            choices = evt.get("choices") or []
                            if choices:
                                delta = choices[0].get("delta") or {}
                                rc = delta.get("reasoning_content")
                                if rc:
                                    if first_tok_t is None:
                                        first_tok_t = time.perf_counter()
                                    est_tokens += max(1, len(rc) // 4)
                                    deltas_seen += 1
                                    _emit(
                                        "thinking",
                                        {
                                            "delta": rc,
                                            "ttft_ms": round((first_tok_t - t0) * 1000.0, 1),
                                            "elapsed_ms": round((time.perf_counter() - t0) * 1000.0, 1),
                                        },
                                    )
                                ct = delta.get("content")
                                if ct:
                                    if first_tok_t is None:
                                        first_tok_t = time.perf_counter()
                                    est_tokens += max(1, len(ct) // 4)
                                    deltas_seen += 1
                                    _emit(
                                        "content",
                                        {
                                            "delta": ct,
                                            "ttft_ms": round((first_tok_t - t0) * 1000.0, 1),
                                            "elapsed_ms": round((time.perf_counter() - t0) * 1000.0, 1),
                                        },
                                    )
                            if deltas_seen > 0 and (deltas_seen % 6 == 0 or usage):
                                now = time.perf_counter()
                                out_tok = int(usage.get("completion_tokens") or est_tokens)
                                dec_tps, ov_tps = _compute_decode_tok_s(out_tok, t0, first_tok_t, now)
                                _emit(
                                    "stats",
                                    {
                                        "ttft_ms": round(((first_tok_t or now) - t0) * 1000.0, 1),
                                        "elapsed_ms": round((now - t0) * 1000.0, 1),
                                        "prompt_tokens": int(usage.get("prompt_tokens") or 0),
                                        "cached_tokens": int(
                                            (usage.get("prompt_tokens_details") or {}).get("cached_tokens") or 0
                                        ),
                                        "completion_tokens": out_tok,
                                        "tok_s": dec_tps,
                                        "overall_tok_s": ov_tps,
                                    },
                                )
                    end_t = time.perf_counter()
                    comp_tok = int(usage.get("completion_tokens") or est_tokens)
                    dec_tps, ov_tps = _compute_decode_tok_s(comp_tok, t0, first_tok_t, end_t)
                    final_stats = {
                        "ttft_ms": round(((first_tok_t or end_t) - t0) * 1000.0, 1),
                        "elapsed_ms": round((end_t - t0) * 1000.0, 1),
                        "prompt_tokens": int(usage.get("prompt_tokens") or 0),
                        "cached_tokens": int(
                            (usage.get("prompt_tokens_details") or {}).get("cached_tokens") or 0
                        ),
                        "completion_tokens": comp_tok,
                        "tok_s": dec_tps,
                        "overall_tok_s": ov_tps,
                    }
                    state.record_chat_telemetry({"model": model, **final_stats})
                    _emit("done", final_stats)
                except Exception as exc:
                    try:
                        _emit("error", {"error": str(exc)})
                    except Exception:
                        pass
                return

            if path == "/api/plan":
                try:
                    repo_p = Path(str(body.get("repo") or _default_repo_path())).expanduser().resolve()
                    goal_txt = str(body.get("goal") or "").strip()
                    if not goal_txt:
                        raise ValueError("goal is required")
                    defaults = load_presets()
                    b_url = str(body.get("backend_url") or defaults[0]["backend_url"])
                    mdl = str(body.get("model") or defaults[0]["model"])
                    key = str(body.get("api_key") or "outrcore-swarm")
                    brief = build_agent_context_brief(repo_p, continuity=state.continuity, runs_dir=state.runs_dir)
                    planned = decompose_goal_to_tasks(
                        repo=repo_p,
                        goal=goal_txt,
                        base_url=b_url,
                        model=mdl,
                        api_key=key,
                        continuity_brief=brief,
                        max_agents=int(body.get("max_agents") or 3),
                    )
                    self._send_json(200, {"tasks": planned, "continuity_brief": brief})
                except Exception as exc:
                    self._send_json(400, {"error": str(exc)})
                return

            if path == "/api/runs":
                try:
                    defaults = load_presets()
                    rec = state.start_run(
                        repo_path=str(body.get("repo") or _default_repo_path()),
                        backend_url=str(body.get("backend_url") or defaults[0]["backend_url"]),
                        model=str(body.get("model") or defaults[0]["model"]),
                        api_key=str(body.get("api_key") or "outrcore-swarm"),
                        spend_cap_usd=float(body.get("spend_cap_usd") or 0.50),
                        tasks=list(body.get("tasks") or []),
                        goal=(str(body["goal"]).strip() if body.get("goal") else None),
                        max_tokens=int(body.get("max_tokens") or 4096),
                        mode=str(body.get("mode") or "swarm"),
                        max_repair_turns=int(body.get("max_repair_turns") if body.get("max_repair_turns") is not None else 2),
                        auto_review=bool(body.get("auto_review", True)),
                        judge_model=(str(body["judge_model"]).strip() if body.get("judge_model") else None),
                        judge_base_url=(str(body["judge_base_url"]).strip() if body.get("judge_base_url") else None),
                    )
                    self._send_json(202, rec)
                except ValueError as exc:
                    self._send_json(400, {"error": str(exc)})
                except Exception as exc:
                    self._send_json(500, {"error": str(exc)})
                return

            if path == "/api/continuity/checkpoint":
                repo_key = str(body.get("repo") or _default_repo_path())
                try:
                    resolved = str(Path(repo_key).expanduser().resolve())
                    cp = state.continuity.publish_checkpoint(
                        project_key=resolved,
                        role=str(body.get("role") or "orchestrator"),
                        summary=str(body.get("summary") or "").strip(),
                        next_step=str(body.get("next_step") or "").strip(),
                        actor=str(body.get("actor") or "matt"),
                        expected_revision=(
                            int(body["expected_revision"]) if body.get("expected_revision") is not None else None
                        ),
                    )
                    self._send_json(200, cp)
                except ContextConflictError as exc:
                    self._send_json(409, {"error": str(exc), "conflicts": exc.conflicts})
                except Exception as exc:
                    self._send_json(400, {"error": str(exc)})
                return

            if path == "/api/continuity/messages":
                repo_key = str(body.get("repo") or _default_repo_path())
                try:
                    resolved = str(Path(repo_key).expanduser().resolve())
                    msg = state.continuity.send_message(
                        project_key=resolved,
                        from_role=str(body.get("from_role") or "orchestrator"),
                        to_role=str(body.get("to_role") or "all"),
                        body=str(body.get("body") or "").strip(),
                    )
                    self._send_json(200, msg)
                except Exception as exc:
                    self._send_json(400, {"error": str(exc)})
                return

            if path.startswith("/api/runs/") and path.endswith("/apply-all"):
                run_id = path[len("/api/runs/") : -len("/apply-all")].strip("/")
                try:
                    out = state.apply_all_passed(run_id, force=bool(body.get("force", False)))
                    self._send_json(200, out)
                except KeyError as exc:
                    self._send_json(404, {"error": str(exc)})
                except Exception as exc:
                    self._send_json(400, {"error": str(exc)})
                return

            if path.startswith("/api/runs/") and path.endswith("/apply"):
                run_id = path[len("/api/runs/") : -len("/apply")].strip("/")
                try:
                    out = state.apply_agent_diff(
                        run_id,
                        str(body.get("agent_id") or ""),
                        force=bool(body.get("force", False)),
                    )
                    self._send_json(200, out)
                except KeyError as exc:
                    self._send_json(404, {"error": str(exc)})
                except ContextConflictError as exc:
                    self._send_json(409, {"error": str(exc), "conflicts": exc.conflicts, "detail": exc.detail})
                except Exception as exc:
                    self._send_json(400, {"error": str(exc)})
                return

            if path.startswith("/api/runs/") and path.endswith("/review"):
                run_id = path[len("/api/runs/") : -len("/review")].strip("/")
                try:
                    out = state.review_agent(
                        run_id,
                        str(body.get("agent_id") or ""),
                        reviewer_model=(str(body["reviewer_model"]).strip() if body.get("reviewer_model") else None),
                        reviewer_base_url=(
                            str(body["reviewer_base_url"]).strip() if body.get("reviewer_base_url") else None
                        ),
                        api_key=(str(body["api_key"]).strip() if body.get("api_key") else None),
                        allow_same_vendor_fresh=bool(body.get("allow_same_vendor_fresh", True)),
                    )
                    self._send_json(200, out)
                except KeyError as exc:
                    self._send_json(404, {"error": str(exc)})
                except Exception as exc:
                    self._send_json(400, {"error": str(exc)})
                return

            if path.startswith("/api/runs/") and path.endswith("/cleanup"):
                run_id = path[len("/api/runs/") : -len("/cleanup")].strip("/")
                try:
                    out = state.cleanup_worktrees(run_id)
                    self._send_json(200, out)
                except KeyError as exc:
                    self._send_json(404, {"error": str(exc)})
                except Exception as exc:
                    self._send_json(400, {"error": str(exc)})
                return

            self._send_json(404, {"error": f"unknown route {path}"})

    return Handler


def start_server(
    host: str = "127.0.0.1",
    port: int = 18790,
    state: DesktopAppState | None = None,
) -> ThreadingHTTPServer:
    if host not in ("127.0.0.1", "localhost"):
        raise ValueError(f"Open Swarm Desktop only binds loopback, got {host!r}")
    app_state = state or DesktopAppState()
    srv = ThreadingHTTPServer((host, port), make_handler(app_state))
    return srv


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Open Swarm Desktop local HTTP backend")
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=18790)
    args = ap.parse_args(argv)
    srv = start_server(host=args.host, port=args.port)
    bound_port = srv.server_address[1]
    print(f"open-swarm-desktop listening on http://{args.host}:{bound_port}", flush=True)
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        srv.server_close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
