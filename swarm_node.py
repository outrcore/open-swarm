#!/usr/bin/env python3
"""Open Swarm v0.3.0 — Remote Worker Daemon (`swarm_node.py`).

Stdlib-only HTTPS/HTTP-over-Tailscale remote worker node that pairs via a
one-time 6-digit PIN, issues SHA-256-pinned bearer device tokens, reports live
hardware/tool/inference capabilities, unpacks `git bundle` payloads in isolated
temporary directories, executes `swarm_run.run_swarm`, and returns metered
unified diffs + test verdicts.
"""
from __future__ import annotations

import argparse
import base64
import binascii
import hashlib
import hmac
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import inspect
import ipaddress
import json
import os
from pathlib import Path
import platform
import secrets
import shutil
import socket
import subprocess
import sys
import tempfile
import threading
import time
from typing import Any
import urllib.error
import urllib.request
import uuid

SWARM_DIR = Path(__file__).resolve().parent
if str(SWARM_DIR) not in sys.path:
    sys.path.insert(0, str(SWARM_DIR))

DEFAULT_PORT = 18799
MAX_BUNDLE_BYTES = 32 * 1024 * 1024  # 32 MiB
MAX_PAIR_ATTEMPTS = 5
TAILSCALE_CGNAT_NET = ipaddress.ip_network("100.64.0.0/10")
DEFAULT_STATE_PATH = Path.home() / ".open_swarm" / "node_state.json"

TAILSCALE_BIN_CANDIDATES = (
    "/usr/local/bin/tailscale",
    "/opt/homebrew/bin/tailscale",
    "/Applications/Tailscale.app/Contents/MacOS/Tailscale",
)


def is_tailscale_ip(ip_str: str) -> bool:
    """Return True if `ip_str` is an IPv4 address inside Tailscale CGNAT `100.64.0.0/10`."""
    try:
        addr = ipaddress.ip_address(str(ip_str or "").strip())
    except ValueError:
        return False
    return isinstance(addr, ipaddress.IPv4Address) and addr in TAILSCALE_CGNAT_NET


def _find_tailscale_cli() -> str | None:
    found = shutil.which("tailscale")
    if found:
        return found
    for cand in TAILSCALE_BIN_CANDIDATES:
        if os.path.isfile(cand) and os.access(cand, os.X_OK):
            return cand
    return None


def detect_default_bind_host() -> str:
    """Detect the node's Tailscale IPv4 (`100.64.0.0/10`) or fall back to `127.0.0.1`."""
    ts_bin = _find_tailscale_cli()
    if not ts_bin:
        return "127.0.0.1"

    try:
        proc = subprocess.run(
            [ts_bin, "ip", "-4"],
            capture_output=True,
            text=True,
            timeout=3,
            check=False,
        )
        if proc.returncode == 0:
            for line in proc.stdout.splitlines():
                cand = line.strip()
                if is_tailscale_ip(cand):
                    return cand
    except (OSError, subprocess.SubprocessError):
        pass

    try:
        proc = subprocess.run(
            [ts_bin, "status", "--json"],
            capture_output=True,
            text=True,
            timeout=3,
            check=False,
        )
        if proc.returncode == 0 and proc.stdout.strip():
            data = json.loads(proc.stdout)
            self_info = data.get("Self") or {}
            ips = self_info.get("TailscaleIPs") or data.get("TailscaleIPs") or []
            for cand in ips:
                if is_tailscale_ip(str(cand)):
                    return str(cand).strip()
    except (OSError, subprocess.SubprocessError, json.JSONDecodeError, TypeError):
        pass

    return "127.0.0.1"


def validate_bind_host(host: str, allow_lan: bool = False) -> str:
    """Validate that `host` is safe to bind (loopback or Tailscale unless `allow_lan=True`)."""
    raw = str(host or "").strip()
    if not raw:
        raise ValueError("Bind host must not be empty.")

    if allow_lan:
        return raw

    if raw.lower() == "localhost":
        return raw

    unwrapped = raw[1:-1] if raw.startswith("[") and raw.endswith("]") else raw
    if "%" in unwrapped:
        unwrapped = unwrapped.split("%", 1)[0]

    try:
        addr = ipaddress.ip_address(unwrapped)
    except ValueError as exc:
        raise ValueError(
            f"Refusing to bind non-IP hostname {raw!r} without --allow-lan."
        ) from exc

    if addr.is_unspecified:
        raise ValueError(
            f"Refusing to bind wildcard address {raw!r} without --allow-lan."
        )
    if addr.is_loopback or is_tailscale_ip(str(addr)):
        return raw

    raise ValueError(
        f"Refusing to bind non-loopback, non-Tailscale address {raw!r} without --allow-lan."
    )


def _hash_sha256_hex(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


class NodeStateStore:
    """Persistent node identity, pairing PIN state, and SHA-256-pinned client tokens."""

    def __init__(self, state_path: str | Path | None = None) -> None:
        self.path = Path(state_path) if state_path else DEFAULT_STATE_PATH
        self._lock = threading.Lock()
        self._state: dict[str, Any] = {}
        self._load_or_init()

    def _load_or_init(self) -> None:
        with self._lock:
            if self.path.is_file():
                try:
                    loaded = json.loads(self.path.read_text(encoding="utf-8"))
                    if isinstance(loaded, dict):
                        self._state = loaded
                except (OSError, json.JSONDecodeError):
                    self._state = {}

            changed = False
            if not self._state.get("node_id"):
                self._state["node_id"] = uuid.uuid4().hex
                changed = True
            if not self._state.get("secret"):
                self._state["secret"] = secrets.token_hex(32)
                changed = True
            expected_fp = _hash_sha256_hex(str(self._state["secret"]))
            if self._state.get("fingerprint") != expected_fp:
                self._state["fingerprint"] = expected_fp
                changed = True
            if not isinstance(self._state.get("paired_clients"), dict):
                self._state["paired_clients"] = {}
                changed = True

            if changed or not self.path.exists():
                self._save_locked()
            else:
                self._ensure_permissions()

    def _ensure_permissions(self) -> None:
        try:
            os.chmod(self.path, 0o600)
        except OSError:
            pass

    def _save_locked(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        try:
            os.chmod(self.path.parent, 0o700)
        except OSError:
            pass
        fd, tmp_name = tempfile.mkstemp(prefix=".node_state_", dir=str(self.path.parent))
        try:
            os.chmod(tmp_name, 0o600)
            with os.fdopen(fd, "w", encoding="utf-8") as fh:
                json.dump(self._state, fh, indent=2, sort_keys=True)
                fh.write("\n")
            os.replace(tmp_name, self.path)
            self._ensure_permissions()
        except Exception:
            try:
                os.unlink(tmp_name)
            except OSError:
                pass
            raise

    @property
    def node_id(self) -> str:
        with self._lock:
            return str(self._state["node_id"])

    @property
    def fingerprint(self) -> str:
        with self._lock:
            return str(self._state["fingerprint"])

    @property
    def paired_clients(self) -> dict[str, dict[str, Any]]:
        with self._lock:
            return dict(self._state.get("paired_clients") or {})

    def generate_pair_code(self, ttl_s: int = 300) -> str:
        """Generate and store a single-use 6-digit numeric pairing code."""
        code = f"{secrets.randbelow(1_000_000):06d}"
        with self._lock:
            self._state["active_pair"] = {
                "code": code,
                "expires_at": time.time() + max(1, int(ttl_s)),
                "attempts": 0,
            }
            self._save_locked()
        return code

    def verify_and_pair(
        self,
        pair_code: str,
        client_label: str = "client",
        host_info: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Verify a 6-digit pairing code and mint a SHA-256-pinned bearer token."""
        candidate = str(pair_code or "").strip()
        label = str(client_label or "client").strip() or "client"

        with self._lock:
            active = self._state.get("active_pair")
            if not isinstance(active, dict) or not active.get("code"):
                return {"ok": False, "status": 403, "error": "No active pairing code."}

            now = time.time()
            expires_at = float(active.get("expires_at") or 0.0)
            attempts = int(active.get("attempts") or 0)
            expected_code = str(active.get("code") or "")

            if attempts >= MAX_PAIR_ATTEMPTS:
                self._state.pop("active_pair", None)
                self._save_locked()
                return {"ok": False, "status": 403, "error": "Pairing code locked out."}

            if now > expires_at:
                self._state.pop("active_pair", None)
                self._save_locked()
                return {"ok": False, "status": 403, "error": "Pairing code expired."}

            if not hmac.compare_digest(candidate, expected_code):
                attempts += 1
                if attempts >= MAX_PAIR_ATTEMPTS:
                    self._state.pop("active_pair", None)
                    msg = "Too many failed pairing attempts; code invalidated."
                else:
                    active["attempts"] = attempts
                    msg = "Invalid pairing code."
                self._save_locked()
                return {"ok": False, "status": 403, "error": msg}

            # Single-use: consume the active code immediately on success.
            self._state.pop("active_pair", None)

            raw_token = f"osw_dev_{secrets.token_urlsafe(32)}"
            token_hash = _hash_sha256_hex(raw_token)
            paired = self._state.setdefault("paired_clients", {})
            paired[token_hash] = {
                "label": label,
                "paired_at": now,
            }
            self._save_locked()

            return {
                "ok": True,
                "status": 200,
                "node_id": str(self._state["node_id"]),
                "fingerprint": str(self._state["fingerprint"]),
                "device_token": raw_token,
                "host_info": dict(host_info or {}),
            }

    def verify_bearer(self, auth_header: str) -> bool:
        """Verify an HTTP `Authorization: Bearer <token>` header against stored token hashes."""
        header = str(auth_header or "").strip()
        if not header:
            return False
        parts = header.split(None, 1)
        if len(parts) != 2 or parts[0].lower() != "bearer":
            return False
        token = parts[1].strip()
        if not token:
            return False

        candidate_hash = _hash_sha256_hex(token)
        with self._lock:
            stored_hashes = list((self._state.get("paired_clients") or {}).keys())

        matched = False
        for stored_hash in stored_hashes:
            if hmac.compare_digest(candidate_hash, str(stored_hash)):
                matched = True
        return matched


def _probe_ram_gb(system_name: str) -> float | None:
    if system_name == "Darwin":
        try:
            proc = subprocess.run(
                ["sysctl", "-n", "hw.memsize"],
                capture_output=True,
                text=True,
                timeout=2,
                check=False,
            )
            if proc.returncode == 0 and proc.stdout.strip().isdigit():
                return round(int(proc.stdout.strip()) / (1024**3), 1)
        except (OSError, subprocess.SubprocessError, ValueError):
            pass
    elif system_name == "Linux":
        try:
            for line in Path("/proc/meminfo").read_text(encoding="utf-8").splitlines():
                if line.startswith("MemTotal:"):
                    parts = line.split()
                    if len(parts) >= 2 and parts[1].isdigit():
                        kib = int(parts[1])
                        return round((kib * 1024) / (1024**3), 1)
        except (OSError, ValueError):
            pass
    return None


def _probe_nvidia_gpu() -> dict[str, Any] | None:
    smi = shutil.which("nvidia-smi") or (
        "/usr/bin/nvidia-smi" if os.path.isfile("/usr/bin/nvidia-smi") else None
    )
    if not smi:
        return None
    try:
        proc = subprocess.run(
            [
                smi,
                "--query-gpu=name,memory.total,memory.free,utilization.gpu",
                "--format=csv,noheader,nounits",
            ],
            capture_output=True,
            text=True,
            timeout=3,
            check=False,
        )
        if proc.returncode != 0:
            return None
        for line in proc.stdout.splitlines():
            raw = line.strip()
            if not raw:
                continue
            parts = [p.strip() for p in raw.split(",")]
            if len(parts) < 4:
                continue
            return {
                "name": parts[0],
                "memory_total_mib": int(float(parts[1])),
                "memory_free_mib": int(float(parts[2])),
                "utilization_pct": int(float(parts[3])),
            }
    except (OSError, subprocess.SubprocessError, ValueError):
        return None
    return None


def _has_tool(*names_or_paths: str) -> bool:
    for item in names_or_paths:
        if item.startswith("/"):
            if os.path.exists(item):
                return True
        else:
            if shutil.which(item):
                return True
    return False


def _normalize_v1_url(url: str) -> str:
    cleaned = str(url or "").strip().rstrip("/")
    if not cleaned:
        return ""
    if cleaned.endswith("/v1"):
        return cleaned
    return f"{cleaned}/v1"


def _probe_http_ok(url: str, timeout_s: float = 0.3) -> bool:
    req = urllib.request.Request(url, method="GET")
    try:
        with urllib.request.urlopen(req, timeout=timeout_s) as resp:
            return 200 <= getattr(resp, "status", 200) < 400
    except (urllib.error.URLError, OSError, TimeoutError, ValueError):
        return False


def _detect_inference_url(explicit_url: str = "") -> str:
    if explicit_url and explicit_url.strip():
        return _normalize_v1_url(explicit_url)

    probes = (
        ("http://127.0.0.1:8000/health", "http://127.0.0.1:8000/v1"),
        ("http://127.0.0.1:8888/health", "http://127.0.0.1:8888/v1"),
        ("http://127.0.0.1:11434/api/tags", "http://127.0.0.1:11434/v1"),
    )
    for probe_url, base_v1 in probes:
        if _probe_http_ok(probe_url, timeout_s=0.3):
            return _normalize_v1_url(base_v1)
    return ""


def probe_local_capabilities(
    inference_url: str = "",
    max_agents: int = 4,
    active_agents: int = 0,
) -> dict[str, Any]:
    """Collect host hardware, tool tags, GPU telemetry, and local inference URL."""
    sys_name = platform.system()
    arch = platform.machine()
    hostname = socket.gethostname()
    cpu_cores = os.cpu_count()
    ram_gb = _probe_ram_gb(sys_name)
    gpu = _probe_nvidia_gpu()

    os_tag = "macos" if sys_name == "Darwin" else ("linux" if sys_name == "Linux" else sys_name.lower())
    caps: list[str] = ["python", os_tag]

    if _has_tool("swift", "swiftc", "/usr/bin/swiftc"):
        caps.append("swift")
    if _has_tool("xcodebuild", "/usr/bin/xcodebuild", "/Applications/Xcode.app"):
        caps.append("xcode")

    has_nvcc = _has_tool("nvcc", "/usr/local/cuda/bin/nvcc")
    has_cuda = gpu is not None or has_nvcc or _has_tool("nvidia-smi", "/usr/local/cuda")
    if has_cuda:
        caps.append("cuda")
    if has_nvcc:
        caps.append("nvcc")
    if _has_tool("docker", "/usr/local/bin/docker", "/opt/homebrew/bin/docker"):
        caps.append("docker")
    if _has_tool("git", "/usr/bin/git", "/opt/homebrew/bin/git"):
        caps.append("git")
    if gpu is not None or "cuda" in caps:
        caps.append("gpu")

    dedup_caps = list(dict.fromkeys(caps))
    detected_inference = _detect_inference_url(inference_url)

    return {
        "hostname": hostname,
        "os": sys_name,
        "arch": arch,
        "cpu_cores": cpu_cores,
        "ram_gb": ram_gb,
        "gpu": gpu,
        "capabilities": dedup_caps,
        "inference_url": detected_inference,
        "max_agents": int(max_agents),
        "active_agents": int(active_agents),
    }


def _git_run(cwd: Path, *args: str, check: bool = True) -> subprocess.CompletedProcess:
    return subprocess.run(
        ["git", *args],
        cwd=str(cwd),
        capture_output=True,
        text=True,
        check=check,
    )


class SwarmNodeServer:
    """Authenticated HTTP server executing remote `git bundle` tasks via `swarm_run`."""

    def __init__(
        self,
        host: str = "127.0.0.1",
        port: int = DEFAULT_PORT,
        state_path: str | Path | None = None,
        inference_url: str = "",
        max_agents: int = 4,
        allow_lan: bool = False,
        node_name: str = "",
    ) -> None:
        self.host = validate_bind_host(host, allow_lan=allow_lan)
        self.port = int(port)
        self.allow_lan = bool(allow_lan)
        self.inference_url = _normalize_v1_url(inference_url) if inference_url else ""
        self.max_agents = max(1, int(max_agents))
        self.node_name = str(node_name or socket.gethostname())
        self.state = NodeStateStore(state_path=state_path)

        self._active_lock = threading.Lock()
        self._active_agents = 0
        self._httpd: ThreadingHTTPServer | None = None
        self._thread: threading.Thread | None = None

    @property
    def active_agents(self) -> int:
        with self._active_lock:
            return self._active_agents

    def _build_handler(self):
        server_ref = self

        class NodeHandler(BaseHTTPRequestHandler):
            protocol_version = "HTTP/1.1"

            def log_message(self, format: str, *args: Any) -> None:  # noqa: A002
                pass

            def _send_json(self, status: int, payload: dict[str, Any]) -> None:
                body = json.dumps(payload).encode("utf-8")
                self.send_response(status)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(body)))
                self.send_header("Cache-Control", "no-store")
                self.end_headers()
                self.wfile.write(body)

            def _read_json_body(self) -> tuple[dict[str, Any] | None, int, str]:
                try:
                    length = int(self.headers.get("Content-Length") or 0)
                except ValueError:
                    return None, 400, "Invalid Content-Length."
                if length <= 0:
                    return {}, 200, ""
                # Base64 of 32 MiB is ~43 MiB; allow small JSON envelope overhead.
                if length > 48 * 1024 * 1024:
                    return None, 413, "Request payload too large."
                raw = self.rfile.read(length)
                try:
                    parsed = json.loads(raw.decode("utf-8"))
                except (UnicodeDecodeError, json.JSONDecodeError):
                    return None, 400, "Invalid JSON payload."
                if not isinstance(parsed, dict):
                    return None, 400, "JSON body must be an object."
                return parsed, 200, ""

            def _require_bearer(self) -> bool:
                auth = self.headers.get("Authorization") or ""
                if not server_ref.state.verify_bearer(auth):
                    self._send_json(401, {"ok": False, "error": "Unauthorized"})
                    return False
                return True

            def do_GET(self) -> None:
                path = self.path.split("?", 1)[0]
                if path != "/node/health":
                    self._send_json(404, {"ok": False, "error": "Not found"})
                    return
                if not self._require_bearer():
                    return
                caps = probe_local_capabilities(
                    inference_url=server_ref.inference_url,
                    max_agents=server_ref.max_agents,
                    active_agents=server_ref.active_agents,
                )
                payload = {
                    "ok": True,
                    "node_id": server_ref.state.node_id,
                    "name": server_ref.node_name,
                    "fingerprint": server_ref.state.fingerprint,
                    "active_agents": caps["active_agents"],
                    "max_agents": caps["max_agents"],
                    "capabilities": caps["capabilities"],
                    "gpu": caps["gpu"],
                    "os": caps["os"],
                    "arch": caps["arch"],
                    "cpu_cores": caps["cpu_cores"],
                    "ram_gb": caps["ram_gb"],
                    "inference_url": caps["inference_url"],
                    "ts": time.time(),
                }
                self._send_json(200, payload)

            def do_POST(self) -> None:
                path = self.path.split("?", 1)[0]
                if path == "/node/pair":
                    body, err_status, err_msg = self._read_json_body()
                    if body is None:
                        self._send_json(err_status, {"ok": False, "error": err_msg})
                        return
                    pair_code = str(body.get("pair_code") or body.get("code") or "")
                    client_label = str(body.get("client_label") or body.get("label") or "client")
                    host_info = probe_local_capabilities(
                        inference_url=server_ref.inference_url,
                        max_agents=server_ref.max_agents,
                        active_agents=server_ref.active_agents,
                    )
                    host_info["name"] = server_ref.node_name
                    res = server_ref.state.verify_and_pair(
                        pair_code=pair_code,
                        client_label=client_label,
                        host_info=host_info,
                    )
                    status = int(res.pop("status", 200 if res.get("ok") else 403))
                    self._send_json(status, res)
                    return

                if path == "/node/exec":
                    if not self._require_bearer():
                        return
                    body, err_status, err_msg = self._read_json_body()
                    if body is None:
                        self._send_json(err_status, {"ok": False, "error": err_msg})
                        return
                    status, resp_payload = server_ref._handle_exec(body)
                    self._send_json(status, resp_payload)
                    return

                self._send_json(404, {"ok": False, "error": "Not found"})

        return NodeHandler

    def _handle_exec(self, body: dict[str, Any]) -> tuple[int, dict[str, Any]]:
        bundle_b64 = body.get("bundle_b64") or body.get("git_bundle_b64")
        if not isinstance(bundle_b64, str) or not bundle_b64.strip():
            return 400, {"ok": False, "error": "Missing or invalid bundle_b64."}

        # Pre-check approximate decoded size before allocating large buffers.
        cleaned_b64 = "".join(bundle_b64.split())
        if (len(cleaned_b64) * 3) // 4 > MAX_BUNDLE_BYTES + 4:
            return 413, {"ok": False, "error": "Bundle exceeds 32 MiB limit."}

        try:
            bundle_bytes = base64.b64decode(cleaned_b64, validate=True)
        except (binascii.Error, ValueError):
            return 400, {"ok": False, "error": "Invalid base64 encoding in bundle_b64."}

        if len(bundle_bytes) > MAX_BUNDLE_BYTES:
            return 413, {"ok": False, "error": "Bundle exceeds 32 MiB limit."}

        task = body.get("task")
        if not isinstance(task, dict) or not task.get("spec") or not task.get("acceptance_cmd"):
            return 400, {"ok": False, "error": "Missing valid task object (requires spec and acceptance_cmd)."}

        base_commit = str(body.get("base_commit") or body.get("head_sha") or "").strip()
        api_key = str(body.get("api_key") or "outrcore-swarm")
        spend_cap_raw = body.get("spend_cap_usd", 1.0)
        spend_cap_usd = None if spend_cap_raw is None else float(spend_cap_raw)
        max_repair_turns = int(body.get("max_repair_turns", 2))
        auto_review = bool(body.get("auto_review", False))
        max_tokens = int(body.get("max_tokens", 4096))

        exec_base_url = str(body.get("base_url") or body.get("backend_url") or "").strip()
        if exec_base_url:
            exec_base_url = _normalize_v1_url(exec_base_url)
        else:
            exec_base_url = _detect_inference_url(self.inference_url)
        if not exec_base_url:
            return 400, {"ok": False, "error": "No inference URL configured or detected on node."}

        exec_model = str(body.get("model") or task.get("model") or "default")

        with self._active_lock:
            if self._active_agents >= self.max_agents:
                return 503, {"ok": False, "error": "Node at max_agents capacity."}
            self._active_agents += 1

        tmp_dir = Path(tempfile.mkdtemp(prefix="swarm_node_exec_"))
        try:
            bundle_path = tmp_dir / "repo.bundle"
            workrepo = tmp_dir / "workrepo"
            bundle_path.write_bytes(bundle_bytes)

            clone_proc = _git_run(tmp_dir, "clone", str(bundle_path), str(workrepo), check=False)
            if clone_proc.returncode != 0:
                return 400, {
                    "ok": False,
                    "error": f"git clone from bundle failed: {(clone_proc.stderr or clone_proc.stdout).strip()}",
                }

            if base_commit:
                co_proc = _git_run(workrepo, "checkout", "--detach", base_commit, check=False)
                if co_proc.returncode != 0:
                    return 400, {
                        "ok": False,
                        "error": f"git checkout {base_commit} failed: {(co_proc.stderr or co_proc.stdout).strip()}",
                    }

            _git_run(workrepo, "config", "user.name", "Open Swarm Node", check=False)
            _git_run(workrepo, "config", "user.email", "node@openswarm.local", check=False)

            import swarm_run  # noqa: E402

            run_kwargs: dict[str, Any] = {
                "repo": workrepo,
                "tasks": [task],
                "base_url": exec_base_url,
                "model": exec_model,
                "api_key": api_key,
                "spend_cap_usd": spend_cap_usd,
                "max_tokens": max_tokens,
                "keep_worktrees": False,
                "max_repair_turns": max_repair_turns,
                "auto_review": auto_review,
            }
            if "route_to_mesh" in inspect.signature(swarm_run.run_swarm).parameters:
                run_kwargs["route_to_mesh"] = False

            run_res = swarm_run.run_swarm(**run_kwargs)
            results = run_res.get("results") or [{}]
            res0 = dict(results[0])
            res0["executed_on"] = self.node_name
            res0["node_id"] = self.state.node_id

            return 200, {
                "ok": True,
                "result": res0,
                "metering": run_res.get("metering") or {},
                "executed_on": self.node_name,
                "node_id": self.state.node_id,
                "fingerprint": self.state.fingerprint,
            }
        except Exception as exc:
            return 500, {"ok": False, "error": f"{type(exc).__name__}: {exc}"}
        finally:
            with self._active_lock:
                self._active_agents = max(0, self._active_agents - 1)
            shutil.rmtree(tmp_dir, ignore_errors=True)

    def start(self) -> int:
        """Start the daemon HTTP server thread and return the bound TCP port."""
        if self._httpd is not None:
            return int(self._httpd.server_address[1])

        class _ReusableHTTPServer(ThreadingHTTPServer):
            allow_reuse_address = True

        self._httpd = _ReusableHTTPServer((self.host, self.port), self._build_handler())
        bound_port = int(self._httpd.server_address[1])
        self.port = bound_port
        self._thread = threading.Thread(target=self._httpd.serve_forever, daemon=True)
        self._thread.start()
        return bound_port

    def stop(self) -> None:
        """Shut down the HTTP server and wait for the background thread."""
        httpd = self._httpd
        thread = self._thread
        self._httpd = None
        self._thread = None
        if httpd is not None:
            try:
                httpd.shutdown()
            except Exception:
                pass
            try:
                httpd.server_close()
            except Exception:
                pass
        if thread is not None:
            thread.join(timeout=5)

    def __enter__(self) -> "SwarmNodeServer":
        self.start()
        return self

    def __exit__(self, exc_type: Any, exc: Any, tb: Any) -> None:
        self.stop()


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Open Swarm v0.3.0 Remote Worker Daemon")
    ap.add_argument("--serve", action="store_true", help="Run the worker daemon HTTP server")
    ap.add_argument("--pair", action="store_true", help="Generate a 6-digit pairing code and serve")
    ap.add_argument("--host", default="", help="Bind host (defaults to Tailscale IPv4 or 127.0.0.1)")
    ap.add_argument("--port", type=int, default=DEFAULT_PORT, help=f"Bind port (default: {DEFAULT_PORT})")
    ap.add_argument("--inference-url", default="", help="Local OpenAI-compatible inference URL")
    ap.add_argument("--state-path", default="", help="Path to node_state.json")
    ap.add_argument("--allow-lan", action="store_true", help="Allow binding non-loopback/non-Tailscale interfaces")
    ap.add_argument("--max-agents", type=int, default=4, help="Max concurrent worker executions (default: 4)")
    ap.add_argument("--node-name", default="", help="Display name for this worker node")
    args = ap.parse_args(argv)

    if not args.serve and not args.pair:
        args.serve = True

    host = args.host.strip() if args.host else detect_default_bind_host()
    try:
        server = SwarmNodeServer(
            host=host,
            port=args.port,
            state_path=args.state_path or None,
            inference_url=args.inference_url,
            max_agents=args.max_agents,
            allow_lan=args.allow_lan,
            node_name=args.node_name,
        )
    except ValueError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 2

    bound_port = server.start()
    caps = probe_local_capabilities(
        inference_url=server.inference_url,
        max_agents=server.max_agents,
        active_agents=0,
    )
    node_url = f"http://{server.host}:{bound_port}"
    short_fp = server.state.fingerprint[:16]
    caps_str = ", ".join(caps["capabilities"])

    if args.pair:
        code = server.state.generate_pair_code(ttl_s=300)
        print(f"Node URL: {node_url}", flush=True)
        print(f"Pair Code: {code}", flush=True)
        print(f"Fingerprint: {short_fp}", flush=True)
        print(f"Capabilities: {caps_str}", flush=True)
    else:
        print(
            f"Swarm Node listening on {node_url} "
            f"(node_id={server.state.node_id[:12]}, fp={short_fp}, caps=[{caps_str}])",
            flush=True,
        )

    try:
        while True:
            time.sleep(1.0)
    except KeyboardInterrupt:
        pass
    finally:
        server.stop()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
