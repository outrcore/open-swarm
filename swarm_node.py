#!/usr/bin/env python3
"""Open Swarm v0.4.0 — Remote Worker Daemon (`swarm_node.py`).

Stdlib-only HTTPS/HTTP-over-Tailscale remote worker node that pairs via a
one-time 6-digit PIN, issues SHA-256-pinned bearer device tokens, reports live
hardware/tool/inference/telemetry capabilities, unpacks `git bundle` payloads in
isolated temporary directories (with optional incremental bare mirror caching),
executes `swarm_run.run_swarm` synchronously or via async SSE jobs, and returns
metered unified diffs + test verdicts.
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
import re
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
import urllib.parse
import urllib.request
import uuid
import warnings

SWARM_DIR = Path(__file__).resolve().parent
if str(SWARM_DIR) not in sys.path:
    sys.path.insert(0, str(SWARM_DIR))

NODE_VERSION = "0.5.0"
DEFAULT_PORT = 18799
MAX_BUNDLE_BYTES = 32 * 1024 * 1024  # 32 MiB
MAX_PAIR_ATTEMPTS = 5
TAILSCALE_CGNAT_NET = ipaddress.ip_network("100.64.0.0/10")
DEFAULT_STATE_PATH = Path.home() / ".open_swarm" / "node_state.json"
DEFAULT_MIRRORS_DIR = Path.home() / ".open_swarm" / "mirrors"

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


def is_loopback_or_tailscale_host(host: str) -> bool:
    """Return True if `host` is localhost, a loopback IP, or a Tailscale CGNAT IP."""
    raw = str(host or "").strip()
    if not raw:
        return False
    if raw.lower() == "localhost":
        return True
    unwrapped = raw[1:-1] if raw.startswith("[") and raw.endswith("]") else raw
    if "%" in unwrapped:
        unwrapped = unwrapped.split("%", 1)[0]
    try:
        addr = ipaddress.ip_address(unwrapped)
    except ValueError:
        return False
    return bool(addr.is_loopback or is_tailscale_ip(str(addr)))


def validate_bind_host(host: str, allow_lan: bool = False) -> str:
    """Validate that `host` is safe to bind (loopback or Tailscale unless `allow_lan=True`)."""
    raw = str(host or "").strip()
    if not raw:
        raise ValueError("Bind host must not be empty.")

    if allow_lan:
        if not is_loopback_or_tailscale_host(raw):
            warnings.warn(
                f"SECURITY WARNING: Binding cleartext HTTP on non-Tailscale interface {raw!r} without TLS.",
                RuntimeWarning,
                stacklevel=2,
            )
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


def _parse_optional_float(val: str) -> float | None:
    cleaned = str(val or "").strip()
    if not cleaned or cleaned.lower() in {"n/a", "na", "[not supported]", "none"}:
        return None
    try:
        return float(cleaned)
    except ValueError:
        return None


def _parse_optional_int(val: str) -> int | None:
    num = _parse_optional_float(val)
    return int(round(num)) if num is not None else None


def _probe_nvidia_telemetry() -> tuple[float | None, int | None, int | None]:
    smi = shutil.which("nvidia-smi") or (
        "/usr/bin/nvidia-smi" if os.path.isfile("/usr/bin/nvidia-smi") else None
    )
    if not smi:
        return None, None, None
    try:
        proc = subprocess.run(
            [
                smi,
                "--query-gpu=utilization.gpu,memory.used,memory.total",
                "--format=csv,noheader,nounits",
            ],
            capture_output=True,
            text=True,
            timeout=2.0,
            check=False,
        )
        if proc.returncode != 0:
            return None, None, None
        for line in proc.stdout.splitlines():
            raw = line.strip()
            if not raw:
                continue
            parts = [p.strip() for p in raw.split(",")]
            if len(parts) < 3:
                continue
            return (
                _parse_optional_float(parts[0]),
                _parse_optional_int(parts[1]),
                _parse_optional_int(parts[2]),
            )
    except (OSError, subprocess.SubprocessError, ValueError):
        return None, None, None
    return None, None, None


def _probe_mem_avail_gb(system_name: str) -> float:
    if system_name == "Linux":
        try:
            meminfo = Path("/proc/meminfo").read_text(encoding="utf-8", errors="replace")
            fields: dict[str, int] = {}
            for line in meminfo.splitlines():
                if ":" not in line:
                    continue
                key, rest = line.split(":", 1)
                parts = rest.split()
                if parts and parts[0].isdigit():
                    fields[key.strip()] = int(parts[0])
            if "MemAvailable" in fields:
                return round((fields["MemAvailable"] * 1024) / (1024**3), 2)
            fallback_kib = fields.get("MemFree", 0) + fields.get("Buffers", 0) + fields.get("Cached", 0)
            if fallback_kib > 0:
                return round((fallback_kib * 1024) / (1024**3), 2)
        except (OSError, ValueError):
            pass
    elif system_name == "Darwin":
        try:
            proc = subprocess.run(
                ["vm_stat"],
                capture_output=True,
                text=True,
                timeout=2.0,
                check=False,
            )
            if proc.returncode == 0 and proc.stdout:
                page_size = 4096
                m_page = re.search(r"page size of (\d+) bytes", proc.stdout)
                if m_page:
                    page_size = int(m_page.group(1))
                else:
                    try:
                        page_size = int(os.sysconf("SC_PAGE_SIZE"))
                    except (OSError, ValueError, AttributeError):
                        page_size = 4096
                avail_pages = 0
                for line in proc.stdout.splitlines():
                    for prefix in ("Pages free:", "Pages inactive:", "Pages speculative:"):
                        if line.startswith(prefix):
                            digits = re.sub(r"[^0-9]", "", line.split(":", 1)[1])
                            if digits:
                                avail_pages += int(digits)
                if avail_pages > 0:
                    return round((avail_pages * page_size) / (1024**3), 2)
        except (OSError, subprocess.SubprocessError, ValueError):
            pass

    try:
        av_pages = os.sysconf("SC_AVPHYS_PAGES")
        page_size = os.sysconf("SC_PAGE_SIZE")
        if av_pages > 0 and page_size > 0:
            return round((av_pages * page_size) / (1024**3), 2)
    except (OSError, ValueError, AttributeError):
        pass
    return 0.0


def _probe_cpu_load_pct() -> float | None:
    try:
        load1 = os.getloadavg()[0]
        cores = max(1, os.cpu_count() or 1)
        return round(max(0.0, (float(load1) / float(cores)) * 100.0), 1)
    except (OSError, AttributeError, ValueError):
        return None


def probe_live_telemetry() -> dict[str, Any]:
    """Collect live GPU, memory, and CPU load telemetry with short timeouts and OS fallbacks."""
    sys_name = platform.system()
    gpu_util_pct, gpu_mem_used_mb, gpu_mem_total_mb = _probe_nvidia_telemetry()
    mem_avail_gb = _probe_mem_avail_gb(sys_name)
    cpu_load_pct = _probe_cpu_load_pct()
    return {
        "gpu_util_pct": gpu_util_pct,
        "gpu_mem_used_mb": gpu_mem_used_mb,
        "gpu_mem_total_mb": gpu_mem_total_mb,
        "mem_avail_gb": float(mem_avail_gb),
        "cpu_load_pct": cpu_load_pct,
    }


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
    """Collect host hardware, tool tags, GPU/host telemetry, and local inference URL."""
    sys_name = platform.system()
    arch = platform.machine()
    hostname = socket.gethostname()
    cpu_cores = os.cpu_count()
    ram_gb = _probe_ram_gb(sys_name)
    gpu = _probe_nvidia_gpu()
    telemetry = probe_live_telemetry()

    if gpu is not None:
        if telemetry["gpu_util_pct"] is None and gpu.get("utilization_pct") is not None:
            telemetry["gpu_util_pct"] = float(gpu["utilization_pct"])
        if telemetry["gpu_mem_total_mb"] is None and gpu.get("memory_total_mib") is not None:
            telemetry["gpu_mem_total_mb"] = int(gpu["memory_total_mib"])
        if (
            telemetry["gpu_mem_used_mb"] is None
            and gpu.get("memory_total_mib") is not None
            and gpu.get("memory_free_mib") is not None
        ):
            telemetry["gpu_mem_used_mb"] = max(
                0, int(gpu["memory_total_mib"]) - int(gpu["memory_free_mib"])
            )

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
        "version": NODE_VERSION,
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
        "telemetry": telemetry,
        "gpu_util_pct": telemetry["gpu_util_pct"],
        "gpu_mem_used_mb": telemetry["gpu_mem_used_mb"],
        "gpu_mem_total_mb": telemetry["gpu_mem_total_mb"],
        "mem_avail_gb": telemetry["mem_avail_gb"],
        "cpu_load_pct": telemetry["cpu_load_pct"],
    }


def _git_run(cwd: Path, *args: str, check: bool = True) -> subprocess.CompletedProcess:
    return subprocess.run(
        [
            "git",
            "-c",
            "core.hooksPath=/dev/null",
            "-c",
            "core.fsmonitor=false",
            *args,
        ],
        cwd=str(cwd),
        capture_output=True,
        text=True,
        check=check,
    )


def _sanitize_repo_key(repo_key: str) -> str:
    raw = str(repo_key or "").strip()
    if not raw:
        raise ValueError("repo_key must not be empty.")
    cleaned = re.sub(r"[^a-zA-Z0-9._-]+", "_", raw)
    cleaned = re.sub(r"\.{2,}", "_", cleaned).strip("._-")
    if not cleaned:
        raise ValueError("repo_key contains no valid identifier characters.")
    return cleaned[:128]


def _bundle_fetch_refspecs(mirror_dir: Path, bundle_path: Path) -> list[str]:
    heads_proc = _git_run(mirror_dir, "bundle", "list-heads", str(bundle_path), check=False)
    refspecs: list[str] = []
    if heads_proc.returncode == 0:
        has_head = False
        for line in heads_proc.stdout.splitlines():
            parts = line.strip().split()
            if len(parts) < 2:
                continue
            ref = parts[1].strip()
            if ref == "HEAD":
                has_head = True
            elif ref.startswith("refs/heads/"):
                refspecs.append(f"+{ref}:{ref}")
        if has_head:
            refspecs.insert(0, "+HEAD:refs/heads/swarm-latest")
    if not refspecs:
        refspecs = ["+HEAD:refs/heads/swarm-latest"]
    return refspecs


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
        mirrors_dir: str | Path | None = None,
    ) -> None:
        self.host = validate_bind_host(host, allow_lan=allow_lan)
        self.port = int(port)
        self.allow_lan = bool(allow_lan)
        self.inference_url = _normalize_v1_url(inference_url) if inference_url else ""
        self.max_agents = max(1, int(max_agents))
        self.node_name = str(node_name or socket.gethostname())
        self.state = NodeStateStore(state_path=state_path)
        self.mirrors_dir = Path(mirrors_dir) if mirrors_dir else DEFAULT_MIRRORS_DIR

        self._active_lock = threading.Lock()
        self._active_agents = 0
        self._mirror_locks_guard = threading.Lock()
        self._mirror_locks: dict[str, threading.Lock] = {}
        self._jobs_lock = threading.Lock()
        self._jobs: dict[str, dict[str, Any]] = {}
        self._httpd: ThreadingHTTPServer | None = None
        self._thread: threading.Thread | None = None

    @property
    def active_agents(self) -> int:
        with self._active_lock:
            return self._active_agents

    def _get_mirror_lock(self, safe_key: str) -> threading.Lock:
        with self._mirror_locks_guard:
            lock = self._mirror_locks.get(safe_key)
            if lock is None:
                lock = threading.Lock()
                self._mirror_locks[safe_key] = lock
            return lock

    def _get_job(self, job_id: str) -> dict[str, Any] | None:
        with self._jobs_lock:
            return self._jobs.get(job_id)

    def _append_job_event(
        self, job: dict[str, Any], event_type: str, data: dict[str, Any]
    ) -> None:
        with job["cond"]:
            seq = len(job["events"]) + 1
            job["events"].append(
                {
                    "seq": seq,
                    "ts": time.time(),
                    "event": str(event_type),
                    "data": dict(data),
                }
            )
            job["updated_at"] = time.time()
            job["cond"].notify_all()

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

            def _handle_job_stream(self, job_id: str) -> None:
                job = server_ref._get_job(job_id)
                if job is None:
                    self._send_json(404, {"ok": False, "error": f"Unknown job_id: {job_id}"})
                    return

                self.close_connection = True
                self.send_response(200)
                self.send_header("Content-Type", "text/event-stream")
                self.send_header("Cache-Control", "no-cache")
                self.send_header("Connection", "close")
                self.end_headers()

                cursor = 0
                try:
                    while True:
                        with job["cond"]:
                            new_events = list(job["events"][cursor:])
                            cursor += len(new_events)
                            is_done = bool(job["done"])
                            final_snapshot = (
                                {
                                    "job_id": job["job_id"],
                                    "status": job["status"],
                                    "result": job["result"],
                                    "error": job["error"],
                                    "metering": job["metering"],
                                    "fingerprint": server_ref.state.fingerprint,
                                }
                                if is_done
                                else None
                            )
                            if not new_events and not is_done:
                                job["cond"].wait(timeout=5.0)
                                new_events = list(job["events"][cursor:])
                                cursor += len(new_events)
                                is_done = bool(job["done"])
                                if is_done:
                                    final_snapshot = {
                                        "job_id": job["job_id"],
                                        "status": job["status"],
                                        "result": job["result"],
                                        "error": job["error"],
                                        "metering": job["metering"],
                                        "fingerprint": server_ref.state.fingerprint,
                                    }

                        if not new_events and not is_done:
                            self.wfile.write(b": keepalive\n\n")
                            self.wfile.flush()
                            continue

                        for ev in new_events:
                            ev_name = str(ev.get("event") or "message")
                            ev_id = ev.get("seq", "")
                            chunk = (
                                f"id: {ev_id}\n"
                                f"event: {ev_name}\n"
                                f"data: {json.dumps(ev)}\n\n"
                            ).encode("utf-8")
                            self.wfile.write(chunk)
                        self.wfile.flush()

                        if is_done:
                            done_chunk = (
                                f"event: done\ndata: {json.dumps(final_snapshot)}\n\n"
                            ).encode("utf-8")
                            self.wfile.write(done_chunk)
                            self.wfile.flush()
                            break
                except (BrokenPipeError, ConnectionResetError, OSError):
                    return

            def do_GET(self) -> None:
                path = self.path.split("?", 1)[0]
                if path == "/node/health":
                    if not self._require_bearer():
                        return
                    caps = probe_local_capabilities(
                        inference_url=server_ref.inference_url,
                        max_agents=server_ref.max_agents,
                        active_agents=server_ref.active_agents,
                    )
                    payload = {
                        "ok": True,
                        "version": NODE_VERSION,
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
                        "telemetry": caps["telemetry"],
                        "gpu_util_pct": caps["gpu_util_pct"],
                        "gpu_mem_used_mb": caps["gpu_mem_used_mb"],
                        "gpu_mem_total_mb": caps["gpu_mem_total_mb"],
                        "mem_avail_gb": caps["mem_avail_gb"],
                        "cpu_load_pct": caps["cpu_load_pct"],
                        "ts": time.time(),
                    }
                    self._send_json(200, payload)
                    return

                if path.startswith("/node/jobs/"):
                    if not self._require_bearer():
                        return
                    rest = path[len("/node/jobs/") :]
                    if rest.endswith("/stream"):
                        job_id = urllib.parse.unquote(rest[: -len("/stream")].strip("/"))
                        if not job_id:
                            self._send_json(404, {"ok": False, "error": "Not found"})
                            return
                        self._handle_job_stream(job_id)
                        return

                    job_id = urllib.parse.unquote(rest.strip("/"))
                    if not job_id or "/" in job_id:
                        self._send_json(404, {"ok": False, "error": "Not found"})
                        return
                    status, resp_payload = server_ref._handle_job_status(job_id)
                    self._send_json(status, resp_payload)
                    return

                self._send_json(404, {"ok": False, "error": "Not found"})

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

                if path == "/node/jobs":
                    if not self._require_bearer():
                        return
                    body, err_status, err_msg = self._read_json_body()
                    if body is None:
                        self._send_json(err_status, {"ok": False, "error": err_msg})
                        return
                    status, resp_payload = server_ref._handle_submit_job(body)
                    self._send_json(status, resp_payload)
                    return

                if path.startswith("/node/jobs/") and path.endswith("/cancel"):
                    if not self._require_bearer():
                        return
                    raw_id = path[len("/node/jobs/") : -len("/cancel")].strip("/")
                    job_id = urllib.parse.unquote(raw_id)
                    if not job_id or "/" in job_id:
                        self._send_json(404, {"ok": False, "error": "Not found"})
                        return
                    status, resp_payload = server_ref._handle_cancel_job(job_id)
                    self._send_json(status, resp_payload)
                    return

                self._send_json(404, {"ok": False, "error": "Not found"})

        return NodeHandler

    def _validate_exec_payload(
        self, body: dict[str, Any]
    ) -> tuple[dict[str, Any] | None, int, dict[str, Any]]:
        bundle_b64 = body.get("bundle_b64") or body.get("git_bundle_b64")
        if not isinstance(bundle_b64, str) or not bundle_b64.strip():
            return None, 400, {"ok": False, "error": "Missing or invalid bundle_b64."}

        # Pre-check approximate decoded size before allocating large buffers.
        cleaned_b64 = "".join(bundle_b64.split())
        if (len(cleaned_b64) * 3) // 4 > MAX_BUNDLE_BYTES + 4:
            return None, 413, {"ok": False, "error": "Bundle exceeds 32 MiB limit."}

        try:
            bundle_bytes = base64.b64decode(cleaned_b64, validate=True)
        except (binascii.Error, ValueError):
            return None, 400, {"ok": False, "error": "Invalid base64 encoding in bundle_b64."}

        if len(bundle_bytes) > MAX_BUNDLE_BYTES:
            return None, 413, {"ok": False, "error": "Bundle exceeds 32 MiB limit."}

        task = body.get("task")
        if not isinstance(task, dict) or not task.get("spec") or not task.get("acceptance_cmd"):
            return None, 400, {
                "ok": False,
                "error": "Missing valid task object (requires spec and acceptance_cmd).",
            }

        repo_key_raw = str(body.get("repo_key") or "").strip()
        safe_repo_key = ""
        if repo_key_raw:
            try:
                safe_repo_key = _sanitize_repo_key(repo_key_raw)
            except ValueError as exc:
                return None, 400, {"ok": False, "error": str(exc)}
        incremental = bool(body.get("incremental", False))

        base_commit = str(body.get("base_commit") or body.get("head_sha") or "").strip()

        # Scoped coordinator LLM proxy forwarding takes precedence over raw api_key.
        llm_proxy_url = str(body.get("llm_proxy_url") or "").strip()
        llm_proxy_token = str(body.get("llm_proxy_token") or "").strip()

        if llm_proxy_token:
            api_key = llm_proxy_token
        else:
            api_key = str(body.get("api_key") or "outrcore-swarm")

        if llm_proxy_url:
            exec_base_url = _normalize_v1_url(llm_proxy_url)
        else:
            raw_base_url = str(body.get("base_url") or body.get("backend_url") or "").strip()
            if raw_base_url:
                exec_base_url = _normalize_v1_url(raw_base_url)
            else:
                exec_base_url = _detect_inference_url(self.inference_url)
        if not exec_base_url:
            return None, 400, {
                "ok": False,
                "error": "No inference URL configured or detected on node.",
            }

        spend_cap_raw = body.get("spend_cap_usd", 1.0)
        spend_cap_usd = None if spend_cap_raw is None else float(spend_cap_raw)
        max_repair_turns = int(body.get("max_repair_turns", 2))
        auto_review = bool(body.get("auto_review", False))
        max_tokens = int(body.get("max_tokens", 4096))
        exec_model = str(body.get("model") or task.get("model") or "default")

        parsed = {
            "bundle_bytes": bundle_bytes,
            "task": task,
            "base_commit": base_commit,
            "api_key": api_key,
            "spend_cap_usd": spend_cap_usd,
            "max_repair_turns": max_repair_turns,
            "auto_review": auto_review,
            "max_tokens": max_tokens,
            "base_url": exec_base_url,
            "model": exec_model,
            "repo_key": safe_repo_key,
            "incremental": incremental,
        }
        return parsed, 200, {}

    def _unpack_bundle_to_workrepo(
        self,
        *,
        tmp_dir: Path,
        bundle_bytes: bytes,
        base_commit: str,
        repo_key: str,
        incremental: bool,
    ) -> tuple[Path | None, int, dict[str, Any]]:
        bundle_path = tmp_dir / "repo.bundle"
        workrepo = tmp_dir / "workrepo"
        bundle_path.write_bytes(bundle_bytes)

        if not repo_key:
            clone_proc = _git_run(tmp_dir, "clone", str(bundle_path), str(workrepo), check=False)
            if clone_proc.returncode != 0:
                return None, 400, {
                    "ok": False,
                    "error": f"git clone from bundle failed: {(clone_proc.stderr or clone_proc.stdout).strip()}",
                }
            if base_commit:
                co_proc = _git_run(workrepo, "checkout", "--detach", base_commit, check=False)
                if co_proc.returncode != 0:
                    return None, 400, {
                        "ok": False,
                        "error": f"git checkout {base_commit} failed: {(co_proc.stderr or co_proc.stdout).strip()}",
                    }
        else:
            mirror_lock = self._get_mirror_lock(repo_key)
            with mirror_lock:
                self.mirrors_dir.mkdir(parents=True, exist_ok=True)
                mirror_dir = (self.mirrors_dir / f"{repo_key}.git").resolve()
                if mirror_dir.parent != self.mirrors_dir.resolve():
                    return None, 400, {"ok": False, "error": "Invalid mirror path."}

                has_mirror = (mirror_dir / "HEAD").is_file()
                if incremental:
                    if not has_mirror:
                        return None, 409, {
                            "ok": False,
                            "mirror_miss": True,
                            "error": f"Mirror cache miss for repo_key={repo_key!r}; full bundle required.",
                        }
                    refspecs = _bundle_fetch_refspecs(mirror_dir, bundle_path)
                    fetch_proc = _git_run(
                        mirror_dir, "fetch", "--force", str(bundle_path), *refspecs, check=False
                    )
                    if fetch_proc.returncode != 0:
                        detail = (fetch_proc.stderr or fetch_proc.stdout).strip()
                        lowered = detail.lower()
                        is_missing_base = (
                            "does not have" in lowered
                            or "lacks the necessary commits" in lowered
                            or "prerequisite" in lowered
                        )
                        return None, 409 if is_missing_base else 400, {
                            "ok": False,
                            "mirror_miss": is_missing_base,
                            "error": f"git fetch incremental bundle failed: {detail}",
                        }
                else:
                    if not has_mirror:
                        if mirror_dir.exists():
                            shutil.rmtree(mirror_dir, ignore_errors=True)
                        bare_proc = _git_run(
                            self.mirrors_dir,
                            "clone",
                            "--bare",
                            str(bundle_path),
                            str(mirror_dir),
                            check=False,
                        )
                        if bare_proc.returncode != 0:
                            shutil.rmtree(mirror_dir, ignore_errors=True)
                            return None, 400, {
                                "ok": False,
                                "error": f"git bare clone failed: {(bare_proc.stderr or bare_proc.stdout).strip()}",
                            }
                    else:
                        refspecs = _bundle_fetch_refspecs(mirror_dir, bundle_path)
                        fetch_proc = _git_run(
                            mirror_dir,
                            "fetch",
                            "--force",
                            str(bundle_path),
                            *refspecs,
                            check=False,
                        )
                        if fetch_proc.returncode != 0:
                            return None, 400, {
                                "ok": False,
                                "error": f"git mirror fetch failed: {(fetch_proc.stderr or fetch_proc.stdout).strip()}",
                            }

                if base_commit:
                    has_commit = _git_run(
                        mirror_dir, "cat-file", "-e", f"{base_commit}^{{commit}}", check=False
                    )
                    if has_commit.returncode != 0:
                        return None, 409 if incremental else 400, {
                            "ok": False,
                            "mirror_miss": incremental,
                            "error": f"base_commit {base_commit} not found in mirror.",
                        }

                clone_proc = _git_run(
                    tmp_dir, "clone", "--no-checkout", str(mirror_dir), str(workrepo), check=False
                )
                if clone_proc.returncode != 0:
                    return None, 400, {
                        "ok": False,
                        "error": f"git clone from mirror failed: {(clone_proc.stderr or clone_proc.stdout).strip()}",
                    }

            if base_commit:
                co_proc = _git_run(workrepo, "checkout", "--detach", base_commit, check=False)
                if co_proc.returncode != 0:
                    return None, 400, {
                        "ok": False,
                        "error": f"git checkout {base_commit} failed: {(co_proc.stderr or co_proc.stdout).strip()}",
                    }
            else:
                co_proc = _git_run(workrepo, "checkout", "--detach", "swarm-latest", check=False)
                if co_proc.returncode != 0:
                    co_proc = _git_run(workrepo, "checkout", "--detach", "HEAD", check=False)
                    if co_proc.returncode != 0:
                        return None, 400, {
                            "ok": False,
                            "error": f"git checkout from mirror failed: {(co_proc.stderr or co_proc.stdout).strip()}",
                        }

        _git_run(workrepo, "config", "core.hooksPath", "/dev/null", check=False)
        _git_run(workrepo, "config", "core.fsmonitor", "false", check=False)
        _git_run(workrepo, "config", "user.name", "Open Swarm Node", check=False)
        _git_run(workrepo, "config", "user.email", "node@openswarm.local", check=False)
        return workrepo, 200, {}

    def _run_swarm_task(
        self,
        *,
        workrepo: Path,
        params: dict[str, Any],
        on_progress: Any = None,
        cancel_event: threading.Event | None = None,
    ) -> dict[str, Any]:
        import swarm_run  # noqa: E402

        sig_params = inspect.signature(swarm_run.run_swarm).parameters
        run_kwargs: dict[str, Any] = {
            "repo": workrepo,
            "tasks": [params["task"]],
            "base_url": params["base_url"],
            "model": params["model"],
            "api_key": params["api_key"],
            "spend_cap_usd": params["spend_cap_usd"],
            "max_tokens": params["max_tokens"],
            "keep_worktrees": False,
            "max_repair_turns": params["max_repair_turns"],
            "auto_review": params["auto_review"],
        }
        if "route_to_mesh" in sig_params:
            run_kwargs["route_to_mesh"] = False
        if on_progress is not None and "on_progress" in sig_params:
            run_kwargs["on_progress"] = on_progress
        if cancel_event is not None and "cancel_event" in sig_params:
            run_kwargs["cancel_event"] = cancel_event

        run_res = swarm_run.run_swarm(**run_kwargs)
        results = run_res.get("results") or [{}]
        res0 = dict(results[0])
        res0["executed_on"] = self.node_name
        res0["node_id"] = self.state.node_id
        return {
            "result": res0,
            "metering": run_res.get("metering") or {},
        }

    def _handle_exec(self, body: dict[str, Any]) -> tuple[int, dict[str, Any]]:
        params, err_status, err_payload = self._validate_exec_payload(body)
        if params is None:
            return err_status, err_payload

        with self._active_lock:
            if self._active_agents >= self.max_agents:
                return 503, {"ok": False, "error": "Node at max_agents capacity."}
            self._active_agents += 1

        tmp_dir = Path(tempfile.mkdtemp(prefix="swarm_node_exec_"))
        try:
            workrepo, unpack_status, unpack_err = self._unpack_bundle_to_workrepo(
                tmp_dir=tmp_dir,
                bundle_bytes=params["bundle_bytes"],
                base_commit=params["base_commit"],
                repo_key=params["repo_key"],
                incremental=params["incremental"],
            )
            if workrepo is None:
                return unpack_status, unpack_err

            ran = self._run_swarm_task(workrepo=workrepo, params=params)
            return 200, {
                "ok": True,
                "result": ran["result"],
                "metering": ran["metering"],
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

    def _handle_submit_job(self, body: dict[str, Any]) -> tuple[int, dict[str, Any]]:
        params, err_status, err_payload = self._validate_exec_payload(body)
        if params is None:
            return err_status, err_payload

        job_id = f"job_{uuid.uuid4().hex[:16]}"
        job: dict[str, Any] = {
            "job_id": job_id,
            "status": "queued",
            "events": [],
            "result": None,
            "error": None,
            "metering": None,
            "cancel_event": threading.Event(),
            "cond": threading.Condition(),
            "created_at": time.time(),
            "updated_at": time.time(),
            "done": False,
        }
        with self._jobs_lock:
            self._jobs[job_id] = job

        self._append_job_event(job, "queued", {"job_id": job_id, "status": "queued"})

        worker = threading.Thread(
            target=self._run_async_job,
            args=(job, params),
            daemon=True,
            name=f"swarm-job-{job_id}",
        )
        worker.start()
        return 202, {"ok": True, "job_id": job_id, "status": "queued"}

    def _run_async_job(self, job: dict[str, Any], params: dict[str, Any]) -> None:
        job_id = job["job_id"]
        cancel_event: threading.Event = job["cancel_event"]

        if cancel_event.is_set():
            with job["cond"]:
                job["status"] = "cancelled"
                job["done"] = True
                job["cond"].notify_all()
            return

        acquired_slot = False
        while not cancel_event.is_set():
            with self._active_lock:
                if self._active_agents < self.max_agents:
                    self._active_agents += 1
                    acquired_slot = True
                    break
            cancel_event.wait(timeout=0.05)

        if not acquired_slot or cancel_event.is_set():
            if acquired_slot:
                with self._active_lock:
                    self._active_agents = max(0, self._active_agents - 1)
            with job["cond"]:
                if job["status"] != "cancelled":
                    job["status"] = "cancelled"
                    job["done"] = True
                    job["cond"].notify_all()
            return

        with job["cond"]:
            if cancel_event.is_set() or job["status"] == "cancelled":
                job["status"] = "cancelled"
                job["done"] = True
                job["cond"].notify_all()
                with self._active_lock:
                    self._active_agents = max(0, self._active_agents - 1)
                return
            job["status"] = "running"
            job["updated_at"] = time.time()
            job["cond"].notify_all()

        self._append_job_event(job, "running", {"job_id": job_id, "status": "running"})

        tmp_dir = Path(tempfile.mkdtemp(prefix="swarm_node_job_"))
        try:
            workrepo, unpack_status, unpack_err = self._unpack_bundle_to_workrepo(
                tmp_dir=tmp_dir,
                bundle_bytes=params["bundle_bytes"],
                base_commit=params["base_commit"],
                repo_key=params["repo_key"],
                incremental=params["incremental"],
            )
            if workrepo is None:
                err_msg = str(unpack_err.get("error") or f"Unpack failed ({unpack_status})")
                with job["cond"]:
                    if not job["done"]:
                        job["status"] = "failed"
                        job["error"] = err_msg
                        job["done"] = True
                self._append_job_event(
                    job,
                    "failed",
                    {"job_id": job_id, "status": "failed", "error": err_msg},
                )
                return

            if cancel_event.is_set():
                with job["cond"]:
                    job["status"] = "cancelled"
                    job["done"] = True
                    job["cond"].notify_all()
                return

            def _on_progress(ev: dict[str, Any]) -> None:
                if cancel_event.is_set():
                    return
                self._append_job_event(job, "progress", dict(ev))

            ran = self._run_swarm_task(
                workrepo=workrepo,
                params=params,
                on_progress=_on_progress,
                cancel_event=cancel_event,
            )

            with job["cond"]:
                if cancel_event.is_set() or job["status"] == "cancelled":
                    job["status"] = "cancelled"
                    job["done"] = True
                    job["cond"].notify_all()
                    return
                job["status"] = "completed"
                job["result"] = ran["result"]
                job["metering"] = ran["metering"]
                job["done"] = True

            self._append_job_event(
                job,
                "completed",
                {"job_id": job_id, "status": "completed", "result": ran["result"]},
            )
        except Exception as exc:
            err_msg = f"{type(exc).__name__}: {exc}"
            with job["cond"]:
                if cancel_event.is_set() or job["status"] == "cancelled":
                    job["status"] = "cancelled"
                    job["done"] = True
                    job["cond"].notify_all()
                    return
                job["status"] = "failed"
                job["error"] = err_msg
                job["done"] = True
            self._append_job_event(
                job,
                "failed",
                {"job_id": job_id, "status": "failed", "error": err_msg},
            )
        finally:
            with self._active_lock:
                self._active_agents = max(0, self._active_agents - 1)
            shutil.rmtree(tmp_dir, ignore_errors=True)

    def _handle_job_status(self, job_id: str) -> tuple[int, dict[str, Any]]:
        job = self._get_job(job_id)
        if job is None:
            return 404, {"ok": False, "error": f"Unknown job_id: {job_id}"}
        with job["cond"]:
            return 200, {
                "ok": True,
                "job_id": job["job_id"],
                "status": job["status"],
                "events": [dict(ev) for ev in job["events"]],
                "result": job["result"],
                "error": job["error"],
                "metering": job["metering"],
                "fingerprint": self.state.fingerprint,
            }

    def _handle_cancel_job(self, job_id: str) -> tuple[int, dict[str, Any]]:
        job = self._get_job(job_id)
        if job is None:
            return 404, {"ok": False, "error": f"Unknown job_id: {job_id}"}

        should_emit = False
        with job["cond"]:
            job["cancel_event"].set()
            if job["status"] in ("queued", "running"):
                job["status"] = "cancelled"
                job["done"] = True
                should_emit = True
            current_status = job["status"]
            job["cond"].notify_all()

        if should_emit:
            self._append_job_event(
                job, "cancelled", {"job_id": job_id, "status": "cancelled"}
            )

        return 200, {"ok": True, "job_id": job_id, "status": current_status}

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
    ap = argparse.ArgumentParser(description=f"Open Swarm v{NODE_VERSION} Remote Worker Daemon")
    ap.add_argument("--serve", action="store_true", help="Run the worker daemon HTTP server")
    ap.add_argument("--pair", action="store_true", help="Generate a 6-digit pairing code and serve")
    ap.add_argument("--host", default="", help="Bind host (defaults to Tailscale IPv4 or 127.0.0.1)")
    ap.add_argument("--port", type=int, default=DEFAULT_PORT, help=f"Bind port (default: {DEFAULT_PORT})")
    ap.add_argument("--inference-url", default="", help="Local OpenAI-compatible inference URL")
    ap.add_argument("--state-path", default="", help="Path to node_state.json")
    ap.add_argument("--mirrors-dir", default="", help="Path to bare git mirrors cache directory")
    ap.add_argument("--allow-lan", action="store_true", help="Allow binding non-loopback/non-Tailscale interfaces")
    ap.add_argument("--max-agents", type=int, default=4, help="Max concurrent worker executions (default: 4)")
    ap.add_argument("--node-name", default="", help="Display name for this worker node")
    args = ap.parse_args(argv)

    if not args.serve and not args.pair:
        args.serve = True

    host = args.host.strip() if args.host else detect_default_bind_host()
    if args.allow_lan and not is_loopback_or_tailscale_host(host):
        print(
            f"SECURITY WARNING: Binding cleartext HTTP on non-Tailscale interface {host!r} without TLS.",
            file=sys.stderr,
            flush=True,
        )

    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            server = SwarmNodeServer(
                host=host,
                port=args.port,
                state_path=args.state_path or None,
                inference_url=args.inference_url,
                max_agents=args.max_agents,
                allow_lan=args.allow_lan,
                node_name=args.node_name,
                mirrors_dir=args.mirrors_dir or None,
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
