#!/usr/bin/env python3
"""Open Swarm v0.4.0 — Verified Device Registry, Tailscale Discovery & Capability-Aware Router.

Pure Python 3 standard library module providing:
- Tailscale CLI discovery (`find_tailscale_bin`, `discover_tailscale_peers`).
- Local machine capability & live telemetry probing (`probe_local_device`).
- Fingerprint-pinned persistent remote device registry (`DeviceRegistry`).
- Capability inference and telemetry/load/capability-aware task routing
  (`infer_required_capabilities`, `route_tasks_to_devices`).
- Authenticated remote git-bundle & async SSE job execution
  (`execute_task_on_remote_device`, `submit_async_remote_job`,
  `get_remote_job_status`, `stream_remote_job_events`, `cancel_remote_job`).
"""
from __future__ import annotations

import base64
from concurrent.futures import ThreadPoolExecutor
import json
import os
from pathlib import Path
import platform
import re
import shutil
import socket
import subprocess
import sys
import tempfile
import threading
import time
from typing import Any, Callable
import urllib.error
import urllib.parse
import urllib.request

SWARM_DIR = Path(__file__).resolve().parent
if str(SWARM_DIR) not in sys.path:
    sys.path.insert(0, str(SWARM_DIR))

from continuity_store import compute_file_hashes, extract_diff_paths  # noqa: E402

DEFAULT_NODE_PORT = 18799
DEFAULT_LOCAL_PORT = 18790
DEFAULT_DEVICES_PATH = Path.home() / ".open_swarm" / "devices.json"

_TAILSCALE_CANDIDATES: tuple[str, ...] = (
    "/usr/local/bin/tailscale",
    "/opt/homebrew/bin/tailscale",
    "/Applications/Tailscale.app/Contents/MacOS/Tailscale",
)

_MACOS_KEYWORDS: tuple[str, ...] = (
    "swift",
    "swiftc",
    "xcode",
    "appkit",
    "webkit",
    "osascript",
    "macos",
    ".swift",
)
_SWIFT_KEYWORDS: tuple[str, ...] = ("swiftc", "xcode", ".swift")
_CUDA_KEYWORDS: tuple[str, ...] = (
    "cuda",
    "nvcc",
    "nvidia",
    "torch",
    "triton",
    "k5cuda",
    ".cu",
    "vram",
)


def _format_host_for_url(host: str) -> str:
    clean = str(host or "").strip()
    if ":" in clean and not (clean.startswith("[") and clean.endswith("]")):
        return f"[{clean}]"
    return clean


def _slugify_device_id(host: str, port: int) -> str:
    raw = f"{host}-{port}"
    slug = re.sub(r"[^a-zA-Z0-9._-]+", "-", raw).strip("-.")
    if not slug or slug == "local":
        return f"node-{port}"
    return slug


def _http_json(
    url: str,
    *,
    method: str = "GET",
    payload: dict[str, Any] | None = None,
    token: str | None = None,
    timeout_s: float = 5.0,
) -> dict[str, Any]:
    headers: dict[str, str] = {"Accept": "application/json"}
    body: bytes | None = None
    if payload is not None:
        body = json.dumps(payload).encode("utf-8")
        headers["Content-Type"] = "application/json"
    if token:
        headers["Authorization"] = f"Bearer {token}"

    req = urllib.request.Request(url, data=body, headers=headers, method=method)
    try:
        with urllib.request.urlopen(req, timeout=timeout_s) as resp:
            raw = resp.read().decode("utf-8", errors="replace")
    except urllib.error.HTTPError as exc:
        detail = ""
        try:
            detail = exc.read().decode("utf-8", errors="replace").strip()
        except Exception:
            detail = ""
        msg = f"HTTP {exc.code}: {exc.reason}"
        if detail:
            msg = f"{msg} ({detail[:300]})"
        raise RuntimeError(msg) from exc
    except urllib.error.URLError as exc:
        raise RuntimeError(f"Connection error: {exc.reason}") from exc

    try:
        parsed = json.loads(raw) if raw.strip() else {}
    except json.JSONDecodeError as exc:
        raise RuntimeError(f"Invalid JSON response from {url}: {exc}") from exc
    if not isinstance(parsed, dict):
        raise RuntimeError(f"Expected JSON object from {url}")
    return parsed


def _pick_tailscale_ip(ips: Any) -> str:
    if not isinstance(ips, list):
        return ""
    cleaned = [str(ip).strip() for ip in ips if isinstance(ip, str) and ip.strip()]
    for ip in cleaned:
        if ip.startswith("100.") and ":" not in ip:
            return ip
    for ip in cleaned:
        if ":" not in ip:
            return ip
    return cleaned[0] if cleaned else ""


def _parse_tailscale_node(
    entry: dict[str, Any],
    *,
    fallback_id: str = "",
    is_self: bool = False,
    default_online: bool = False,
) -> dict[str, Any]:
    dns_name = str(entry.get("DNSName") or "").strip().rstrip(".")
    return {
        "tailscale_id": str(entry.get("ID") or fallback_id or ""),
        "hostname": str(entry.get("HostName") or ""),
        "dns_name": dns_name,
        "ip": _pick_tailscale_ip(entry.get("TailscaleIPs")),
        "os": str(entry.get("OS") or ""),
        "online": bool(entry.get("Online", default_online)),
        "is_self": is_self,
        "suggested_port": DEFAULT_NODE_PORT,
    }


def find_tailscale_bin() -> str | None:
    """Locate the Tailscale CLI binary if installed."""
    which_path = shutil.which("tailscale")
    if which_path:
        return which_path
    for candidate in _TAILSCALE_CANDIDATES:
        if os.path.isfile(candidate) and os.access(candidate, os.X_OK):
            return candidate
    return None


def discover_tailscale_peers(timeout_s: float = 3.0) -> dict[str, Any]:
    """Run ``tailscale status --json`` and return normalized self and peer records."""
    empty = {"available": False, "running": False, "self": None, "peers": []}
    ts_bin = find_tailscale_bin()
    if not ts_bin:
        return empty

    try:
        proc = subprocess.run(
            [ts_bin, "status", "--json"],
            capture_output=True,
            text=True,
            timeout=timeout_s,
            check=False,
        )
    except (FileNotFoundError, subprocess.TimeoutExpired, OSError):
        return empty

    if proc.returncode != 0:
        return empty

    try:
        data = json.loads(proc.stdout or "{}")
    except json.JSONDecodeError:
        return empty
    if not isinstance(data, dict):
        return empty

    backend_state = str(data.get("BackendState") or "").strip().lower()
    running = backend_state == "running"

    self_raw = data.get("Self")
    self_node = (
        _parse_tailscale_node(self_raw, is_self=True, default_online=running)
        if isinstance(self_raw, dict)
        else None
    )

    peers: list[dict[str, Any]] = []
    peers_raw = data.get("Peer")
    if isinstance(peers_raw, dict):
        for key, entry in peers_raw.items():
            if not isinstance(entry, dict):
                continue
            peers.append(
                _parse_tailscale_node(
                    entry,
                    fallback_id=str(key),
                    is_self=False,
                    default_online=False,
                )
            )

    peers.sort(key=lambda p: (0 if p["online"] else 1, p["hostname"].lower(), p["ip"]))
    return {
        "available": True,
        "running": running,
        "self": self_node,
        "peers": peers,
    }


def _detect_ram_gb() -> float | None:
    system = platform.system()
    if system == "Darwin":
        try:
            proc = subprocess.run(
                ["sysctl", "-n", "hw.memsize"],
                capture_output=True,
                text=True,
                timeout=1.5,
                check=False,
            )
            if proc.returncode == 0 and proc.stdout.strip().isdigit():
                return round(int(proc.stdout.strip()) / (1024**3), 1)
        except (OSError, subprocess.TimeoutExpired):
            pass
    elif system == "Linux":
        try:
            meminfo = Path("/proc/meminfo").read_text(encoding="utf-8", errors="replace")
            for line in meminfo.splitlines():
                if line.startswith("MemTotal:"):
                    kb = int(line.split()[1])
                    return round(kb / (1024**2), 1)
        except (OSError, ValueError, IndexError):
            pass
    try:
        pages = os.sysconf("SC_PHYS_PAGES")
        page_size = os.sysconf("SC_PAGE_SIZE")
        if pages > 0 and page_size > 0:
            return round((pages * page_size) / (1024**3), 1)
    except (OSError, ValueError, AttributeError):
        pass
    return None


def _probe_live_telemetry_fallback() -> dict[str, Any]:
    sys_name = platform.system()
    gpu_util_pct: float | None = None
    gpu_mem_used_mb: int | None = None
    gpu_mem_total_mb: int | None = None

    smi = shutil.which("nvidia-smi") or (
        "/usr/bin/nvidia-smi" if os.path.isfile("/usr/bin/nvidia-smi") else None
    )
    if smi:
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
            if proc.returncode == 0:
                for line in proc.stdout.splitlines():
                    raw = line.strip()
                    if not raw:
                        continue
                    parts = [p.strip() for p in raw.split(",")]
                    if len(parts) >= 3:
                        try:
                            gpu_util_pct = float(parts[0])
                        except ValueError:
                            gpu_util_pct = None
                        try:
                            gpu_mem_used_mb = int(round(float(parts[1])))
                        except ValueError:
                            gpu_mem_used_mb = None
                        try:
                            gpu_mem_total_mb = int(round(float(parts[2])))
                        except ValueError:
                            gpu_mem_total_mb = None
                        break
        except (OSError, subprocess.SubprocessError):
            pass

    mem_avail_gb = 0.0
    if sys_name == "Linux":
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
                mem_avail_gb = round((fields["MemAvailable"] * 1024) / (1024**3), 2)
            else:
                kib = fields.get("MemFree", 0) + fields.get("Buffers", 0) + fields.get("Cached", 0)
                if kib > 0:
                    mem_avail_gb = round((kib * 1024) / (1024**3), 2)
        except (OSError, ValueError):
            pass
    elif sys_name == "Darwin":
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
                avail_pages = 0
                for line in proc.stdout.splitlines():
                    for prefix in ("Pages free:", "Pages inactive:", "Pages speculative:"):
                        if line.startswith(prefix):
                            digits = re.sub(r"[^0-9]", "", line.split(":", 1)[1])
                            if digits:
                                avail_pages += int(digits)
                if avail_pages > 0:
                    mem_avail_gb = round((avail_pages * page_size) / (1024**3), 2)
        except (OSError, subprocess.SubprocessError, ValueError):
            pass

    if mem_avail_gb == 0.0:
        try:
            av_pages = os.sysconf("SC_AVPHYS_PAGES")
            page_size = os.sysconf("SC_PAGE_SIZE")
            if av_pages > 0 and page_size > 0:
                mem_avail_gb = round((av_pages * page_size) / (1024**3), 2)
        except (OSError, ValueError, AttributeError):
            pass

    cpu_load_pct: float | None = None
    try:
        load1 = os.getloadavg()[0]
        cores = max(1, os.cpu_count() or 1)
        cpu_load_pct = round(max(0.0, (float(load1) / float(cores)) * 100.0), 1)
    except (OSError, AttributeError, ValueError):
        cpu_load_pct = None

    return {
        "gpu_util_pct": gpu_util_pct,
        "gpu_mem_used_mb": gpu_mem_used_mb,
        "gpu_mem_total_mb": gpu_mem_total_mb,
        "mem_avail_gb": float(mem_avail_gb),
        "cpu_load_pct": cpu_load_pct,
    }


def _fallback_local_capabilities() -> dict[str, Any]:
    system = platform.system()
    arch = platform.machine()
    caps: list[str] = []
    if shutil.which("python3") or sys.executable:
        caps.append("python")
    if shutil.which("git"):
        caps.append("git")
    if system == "Darwin":
        caps.append("macos")
        if shutil.which("swiftc") or shutil.which("swift") or shutil.which("xcodebuild"):
            caps.append("swift")
        if arch == "arm64":
            caps.append("apple_silicon")
    elif system == "Linux":
        caps.append("linux")

    gpu: str | None = None
    if shutil.which("nvcc") or shutil.which("nvidia-smi"):
        caps.append("cuda")
        caps.append("gpu")
        gpu = "NVIDIA GPU"
    elif system == "Darwin" and arch == "arm64":
        gpu = "Apple Silicon GPU"

    telemetry = _probe_live_telemetry_fallback()
    return {
        "version": "0.4.0",
        "capabilities": caps,
        "gpu": gpu,
        "os": system.lower(),
        "arch": arch,
        "cpu_cores": os.cpu_count() or 1,
        "ram_gb": _detect_ram_gb(),
        "inference_url": None,
        "telemetry": telemetry,
        **telemetry,
    }


def probe_local_device() -> dict[str, Any]:
    """Return a standardized device descriptor with live telemetry for the local host."""
    caps_info: dict[str, Any] | None = None
    try:
        from swarm_node import probe_local_capabilities  # type: ignore[import-not-found]

        raw_caps = probe_local_capabilities()
        if isinstance(raw_caps, dict):
            caps_info = raw_caps
    except Exception:
        caps_info = None

    if caps_info is None:
        caps_info = _fallback_local_capabilities()

    telemetry = caps_info.get("telemetry")
    if not isinstance(telemetry, dict):
        telemetry = _probe_live_telemetry_fallback()
    else:
        telemetry = dict(telemetry)

    for key in (
        "gpu_util_pct",
        "gpu_mem_used_mb",
        "gpu_mem_total_mb",
        "mem_avail_gb",
        "cpu_load_pct",
    ):
        if key not in telemetry or (key == "mem_avail_gb" and telemetry.get(key) is None):
            val = caps_info.get(key)
            if val is not None:
                telemetry[key] = val

    if telemetry.get("mem_avail_gb") is None:
        telemetry["mem_avail_gb"] = 0.0

    system = platform.system()
    default_name = "This Mac" if system == "Darwin" else socket.gethostname()
    return {
        "device_id": "local",
        "name": str(caps_info.get("name") or default_name),
        "version": str(caps_info.get("version") or "0.4.0"),
        "is_local": True,
        "online": True,
        "tampered": False,
        "has_token": False,
        "host": "127.0.0.1",
        "port": int(caps_info.get("port") or DEFAULT_LOCAL_PORT),
        "active_agents": int(caps_info.get("active_agents") or 0),
        "max_agents": int(caps_info.get("max_agents") or 4),
        "capabilities": list(caps_info.get("capabilities") or []),
        "gpu": caps_info.get("gpu"),
        "os": str(caps_info.get("os") or system.lower()),
        "arch": str(caps_info.get("arch") or platform.machine()),
        "cpu_cores": caps_info.get("cpu_cores") or (os.cpu_count() or 1),
        "ram_gb": caps_info.get("ram_gb"),
        "inference_url": caps_info.get("inference_url"),
        "fingerprint": caps_info.get("fingerprint"),
        "telemetry": telemetry,
        "gpu_util_pct": telemetry.get("gpu_util_pct"),
        "gpu_mem_used_mb": telemetry.get("gpu_mem_used_mb"),
        "gpu_mem_total_mb": telemetry.get("gpu_mem_total_mb"),
        "mem_avail_gb": float(telemetry.get("mem_avail_gb") or 0.0),
        "cpu_load_pct": telemetry.get("cpu_load_pct"),
        "latency_ms": 0.0,
        "error": None,
    }


class DeviceRegistry:
    """Thread-safe, fingerprint-pinned registry of paired remote swarm nodes."""

    def __init__(self, path: Path | None = None) -> None:
        self.path = Path(path) if path is not None else DEFAULT_DEVICES_PATH
        self._lock = threading.Lock()

    def _load_locked(self) -> dict[str, dict[str, Any]]:
        if not self.path.exists():
            return {}
        try:
            raw = json.loads(self.path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return {}
        if not isinstance(raw, dict):
            return {}
        if "devices" in raw and isinstance(raw["devices"], dict):
            source = raw["devices"]
        else:
            source = raw
        devices: dict[str, dict[str, Any]] = {}
        for dev_id, rec in source.items():
            if isinstance(dev_id, str) and dev_id != "local" and isinstance(rec, dict):
                devices[dev_id] = dict(rec)
        return devices

    def _save_locked(self, devices: dict[str, dict[str, Any]]) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        payload = json.dumps(devices, indent=2, sort_keys=True) + "\n"
        fd, tmp_name = tempfile.mkstemp(
            prefix=".devices.",
            suffix=".tmp",
            dir=str(self.path.parent),
        )
        tmp_path = Path(tmp_name)
        try:
            os.chmod(tmp_path, 0o600)
            with os.fdopen(fd, "w", encoding="utf-8") as fh:
                fh.write(payload)
            os.chmod(tmp_path, 0o600)
            os.replace(tmp_path, self.path)
            os.chmod(self.path, 0o600)
        except Exception:
            try:
                tmp_path.unlink(missing_ok=True)
            except OSError:
                pass
            raise

    @staticmethod
    def _sanitize_device(rec: dict[str, Any]) -> dict[str, Any]:
        cleaned = dict(rec)
        raw_token = cleaned.pop("token", None)
        cleaned.pop("device_token", None)
        cleaned["has_token"] = bool(raw_token)
        cleaned["is_local"] = bool(cleaned.get("is_local", False))
        return cleaned

    def pair_device(
        self,
        *,
        host: str,
        pair_code: str,
        name: str = "",
        port: int = DEFAULT_NODE_PORT,
        client_label: str = "Open Swarm",
    ) -> dict[str, Any]:
        """Pair with a remote ``swarm_node`` via its one-time ``pair_code``."""
        clean_host = str(host or "").strip()
        clean_code = str(pair_code or "").strip()
        if not clean_host:
            raise ValueError("host is required")
        if not clean_code:
            raise ValueError("pair_code is required")

        port_int = int(port)
        url = f"http://{_format_host_for_url(clean_host)}:{port_int}/node/pair"
        resp = _http_json(
            url,
            method="POST",
            payload={"pair_code": clean_code, "client_label": client_label},
            timeout_s=10.0,
        )

        token = str(resp.get("device_token") or "").strip()
        fingerprint = str(resp.get("fingerprint") or "").strip()
        if not token or not fingerprint:
            raise RuntimeError("Pairing response missing required device_token or fingerprint")

        raw_id = str(resp.get("node_id") or resp.get("device_id") or "").strip()
        device_id = raw_id if raw_id and raw_id != "local" else _slugify_device_id(clean_host, port_int)
        dev_name = str(name or "").strip() or str(resp.get("name") or resp.get("hostname") or clean_host)

        rec: dict[str, Any] = {
            "device_id": device_id,
            "name": dev_name,
            "host": clean_host,
            "port": port_int,
            "token": token,
            "fingerprint": fingerprint,
            "capabilities": list(resp.get("capabilities") or []),
            "gpu": resp.get("gpu"),
            "os": str(resp.get("os") or ""),
            "arch": str(resp.get("arch") or ""),
            "cpu_cores": resp.get("cpu_cores"),
            "ram_gb": resp.get("ram_gb"),
            "inference_url": resp.get("inference_url"),
            "paired_at": time.time(),
        }

        with self._lock:
            devices = self._load_locked()
            devices[device_id] = rec
            self._save_locked(devices)

        sanitized = self._sanitize_device(rec)
        sanitized["ok"] = True
        sanitized["online"] = True
        sanitized["tampered"] = False
        return sanitized

    def remove_device(self, device_id: str) -> bool:
        """Delete a paired remote device from the registry (refuses ``'local'``)."""
        if not device_id or device_id == "local":
            return False
        with self._lock:
            devices = self._load_locked()
            if device_id not in devices:
                return False
            del devices[device_id]
            self._save_locked(devices)
            return True

    def get_device_credentials(self, device_id: str) -> dict[str, Any] | None:
        """Return full stored credentials (including bearer token) for internal dispatch."""
        if not device_id or device_id == "local":
            return None
        with self._lock:
            devices = self._load_locked()
            rec = devices.get(device_id)
            return dict(rec) if rec is not None else None

    def _probe_remote(self, rec: dict[str, Any], timeout_s: float) -> dict[str, Any]:
        host = str(rec.get("host") or "")
        port = int(rec.get("port") or DEFAULT_NODE_PORT)
        token = str(rec.get("token") or "")
        expected_fp = str(rec.get("fingerprint") or "")
        url = f"http://{_format_host_for_url(host)}:{port}/node/health"

        out = dict(rec)
        out["is_local"] = False
        t0 = time.monotonic()
        try:
            health = _http_json(url, method="GET", token=token, timeout_s=timeout_s)
        except Exception as exc:
            out["online"] = False
            out["tampered"] = False
            out["error"] = str(exc)
            return self._sanitize_device(out)

        latency_ms = round((time.monotonic() - t0) * 1000.0, 2)
        actual_fp = str(health.get("fingerprint") or "")
        if not expected_fp or actual_fp != expected_fp:
            out["online"] = False
            out["tampered"] = True
            out["latency_ms"] = latency_ms
            out["error"] = "Fingerprint mismatch"
            return self._sanitize_device(out)

        out["online"] = True
        out["tampered"] = False
        out["error"] = None
        out["latency_ms"] = latency_ms
        if health.get("version") is not None:
            out["version"] = str(health["version"])
        if "capabilities" in health and health["capabilities"] is not None:
            out["capabilities"] = list(health["capabilities"])
        if "gpu" in health:
            out["gpu"] = health["gpu"]
        if "active_agents" in health:
            out["active_agents"] = int(health.get("active_agents") or 0)
        else:
            out.setdefault("active_agents", 0)
        if "max_agents" in health:
            out["max_agents"] = int(health.get("max_agents") or 4)
        else:
            out.setdefault("max_agents", 4)
        if "inference_url" in health:
            out["inference_url"] = health["inference_url"]
        for field in ("os", "arch", "cpu_cores", "ram_gb"):
            if health.get(field) is not None:
                out[field] = health[field]

        raw_telemetry = health.get("telemetry") if isinstance(health.get("telemetry"), dict) else {}
        telemetry: dict[str, Any] = {}
        for tfield in (
            "gpu_util_pct",
            "gpu_mem_used_mb",
            "gpu_mem_total_mb",
            "mem_avail_gb",
            "cpu_load_pct",
        ):
            val = health.get(tfield)
            if val is None:
                val = raw_telemetry.get(tfield)
            telemetry[tfield] = val
            out[tfield] = val
        out["telemetry"] = telemetry
        return self._sanitize_device(out)

    def list_devices(self, *, probe: bool = True, timeout_s: float = 2.5) -> list[dict[str, Any]]:
        """Return sanitized local and paired remote device records."""
        local_dev = probe_local_device()
        with self._lock:
            remotes = list(self._load_locked().values())

        if not remotes:
            return [local_dev]

        if not probe:
            offline_remotes = [
                self._sanitize_device(
                    {
                        **rec,
                        "is_local": False,
                        "online": False,
                        "tampered": False,
                        "active_agents": int(rec.get("active_agents") or 0),
                        "max_agents": int(rec.get("max_agents") or 4),
                        "error": None,
                    }
                )
                for rec in remotes
            ]
            return [local_dev, *offline_remotes]

        max_workers = min(8, len(remotes))
        with ThreadPoolExecutor(max_workers=max_workers) as pool:
            probed = list(pool.map(lambda r: self._probe_remote(r, timeout_s), remotes))
        return [local_dev, *probed]


def infer_required_capabilities(task: dict[str, Any], mode: str = "swarm") -> list[str]:
    """Infer required device capabilities from explicit task tags and task text."""
    required: list[str] = []
    seen: set[str] = set()

    def _add(cap: str) -> None:
        norm = cap.strip().lower()
        if norm and norm not in seen:
            seen.add(norm)
            required.append(norm)

    explicit = task.get("required_capabilities")
    if isinstance(explicit, str):
        _add(explicit)
    elif isinstance(explicit, (list, tuple, set)):
        for item in explicit:
            if isinstance(item, str):
                _add(item)

    owned_raw = task.get("owned_files")
    if isinstance(owned_raw, (list, tuple)):
        owned_txt = " ".join(str(x) for x in owned_raw)
    else:
        owned_txt = str(owned_raw or "")

    text = " ".join(
        [
            str(task.get("title") or ""),
            str(task.get("spec") or ""),
            str(task.get("acceptance_cmd") or ""),
            owned_txt,
        ]
    ).lower()

    if any(kw in text for kw in _MACOS_KEYWORDS):
        _add("macos")
        if any(kw in text for kw in _SWIFT_KEYWORDS):
            _add("swift")

    if any(kw in text for kw in _CUDA_KEYWORDS):
        _add("cuda")
    elif "gpu" in text and ("kernel" in text or "benchmark" in text or mode == "research"):
        _add("cuda")

    return required


def _device_satisfies(device: dict[str, Any], required: list[str]) -> bool:
    caps = {str(c).strip().lower() for c in (device.get("capabilities") or []) if isinstance(c, str)}
    gpu_val = device.get("gpu")
    gpu_str = str(gpu_val or "").lower()

    for req in required:
        if req == "gpu":
            if "gpu" in caps or "cuda" in caps or gpu_val is not None:
                continue
            return False
        if req == "cuda":
            if "cuda" in caps or ("gpu" in caps and "nvidia" in gpu_str) or "nvidia" in gpu_str:
                continue
            return False
        if req not in caps:
            return False
    return True


def _device_telemetry_value(device: dict[str, Any], key: str) -> Any:
    val = device.get(key)
    if val is not None:
        return val
    telemetry = device.get("telemetry")
    if isinstance(telemetry, dict):
        return telemetry.get(key)
    return None


def _pick_inference_url(
    selected: dict[str, Any],
    online_devices: list[dict[str, Any]],
    prefer_local_inference: bool,
) -> str | None:
    if not prefer_local_inference:
        return None
    if selected.get("inference_url"):
        return str(selected["inference_url"])
    for dev in online_devices:
        if dev.get("inference_url"):
            return str(dev["inference_url"])
    return None


def route_tasks_to_devices(
    tasks: list[dict[str, Any]],
    devices: list[dict[str, Any]] | dict[str, Any],
    mode: str | list[dict[str, Any]] = "swarm",
    prefer_local_inference: bool = True,
    *,
    remote_devices: list[dict[str, Any]] | None = None,
    max_gpu_util_pct: float = 75.0,
    min_mem_avail_gb: float = 2.0,
) -> list[dict[str, Any]]:
    """Assign each task in ``tasks`` to an online verified device using capabilities, telemetry, and load."""
    if isinstance(devices, dict):
        merged_devices: list[dict[str, Any]] = [devices]
        if isinstance(mode, list):
            merged_devices.extend(mode)
            actual_mode = "swarm"
        else:
            actual_mode = str(mode)
        if remote_devices:
            merged_devices.extend(remote_devices)
    else:
        merged_devices = list(devices)
        if remote_devices:
            merged_devices.extend(remote_devices)
        actual_mode = str(mode)

    online_devices = [
        d for d in merged_devices if d.get("online") and not d.get("tampered")
    ]
    local_fallback = next(
        (d for d in online_devices if d.get("device_id") == "local" or d.get("is_local")),
        None,
    ) or next(
        (d for d in merged_devices if d.get("device_id") == "local" or d.get("is_local")),
        None,
    ) or probe_local_device()

    if not online_devices:
        online_devices = [local_fallback]

    device_index = {d["device_id"]: idx for idx, d in enumerate(online_devices)}
    assigned_counts: dict[str, int] = {d["device_id"]: 0 for d in online_devices}
    assigned_counts.setdefault(local_fallback["device_id"], 0)
    local_id = local_fallback["device_id"]

    assignments: list[dict[str, Any]] = []
    multi_task = len(tasks) > 1

    for idx, task in enumerate(tasks):
        required = infer_required_capabilities(task, mode=actual_mode)
        needs_gpu = any(c in ("cuda", "gpu") for c in required) or bool(task.get("requires_gpu"))
        is_heavy = bool(task.get("heavy"))
        target_id = str(task.get("target_device") or "").strip()
        selected: dict[str, Any] | None = None
        reason = ""
        routing_notes: list[dict[str, str]] = []

        eligible_devices: list[dict[str, Any]] = []
        for dev in online_devices:
            gpu_util = _device_telemetry_value(dev, "gpu_util_pct")
            mem_avail = _device_telemetry_value(dev, "mem_avail_gb")
            if (needs_gpu or is_heavy) and gpu_util is not None and float(gpu_util) >= float(max_gpu_util_pct):
                routing_notes.append({"device_id": dev["device_id"], "reason": "skipped_busy_gpu"})
                continue
            if mem_avail is not None and float(mem_avail) < float(min_mem_avail_gb):
                routing_notes.append({"device_id": dev["device_id"], "reason": "skipped_low_mem"})
                continue
            eligible_devices.append(dev)

        if target_id:
            pinned = next((d for d in online_devices if d.get("device_id") == target_id), None)
            if pinned is not None:
                selected = pinned
                reason = "pinned"

        if selected is None and required:
            capable = [d for d in eligible_devices if _device_satisfies(d, required)]
            if capable:
                prefer_local = any(c in ("macos", "swift") for c in required)
                prefer_gpu = any(c in ("cuda", "gpu") for c in required)

                def _cap_key(d: dict[str, Any]) -> tuple[int, int, int]:
                    load = int(d.get("active_agents") or 0) + assigned_counts.get(d["device_id"], 0)
                    if prefer_local:
                        tie = 0 if d.get("is_local") else 1
                    elif prefer_gpu:
                        if not d.get("is_local") and d.get("gpu") is not None:
                            tie = 0
                        elif d.get("gpu") is not None:
                            tie = 1
                        else:
                            tie = 2
                    else:
                        tie = 0 if d.get("is_local") else 1
                    return (load, tie, device_index.get(d["device_id"], 0))

                selected = min(capable, key=_cap_key)
                reason = f"capability:{','.join(required)}"
            else:
                selected = eligible_devices[0] if eligible_devices and local_fallback not in eligible_devices else local_fallback
                reason = "fallback_no_capable_device"

        if selected is None:
            candidate_pool = eligible_devices or [local_fallback]
            if not multi_task:
                selected = min(
                    candidate_pool,
                    key=lambda d: (
                        int(d.get("active_agents") or 0) + assigned_counts.get(d["device_id"], 0),
                        0 if d.get("is_local") else 1,
                        device_index.get(d["device_id"], 0),
                    ),
                )
            else:
                assigned_local = assigned_counts.get(local_id, 0)

                def _spread_key(d: dict[str, Any]) -> tuple[float, int]:
                    base_load = float(d.get("active_agents") or 0) + float(
                        assigned_counts.get(d["device_id"], 0)
                    )
                    if d.get("is_local"):
                        bias = -0.1 if assigned_local == 0 else 0.5
                    else:
                        bias = 0.0
                    return (base_load + bias, device_index.get(d["device_id"], 0))

                selected = min(candidate_pool, key=_spread_key)
            reason = "load_balanced" if len(candidate_pool) > 1 else "local_default"

        sel_id = selected["device_id"]
        assigned_counts[sel_id] = assigned_counts.get(sel_id, 0) + 1
        assignments.append(
            {
                "task_index": idx,
                "device_id": sel_id,
                "device_name": str(selected.get("name") or sel_id),
                "is_local": bool(selected.get("is_local")),
                "reason": reason,
                "required_capabilities": required,
                "suggested_inference_url": _pick_inference_url(
                    selected, online_devices, prefer_local_inference
                ),
                "routing_notes": routing_notes,
            }
        )

    return assignments


_MIRROR_SYNC_LOCK = threading.Lock()
_MIRROR_SYNC_STATE: dict[tuple[str, str], str] = {}


def _create_git_bundle_b64(
    repo_path: Path,
    *,
    head_sha: str,
    since_commit: str = "",
) -> tuple[str, bool]:
    with tempfile.TemporaryDirectory(prefix="open_swarm_bundle_") as tmpdir:
        bundle_path = Path(tmpdir) / "repo.bundle"
        if since_commit and since_commit != head_sha:
            rev_range = f"{since_commit}..HEAD"
            proc = subprocess.run(
                ["git", "bundle", "create", str(bundle_path), rev_range],
                cwd=str(repo_path),
                capture_output=True,
                text=True,
                check=False,
            )
            if proc.returncode == 0 and bundle_path.is_file() and bundle_path.stat().st_size > 0:
                return base64.b64encode(bundle_path.read_bytes()).decode("ascii"), True

        subprocess.run(
            ["git", "bundle", "create", str(bundle_path), "HEAD"],
            cwd=str(repo_path),
            capture_output=True,
            text=True,
            check=True,
        )
        return base64.b64encode(bundle_path.read_bytes()).decode("ascii"), False


def _build_remote_exec_payload(
    *,
    repo_path: Path,
    device_id: str,
    task: dict[str, Any],
    base_url: str,
    model: str,
    api_key: str,
    spend_cap_usd: float,
    max_repair_turns: int,
    auto_review: bool,
    timeout_s: int,
    llm_proxy_url: str = "",
    llm_proxy_token: str = "",
    repo_key: str = "",
    incremental: bool | None = None,
    since_commit: str = "",
) -> tuple[str, dict[str, Any], bool]:
    head_proc = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=str(repo_path),
        capture_output=True,
        text=True,
        check=True,
    )
    head_sha = head_proc.stdout.strip()

    clean_repo_key = str(repo_key or "").strip()
    effective_since = str(since_commit or "").strip()
    if not effective_since and clean_repo_key and incremental is not False:
        with _MIRROR_SYNC_LOCK:
            effective_since = _MIRROR_SYNC_STATE.get((device_id, clean_repo_key), "")

    want_incremental = bool(incremental) if incremental is not None else bool(effective_since)
    bundle_b64, is_incremental = _create_git_bundle_b64(
        repo_path,
        head_sha=head_sha,
        since_commit=effective_since if want_incremental else "",
    )

    clean_proxy_url = str(llm_proxy_url or "").strip()
    clean_proxy_token = str(llm_proxy_token or "").strip()

    payload: dict[str, Any] = {
        "base_commit": head_sha,
        "bundle_b64": bundle_b64,
        "task": task,
        "base_url": clean_proxy_url or base_url,
        "model": model,
        "spend_cap_usd": float(spend_cap_usd),
        "max_repair_turns": int(max_repair_turns),
        "auto_review": bool(auto_review),
        "timeout_s": int(timeout_s),
    }
    if clean_proxy_url:
        payload["llm_proxy_url"] = clean_proxy_url
    if clean_proxy_token:
        # Send scoped proxy token instead of raw upstream api_key.
        payload["llm_proxy_token"] = clean_proxy_token
    else:
        payload["api_key"] = api_key

    if clean_repo_key:
        payload["repo_key"] = clean_repo_key
        payload["incremental"] = is_incremental

    return head_sha, payload, is_incremental


def _post_bundle_with_mirror_fallback(
    *,
    url: str,
    token: str,
    repo_path: Path,
    device_id: str,
    head_sha: str,
    payload: dict[str, Any],
    is_incremental: bool,
    timeout_s: float,
) -> dict[str, Any]:
    try:
        resp = _http_json(url, method="POST", payload=payload, token=token, timeout_s=timeout_s)
    except RuntimeError as exc:
        msg = str(exc)
        if is_incremental and ("409" in msg or "mirror_miss" in msg.lower() or "mirror cache miss" in msg.lower()):
            full_b64, _ = _create_git_bundle_b64(repo_path, head_sha=head_sha, since_commit="")
            retry_payload = dict(payload)
            retry_payload["bundle_b64"] = full_b64
            retry_payload["incremental"] = False
            resp = _http_json(
                url, method="POST", payload=retry_payload, token=token, timeout_s=timeout_s
            )
        else:
            raise

    clean_repo_key = str(payload.get("repo_key") or "").strip()
    if clean_repo_key:
        with _MIRROR_SYNC_LOCK:
            _MIRROR_SYNC_STATE[(device_id, clean_repo_key)] = head_sha
    return resp


def execute_task_on_remote_device(
    *,
    repo: Path,
    device_creds: dict[str, Any],
    task: dict[str, Any],
    base_url: str,
    model: str,
    api_key: str = "outrcore-swarm",
    spend_cap_usd: float = 0.5,
    max_repair_turns: int = 2,
    auto_review: bool = False,
    timeout_s: int = 240,
    llm_proxy_url: str = "",
    llm_proxy_token: str = "",
    repo_key: str = "",
    incremental: bool | None = None,
    since_commit: str = "",
) -> dict[str, Any]:
    """Bundle ``repo`` HEAD, execute ``task`` on a paired remote node, and recompute local file hashes."""
    repo_path = Path(repo).resolve()
    device_id = str(device_creds["device_id"])
    head_sha, payload, is_incremental = _build_remote_exec_payload(
        repo_path=repo_path,
        device_id=device_id,
        task=task,
        base_url=base_url,
        model=model,
        api_key=api_key,
        spend_cap_usd=spend_cap_usd,
        max_repair_turns=max_repair_turns,
        auto_review=auto_review,
        timeout_s=timeout_s,
        llm_proxy_url=llm_proxy_url,
        llm_proxy_token=llm_proxy_token,
        repo_key=repo_key,
        incremental=incremental,
        since_commit=since_commit,
    )

    host = str(device_creds["host"])
    port = int(device_creds.get("port") or DEFAULT_NODE_PORT)
    token = str(device_creds.get("token") or "")
    url = f"http://{_format_host_for_url(host)}:{port}/node/exec"

    resp = _post_bundle_with_mirror_fallback(
        url=url,
        token=token,
        repo_path=repo_path,
        device_id=device_id,
        head_sha=head_sha,
        payload=payload,
        is_incremental=is_incremental,
        timeout_s=float(timeout_s) + 15.0,
    )
    expected_fp = str(device_creds.get("fingerprint") or "")
    resp_fp = str(resp.get("fingerprint") or "")
    if expected_fp and resp_fp and resp_fp != expected_fp:
        raise RuntimeError("Fingerprint mismatch on remote execution response")

    raw_result = resp.get("result")
    if not isinstance(raw_result, dict):
        raise RuntimeError("Remote execution response missing 'result' object")

    res = dict(raw_result)
    written = [str(p) for p in (res.get("written_files") or []) if isinstance(p, str)]
    if not written and isinstance(res.get("diff"), str) and res["diff"].strip():
        written = extract_diff_paths(res["diff"])
    res["base_file_hashes"] = compute_file_hashes(repo_path, written) if written else {}
    res["device_id"] = device_id
    res["device_name"] = str(device_creds.get("name") or device_id)
    return res


def submit_async_remote_job(
    *,
    repo: Path,
    device_creds: dict[str, Any],
    task: dict[str, Any],
    base_url: str,
    model: str,
    api_key: str = "outrcore-swarm",
    spend_cap_usd: float = 0.5,
    max_repair_turns: int = 2,
    auto_review: bool = False,
    timeout_s: int = 240,
    llm_proxy_url: str = "",
    llm_proxy_token: str = "",
    repo_key: str = "",
    incremental: bool | None = None,
    since_commit: str = "",
) -> dict[str, Any]:
    """Submit an asynchronous job to ``POST /node/jobs`` and return the queued job handle."""
    repo_path = Path(repo).resolve()
    device_id = str(device_creds["device_id"])
    head_sha, payload, is_incremental = _build_remote_exec_payload(
        repo_path=repo_path,
        device_id=device_id,
        task=task,
        base_url=base_url,
        model=model,
        api_key=api_key,
        spend_cap_usd=spend_cap_usd,
        max_repair_turns=max_repair_turns,
        auto_review=auto_review,
        timeout_s=timeout_s,
        llm_proxy_url=llm_proxy_url,
        llm_proxy_token=llm_proxy_token,
        repo_key=repo_key,
        incremental=incremental,
        since_commit=since_commit,
    )

    host = str(device_creds["host"])
    port = int(device_creds.get("port") or DEFAULT_NODE_PORT)
    token = str(device_creds.get("token") or "")
    url = f"http://{_format_host_for_url(host)}:{port}/node/jobs"

    resp = _post_bundle_with_mirror_fallback(
        url=url,
        token=token,
        repo_path=repo_path,
        device_id=device_id,
        head_sha=head_sha,
        payload=payload,
        is_incremental=is_incremental,
        timeout_s=15.0,
    )
    out = dict(resp)
    out["device_id"] = device_id
    out["device_name"] = str(device_creds.get("name") or device_id)
    out["base_commit"] = head_sha
    return out


def get_remote_job_status(
    *,
    device_creds: dict[str, Any],
    job_id: str,
    timeout_s: float = 10.0,
) -> dict[str, Any]:
    """Fetch status, events, and result for an async remote job via ``GET /node/jobs/<id>``."""
    host = str(device_creds["host"])
    port = int(device_creds.get("port") or DEFAULT_NODE_PORT)
    token = str(device_creds.get("token") or "")
    safe_id = urllib.parse.quote(str(job_id), safe="")
    url = f"http://{_format_host_for_url(host)}:{port}/node/jobs/{safe_id}"

    resp = _http_json(url, method="GET", token=token, timeout_s=timeout_s)
    expected_fp = str(device_creds.get("fingerprint") or "")
    resp_fp = str(resp.get("fingerprint") or "")
    if expected_fp and resp_fp and resp_fp != expected_fp:
        raise RuntimeError("Fingerprint mismatch on remote job status response")
    return resp


def stream_remote_job_events(
    *,
    device_creds: dict[str, Any],
    job_id: str,
    on_event: Callable[[dict[str, Any]], None] | None = None,
    timeout_s: float = 240.0,
) -> list[dict[str, Any]]:
    """Consume SSE events from ``GET /node/jobs/<id>/stream`` until ``event: done``."""
    host = str(device_creds["host"])
    port = int(device_creds.get("port") or DEFAULT_NODE_PORT)
    token = str(device_creds.get("token") or "")
    safe_id = urllib.parse.quote(str(job_id), safe="")
    url = f"http://{_format_host_for_url(host)}:{port}/node/jobs/{safe_id}/stream"

    headers = {
        "Accept": "text/event-stream",
        "Authorization": f"Bearer {token}",
    }
    req = urllib.request.Request(url, headers=headers, method="GET")
    events: list[dict[str, Any]] = []

    def _emit(ev_name: str, ev_id: str, data_lines: list[str]) -> bool:
        if not data_lines and not ev_name:
            return False
        raw_data = "\n".join(data_lines)
        try:
            parsed_data: Any = json.loads(raw_data) if raw_data else {}
        except json.JSONDecodeError:
            parsed_data = raw_data
        record = {
            "event": ev_name or "message",
            "id": ev_id or None,
            "data": parsed_data,
        }
        events.append(record)
        if on_event is not None:
            on_event(record)
        return record["event"] == "done"

    try:
        with urllib.request.urlopen(req, timeout=timeout_s) as resp:
            ev_name = ""
            ev_id = ""
            data_lines: list[str] = []
            while True:
                raw_line = resp.readline()
                if not raw_line:
                    _emit(ev_name, ev_id, data_lines)
                    break
                line = raw_line.decode("utf-8", errors="replace").rstrip("\r\n")
                if line == "":
                    done = _emit(ev_name, ev_id, data_lines)
                    ev_name = ""
                    ev_id = ""
                    data_lines = []
                    if done:
                        break
                    continue
                if line.startswith(":"):
                    continue
                if line.startswith("event:"):
                    ev_name = line[len("event:") :].strip()
                elif line.startswith("id:"):
                    ev_id = line[len("id:") :].strip()
                elif line.startswith("data:"):
                    data_lines.append(line[len("data:") :].lstrip())
    except urllib.error.HTTPError as exc:
        detail = ""
        try:
            detail = exc.read().decode("utf-8", errors="replace").strip()
        except Exception:
            pass
        raise RuntimeError(f"HTTP {exc.code}: {exc.reason} ({detail[:300]})") from exc
    except urllib.error.URLError as exc:
        raise RuntimeError(f"Connection error: {exc.reason}") from exc

    return events


def cancel_remote_job(
    *,
    device_creds: dict[str, Any],
    job_id: str,
    timeout_s: float = 10.0,
) -> dict[str, Any]:
    """Request cancellation of a remote async job via ``POST /node/jobs/<id>/cancel``."""
    host = str(device_creds["host"])
    port = int(device_creds.get("port") or DEFAULT_NODE_PORT)
    token = str(device_creds.get("token") or "")
    safe_id = urllib.parse.quote(str(job_id), safe="")
    url = f"http://{_format_host_for_url(host)}:{port}/node/jobs/{safe_id}/cancel"
    return _http_json(url, method="POST", payload={}, token=token, timeout_s=timeout_s)
