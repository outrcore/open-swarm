#!/usr/bin/env python3
"""Open Swarm v0.3.0 — Verified Device Registry, Tailscale Discovery & Capability-Aware Router.

Pure Python 3 standard library module providing:
- Tailscale CLI discovery (`find_tailscale_bin`, `discover_tailscale_peers`).
- Local machine capability probing (`probe_local_device`).
- Fingerprint-pinned persistent remote device registry (`DeviceRegistry`).
- Capability inference and load/capability-aware task routing
  (`infer_required_capabilities`, `route_tasks_to_devices`).
- Authenticated remote git-bundle task execution (`execute_task_on_remote_device`).
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
from typing import Any
import urllib.error
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

    return {
        "capabilities": caps,
        "gpu": gpu,
        "os": system.lower(),
        "arch": arch,
        "cpu_cores": os.cpu_count() or 1,
        "ram_gb": _detect_ram_gb(),
        "inference_url": None,
    }


def probe_local_device() -> dict[str, Any]:
    """Return a standardized device descriptor for the local host."""
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

    system = platform.system()
    default_name = "This Mac" if system == "Darwin" else socket.gethostname()
    return {
        "device_id": "local",
        "name": str(caps_info.get("name") or default_name),
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
    devices: list[dict[str, Any]],
    mode: str = "swarm",
    prefer_local_inference: bool = True,
) -> list[dict[str, Any]]:
    """Assign each task in ``tasks`` to an online verified device using capabilities and load."""
    online_devices = [
        d for d in devices if d.get("online") and not d.get("tampered")
    ]
    local_fallback = next(
        (d for d in online_devices if d.get("device_id") == "local" or d.get("is_local")),
        None,
    ) or next(
        (d for d in devices if d.get("device_id") == "local" or d.get("is_local")),
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
        required = infer_required_capabilities(task, mode=mode)
        target_id = str(task.get("target_device") or "").strip()
        selected: dict[str, Any] | None = None
        reason = ""

        if target_id:
            pinned = next((d for d in online_devices if d.get("device_id") == target_id), None)
            if pinned is not None:
                selected = pinned
                reason = "pinned"

        if selected is None and required:
            capable = [d for d in online_devices if _device_satisfies(d, required)]
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
                selected = local_fallback
                reason = "fallback_no_capable_device"

        if selected is None:
            if not multi_task:
                selected = min(
                    online_devices,
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

                selected = min(online_devices, key=_spread_key)
            reason = "load_balanced" if len(online_devices) > 1 else "local_default"

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
            }
        )

    return assignments


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
) -> dict[str, Any]:
    """Bundle ``repo`` HEAD, execute ``task`` on a paired remote node, and recompute local file hashes."""
    repo_path = Path(repo).resolve()
    head_proc = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=str(repo_path),
        capture_output=True,
        text=True,
        check=True,
    )
    head_sha = head_proc.stdout.strip()

    with tempfile.TemporaryDirectory(prefix="open_swarm_bundle_") as tmpdir:
        bundle_path = Path(tmpdir) / "repo.bundle"
        subprocess.run(
            ["git", "bundle", "create", str(bundle_path), "HEAD"],
            cwd=str(repo_path),
            capture_output=True,
            text=True,
            check=True,
        )
        bundle_b64 = base64.b64encode(bundle_path.read_bytes()).decode("ascii")

    host = str(device_creds["host"])
    port = int(device_creds.get("port") or DEFAULT_NODE_PORT)
    token = str(device_creds.get("token") or "")
    url = f"http://{_format_host_for_url(host)}:{port}/node/exec"

    payload = {
        "base_commit": head_sha,
        "bundle_b64": bundle_b64,
        "task": task,
        "base_url": base_url,
        "model": model,
        "api_key": api_key,
        "spend_cap_usd": float(spend_cap_usd),
        "max_repair_turns": int(max_repair_turns),
        "auto_review": bool(auto_review),
        "timeout_s": int(timeout_s),
    }

    resp = _http_json(
        url,
        method="POST",
        payload=payload,
        token=token,
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
    res["device_id"] = str(device_creds["device_id"])
    res["device_name"] = str(device_creds.get("name") or device_creds["device_id"])
    return res
