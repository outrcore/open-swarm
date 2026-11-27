"""Unit tests for open_swarm.device_mesh."""

from __future__ import annotations

import base64
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
import os
from pathlib import Path
import stat
import subprocess
import sys
import tempfile
import threading
import unittest
from unittest.mock import patch

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from open_swarm.continuity_store import compute_file_hashes  # noqa: E402
from open_swarm.device_mesh import (  # noqa: E402
    DeviceRegistry,
    cancel_remote_job,
    discover_tailscale_peers,
    execute_task_on_remote_device,
    find_tailscale_bin,
    get_remote_job_status,
    infer_required_capabilities,
    probe_local_device,
    route_tasks_to_devices,
    stream_remote_job_events,
    submit_async_remote_job,
)


class TailscaleDiscoveryTests(unittest.TestCase):
    def test_discover_tailscale_peers_parses_mac_and_spark(self) -> None:
        fake_status = {
            "BackendState": "Running",
            "Self": {
                "ID": "self-node-1",
                "HostName": "macbook-pro",
                "DNSName": "macbook-pro.tail123.ts.net.",
                "TailscaleIPs": ["100.64.0.10", "fd7a:115c:a1e0::10"],
                "OS": "macOS",
                "Online": True,
            },
            "Peer": {
                "nodekey:spark": {
                    "ID": "peer-spark-1",
                    "HostName": "dgx-spark",
                    "DNSName": "dgx-spark.tail123.ts.net.",
                    "TailscaleIPs": ["fd7a:115c:a1e0::20", "100.64.0.20"],
                    "OS": "linux",
                    "Online": True,
                },
                "nodekey:offline": {
                    "ID": "peer-old-box",
                    "HostName": "old-linux",
                    "DNSName": "old-linux.tail123.ts.net.",
                    "TailscaleIPs": ["100.64.0.99"],
                    "OS": "linux",
                    "Online": False,
                },
            },
        }
        proc = subprocess.CompletedProcess(
            args=["/opt/homebrew/bin/tailscale", "status", "--json"],
            returncode=0,
            stdout=json.dumps(fake_status),
            stderr="",
        )
        with (
            patch("open_swarm.device_mesh.find_tailscale_bin", return_value="/opt/homebrew/bin/tailscale"),
            patch("open_swarm.device_mesh.subprocess.run", return_value=proc),
        ):
            res = discover_tailscale_peers(timeout_s=2.0)

        self.assertTrue(res["available"])
        self.assertTrue(res["running"])
        self.assertIsNotNone(res["self"])
        self.assertEqual(res["self"]["ip"], "100.64.0.10")
        self.assertTrue(res["self"]["is_self"])
        self.assertEqual(len(res["peers"]), 2)

        spark = next(p for p in res["peers"] if p["hostname"] == "dgx-spark")
        self.assertEqual(spark["tailscale_id"], "peer-spark-1")
        self.assertEqual(spark["dns_name"], "dgx-spark.tail123.ts.net")
        self.assertEqual(spark["ip"], "100.64.0.20")
        self.assertEqual(spark["os"], "linux")
        self.assertTrue(spark["online"])
        self.assertFalse(spark["is_self"])
        self.assertEqual(spark["suggested_port"], 18799)

        offline = next(p for p in res["peers"] if p["hostname"] == "old-linux")
        self.assertFalse(offline["online"])
        self.assertEqual(offline["ip"], "100.64.0.99")

    def test_discover_tailscale_missing_binary_graceful(self) -> None:
        with patch("open_swarm.device_mesh.find_tailscale_bin", return_value=None):
            res = discover_tailscale_peers()
        self.assertEqual(
            res,
            {"available": False, "running": False, "self": None, "peers": []},
        )

    def test_discover_tailscale_timeout_and_nonzero_exit(self) -> None:
        with (
            patch("open_swarm.device_mesh.find_tailscale_bin", return_value="/usr/local/bin/tailscale"),
            patch(
                "open_swarm.device_mesh.subprocess.run",
                side_effect=subprocess.TimeoutExpired(cmd="tailscale", timeout=1.0),
            ),
        ):
            res_timeout = discover_tailscale_peers(timeout_s=1.0)
        self.assertFalse(res_timeout["available"])
        self.assertFalse(res_timeout["running"])
        self.assertEqual(res_timeout["peers"], [])

        fail_proc = subprocess.CompletedProcess(
            args=["tailscale", "status", "--json"], returncode=1, stdout="", stderr="stopped"
        )
        with (
            patch("open_swarm.device_mesh.find_tailscale_bin", return_value="/usr/local/bin/tailscale"),
            patch("open_swarm.device_mesh.subprocess.run", return_value=fail_proc),
        ):
            res_err = discover_tailscale_peers()
        self.assertFalse(res_err["available"])
        self.assertEqual(res_err["peers"], [])

    def test_find_tailscale_bin_checks_candidates(self) -> None:
        with (
            patch("open_swarm.device_mesh.shutil.which", return_value=None),
            patch(
                "open_swarm.device_mesh.os.path.isfile",
                side_effect=lambda p: p == "/Applications/Tailscale.app/Contents/MacOS/Tailscale",
            ),
            patch(
                "open_swarm.device_mesh.os.access",
                side_effect=lambda p, m: p == "/Applications/Tailscale.app/Contents/MacOS/Tailscale",
            ),
        ):
            self.assertEqual(
                find_tailscale_bin(),
                "/Applications/Tailscale.app/Contents/MacOS/Tailscale",
            )


class DeviceRegistryTests(unittest.TestCase):
    def setUp(self) -> None:
        self._tmpdir = tempfile.TemporaryDirectory()
        self.root = Path(self._tmpdir.name)
        self.devices_path = self.root / "state" / "devices.json"
        self.registry = DeviceRegistry(path=self.devices_path)

        state = {
            "node_id": "spark-gb10",
            "name": "DGX Spark",
            "device_token": "secret-bearer-token-123",
            "fingerprint": "sha256:fp-legit-001",
            "health_fingerprint": "sha256:fp-legit-001",
            "capabilities": ["linux", "cuda", "gpu", "python"],
            "gpu": "NVIDIA GB10",
            "os": "linux",
            "arch": "aarch64",
            "cpu_cores": 20,
            "ram_gb": 128.0,
            "inference_url": "http://100.64.0.20:8000/v1",
            "active_agents": 0,
            "max_agents": 8,
            "last_auth": None,
            "last_exec": None,
        }
        self.server_state = state

        class _NodeHandler(BaseHTTPRequestHandler):
            def log_message(self, format: str, *args: object) -> None:
                return

            def _send_json(self, code: int, body: dict) -> None:
                raw = json.dumps(body).encode("utf-8")
                self.send_response(code)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(raw)))
                self.end_headers()
                self.wfile.write(raw)

            def do_POST(self) -> None:
                length = int(self.headers.get("Content-Length", "0"))
                body = json.loads(self.rfile.read(length).decode("utf-8"))
                if self.path == "/node/pair":
                    if body.get("pair_code") != "PAIR-999":
                        self._send_json(403, {"error": "invalid pair code"})
                        return
                    self._send_json(
                        200,
                        {
                            "node_id": state["node_id"],
                            "name": state["name"],
                            "device_token": state["device_token"],
                            "fingerprint": state["fingerprint"],
                            "capabilities": state["capabilities"],
                            "gpu": state["gpu"],
                            "os": state["os"],
                            "arch": state["arch"],
                            "cpu_cores": state["cpu_cores"],
                            "ram_gb": state["ram_gb"],
                            "inference_url": state["inference_url"],
                        },
                    )
                    return

                if self.path == "/node/exec":
                    state["last_auth"] = self.headers.get("Authorization")
                    state["last_exec"] = body
                    self._send_json(
                        200,
                        {
                            "ok": True,
                            "fingerprint": state["health_fingerprint"],
                            "result": {
                                "agent_id": "coder-a",
                                "ticket": body["task"].get("id", "T-001"),
                                "written_files": ["app.py"],
                                "tests_ok": True,
                                "diff": "diff --git a/app.py b/app.py\n",
                            },
                        },
                    )
                    return

                self._send_json(404, {"error": "not found"})

            def do_GET(self) -> None:
                if self.path == "/node/health":
                    state["last_auth"] = self.headers.get("Authorization")
                    self._send_json(
                        200,
                        {
                            "ok": True,
                            "fingerprint": state["health_fingerprint"],
                            "capabilities": state["capabilities"],
                            "gpu": state["gpu"],
                            "active_agents": state["active_agents"],
                            "max_agents": state["max_agents"],
                            "inference_url": state["inference_url"],
                        },
                    )
                    return
                self._send_json(404, {"error": "not found"})

        self.server = ThreadingHTTPServer(("127.0.0.1", 0), _NodeHandler)
        self.port = self.server.server_address[1]
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)
        self.thread.start()

    def tearDown(self) -> None:
        self.server.shutdown()
        self.server.server_close()
        self._tmpdir.cleanup()

    def test_pair_permissions_sanitization_fingerprint_pin_and_remove(self) -> None:
        paired = self.registry.pair_device(
            host="127.0.0.1",
            port=self.port,
            pair_code="PAIR-999",
            name="Spark Lab",
        )
        self.assertTrue(paired["ok"])
        self.assertEqual(paired["device_id"], "spark-gb10")
        self.assertEqual(paired["name"], "Spark Lab")
        self.assertTrue(paired["has_token"])
        self.assertNotIn("token", paired)
        self.assertNotIn("device_token", paired)

        # Verify 0o600 file permissions
        mode = stat.S_IMODE(os.stat(self.devices_path).st_mode)
        self.assertEqual(mode, 0o600)

        # Internal credentials retain token and fingerprint
        creds = self.registry.get_device_credentials("spark-gb10")
        self.assertIsNotNone(creds)
        self.assertEqual(creds["token"], "secret-bearer-token-123")
        self.assertEqual(creds["fingerprint"], "sha256:fp-legit-001")

        # list_devices returns local + sanitized online remote
        devices = self.registry.list_devices(probe=True)
        self.assertEqual(devices[0]["device_id"], "local")
        spark = next(d for d in devices if d["device_id"] == "spark-gb10")
        self.assertTrue(spark["online"])
        self.assertFalse(spark["tampered"])
        self.assertTrue(spark["has_token"])
        self.assertNotIn("token", spark)
        self.assertEqual(self.server_state["last_auth"], "Bearer secret-bearer-token-123")

        # Fingerprint mismatch triggers tampered=True and online=False
        self.server_state["health_fingerprint"] = "sha256:fp-rogue-999"
        tampered_devices = self.registry.list_devices(probe=True)
        spark_tampered = next(d for d in tampered_devices if d["device_id"] == "spark-gb10")
        self.assertFalse(spark_tampered["online"])
        self.assertTrue(spark_tampered["tampered"])
        self.assertIn("Fingerprint mismatch", spark_tampered["error"])
        self.assertNotIn("token", spark_tampered)

        # remove_device refuses 'local' and removes remote device
        self.assertFalse(self.registry.remove_device("local"))
        self.assertTrue(self.registry.remove_device("spark-gb10"))
        self.assertFalse(self.registry.remove_device("spark-gb10"))
        self.assertIsNone(self.registry.get_device_credentials("spark-gb10"))
        self.assertEqual(len(self.registry.list_devices(probe=False)), 1)

    def test_execute_task_on_remote_device_bundles_and_recomputes_hashes(self) -> None:
        repo = self.root / "repo"
        repo.mkdir()
        subprocess.run(["git", "init"], cwd=str(repo), check=True, capture_output=True)
        subprocess.run(["git", "config", "user.email", "test@example.com"], cwd=str(repo), check=True)
        subprocess.run(["git", "config", "user.name", "Test"], cwd=str(repo), check=True)
        (repo / "app.py").write_text("print('hello')\n", encoding="utf-8")
        subprocess.run(["git", "add", "app.py"], cwd=str(repo), check=True)
        subprocess.run(["git", "commit", "-m", "init"], cwd=str(repo), check=True, capture_output=True)

        self.registry.pair_device(host="127.0.0.1", port=self.port, pair_code="PAIR-999")
        creds = self.registry.get_device_credentials("spark-gb10")
        assert creds is not None

        res = execute_task_on_remote_device(
            repo=repo,
            device_creds=creds,
            task={"id": "T-007", "title": "CUDA kernel", "spec": "Optimize", "acceptance_cmd": "pytest"},
            base_url="http://127.0.0.1:8000/v1",
            model="qwen-coder",
        )
        self.assertEqual(res["device_id"], "spark-gb10")
        self.assertEqual(res["device_name"], "DGX Spark")
        self.assertEqual(res["base_file_hashes"], compute_file_hashes(repo, ["app.py"]))
        self.assertEqual(self.server_state["last_auth"], "Bearer secret-bearer-token-123")
        bundle_bytes = base64.b64decode(self.server_state["last_exec"]["bundle_b64"])
        self.assertTrue(bundle_bytes.startswith(b"# v2 git bundle"))


class RoutingTests(unittest.TestCase):
    def _sample_devices(self, *, spark_online: bool = True) -> list[dict]:
        return [
            {
                "device_id": "local",
                "name": "This Mac",
                "is_local": True,
                "online": True,
                "tampered": False,
                "capabilities": ["python", "git", "macos", "swift"],
                "gpu": "Apple Silicon GPU",
                "active_agents": 0,
                "max_agents": 4,
                "inference_url": None,
            },
            {
                "device_id": "spark-1",
                "name": "DGX Spark",
                "is_local": False,
                "online": spark_online,
                "tampered": False,
                "capabilities": ["python", "git", "linux", "cuda", "gpu"],
                "gpu": "NVIDIA GB10",
                "active_agents": 0,
                "max_agents": 8,
                "inference_url": "http://100.64.0.20:8000/v1",
            },
        ]

    def test_swift_xcode_task_routes_to_mac(self) -> None:
        devices = self._sample_devices(spark_online=True)
        tasks = [
            {
                "id": "T-1",
                "title": "Build macOS menu bar UI",
                "spec": "Update AppKit controller in Sources/MenuBar.swift",
                "acceptance_cmd": "swiftc -parse Sources/MenuBar.swift",
            }
        ]
        self.assertEqual(infer_required_capabilities(tasks[0]), ["macos", "swift"])
        routed = route_tasks_to_devices(tasks, devices)
        self.assertEqual(len(routed), 1)
        self.assertEqual(routed[0]["device_id"], "local")
        self.assertTrue(routed[0]["is_local"])
        # Still falls back to online Spark inference URL if local has none
        self.assertEqual(routed[0]["suggested_inference_url"], "http://100.64.0.20:8000/v1")

    def test_cuda_pytorch_task_routes_to_spark(self) -> None:
        devices = self._sample_devices(spark_online=True)
        tasks = [
            {
                "id": "T-2",
                "title": "Optimize fused Triton kernel",
                "spec": "Benchmark PyTorch torch.compile CUDA kernel throughput",
                "acceptance_cmd": "python3 -m pytest tests/test_kernel.py",
            }
        ]
        self.assertEqual(infer_required_capabilities(tasks[0]), ["cuda"])
        routed = route_tasks_to_devices(tasks, devices)
        self.assertEqual(routed[0]["device_id"], "spark-1")
        self.assertFalse(routed[0]["is_local"])
        self.assertEqual(routed[0]["suggested_inference_url"], "http://100.64.0.20:8000/v1")

    def test_cuda_task_falls_back_to_local_when_spark_offline(self) -> None:
        devices = self._sample_devices(spark_online=False)
        tasks = [
            {
                "id": "T-3",
                "title": "Write CUDA matmul",
                "spec": "Add nvcc kernel in kernels/matmul.cu",
                "acceptance_cmd": "nvcc --version",
            }
        ]
        routed = route_tasks_to_devices(tasks, devices)
        self.assertEqual(routed[0]["device_id"], "local")
        self.assertEqual(routed[0]["reason"], "fallback_no_capable_device")
        self.assertIsNone(routed[0]["suggested_inference_url"])

    def test_three_general_python_tasks_spread_across_local_and_spark(self) -> None:
        devices = self._sample_devices(spark_online=True)
        tasks = [
            {"id": f"T-{i}", "title": f"Refactor Python module {i}", "spec": "Clean up helpers", "acceptance_cmd": "pytest"}
            for i in range(1, 4)
        ]
        routed = route_tasks_to_devices(tasks, devices)
        assigned_ids = [r["device_id"] for r in routed]
        self.assertIn("local", assigned_ids)
        self.assertIn("spark-1", assigned_ids)
        self.assertEqual(len(assigned_ids), 3)
        # First task starts locally, remaining spread onto idle Spark
        self.assertEqual(assigned_ids[0], "local")
        for r in routed:
            self.assertEqual(r["suggested_inference_url"], "http://100.64.0.20:8000/v1")

    def test_probe_local_device_shape(self) -> None:
        dev = probe_local_device()
        self.assertEqual(dev["device_id"], "local")
        self.assertTrue(dev["is_local"])
        self.assertTrue(dev["online"])
        self.assertEqual(dev["host"], "127.0.0.1")
        self.assertEqual(dev["port"], 18790)
        self.assertIn("telemetry", dev)
        self.assertIsInstance(dev["mem_avail_gb"], float)
        self.assertIn("gpu_util_pct", dev)
        self.assertIn("cpu_load_pct", dev)


class Phase2DDeviceMeshTests(unittest.TestCase):
    def setUp(self) -> None:
        self._tmpdir = tempfile.TemporaryDirectory()
        self.root = Path(self._tmpdir.name)
        self.devices_path = self.root / "devices.json"
        self.registry = DeviceRegistry(path=self.devices_path)

    def tearDown(self) -> None:
        self._tmpdir.cleanup()

    def test_busy_gpu_and_low_mem_routing_guards(self) -> None:
        local_dev = {
            "device_id": "local",
            "name": "This Mac",
            "is_local": True,
            "online": True,
            "tampered": False,
            "capabilities": ["python", "git", "macos", "swift"],
            "gpu": "Apple Silicon GPU",
            "gpu_util_pct": None,
            "mem_avail_gb": 16.0,
            "active_agents": 0,
            "max_agents": 4,
            "inference_url": None,
        }
        spark_busy = {
            "device_id": "spark-busy",
            "name": "Spark Busy",
            "is_local": False,
            "online": True,
            "tampered": False,
            "capabilities": ["python", "git", "linux", "cuda", "gpu"],
            "gpu": "NVIDIA GB10",
            "gpu_util_pct": 89.0,
            "mem_avail_gb": 64.0,
            "active_agents": 0,
            "max_agents": 8,
            "inference_url": "http://100.64.0.21:8000/v1",
        }
        spark_idle = {
            "device_id": "spark-idle",
            "name": "Spark Idle",
            "is_local": False,
            "online": True,
            "tampered": False,
            "capabilities": ["python", "git", "linux", "cuda", "gpu"],
            "gpu": "NVIDIA GB10",
            "gpu_util_pct": 12.0,
            "mem_avail_gb": 64.0,
            "active_agents": 0,
            "max_agents": 8,
            "inference_url": "http://100.64.0.22:8000/v1",
        }
        cuda_task = [
            {
                "id": "T-GPU-1",
                "title": "Train CUDA kernel",
                "spec": "Benchmark torch.compile CUDA throughput",
                "acceptance_cmd": "pytest",
            }
        ]

        # 1. Busy Spark skipped in favor of idle Spark, using (tasks, local, remote_devices=...) form
        routed = route_tasks_to_devices(
            cuda_task,
            local_dev,
            remote_devices=[spark_busy, spark_idle],
            max_gpu_util_pct=75.0,
            min_mem_avail_gb=2.0,
        )
        self.assertEqual(routed[0]["device_id"], "spark-idle")
        self.assertIn(
            {"device_id": "spark-busy", "reason": "skipped_busy_gpu"},
            routed[0]["routing_notes"],
        )

        # 2. Only busy Spark available -> fallback to local with skipped_busy_gpu note
        routed_fb = route_tasks_to_devices(
            cuda_task,
            [local_dev, spark_busy],
            max_gpu_util_pct=75.0,
        )
        self.assertEqual(routed_fb[0]["device_id"], "local")
        self.assertEqual(routed_fb[0]["reason"], "fallback_no_capable_device")
        self.assertIn(
            {"device_id": "spark-busy", "reason": "skipped_busy_gpu"},
            routed_fb[0]["routing_notes"],
        )

        # 3. Low-memory device skipped for general tasks too
        low_mem_spark = dict(spark_idle, device_id="spark-lowmem", mem_avail_gb=0.9)
        py_task = [
            {"id": "T-PY-1", "title": "Clean utils", "spec": "Refactor python helpers", "acceptance_cmd": "pytest"}
        ]
        routed_mem = route_tasks_to_devices(
            py_task,
            [local_dev, low_mem_spark],
            min_mem_avail_gb=2.0,
        )
        self.assertEqual(routed_mem[0]["device_id"], "local")
        self.assertIn(
            {"device_id": "spark-lowmem", "reason": "skipped_low_mem"},
            routed_mem[0]["routing_notes"],
        )

    def test_remote_telemetry_proxy_forwarding_mirror_fallback_and_async_sse(self) -> None:
        state = {
            "exec_payloads": [],
            "job_payloads": [],
            "mirror_misses_left": 1,
            "cancelled": False,
        }

        class _Handler(BaseHTTPRequestHandler):
            def log_message(self, format: str, *args: object) -> None:
                return

            def _send_json(self, code: int, body: dict) -> None:
                raw = json.dumps(body).encode("utf-8")
                self.send_response(code)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(raw)))
                self.end_headers()
                self.wfile.write(raw)

            def do_GET(self) -> None:
                if self.path == "/node/health":
                    self._send_json(
                        200,
                        {
                            "ok": True,
                            "version": "0.4.0",
                            "fingerprint": "fp-2d",
                            "capabilities": ["linux", "cuda", "gpu", "python"],
                            "gpu": "NVIDIA GB10",
                            "active_agents": 1,
                            "max_agents": 8,
                            "inference_url": "http://100.64.0.20:8000/v1",
                            "telemetry": {
                                "gpu_util_pct": 22.5,
                                "gpu_mem_used_mb": 1024,
                                "gpu_mem_total_mb": 24576,
                                "mem_avail_gb": 96.4,
                                "cpu_load_pct": 18.0,
                            },
                            "gpu_util_pct": 22.5,
                            "gpu_mem_used_mb": 1024,
                            "gpu_mem_total_mb": 24576,
                            "mem_avail_gb": 96.4,
                            "cpu_load_pct": 18.0,
                        },
                    )
                    return

                if self.path == "/node/jobs/job_123":
                    self._send_json(
                        200,
                        {
                            "ok": True,
                            "job_id": "job_123",
                            "status": "cancelled" if state["cancelled"] else "completed",
                            "events": [{"seq": 1, "event": "queued", "data": {"status": "queued"}}],
                            "result": {"tests_ok": True, "written_files": ["app.py"], "diff": ""},
                            "error": None,
                            "fingerprint": "fp-2d",
                        },
                    )
                    return

                if self.path == "/node/jobs/job_123/stream":
                    self.close_connection = True
                    self.send_response(200)
                    self.send_header("Content-Type", "text/event-stream")
                    self.send_header("Cache-Control", "no-cache")
                    self.send_header("Connection", "close")
                    self.end_headers()
                    chunks = (
                        b"id: 1\nevent: queued\ndata: {\"seq\": 1, \"event\": \"queued\"}\n\n"
                        b"id: 2\nevent: progress\ndata: {\"seq\": 2, \"event\": \"progress\", \"data\": {\"stage\": \"coding\"}}\n\n"
                        b"event: done\ndata: {\"job_id\": \"job_123\", \"status\": \"completed\"}\n\n"
                    )
                    self.wfile.write(chunks)
                    self.wfile.flush()
                    return

                self._send_json(404, {"error": "not found"})

            def do_POST(self) -> None:
                n = int(self.headers.get("Content-Length") or 0)
                body = json.loads(self.rfile.read(n).decode("utf-8")) if n else {}
                if self.path == "/node/pair":
                    self._send_json(
                        200,
                        {
                            "node_id": "spark-2d",
                            "name": "Spark 2D",
                            "device_token": "tok-2d",
                            "fingerprint": "fp-2d",
                            "capabilities": ["linux", "cuda", "gpu", "python"],
                        },
                    )
                    return
                if self.path == "/node/exec":
                    state["exec_payloads"].append(body)
                    if body.get("incremental") and state["mirror_misses_left"] > 0:
                        state["mirror_misses_left"] -= 1
                        self._send_json(409, {"ok": False, "mirror_miss": True, "error": "Mirror cache miss"})
                        return
                    self._send_json(
                        200,
                        {
                            "ok": True,
                            "fingerprint": "fp-2d",
                            "result": {"written_files": ["app.py"], "tests_ok": True, "diff": ""},
                        },
                    )
                    return
                if self.path == "/node/jobs":
                    state["job_payloads"].append(body)
                    self._send_json(202, {"ok": True, "job_id": "job_123", "status": "queued"})
                    return
                if self.path == "/node/jobs/job_123/cancel":
                    state["cancelled"] = True
                    self._send_json(200, {"ok": True, "job_id": "job_123", "status": "cancelled"})
                    return
                self._send_json(404, {"error": "not found"})

        srv = ThreadingHTTPServer(("127.0.0.1", 0), _Handler)
        port = srv.server_address[1]
        thr = threading.Thread(target=srv.serve_forever, daemon=True)
        thr.start()
        try:
            self.registry.pair_device(host="127.0.0.1", port=port, pair_code="PAIR-OK")
            devices = self.registry.list_devices(probe=True)
            spark = next(d for d in devices if d["device_id"] == "spark-2d")
            self.assertEqual(spark["version"], "0.4.0")
            self.assertEqual(spark["gpu_util_pct"], 22.5)
            self.assertEqual(spark["gpu_mem_used_mb"], 1024)
            self.assertEqual(spark["gpu_mem_total_mb"], 24576)
            self.assertEqual(spark["mem_avail_gb"], 96.4)
            self.assertEqual(spark["cpu_load_pct"], 18.0)

            repo = self.root / "repo"
            repo.mkdir()
            subprocess.run(["git", "init"], cwd=str(repo), check=True, capture_output=True)
            subprocess.run(["git", "config", "user.email", "t@t"], cwd=str(repo), check=True)
            subprocess.run(["git", "config", "user.name", "T"], cwd=str(repo), check=True)
            (repo / "app.py").write_text("print(1)\n", encoding="utf-8")
            subprocess.run(["git", "add", "app.py"], cwd=str(repo), check=True)
            subprocess.run(["git", "commit", "-m", "c1"], cwd=str(repo), check=True, capture_output=True)
            c1 = subprocess.run(
                ["git", "rev-parse", "HEAD"], cwd=str(repo), check=True, capture_output=True, text=True
            ).stdout.strip()
            (repo / "app.py").write_text("print(2)\n", encoding="utf-8")
            subprocess.run(["git", "add", "app.py"], cwd=str(repo), check=True)
            subprocess.run(["git", "commit", "-m", "c2"], cwd=str(repo), check=True, capture_output=True)

            creds = self.registry.get_device_credentials("spark-2d")
            assert creds is not None

            # Scoped LLM proxy + incremental fallback on 409 mirror_miss
            execute_task_on_remote_device(
                repo=repo,
                device_creds=creds,
                task={"id": "T-1", "spec": "Do", "acceptance_cmd": "pytest"},
                base_url="https://api.upstream.example/v1",
                model="coder",
                api_key="sk-raw-upstream-secret",
                llm_proxy_url="http://100.64.0.1:18790/v1",
                llm_proxy_token="scoped-proxy-tok",
                repo_key="test/repo",
                incremental=True,
                since_commit=c1,
            )
            self.assertEqual(len(state["exec_payloads"]), 2)
            self.assertTrue(state["exec_payloads"][0]["incremental"])
            self.assertFalse(state["exec_payloads"][1]["incremental"])
            for p in state["exec_payloads"]:
                self.assertEqual(p["llm_proxy_url"], "http://100.64.0.1:18790/v1")
                self.assertEqual(p["llm_proxy_token"], "scoped-proxy-tok")
                self.assertNotIn("api_key", p)

            # Async submit + SSE stream + status + cancel
            sub = submit_async_remote_job(
                repo=repo,
                device_creds=creds,
                task={"id": "T-2", "spec": "Do", "acceptance_cmd": "pytest"},
                base_url="http://127.0.0.1:8000/v1",
                model="coder",
                api_key="sk-raw-upstream-secret",
                llm_proxy_url="http://100.64.0.1:18790/v1",
                llm_proxy_token="scoped-job-tok",
            )
            self.assertEqual(sub["job_id"], "job_123")
            self.assertEqual(sub["status"], "queued")
            self.assertNotIn("api_key", state["job_payloads"][0])
            self.assertEqual(state["job_payloads"][0]["llm_proxy_token"], "scoped-job-tok")

            seen: list[str] = []
            events = stream_remote_job_events(
                device_creds=creds,
                job_id="job_123",
                on_event=lambda ev: seen.append(ev["event"]),
            )
            self.assertEqual(seen, ["queued", "progress", "done"])
            self.assertEqual(events[-1]["data"]["status"], "completed")

            st = get_remote_job_status(device_creds=creds, job_id="job_123")
            self.assertEqual(st["status"], "completed")

            can = cancel_remote_job(device_creds=creds, job_id="job_123")
            self.assertEqual(can["status"], "cancelled")
        finally:
            srv.shutdown()
            srv.server_close()


if __name__ == "__main__":
    unittest.main()
