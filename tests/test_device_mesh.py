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
    discover_tailscale_peers,
    execute_task_on_remote_device,
    find_tailscale_bin,
    infer_required_capabilities,
    probe_local_device,
    route_tasks_to_devices,
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


if __name__ == "__main__":
    unittest.main()
