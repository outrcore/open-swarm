"""Unit and end-to-end tests for open_swarm/swarm_node.py."""
from __future__ import annotations

import base64
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
import os
from pathlib import Path
import shutil
import stat
import subprocess
import sys
import tempfile
import threading
import unittest
from unittest.mock import patch
import urllib.error
import urllib.request

SWARM_DIR = Path(__file__).resolve().parent.parent
if str(SWARM_DIR) not in sys.path:
    sys.path.insert(0, str(SWARM_DIR))

import swarm_node  # noqa: E402


def _http_json(
    url: str,
    *,
    method: str = "GET",
    payload: dict | None = None,
    headers: dict[str, str] | None = None,
) -> tuple[int, dict]:
    data = json.dumps(payload).encode("utf-8") if payload is not None else None
    req_headers = dict(headers or {})
    if payload is not None:
        req_headers.setdefault("Content-Type", "application/json")
    req = urllib.request.Request(url, data=data, headers=req_headers, method=method)
    try:
        with urllib.request.urlopen(req, timeout=30) as resp:
            raw = resp.read().decode("utf-8")
            return resp.status, json.loads(raw)
    except urllib.error.HTTPError as err:
        raw = err.read().decode("utf-8")
        return err.code, json.loads(raw)


class MockOpenAIChatServer:
    """Minimal local OpenAI `/v1/chat/completions` mock server returning SSE or JSON."""

    def __init__(self, reply_content: str) -> None:
        self.reply_content = reply_content
        self.requests: list[dict] = []
        self._lock = threading.Lock()
        parent = self

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, *args):
                pass

            def do_POST(self):
                if self.path != "/v1/chat/completions":
                    self.send_response(404)
                    self.end_headers()
                    return
                n = int(self.headers.get("Content-Length") or 0)
                body = json.loads(self.rfile.read(n).decode("utf-8"))
                with parent._lock:
                    parent.requests.append(body)

                if body.get("stream"):
                    chunks = [
                        {
                            "id": "chatcmpl-node-e2e",
                            "object": "chat.completion.chunk",
                            "model": body.get("model", "mock"),
                            "choices": [
                                {
                                    "index": 0,
                                    "delta": {"role": "assistant", "content": parent.reply_content},
                                    "finish_reason": None,
                                }
                            ],
                        },
                        {
                            "id": "chatcmpl-node-e2e",
                            "object": "chat.completion.chunk",
                            "model": body.get("model", "mock"),
                            "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}],
                            "usage": {
                                "prompt_tokens": 50,
                                "completion_tokens": 30,
                                "prompt_tokens_details": {"cached_tokens": 0},
                            },
                        },
                    ]
                    payload = ("".join(f"data: {json.dumps(c)}\n\n" for c in chunks) + "data: [DONE]\n\n").encode("utf-8")
                    self.send_response(200)
                    self.send_header("Content-Type", "text/event-stream")
                    self.send_header("Content-Length", str(len(payload)))
                    self.end_headers()
                    self.wfile.write(payload)
                else:
                    resp_obj = {
                        "id": "chatcmpl-node-e2e",
                        "object": "chat.completion",
                        "model": body.get("model", "mock"),
                        "choices": [
                            {
                                "index": 0,
                                "message": {"role": "assistant", "content": parent.reply_content},
                                "finish_reason": "stop",
                            }
                        ],
                        "usage": {"prompt_tokens": 50, "completion_tokens": 30},
                    }
                    payload = json.dumps(resp_obj).encode("utf-8")
                    self.send_response(200)
                    self.send_header("Content-Type", "application/json")
                    self.send_header("Content-Length", str(len(payload)))
                    self.end_headers()
                    self.wfile.write(payload)

        self.srv = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        self.port = self.srv.socket.getsockname()[1]
        self.url = f"http://127.0.0.1:{self.port}/v1"
        self.thread = threading.Thread(target=self.srv.serve_forever, daemon=True)

    def __enter__(self) -> "MockOpenAIChatServer":
        self.thread.start()
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.srv.shutdown()
        self.srv.server_close()
        self.thread.join(timeout=5)


class TestSwarmNodeBindSafety(unittest.TestCase):
    def test_bind_safety_rejects_wildcard_and_lan_unless_allowed(self):
        with self.assertRaises(ValueError):
            swarm_node.validate_bind_host("0.0.0.0")
        with self.assertRaises(ValueError):
            swarm_node.validate_bind_host("::")
        with self.assertRaises(ValueError):
            swarm_node.validate_bind_host("192.168.1.50")

        self.assertEqual(swarm_node.validate_bind_host("127.0.0.1"), "127.0.0.1")
        self.assertEqual(swarm_node.validate_bind_host("100.100.10.10"), "100.100.10.10")
        self.assertEqual(swarm_node.validate_bind_host("0.0.0.0", allow_lan=True), "0.0.0.0")

    def test_is_tailscale_ip(self):
        self.assertTrue(swarm_node.is_tailscale_ip("100.64.0.1"))
        self.assertTrue(swarm_node.is_tailscale_ip("100.100.10.10"))
        self.assertTrue(swarm_node.is_tailscale_ip("100.127.255.254"))
        self.assertFalse(swarm_node.is_tailscale_ip("100.63.255.255"))
        self.assertFalse(swarm_node.is_tailscale_ip("100.128.0.1"))
        self.assertFalse(swarm_node.is_tailscale_ip("192.168.1.1"))
        self.assertFalse(swarm_node.is_tailscale_ip("not-an-ip"))

    def test_detect_default_bind_host_fallback_and_tailscale(self):
        with patch.object(swarm_node, "_find_tailscale_cli", return_value=None):
            self.assertEqual(swarm_node.detect_default_bind_host(), "127.0.0.1")

        mock_proc = subprocess.CompletedProcess(
            args=["tailscale", "ip", "-4"],
            returncode=0,
            stdout="100.100.10.10\n",
            stderr="",
        )
        with patch.object(swarm_node, "_find_tailscale_cli", return_value="/usr/local/bin/tailscale"):
            with patch("subprocess.run", return_value=mock_proc):
                self.assertEqual(swarm_node.detect_default_bind_host(), "100.100.10.10")


class TestSwarmNodePairingAndAuth(unittest.TestCase):
    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp(prefix="swarm_node_test_"))
        self.state_path = self.tmp / "node_state.json"
        self.server = swarm_node.SwarmNodeServer(
            host="127.0.0.1",
            port=0,
            state_path=self.state_path,
            node_name="test-worker",
        )
        self.port = self.server.start()
        self.base = f"http://127.0.0.1:{self.port}"

    def tearDown(self):
        self.server.stop()
        shutil.rmtree(self.tmp, ignore_errors=True)

    def test_state_file_permissions_and_hash_only_storage(self):
        mode = stat.S_IMODE(self.state_path.stat().st_mode)
        self.assertEqual(mode, 0o600)
        code = self.server.state.generate_pair_code()
        status, body = _http_json(
            f"{self.base}/node/pair",
            method="POST",
            payload={"pair_code": code, "client_label": "mac-studio"},
        )
        self.assertEqual(status, 200)
        raw_token = body["device_token"]
        self.assertTrue(raw_token.startswith("osw_dev_"))
        disk_text = self.state_path.read_text(encoding="utf-8")
        self.assertNotIn(raw_token, disk_text)

    def test_one_time_pairing_and_lockout(self):
        code = self.server.state.generate_pair_code(ttl_s=300)
        self.assertEqual(len(code), 6)
        self.assertTrue(code.isdigit())

        wrong_code = "000000" if code != "000000" else "999999"

        # Wrong code -> 403
        st, resp = _http_json(
            f"{self.base}/node/pair",
            method="POST",
            payload={"pair_code": wrong_code, "client_label": "c1"},
        )
        self.assertEqual(st, 403)
        self.assertFalse(resp["ok"])

        # Valid code -> 200 with device_token + fingerprint
        st, resp = _http_json(
            f"{self.base}/node/pair",
            method="POST",
            payload={"pair_code": code, "client_label": "c1"},
        )
        self.assertEqual(st, 200)
        self.assertTrue(resp["ok"])
        self.assertTrue(resp["device_token"].startswith("osw_dev_"))
        self.assertEqual(resp["fingerprint"], self.server.state.fingerprint)

        # Reusing the same code -> 403
        st, resp = _http_json(
            f"{self.base}/node/pair",
            method="POST",
            payload={"pair_code": code, "client_label": "c2"},
        )
        self.assertEqual(st, 403)
        self.assertFalse(resp["ok"])

        # 5 bad guesses invalidate the active code
        code2 = self.server.state.generate_pair_code(ttl_s=300)
        wrong_code2 = "111111" if code2 != "111111" else "222222"
        for _ in range(5):
            st, resp = _http_json(
                f"{self.base}/node/pair",
                method="POST",
                payload={"pair_code": wrong_code2, "client_label": "attacker"},
            )
            self.assertEqual(st, 403)

        # Even the previously valid code is now invalidated -> 403
        st, resp = _http_json(
            f"{self.base}/node/pair",
            method="POST",
            payload={"pair_code": code2, "client_label": "legit"},
        )
        self.assertEqual(st, 403)
        self.assertFalse(resp["ok"])

    def test_auth_enforcement_health_and_exec(self):
        # No token -> 401
        st, _ = _http_json(f"{self.base}/node/health", method="GET")
        self.assertEqual(st, 401)
        st, _ = _http_json(f"{self.base}/node/exec", method="POST", payload={})
        self.assertEqual(st, 401)

        # Wrong token -> 401
        bad_headers = {"Authorization": "Bearer osw_dev_invalid_token"}
        st, _ = _http_json(f"{self.base}/node/health", method="GET", headers=bad_headers)
        self.assertEqual(st, 401)
        st, _ = _http_json(f"{self.base}/node/exec", method="POST", payload={}, headers=bad_headers)
        self.assertEqual(st, 401)

        # Pair to obtain valid bearer token
        code = self.server.state.generate_pair_code()
        st, pair_body = _http_json(
            f"{self.base}/node/pair",
            method="POST",
            payload={"pair_code": code, "client_label": "controller"},
        )
        self.assertEqual(st, 200)
        token = pair_body["device_token"]

        # Valid bearer token -> 200 with live capabilities & fingerprint
        st, health = _http_json(
            f"{self.base}/node/health",
            method="GET",
            headers={"Authorization": f"Bearer {token}"},
        )
        self.assertEqual(st, 200)
        self.assertTrue(health["ok"])
        self.assertEqual(health["fingerprint"], self.server.state.fingerprint)
        self.assertEqual(health["node_id"], self.server.state.node_id)
        self.assertIn("python", health["capabilities"])
        self.assertIsInstance(health["cpu_cores"], int)


class TestSwarmNodeRemoteBundleE2E(unittest.TestCase):
    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp(prefix="swarm_node_e2e_"))
        self.repo = self.tmp / "origin_repo"
        self.repo.mkdir()
        subprocess.run(["git", "init"], cwd=self.repo, check=True, capture_output=True)
        subprocess.run(["git", "config", "user.name", "Tester"], cwd=self.repo, check=True, capture_output=True)
        subprocess.run(["git", "config", "user.email", "tester@example.com"], cwd=self.repo, check=True, capture_output=True)
        (self.repo / "README.md").write_text("# Remote Bundle Repo\n", encoding="utf-8")
        subprocess.run(["git", "add", "README.md"], cwd=self.repo, check=True, capture_output=True)
        subprocess.run(["git", "commit", "-m", "initial commit"], cwd=self.repo, check=True, capture_output=True)
        self.base_commit = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=self.repo,
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()

        self.bundle_path = self.tmp / "repo.bundle"
        subprocess.run(
            ["git", "bundle", "create", str(self.bundle_path), "HEAD"],
            cwd=self.repo,
            check=True,
            capture_output=True,
        )
        self.bundle_b64 = base64.b64encode(self.bundle_path.read_bytes()).decode("ascii")

    def tearDown(self):
        shutil.rmtree(self.tmp, ignore_errors=True)

    def test_remote_git_bundle_exec_and_clean_diff_apply(self):
        pkg_code = (
            "```file:pkg_remote.py\n"
            "import unittest\n\n\n"
            "def multiply(a: int, b: int) -> int:\n"
            "    return a * b\n\n\n"
            "class TestPkgRemote(unittest.TestCase):\n"
            "    def test_multiply(self) -> None:\n"
            "        self.assertEqual(multiply(6, 7), 42)\n\n\n"
            "if __name__ == '__main__':\n"
            "    unittest.main()\n"
            "```\n"
        )

        with MockOpenAIChatServer(pkg_code) as mock_llm:
            state_path = self.tmp / "node_state.json"
            node = swarm_node.SwarmNodeServer(
                host="127.0.0.1",
                port=0,
                state_path=state_path,
                inference_url=mock_llm.url,
                node_name="remote-worker-1",
            )
            port = node.start()
            try:
                base = f"http://127.0.0.1:{port}"
                code = node.state.generate_pair_code()
                st, pair_resp = _http_json(
                    f"{base}/node/pair",
                    method="POST",
                    payload={"pair_code": code, "client_label": "orchestrator"},
                )
                self.assertEqual(st, 200)
                token = pair_resp["device_token"]

                st, exec_resp = _http_json(
                    f"{base}/node/exec",
                    method="POST",
                    headers={"Authorization": f"Bearer {token}"},
                    payload={
                        "bundle_b64": self.bundle_b64,
                        "base_commit": self.base_commit,
                        "task": {
                            "id": "T-REMOTE-1",
                            "title": "Create pkg_remote.py",
                            "spec": "Write pkg_remote.py with multiply(a, b) and a unittest test case.",
                            "acceptance_cmd": "python3 -m unittest pkg_remote.py",
                        },
                        "model": "mock-coder",
                        "api_key": "test-key",
                        "spend_cap_usd": 1.0,
                        "max_repair_turns": 1,
                        "auto_review": False,
                    },
                )
                self.assertEqual(st, 200, exec_resp)
                self.assertTrue(exec_resp["ok"])
                res = exec_resp["result"]
                self.assertTrue(res["tests_ok"], res.get("test_output"))
                self.assertEqual(res["written_files"], ["pkg_remote.py"])
                self.assertEqual(res["executed_on"], "remote-worker-1")
                self.assertEqual(res["node_id"], node.state.node_id)
                self.assertIn("metering", exec_resp)
                self.assertTrue(res["diff"].strip())

                # Verify unified diff applies cleanly onto the original repo
                patch_file = self.tmp / "remote.patch"
                patch_file.write_text(res["diff"], encoding="utf-8")
                chk = subprocess.run(
                    ["git", "apply", "--check", str(patch_file)],
                    cwd=self.repo,
                    capture_output=True,
                    text=True,
                )
                self.assertEqual(chk.returncode, 0, chk.stderr)
                self.assertEqual(node.active_agents, 0)
            finally:
                node.stop()


class TestSwarmNodePhase2D(unittest.TestCase):
    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp(prefix="swarm_node_2d_"))
        self.state_path = self.tmp / "node_state.json"
        self.mirrors_dir = self.tmp / "mirrors"
        self.repo = self.tmp / "origin_repo"
        self.repo.mkdir()
        subprocess.run(["git", "init"], cwd=self.repo, check=True, capture_output=True)
        subprocess.run(["git", "config", "user.name", "Tester"], cwd=self.repo, check=True)
        subprocess.run(["git", "config", "user.email", "tester@example.com"], cwd=self.repo, check=True)
        (self.repo / "base.txt").write_text("v1\n", encoding="utf-8")
        subprocess.run(["git", "add", "base.txt"], cwd=self.repo, check=True)
        subprocess.run(["git", "commit", "-m", "c1"], cwd=self.repo, check=True, capture_output=True)
        self.commit1 = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=self.repo,
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
        full_bundle = self.tmp / "full.bundle"
        subprocess.run(
            ["git", "bundle", "create", str(full_bundle), "HEAD"],
            cwd=self.repo,
            check=True,
            capture_output=True,
        )
        self.full_b64 = base64.b64encode(full_bundle.read_bytes()).decode("ascii")

    def tearDown(self):
        shutil.rmtree(self.tmp, ignore_errors=True)

    def _pair(self, base: str, server: swarm_node.SwarmNodeServer) -> str:
        code = server.state.generate_pair_code()
        st, resp = _http_json(
            f"{base}/node/pair",
            method="POST",
            payload={"pair_code": code, "client_label": "test-client"},
        )
        self.assertEqual(st, 200)
        return resp["device_token"]

    def test_live_telemetry_and_health_v040(self):
        fake_smi = subprocess.CompletedProcess(
            args=["nvidia-smi"],
            returncode=0,
            stdout="55.5, 2048, 16384\n",
            stderr="",
        )
        with (
            patch.object(swarm_node.shutil, "which", return_value="/usr/bin/nvidia-smi"),
            patch.object(swarm_node.os.path, "isfile", return_value=True),
            patch.object(swarm_node.subprocess, "run", return_value=fake_smi),
            patch.object(swarm_node.os, "getloadavg", return_value=(2.0, 1.5, 1.0)),
            patch.object(swarm_node.os, "cpu_count", return_value=8),
            patch.object(swarm_node, "_probe_mem_avail_gb", return_value=14.25),
        ):
            telem = swarm_node.probe_live_telemetry()

        self.assertEqual(telem["gpu_util_pct"], 55.5)
        self.assertEqual(telem["gpu_mem_used_mb"], 2048)
        self.assertEqual(telem["gpu_mem_total_mb"], 16384)
        self.assertEqual(telem["mem_avail_gb"], 14.25)
        self.assertEqual(telem["cpu_load_pct"], 25.0)

        with swarm_node.SwarmNodeServer(
            host="127.0.0.1",
            port=0,
            state_path=self.state_path,
            mirrors_dir=self.mirrors_dir,
            node_name="telem-node",
        ) as srv:
            base = f"http://127.0.0.1:{srv.port}"
            token = self._pair(base, srv)
            st, health = _http_json(
                f"{base}/node/health",
                method="GET",
                headers={"Authorization": f"Bearer {token}"},
            )
            self.assertEqual(st, 200)
            self.assertEqual(health["version"], "0.5.1")
            self.assertIsInstance(health["capabilities"], list)
            self.assertIn("telemetry", health)
            self.assertIsInstance(health["mem_avail_gb"], float)
            self.assertIn("gpu_util_pct", health["telemetry"])
            self.assertIn("cpu_load_pct", health["telemetry"])

    def test_allow_lan_warning_and_git_hook_hardening(self):
        with self.assertWarns(RuntimeWarning) as cm:
            res = swarm_node.validate_bind_host("0.0.0.0", allow_lan=True)
        self.assertEqual(res, "0.0.0.0")
        self.assertIn("SECURITY WARNING", str(cm.warning))
        self.assertIn("non-Tailscale", str(cm.warning))

        # Verify _git_run enforces core.hooksPath=/dev/null and core.fsmonitor=false
        calls = []
        orig_run = subprocess.run

        def _spy_run(cmd, *args, **kwargs):
            calls.append(list(cmd))
            return orig_run(cmd, *args, **kwargs)

        with patch.object(swarm_node.subprocess, "run", side_effect=_spy_run):
            p = swarm_node._git_run(self.repo, "status", "--short")
            self.assertEqual(p.returncode, 0)

        self.assertTrue(calls)
        git_cmd = calls[0]
        self.assertEqual(git_cmd[:5], ["git", "-c", "core.hooksPath=/dev/null", "-c", "core.fsmonitor=false"])

    def test_incremental_bare_mirror_and_scoped_llm_proxy(self):
        pkg_code = (
            "```file:check_ver.py\n"
            "import unittest\n"
            "from pathlib import Path\n\n"
            "class TestVer(unittest.TestCase):\n"
            "    def test_v2(self):\n"
            "        self.assertEqual(Path('base.txt').read_text().strip(), 'v2')\n\n"
            "if __name__ == '__main__':\n"
            "    unittest.main()\n"
            "```\n"
        )

        # Create commit 2 and an incremental bundle commit1..HEAD
        (self.repo / "base.txt").write_text("v2\n", encoding="utf-8")
        subprocess.run(["git", "add", "base.txt"], cwd=self.repo, check=True)
        subprocess.run(["git", "commit", "-m", "c2"], cwd=self.repo, check=True, capture_output=True)
        commit2 = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=self.repo,
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
        inc_bundle = self.tmp / "inc.bundle"
        subprocess.run(
            ["git", "bundle", "create", str(inc_bundle), f"{self.commit1}..HEAD"],
            cwd=self.repo,
            check=True,
            capture_output=True,
        )
        inc_b64 = base64.b64encode(inc_bundle.read_bytes()).decode("ascii")

        seen_auth_headers: list[str] = []

        with MockOpenAIChatServer(pkg_code) as mock_llm:
            # Wrap mock handler to record Authorization header sent to LLM
            orig_do_post = mock_llm.srv.RequestHandlerClass.do_POST

            def _wrapped_post(handler_self):
                seen_auth_headers.append(handler_self.headers.get("Authorization") or "")
                return orig_do_post(handler_self)

            mock_llm.srv.RequestHandlerClass.do_POST = _wrapped_post

            with swarm_node.SwarmNodeServer(
                host="127.0.0.1",
                port=0,
                state_path=self.state_path,
                mirrors_dir=self.mirrors_dir,
                node_name="mirror-node",
            ) as srv:
                base = f"http://127.0.0.1:{srv.port}"
                token = self._pair(base, srv)
                auth = {"Authorization": f"Bearer {token}"}

                # 1. Incremental before mirror seeded -> 409 mirror_miss
                st_miss, resp_miss = _http_json(
                    f"{base}/node/exec",
                    method="POST",
                    headers=auth,
                    payload={
                        "bundle_b64": inc_b64,
                        "base_commit": commit2,
                        "repo_key": "org/demo-repo",
                        "incremental": True,
                        "llm_proxy_url": mock_llm.url,
                        "llm_proxy_token": "scoped-tok-abc",
                        "task": {
                            "id": "T-MISS",
                            "title": "Check v2",
                            "spec": "Write check_ver.py",
                            "acceptance_cmd": "python3 -m unittest check_ver.py",
                        },
                        "model": "mock-coder",
                    },
                )
                self.assertEqual(st_miss, 409)
                self.assertTrue(resp_miss.get("mirror_miss"))

                # 2. Seed bare mirror with full bundle
                st_seed, resp_seed = _http_json(
                    f"{base}/node/exec",
                    method="POST",
                    headers=auth,
                    payload={
                        "bundle_b64": self.full_b64,
                        "base_commit": self.commit1,
                        "repo_key": "org/demo-repo",
                        "incremental": False,
                        "llm_proxy_url": mock_llm.url,
                        "llm_proxy_token": "scoped-tok-seed",
                        "task": {
                            "id": "T-SEED",
                            "title": "Echo test",
                            "spec": "Write check_ver.py",
                            "acceptance_cmd": "python3 -c \"from pathlib import Path; assert Path('base.txt').read_text().strip() == 'v1'\"",
                        },
                        "model": "mock-coder",
                    },
                )
                self.assertEqual(st_seed, 200, resp_seed)
                self.assertTrue((self.mirrors_dir / "org_demo-repo.git" / "HEAD").is_file())

                # 3. Incremental delta bundle updates mirror and sees base.txt == 'v2' + uses scoped proxy token
                st_inc, resp_inc = _http_json(
                    f"{base}/node/exec",
                    method="POST",
                    headers=auth,
                    payload={
                        "bundle_b64": inc_b64,
                        "base_commit": commit2,
                        "repo_key": "org/demo-repo",
                        "incremental": True,
                        "llm_proxy_url": mock_llm.url,
                        "llm_proxy_token": "scoped-tok-xyz",
                        "task": {
                            "id": "T-INC",
                            "title": "Verify v2",
                            "spec": "Write check_ver.py checking base.txt == v2",
                            "acceptance_cmd": "python3 -m unittest check_ver.py",
                        },
                        "model": "mock-coder",
                    },
                )
                self.assertEqual(st_inc, 200, resp_inc)
                self.assertTrue(resp_inc["result"]["tests_ok"])
                self.assertIn("Bearer scoped-tok-xyz", seen_auth_headers)

    def test_async_jobs_sse_stream_and_cancel(self):
        pkg_code = (
            "```file:job_mod.py\n"
            "import unittest\n\n"
            "def add(a, b):\n"
            "    return a + b\n\n"
            "class T(unittest.TestCase):\n"
            "    def test_add(self):\n"
            "        self.assertEqual(add(2, 3), 5)\n\n"
            "if __name__ == '__main__':\n"
            "    unittest.main()\n"
            "```\n"
        )

        with MockOpenAIChatServer(pkg_code) as mock_llm:
            with swarm_node.SwarmNodeServer(
                host="127.0.0.1",
                port=0,
                state_path=self.state_path,
                mirrors_dir=self.mirrors_dir,
                inference_url=mock_llm.url,
                node_name="async-node",
            ) as srv:
                base = f"http://127.0.0.1:{srv.port}"
                token = self._pair(base, srv)
                auth = {"Authorization": f"Bearer {token}"}

                # Unauthorized -> 401
                st_unauth, _ = _http_json(f"{base}/node/jobs", method="POST", payload={})
                self.assertEqual(st_unauth, 401)

                # Submit async job -> 202 queued
                st_sub, sub_body = _http_json(
                    f"{base}/node/jobs",
                    method="POST",
                    headers=auth,
                    payload={
                        "bundle_b64": self.full_b64,
                        "base_commit": self.commit1,
                        "task": {
                            "id": "T-ASYNC-1",
                            "title": "Write job_mod.py",
                            "spec": "Create job_mod.py with add(a, b).",
                            "acceptance_cmd": "python3 -m unittest job_mod.py",
                        },
                        "model": "mock-coder",
                        "api_key": "test-key",
                    },
                )
                self.assertEqual(st_sub, 202, sub_body)
                job_id = sub_body["job_id"]
                self.assertEqual(sub_body["status"], "queued")

                # Consume SSE stream until EOF (terminates after event: done)
                req = urllib.request.Request(
                    f"{base}/node/jobs/{job_id}/stream",
                    headers={**auth, "Accept": "text/event-stream"},
                    method="GET",
                )
                with urllib.request.urlopen(req, timeout=30) as resp:
                    self.assertIn("text/event-stream", resp.headers.get("Content-Type", ""))
                    raw_sse = resp.read().decode("utf-8")

                self.assertIn("event: queued", raw_sse)
                self.assertIn("event: done", raw_sse)

                st_get, job_state = _http_json(
                    f"{base}/node/jobs/{job_id}",
                    method="GET",
                    headers=auth,
                )
                self.assertEqual(st_get, 200)
                self.assertEqual(job_state["status"], "completed")
                self.assertTrue(job_state["result"]["tests_ok"])
                self.assertGreaterEqual(len(job_state["events"]), 2)

                # Test cancellation on a blocked LLM call
                gate = threading.Event()
                orig_do_post = mock_llm.srv.RequestHandlerClass.do_POST

                def _slow_post(handler_self):
                    gate.wait(timeout=5.0)
                    return orig_do_post(handler_self)

                mock_llm.srv.RequestHandlerClass.do_POST = _slow_post
                try:
                    st_csub, csub = _http_json(
                        f"{base}/node/jobs",
                        method="POST",
                        headers=auth,
                        payload={
                            "bundle_b64": self.full_b64,
                            "base_commit": self.commit1,
                            "task": {
                                "id": "T-CANCEL-1",
                                "title": "Slow job",
                                "spec": "Write job_mod.py",
                                "acceptance_cmd": "python3 -m unittest job_mod.py",
                            },
                            "model": "mock-coder",
                        },
                    )
                    self.assertEqual(st_csub, 202)
                    cjob_id = csub["job_id"]

                    st_can, can_body = _http_json(
                        f"{base}/node/jobs/{cjob_id}/cancel",
                        method="POST",
                        headers=auth,
                        payload={},
                    )
                    self.assertEqual(st_can, 200)
                    self.assertEqual(can_body["status"], "cancelled")

                    req2 = urllib.request.Request(
                        f"{base}/node/jobs/{cjob_id}/stream",
                        headers={**auth, "Accept": "text/event-stream"},
                        method="GET",
                    )
                    with urllib.request.urlopen(req2, timeout=15) as resp2:
                        sse2 = resp2.read().decode("utf-8")
                    self.assertIn("event: cancelled", sse2)
                    self.assertIn("event: done", sse2)
                finally:
                    gate.set()
                    mock_llm.srv.RequestHandlerClass.do_POST = orig_do_post


if __name__ == "__main__":
    unittest.main(verbosity=2)
