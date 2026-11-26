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
        self.assertEqual(swarm_node.validate_bind_host("100.94.164.64"), "100.94.164.64")
        self.assertEqual(swarm_node.validate_bind_host("0.0.0.0", allow_lan=True), "0.0.0.0")

    def test_is_tailscale_ip(self):
        self.assertTrue(swarm_node.is_tailscale_ip("100.64.0.1"))
        self.assertTrue(swarm_node.is_tailscale_ip("100.94.164.64"))
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
            stdout="100.94.164.64\n",
            stderr="",
        )
        with patch.object(swarm_node, "_find_tailscale_cli", return_value="/usr/local/bin/tailscale"):
            with patch("subprocess.run", return_value=mock_proc):
                self.assertEqual(swarm_node.detect_default_bind_host(), "100.94.164.64")


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


if __name__ == "__main__":
    unittest.main(verbosity=2)
