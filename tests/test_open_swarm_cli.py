#!/usr/bin/env python3
"""Unit tests for open_swarm/open_swarm_cli.py (v0.5.0)."""
from __future__ import annotations

import io
import json
import sys
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path

SWARM_DIR = Path(__file__).resolve().parents[1]
if str(SWARM_DIR) not in sys.path:
    sys.path.insert(0, str(SWARM_DIR))

from open_swarm_cli import CLI_VERSION, main, run_doctor, run_offline_benchmark  # noqa: E402


class OpenSwarmCLITests(unittest.TestCase):
    def test_doctor_reports_runtime_checks(self) -> None:
        rep = run_doctor(port=18799)
        self.assertTrue(rep["ok"])
        self.assertEqual(rep["version"], CLI_VERSION)
        names = {c["name"] for c in rep["checks"]}
        self.assertTrue({"python", "git", "os_sandbox", "tailscale", "device_registry", "desktop_server"}.issubset(names))

    def test_offline_benchmark_scorecard_passes(self) -> None:
        res = run_offline_benchmark()
        self.assertTrue(res["ok"], res)
        self.assertEqual(res["passed_count"], 2)
        self.assertEqual(res["applied_count"], 2)
        self.assertTrue(res["integration_gate_ok"])
        self.assertTrue(res["repair_recovery_ok"])
        self.assertGreaterEqual(res["board_messages"], 2)
        self.assertGreaterEqual(res["total_tool_calls"], 4)

    def test_cli_board_and_doctor_json_commands(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            db = Path(td) / "cli_board.sqlite3"
            buf = io.StringIO()
            with redirect_stdout(buf):
                rc = main(["board", "--run-id", "cli-r1", "--db", str(db), "--post-kind", "FINDING", "--body", "API contract ready", "--json"])
            self.assertEqual(rc, 0)
            posted = json.loads(buf.getvalue())
            self.assertEqual(posted["kind"], "FINDING")

            buf2 = io.StringIO()
            with redirect_stdout(buf2):
                rc2 = main(["board", "--run-id", "cli-r1", "--db", str(db), "--json"])
            self.assertEqual(rc2, 0)
            listed = json.loads(buf2.getvalue())
            self.assertEqual(len(listed["messages"]), 1)

            buf3 = io.StringIO()
            with redirect_stdout(buf3):
                rc3 = main(["doctor", "--json"])
            self.assertEqual(rc3, 0)
            doc = json.loads(buf3.getvalue())
            self.assertTrue(doc["ok"])


if __name__ == "__main__":
    unittest.main()
