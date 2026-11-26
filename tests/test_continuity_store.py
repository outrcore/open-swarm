"""Unit tests for open_swarm.continuity_store."""

from __future__ import annotations

import hashlib
import os
import sqlite3
import sys
import tempfile
import time
import unittest
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from open_swarm.continuity_store import (
    ContextConflictError,
    ContinuityStore,
    compute_file_hashes,
    extract_diff_paths,
)


class ContinuityStoreTests(unittest.TestCase):
    def setUp(self) -> None:
        self._tmpdir = tempfile.TemporaryDirectory()
        self.root = Path(self._tmpdir.name)
        self.repo = self.root / "repo"
        self.repo.mkdir()
        self.db_path = self.root / "state" / "continuity.db"
        self.store = ContinuityStore(self.db_path)

    def tearDown(self) -> None:
        self._tmpdir.cleanup()

    def test_sqlite_wal_mode_and_tables_initialized(self) -> None:
        self.assertTrue(self.db_path.exists())
        conn = sqlite3.connect(str(self.db_path), timeout=10.0)
        try:
            mode = conn.execute("PRAGMA journal_mode").fetchone()[0]
            self.assertEqual(mode.lower(), "wal")
            tables = {
                row[0]
                for row in conn.execute(
                    "SELECT name FROM sqlite_master WHERE type='table'"
                ).fetchall()
            }
            self.assertTrue(
                {"checkpoints", "action_journal", "messages"}.issubset(tables)
            )
        finally:
            conn.close()

    def test_checkpoint_revision_increment_and_conflict(self) -> None:
        cp1 = self.store.publish_checkpoint(
            project_key="proj-a",
            role="planner",
            summary="Initial plan",
            next_step="Implement store",
            actor="agent-1",
            expected_revision=0,
            file_hashes={"src/main.py": "abc123"},
        )
        self.assertEqual(cp1["revision"], 1)
        self.assertEqual(cp1["summary"], "Initial plan")
        self.assertEqual(cp1["next_step"], "Implement store")
        self.assertEqual(cp1["actor"], "agent-1")
        self.assertEqual(cp1["file_hashes"], {"src/main.py": "abc123"})
        self.assertEqual(len(cp1["context_guard"]), 64)

        cp2 = self.store.publish_checkpoint(
            project_key="proj-a",
            role="planner",
            summary="Updated plan",
            expected_revision=1,
        )
        self.assertEqual(cp2["revision"], 2)

        with self.assertRaises(ContextConflictError) as ctx:
            self.store.publish_checkpoint(
                project_key="proj-a",
                role="planner",
                summary="Stale write",
                expected_revision=1,
            )
        self.assertIsInstance(ctx.exception, RuntimeError)
        self.assertEqual(ctx.exception.conflicts[0]["actual_revision"], 2)
        self.assertEqual(ctx.exception.conflicts[0]["expected_revision"], 1)

        # Ensure stale write did not insert a new revision
        time.sleep(0.01)
        cp_worker = self.store.publish_checkpoint(
            project_key="proj-a",
            role="worker",
            summary="Worker ready",
        )
        self.assertEqual(cp_worker["revision"], 1)

        latest = self.store.get_latest_checkpoints("proj-a")
        self.assertEqual([c["role"] for c in latest], ["worker", "planner"])
        self.assertEqual(latest[1]["revision"], 2)

    def test_compute_file_hashes_sha256_missing_and_traversal_rejection(self) -> None:
        outside_file = self.root / "secret.txt"
        outside_file.write_text("top secret", encoding="utf-8")

        target_file = self.repo / "sub" / "hello.txt"
        target_file.parent.mkdir(parents=True)
        payload = b"hello continuity\n"
        target_file.write_bytes(payload)
        expected_sha = hashlib.sha256(payload).hexdigest()

        symlink_escape = self.repo / "escape_link.txt"
        os.symlink(outside_file, symlink_escape)

        hashes = compute_file_hashes(
            self.repo,
            [
                "sub/hello.txt",
                "missing.txt",
                "../secret.txt",
                "sub/../../secret.txt",
                str(outside_file),
                "escape_link.txt",
            ],
        )

        self.assertEqual(
            hashes,
            {
                "sub/hello.txt": expected_sha,
                "missing.txt": None,
            },
        )
        self.assertNotIn("../secret.txt", hashes)
        self.assertNotIn("sub/../../secret.txt", hashes)
        self.assertNotIn(str(outside_file), hashes)
        self.assertNotIn("escape_link.txt", hashes)

    def test_extract_diff_paths_parsing_and_dedup(self) -> None:
        diff = """\
diff --git a/open_swarm/a.py b/open_swarm/a.py
index 1111111..2222222 100644
--- a/open_swarm/a.py
+++ b/open_swarm/a.py
@@ -1 +1 @@
-old
+new
diff --git a/open_swarm/new.py b/open_swarm/new.py
new file mode 100644
--- /dev/null
+++ b/open_swarm/new.py
@@ -0,0 +1 @@
+created
diff --git a/open_swarm/deleted.py b/open_swarm/deleted.py
deleted file mode 100644
--- a/open_swarm/deleted.py
+++ /dev/null
@@ -1 +0,0 @@
-gone
diff --git a/open_swarm/old_name.py b/open_swarm/new_name.py
rename from open_swarm/old_name.py
rename to open_swarm/new_name.py
--- a/open_swarm/old_name.py
+++ b/open_swarm/new_name.py
"""
        paths = extract_diff_paths(diff)
        self.assertEqual(
            paths,
            [
                "open_swarm/a.py",
                "open_swarm/new.py",
                "open_swarm/deleted.py",
                "open_swarm/old_name.py",
                "open_swarm/new_name.py",
            ],
        )

    def test_verify_context_guard_untouched_and_modified(self) -> None:
        f1 = self.repo / "alpha.py"
        f1.write_text("v1", encoding="utf-8")

        expected = compute_file_hashes(self.repo, ["alpha.py", "beta.py"])
        ok_res = self.store.verify_context_guard(self.repo, expected)
        self.assertTrue(ok_res["ok"])
        self.assertEqual(ok_res["conflicts"], [])
        self.assertEqual(ok_res["checked_count"], 2)

        # Modify alpha.py externally and create beta.py
        f1.write_text("v2", encoding="utf-8")
        (self.repo / "beta.py").write_text("new", encoding="utf-8")

        bad_res = self.store.verify_context_guard(self.repo, expected)
        self.assertFalse(bad_res["ok"])
        self.assertEqual(bad_res["checked_count"], 2)
        self.assertEqual(len(bad_res["conflicts"]), 2)
        conflict_paths = {c["path"] for c in bad_res["conflicts"]}
        self.assertEqual(conflict_paths, {"alpha.py", "beta.py"})

    def test_record_action_validation_and_evidence_hashing(self) -> None:
        with self.assertRaises(ValueError):
            self.store.record_action(
                project_key="proj-a",
                role="tester",
                action="run_tests",
                status="pending",
            )

        artifact = self.repo / "output.log"
        artifact.write_bytes(b"all green")
        expected_sha = hashlib.sha256(b"all green").hexdigest()

        a1 = self.store.record_action(
            project_key="proj-a",
            role="worker",
            action="write_output",
            status="completed",
            detail="wrote output.log",
            run_id="run-42",
            evidence=[
                {"kind": "file", "path": "output.log"},
                {"kind": "file", "path": "missing.log"},
                {"kind": "command", "cmd": "pytest"},
            ],
            repo=self.repo,
        )
        self.assertEqual(a1["status"], "completed")
        self.assertEqual(a1["evidence"][0]["sha256"], expected_sha)
        self.assertIsNone(a1["evidence"][1]["sha256"])
        self.assertNotIn("sha256", a1["evidence"][2])

        a2 = self.store.record_action(
            project_key="proj-a",
            role="worker",
            action="check_env",
            status="failed",
        )
        a3 = self.store.record_action(
            project_key="proj-a",
            role="worker",
            action="probe",
            status="unknown",
        )
        self.assertEqual(a2["status"], "failed")
        self.assertEqual(a3["status"], "unknown")

        listed = self.store.list_actions("proj-a", limit=2)
        self.assertEqual(len(listed), 2)
        self.assertEqual([a["id"] for a in listed], [a3["id"], a2["id"]])

    def test_advisory_messaging_cursor_and_project_status(self) -> None:
        m1 = self.store.send_message("proj-a", "planner", "worker", "Start task 1")
        m2 = self.store.send_message("proj-a", "planner", "reviewer", "Heads up")
        m3 = self.store.send_message("proj-a", "worker", "all", "Broadcast update")
        m4 = self.store.send_message("proj-a", "worker", "*", "Wildcard ping")

        worker_msgs = self.store.read_messages("proj-a", to_role="worker")
        self.assertEqual(
            [m["id"] for m in worker_msgs], [m1["id"], m3["id"], m4["id"]]
        )

        paged = self.store.read_messages("proj-a", to_role="worker", after_id=m1["id"])
        self.assertEqual([m["id"] for m in paged], [m3["id"], m4["id"]])

        all_msgs = self.store.read_messages("proj-a")
        self.assertEqual(
            [m["id"] for m in all_msgs], [m1["id"], m2["id"], m3["id"], m4["id"]]
        )

        self.store.publish_checkpoint("proj-a", "planner", "Summary")
        self.store.record_action("proj-a", "worker", "build", "completed")

        status = self.store.project_status("proj-a")
        self.assertEqual(status["project_key"], "proj-a")
        self.assertEqual(len(status["checkpoints"]), 1)
        self.assertEqual(len(status["actions"]), 1)
        self.assertEqual(len(status["messages"]), 4)


if __name__ == "__main__":
    unittest.main()
