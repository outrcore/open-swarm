"""Unit tests for open_swarm.continuity_store."""

from __future__ import annotations

import hashlib
import os
import sqlite3
import sys
import tempfile
import threading
import time
import unittest
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from open_swarm.continuity_store import (
    VALID_BOARD_KINDS,
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

    def test_board_schema_tables_and_index_initialized(self) -> None:
        conn = sqlite3.connect(str(self.db_path), timeout=10.0)
        try:
            tables = {
                row[0]
                for row in conn.execute(
                    "SELECT name FROM sqlite_master WHERE type='table'"
                ).fetchall()
            }
            self.assertTrue({"board_messages", "file_leases"}.issubset(tables))
            indexes = {
                row[0]
                for row in conn.execute(
                    "SELECT name FROM sqlite_master WHERE type='index'"
                ).fetchall()
            }
            self.assertIn("idx_board_messages_run_seq", indexes)
        finally:
            conn.close()

    def test_board_validation_post_read_and_filtering(self) -> None:
        self.assertEqual(
            VALID_BOARD_KINDS,
            (
                "CLAIM_FILE",
                "CONTRACT",
                "FINDING",
                "QUESTION",
                "NEED_DECISION",
                "BLOCKED",
                "DONE",
                "DECISION",
            ),
        )

        with self.assertRaises(ValueError):
            self.store.post_board_message("run-1", "agent-a", "UNKNOWN", "bad")
        with self.assertRaises(ValueError):
            self.store.post_board_message("", "agent-a", "FINDING", "missing run")
        with self.assertRaises(ValueError):
            self.store.post_board_message("run-1", "  ", "FINDING", "missing sender")

        m1 = self.store.post_board_message(
            "run-1",
            "planner",
            "CONTRACT",
            "API frozen",
            meta={"version": 1},
        )
        m2 = self.store.post_board_message(
            "run-1",
            "coder-1",
            "QUESTION",
            "Which port?",
            to_agent="planner",
        )
        m3 = self.store.post_board_message(
            "run-1",
            "coder-2",
            "FINDING",
            "Found race condition",
            to_agent="reviewer",
        )
        m4 = self.store.post_board_message(
            "run-2",
            "other",
            "DONE",
            "Unrelated run",
        )

        self.assertEqual(m1["to_agent"], "all")
        self.assertEqual(m1["meta"], {"version": 1})
        self.assertFalse(m1["resolved"])

        # Direct to planner + broadcast ('all'), excluding reviewer-only m3 and run-2 m4
        planner_msgs = self.store.read_board_messages("run-1", to_agent="planner")
        self.assertEqual([m["seq"] for m in planner_msgs], [m1["seq"], m2["seq"]])

        # since_seq paging
        paged = self.store.read_board_messages(
            "run-1", since_seq=m1["seq"], to_agent="planner"
        )
        self.assertEqual([m["seq"] for m in paged], [m2["seq"]])

        # to_agent="all" or None returns all run-1 messages
        all_run1 = self.store.read_board_messages("run-1", to_agent="all")
        self.assertEqual(
            [m["seq"] for m in all_run1], [m1["seq"], m2["seq"], m3["seq"]]
        )
        self.assertNotIn(m4["seq"], [m["seq"] for m in all_run1])

        # kind filter and limit
        findings = self.store.read_board_messages("run-1", kind="FINDING")
        self.assertEqual([m["seq"] for m in findings], [m3["seq"]])
        limited = self.store.read_board_messages("run-1", limit=2)
        self.assertEqual([m["seq"] for m in limited], [m1["seq"], m2["seq"]])

    def test_resolve_board_decision(self) -> None:
        q = self.store.post_board_message(
            "run-dec",
            "coder-1",
            "NEED_DECISION",
            "Use WAL or DELETE journal mode?",
            to_agent="lead",
        )
        resolved = self.store.resolve_board_decision(
            q["seq"], "Use WAL mode", resolver="lead"
        )
        self.assertTrue(resolved["resolved"])
        self.assertEqual(resolved["resolution"], "Use WAL mode")
        self.assertEqual(resolved["resolved_by"], "lead")
        self.assertIsNotNone(resolved["resolved_at"])

        coder_inbox = self.store.read_board_messages(
            "run-dec", since_seq=q["seq"], to_agent="coder-1"
        )
        self.assertEqual(len(coder_inbox), 1)
        reply = coder_inbox[0]
        self.assertEqual(reply["kind"], "DECISION")
        self.assertEqual(reply["from_agent"], "lead")
        self.assertEqual(reply["to_agent"], "coder-1")
        self.assertEqual(reply["body"], "Use WAL mode")
        self.assertEqual(reply["meta"], {"reply_to_seq": q["seq"]})

        with self.assertRaises(ValueError):
            self.store.resolve_board_decision(999999, "nope")

    def test_file_leases_claim_conflict_renew_expire_and_release(self) -> None:
        # Fresh claim normalizes path and emits a CLAIM_FILE board message
        c1 = self.store.claim_file_lease(
            "run-lease", "./open_swarm//app.py", "agent-a", ttl_s=30.0
        )
        self.assertTrue(c1["claimed"])
        self.assertEqual(c1["rel_path"], "open_swarm/app.py")
        self.assertEqual(c1["owner"], "agent-a")

        board_claims = self.store.read_board_messages("run-lease", kind="CLAIM_FILE")
        self.assertEqual(len(board_claims), 1)
        self.assertEqual(board_claims[0]["from_agent"], "agent-a")

        # Conflict for another agent
        c2 = self.store.claim_file_lease(
            "run-lease", "open_swarm/app.py", "agent-b", ttl_s=30.0
        )
        self.assertFalse(c2["claimed"])
        self.assertEqual(c2["owner"], "agent-a")
        self.assertEqual(c2["expires_at"], c1["expires_at"])

        # Same owner renews expiry without emitting a duplicate CLAIM_FILE event
        time.sleep(0.01)
        c3 = self.store.claim_file_lease(
            "run-lease", "open_swarm/app.py", "agent-a", ttl_s=60.0
        )
        self.assertTrue(c3["claimed"])
        self.assertGreater(c3["expires_at"], c1["expires_at"])
        self.assertEqual(
            len(self.store.read_board_messages("run-lease", kind="CLAIM_FILE")), 1
        )

        # Release restricted by owner_agent
        self.assertFalse(
            self.store.release_file_lease(
                "run-lease", "open_swarm/app.py", owner_agent="agent-b"
            )
        )
        self.assertTrue(
            self.store.release_file_lease(
                "run-lease", "./open_swarm/app.py", owner_agent="agent-a"
            )
        )
        self.assertFalse(
            self.store.release_file_lease("run-lease", "open_swarm/app.py")
        )

        # Short TTL expiration allows re-claim by another agent
        exp_claim = self.store.claim_file_lease(
            "run-lease", "z_last.py", "agent-a", ttl_s=0.02
        )
        self.assertTrue(exp_claim["claimed"])
        time.sleep(0.03)
        reclaimed = self.store.claim_file_lease(
            "run-lease", "z_last.py", "agent-b", ttl_s=30.0
        )
        self.assertTrue(reclaimed["claimed"])
        self.assertEqual(reclaimed["owner"], "agent-b")

        # list_file_leases prunes expired and returns active ordered by rel_path
        self.store.claim_file_lease("run-lease", "expired.py", "agent-a", ttl_s=0.01)
        self.store.claim_file_lease("run-lease", "a_first.py", "agent-c", ttl_s=30.0)
        time.sleep(0.02)
        active = self.store.list_file_leases("run-lease")
        self.assertEqual([l["rel_path"] for l in active], ["a_first.py", "z_last.py"])
        self.assertEqual([l["owner_agent"] for l in active], ["agent-c", "agent-b"])

    def test_file_lease_path_traversal_and_absolute_rejection(self) -> None:
        bad_paths = [
            "",
            "   ",
            ".",
            "../secret.py",
            "a/../../secret.py",
            "a/../b.py",
            "/etc/passwd",
            "foo\x00bar.py",
        ]
        for bad in bad_paths:
            with self.subTest(bad=bad):
                with self.assertRaises(ValueError):
                    self.store.claim_file_lease("run-sec", bad, "agent-a")
                with self.assertRaises(ValueError):
                    self.store.release_file_lease("run-sec", bad)

    def test_file_lease_concurrent_claim_atomicity(self) -> None:
        barrier = threading.Barrier(4)
        results: list[dict[str, object]] = []
        lock = threading.Lock()

        def worker(agent_name: str) -> None:
            barrier.wait()
            res = self.store.claim_file_lease("run-race", "shared.py", agent_name)
            with lock:
                results.append(res)

        threads = [
            threading.Thread(target=worker, args=(f"agent-{i}",)) for i in range(4)
        ]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        winners = [r for r in results if r["claimed"]]
        losers = [r for r in results if not r["claimed"]]
        self.assertEqual(len(winners), 1)
        self.assertEqual(len(losers), 3)
        winner_owner = winners[0]["owner"]
        for loser in losers:
            self.assertEqual(loser["owner"], winner_owner)


if __name__ == "__main__":
    unittest.main()
