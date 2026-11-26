#!/usr/bin/env python3
"""Unit tests for open_swarm/fleet_scanner.py."""
from __future__ import annotations

import json
import os
from pathlib import Path
import sqlite3
import sys
import tempfile
import time
import unittest

SWARM_DIR = Path(__file__).resolve().parents[1]
if str(SWARM_DIR) not in sys.path:
    sys.path.insert(0, str(SWARM_DIR))

from fleet_scanner import scan_fleet  # noqa: E402


def _write_jsonl(path: Path, records: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    body = "".join(json.dumps(r) + "\n" for r in records)
    path.write_text(body, encoding="utf-8")


def _iso_offset(seconds_ago: float) -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(time.time() - seconds_ago))


class FleetScannerTests(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        self.claude_dir = self.root / "claude_projects"
        self.codex_dir = self.root / "codex_home"
        self.runs_dir = self.root / "open_swarm_runs"
        self.claude_dir.mkdir()
        (self.codex_dir / "sessions").mkdir(parents=True)
        self.runs_dir.mkdir()

    def tearDown(self) -> None:
        self.tmp.cleanup()

    def test_claude_classifications_and_bounded_head_tail(self) -> None:
        proj_alpha = str(self.root / "repos" / "alpha")

        # 1. Claude usage_limit (also >150KB to verify bounded 64KB head + 64KB tail)
        s1 = "11111111-1111-1111-1111-111111111111"
        head_records = [
            {
                "type": "user",
                "sessionId": s1,
                "cwd": proj_alpha,
                "gitBranch": "feat/scanner",
                "timestamp": _iso_offset(1200),
                "message": {"role": "user", "content": "Investigate rate limit handling"},
            }
        ]
        # Pad middle > 150 KB with assistant/user noise that would look "finished" if read without tail override
        padding_line = json.dumps(
            {
                "type": "progress",
                "sessionId": s1,
                "timestamp": _iso_offset(1150),
                "payload": "x" * 2048,
            }
        )
        middle_lines = [padding_line] * 80  # ~165 KB
        tail_records = [
            {
                "type": "assistant",
                "sessionId": s1,
                "timestamp": _iso_offset(900),
                "isApiErrorMessage": True,
                "error": "rate_limit",
                "message": {"role": "assistant", "content": [{"type": "text", "text": "Rate limit reached"}]},
            }
        ]
        large_file = self.claude_dir / "alpha" / f"{s1}.jsonl"
        large_file.parent.mkdir(parents=True, exist_ok=True)
        large_file.write_text(
            "\n".join([json.dumps(r) for r in head_records] + middle_lines + [json.dumps(r) for r in tail_records]) + "\n",
            encoding="utf-8",
        )

        # 2. Claude cut_off (tool_use with no terminal reply)
        s2 = "22222222-2222-2222-2222-222222222222"
        _write_jsonl(
            self.claude_dir / "alpha" / f"{s2}.jsonl",
            [
                {
                    "type": "user",
                    "sessionId": s2,
                    "cwd": proj_alpha,
                    "gitBranch": "main",
                    "timestamp": _iso_offset(800),
                    "message": {"role": "user", "content": [{"type": "text", "text": "Fix failing test"}]},
                },
                {
                    "type": "assistant",
                    "sessionId": s2,
                    "timestamp": _iso_offset(790),
                    "message": {
                        "role": "assistant",
                        "stop_reason": "tool_use",
                        "content": [{"type": "tool_use", "name": "Bash"}],
                    },
                },
            ],
        )

        # 3. Claude finished
        s3 = "33333333-3333-3333-3333-333333333333"
        _write_jsonl(
            self.claude_dir / "beta" / f"{s3}.jsonl",
            [
                {
                    "type": "user",
                    "sessionId": s3,
                    "cwd": str(self.root / "repos" / "beta"),
                    "gitBranch": "main",
                    "timestamp": _iso_offset(3600),
                    "message": {"role": "user", "content": "Summarize architecture"},
                },
                {
                    "type": "assistant",
                    "sessionId": s3,
                    "timestamp": _iso_offset(3590),
                    "message": {
                        "role": "assistant",
                        "stop_reason": "end_turn",
                        "content": [{"type": "text", "text": "Here is the summary."}],
                    },
                },
            ],
        )

        # 4. Claude errored (tool error)
        s4 = "44444444-4444-4444-4444-444444444444"
        _write_jsonl(
            self.claude_dir / "beta" / f"{s4}.jsonl",
            [
                {
                    "type": "user",
                    "sessionId": s4,
                    "cwd": str(self.root / "repos" / "beta"),
                    "timestamp": _iso_offset(1800),
                    "toolUseResult": {"is_error": True, "error": "Permission denied"},
                    "message": {
                        "role": "user",
                        "content": [{"type": "tool_result", "is_error": True, "content": "Permission denied"}],
                    },
                }
            ],
        )

        res = scan_fleet(
            claude_dir=self.claude_dir,
            codex_dir=self.codex_dir,
            open_swarm_runs_dir=self.runs_dir,
        )
        self.assertEqual(res["warnings"], [])
        self.assertEqual(res["counts"]["total"], 4)
        self.assertEqual(res["counts"]["claude"], 4)
        self.assertEqual(res["counts"]["needs_attention"], 3)

        by_sid = {
            s["session_id"]: s
            for p in res["projects"]
            for s in p["sessions"]
        }
        self.assertEqual(by_sid[s1]["status"], "usage_limit")
        self.assertTrue(by_sid[s1]["needs_attention"])
        self.assertEqual(by_sid[s1]["title"], "Investigate rate limit handling")
        self.assertEqual(by_sid[s1]["branch"], "feat/scanner")
        self.assertEqual(by_sid[s1]["resume_cmd"], f"claude --resume {s1}")

        self.assertEqual(by_sid[s2]["status"], "cut_off")
        self.assertTrue(by_sid[s2]["needs_attention"])

        self.assertEqual(by_sid[s3]["status"], "finished")
        self.assertFalse(by_sid[s3]["needs_attention"])

        self.assertEqual(by_sid[s4]["status"], "errored")
        self.assertTrue(by_sid[s4]["needs_attention"])

    def test_codex_jsonl_and_wal_sqlite_strictly_read_only(self) -> None:
        proj_repo = str(self.root / "repos" / "shared")

        # Codex rollout 1: finished
        x1 = "aaaa1111-1111-1111-1111-111111111111"
        r1_path = self.codex_dir / "sessions" / "2026" / "11" / f"rollout-{x1}.jsonl"
        _write_jsonl(
            r1_path,
            [
                {
                    "type": "session_meta",
                    "timestamp": _iso_offset(5000),
                    "payload": {"id": x1, "cwd": proj_repo, "git": {"branch": "codex-branch"}},
                },
                {
                    "type": "event_msg",
                    "timestamp": _iso_offset(4950),
                    "payload": {"type": "user_message", "message": "Refactor parser"},
                },
                {
                    "type": "event_msg",
                    "timestamp": _iso_offset(4900),
                    "payload": {"type": "task_complete", "turn_id": "t1"},
                },
            ],
        )

        # Codex rollout 2: usage_limit
        x2 = "bbbb2222-2222-2222-2222-222222222222"
        r2_path = self.codex_dir / "sessions" / "2026" / "11" / f"rollout-{x2}.jsonl"
        _write_jsonl(
            r2_path,
            [
                {
                    "type": "session_meta",
                    "timestamp": _iso_offset(1000),
                    "payload": {"id": x2, "cwd": proj_repo},
                },
                {
                    "type": "event_msg",
                    "timestamp": _iso_offset(950),
                    "payload": {
                        "type": "task_complete",
                        "turn_id": "t2",
                        "error": {"codex_error_info": "usage_limit_exceeded"},
                    },
                },
            ],
        )

        # Codex state_5.sqlite in WAL mode with an uncheckpointed row in -wal
        x3 = "cccc3333-3333-3333-3333-333333333333"
        db_path = self.codex_dir / "state_5.sqlite"
        writer = sqlite3.connect(str(db_path))
        try:
            writer.execute("PRAGMA journal_mode=WAL;")
            writer.execute("PRAGMA wal_autocheckpoint=0;")
            writer.execute(
                "CREATE TABLE threads (id TEXT PRIMARY KEY, rollout_path TEXT, cwd TEXT, title TEXT, git_branch TEXT, updated_at_ms INTEGER, thread_source TEXT)"
            )
            writer.execute(
                "INSERT INTO threads VALUES (?, ?, ?, ?, ?, ?, ?)",
                (x1, str(r1_path), proj_repo, "Catalog Title For X1", "codex-branch", int((time.time() - 4900) * 1000), "cli"),
            )
            writer.commit()
            # Force checkpoint of first row, then write x3 purely into -wal without checkpointing
            writer.execute("PRAGMA wal_checkpoint(TRUNCATE);")
            writer.execute(
                "INSERT INTO threads VALUES (?, ?, ?, ?, ?, ?, ?)",
                (x3, None, proj_repo, "Uncheckpointed WAL Thread", "wal-branch", int((time.time() - 120) * 1000), "cli"),
            )
            writer.commit()

            wal_path = Path(str(db_path) + "-wal")
            self.assertTrue(wal_path.exists())
            self.assertGreater(wal_path.stat().st_size, 0)

            before_files = sorted(p.name for p in self.codex_dir.iterdir())
            before_db_stat = db_path.stat()
            before_wal_stat = wal_path.stat()

            res = scan_fleet(
                claude_dir=self.claude_dir,
                codex_dir=self.codex_dir,
                open_swarm_runs_dir=self.runs_dir,
            )

            after_files = sorted(p.name for p in self.codex_dir.iterdir())
            after_db_stat = db_path.stat()
            after_wal_stat = wal_path.stat()
            self.assertEqual(before_files, after_files)
            self.assertEqual(before_db_stat.st_mtime_ns, after_db_stat.st_mtime_ns)
            self.assertEqual(before_wal_stat.st_mtime_ns, after_wal_stat.st_mtime_ns)
        finally:
            writer.close()

        self.assertEqual(res["counts"]["codex"], 3)
        by_sid = {s["session_id"]: s for p in res["projects"] for s in p["sessions"]}

        self.assertEqual(by_sid[x1]["status"], "finished")
        self.assertEqual(by_sid[x1]["title"], "Catalog Title For X1")
        self.assertEqual(by_sid[x1]["resume_cmd"], f"codex resume {x1}")

        self.assertEqual(by_sid[x2]["status"], "usage_limit")
        self.assertTrue(by_sid[x2]["needs_attention"])

        # x3 came only from uncommitted WAL frames; age=120s (<600s) -> "recent"
        self.assertEqual(by_sid[x3]["title"], "Uncheckpointed WAL Thread")
        self.assertEqual(by_sid[x3]["branch"], "wal-branch")
        self.assertEqual(by_sid[x3]["status"], "recent")
        self.assertFalse(by_sid[x3]["needs_attention"])

    def test_no_wal_shm_sidecars_created_for_delete_mode_sqlite(self) -> None:
        db_path = self.codex_dir / "state_5.sqlite"
        conn = sqlite3.connect(str(db_path))
        try:
            conn.execute("CREATE TABLE threads (id TEXT PRIMARY KEY, rollout_path TEXT, cwd TEXT, title TEXT, git_branch TEXT, updated_at_ms INTEGER)")
            conn.execute(
                "INSERT INTO threads VALUES (?, ?, ?, ?, ?, ?)",
                ("dddd4444-4444-4444-4444-444444444444", None, str(self.root / "repo_ro"), "RO Check", "main", int((time.time() - 3600) * 1000)),
            )
            conn.commit()
        finally:
            conn.close()

        # Make codex_dir read-only to prove zero writes/locks occur inside codex_dir
        os.chmod(self.codex_dir, 0o555)
        try:
            res = scan_fleet(
                claude_dir=self.claude_dir,
                codex_dir=self.codex_dir,
                open_swarm_runs_dir=self.runs_dir,
            )
        finally:
            os.chmod(self.codex_dir, 0o755)

        self.assertFalse(Path(str(db_path) + "-wal").exists())
        self.assertFalse(Path(str(db_path) + "-shm").exists())
        self.assertEqual(res["counts"]["codex"], 1)

    def test_open_swarm_runs_and_project_grouping_sorting(self) -> None:
        repo_a = str(self.root / "repos" / "proj_a")
        repo_b = str(self.root / "repos" / "proj_b")

        # Claude finished session in proj_b (newer overall, but NOT needing attention)
        _write_jsonl(
            self.claude_dir / "b" / "sess-b.jsonl",
            [
                {
                    "type": "user",
                    "sessionId": "sess-b",
                    "cwd": repo_b + "/",  # trailing slash tests normalization
                    "timestamp": _iso_offset(60),
                    "message": {"role": "user", "content": "Check status"},
                },
                {
                    "type": "assistant",
                    "sessionId": "sess-b",
                    "timestamp": _iso_offset(50),
                    "message": {"role": "assistant", "stop_reason": "end_turn", "content": [{"type": "text", "text": "Ok"}]},
                },
            ],
        )

        # Open Swarm running in proj_b
        (self.runs_dir / "run-running.json").write_text(
            json.dumps(
                {
                    "run_id": "run-running",
                    "repo": repo_b,
                    "status": "running",
                    "model": "org/Mini-Model",
                    "agents_count": 2,
                    "passed_count": 0,
                    "created_at": time.time() - 30,
                }
            ),
            encoding="utf-8",
        )

        # Open Swarm completed in proj_b
        (self.runs_dir / "run-done.json").write_text(
            json.dumps(
                {
                    "run_id": "run-done",
                    "repo": repo_b,
                    "status": "completed",
                    "model": "org/Mini-Model",
                    "agents_count": 3,
                    "passed_count": 3,
                    "created_at": time.time() - 300,
                    "finished_at": time.time() - 200,
                }
            ),
            encoding="utf-8",
        )

        # Open Swarm errored in proj_a (older, but needs_attention -> proj_a must sort FIRST)
        (self.runs_dir / "run-err.json").write_text(
            json.dumps(
                {
                    "run_id": "run-err",
                    "repo": repo_a,
                    "status": "error",
                    "error": "SpendCapExceeded: cap reached",
                    "model": "org/Mini-Model",
                    "agents_count": 2,
                    "passed_count": 1,
                    "created_at": time.time() - 7200,
                    "finished_at": time.time() - 7100,
                }
            ),
            encoding="utf-8",
        )

        # Old Open Swarm completed > 7 days ago -> should NOT count in active_7d
        (self.runs_dir / "run-old.json").write_text(
            json.dumps(
                {
                    "run_id": "run-old",
                    "repo": "",
                    "status": "completed",
                    "created_at": time.time() - (10 * 86400),
                    "finished_at": time.time() - (10 * 86400),
                }
            ),
            encoding="utf-8",
        )

        res = scan_fleet(
            claude_dir=self.claude_dir,
            codex_dir=self.codex_dir,
            open_swarm_runs_dir=self.runs_dir,
        )

        self.assertEqual(res["counts"]["total"], 5)
        self.assertEqual(res["counts"]["claude"], 1)
        self.assertEqual(res["counts"]["open_swarm"], 4)
        self.assertEqual(res["counts"]["active_7d"], 4)
        self.assertEqual(res["counts"]["needs_attention"], 1)

        projects = res["projects"]
        self.assertEqual([p["name"] for p in projects], ["proj_a", "proj_b", "unknown"])

        pa = projects[0]
        self.assertTrue(pa["needs_attention"])
        self.assertEqual(pa["counts"], {"total": 1, "claude": 0, "codex": 0, "open_swarm": 1, "needs_attention": 1})
        self.assertEqual(pa["sessions"][0]["status"], "errored")
        self.assertEqual(pa["sessions"][0]["resume_cmd"], "open-swarm run run-err")

        pb = projects[1]
        self.assertFalse(pb["needs_attention"])
        self.assertEqual(pb["counts"], {"total": 3, "claude": 1, "codex": 0, "open_swarm": 2, "needs_attention": 0})
        pb_by_id = {s["session_id"]: s for s in pb["sessions"]}
        self.assertEqual(pb_by_id["run-running"]["status"], "running")
        self.assertEqual(pb_by_id["run-done"]["status"], "completed")

    def test_missing_directories_handled_cleanly_with_warnings(self) -> None:
        missing_c = self.root / "no_claude"
        missing_x = self.root / "no_codex"
        missing_s = self.root / "no_runs"

        res = scan_fleet(
            claude_dir=missing_c,
            codex_dir=missing_x,
            open_swarm_runs_dir=missing_s,
        )
        self.assertEqual(res["counts"], {
            "total": 0,
            "claude": 0,
            "codex": 0,
            "open_swarm": 0,
            "active_7d": 0,
            "needs_attention": 0,
        })
        self.assertEqual(res["projects"], [])
        self.assertEqual(len(res["warnings"]), 3)
        self.assertTrue(any("claude_dir missing" in w for w in res["warnings"]))
        self.assertTrue(any("codex_dir missing" in w for w in res["warnings"]))
        self.assertTrue(any("open_swarm_runs_dir missing" in w for w in res["warnings"]))


if __name__ == "__main__":
    unittest.main()
