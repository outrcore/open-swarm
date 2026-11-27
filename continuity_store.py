"""SQLite Continuity Store & Context Guard for Open Swarm.

Provides durable checkpoints with optimistic concurrency control, file-hash
context verification, action journaling with file evidence hashing, and
advisory inter-role messaging using only the Python 3 standard library.
"""

from __future__ import annotations

import hashlib
import json
import os
import posixpath
import re
import sqlite3
import time
from pathlib import Path, PurePosixPath
from typing import Any

VALID_ACTION_STATUSES = ("completed", "failed", "unknown")
VALID_BOARD_KINDS = (
    "CLAIM_FILE",
    "CONTRACT",
    "FINDING",
    "QUESTION",
    "NEED_DECISION",
    "BLOCKED",
    "DONE",
    "DECISION",
)

_DIFF_OLD_RE = re.compile(r"^---\s+(.*)$")
_DIFF_NEW_RE = re.compile(r"^\+\+\+\s+(.*)$")


class ContextConflictError(RuntimeError):
    """Raised when a checkpoint publish conflicts with the current revision."""

    def __init__(
        self,
        message: str = "",
        *,
        conflicts: list[dict[str, Any]] | None = None,
        detail: Any = None,
    ) -> None:
        super().__init__(message)
        self.conflicts: list[dict[str, Any]] = list(conflicts) if conflicts else []
        self.detail: Any = detail


def _is_confined_relative_path(repo_root: Path, rel_path: str) -> Path | None:
    """Resolve *rel_path* inside *repo_root*, returning None if it escapes."""
    if not rel_path or "\x00" in rel_path:
        return None
    candidate_rel = Path(rel_path)
    if candidate_rel.is_absolute() or ".." in candidate_rel.parts:
        return None
    try:
        resolved = (repo_root / candidate_rel).resolve()
    except OSError:
        return None
    if resolved == repo_root or not resolved.is_relative_to(repo_root):
        return None
    return resolved


def _require_non_empty(value: str, field_name: str) -> str:
    """Validate that *value* is a non-empty string."""
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{field_name} must be a non-empty string")
    return value


def _normalize_rel_path(rel_path: str) -> str:
    """Normalize a relative repo path and reject absolute paths or ``..`` traversal."""
    if not isinstance(rel_path, str) or not rel_path.strip() or "\x00" in rel_path:
        raise ValueError("rel_path must be a non-empty relative path")
    if rel_path.startswith(("/", "\\")) or Path(rel_path).is_absolute():
        raise ValueError(f"Absolute path is not allowed: {rel_path!r}")
    slash_path = rel_path.replace("\\", "/")
    if len(slash_path) >= 2 and slash_path[1] == ":" and slash_path[0].isalpha():
        raise ValueError(f"Absolute path is not allowed: {rel_path!r}")
    if ".." in PurePosixPath(slash_path).parts:
        raise ValueError(f"Path traversal '..' is not allowed: {rel_path!r}")
    norm = posixpath.normpath(slash_path)
    if norm in ("", ".") or norm.startswith("/") or norm.startswith(".."):
        raise ValueError(f"Invalid relative path: {rel_path!r}")
    return norm


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(65536), b""):
            digest.update(chunk)
    return digest.hexdigest()


def compute_file_hashes(repo: Path, rel_paths: list[str]) -> dict[str, str | None]:
    """Compute SHA-256 hex digests for *rel_paths* confined inside *repo*.

    Paths attempting traversal (e.g. ``..``) or escaping ``repo.resolve()``
    (including via symlinks) are omitted from the returned mapping. Non-existent
    confined paths map to ``None``.
    """
    repo_root = Path(repo).resolve()
    hashes: dict[str, str | None] = {}
    for rel_path in rel_paths:
        resolved = _is_confined_relative_path(repo_root, rel_path)
        if resolved is None:
            continue
        if resolved.is_file():
            hashes[rel_path] = _sha256_file(resolved)
        elif not resolved.exists():
            hashes[rel_path] = None
    return hashes


def extract_diff_paths(diff_text: str) -> list[str]:
    """Parse ``--- a/...`` and ``+++ b/...`` paths from a unified git diff.

    Ignores ``/dev/null``, preserves first-seen order, and deduplicates paths.
    """
    paths: list[str] = []
    seen: set[str] = set()

    for raw_line in diff_text.splitlines():
        line = raw_line.rstrip("\r\n")
        path: str | None = None

        old_match = _DIFF_OLD_RE.match(line)
        if old_match:
            target = old_match.group(1)
            if target != "/dev/null" and target.startswith("a/"):
                path = target[2:]
        else:
            new_match = _DIFF_NEW_RE.match(line)
            if new_match:
                target = new_match.group(1)
                if target != "/dev/null" and target.startswith("b/"):
                    path = target[2:]

        if path and path not in seen:
            seen.add(path)
            paths.append(path)

    return paths


def _compute_context_guard(
    project_key: str,
    role: str,
    revision: int,
    file_hashes: dict[str, str | None],
) -> str:
    payload = json.dumps(
        {
            "project_key": project_key,
            "role": role,
            "revision": revision,
            "file_hashes": file_hashes,
        },
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


class ContinuityStore:
    """SQLite-backed continuity store for checkpoints, actions, and messages."""

    def __init__(self, db_path: Path) -> None:
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_schema()

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(str(self.db_path), timeout=10.0)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("PRAGMA busy_timeout=10000")
        return conn

    def _init_schema(self) -> None:
        conn = self._connect()
        try:
            with conn:
                conn.executescript(
                    """
                    CREATE TABLE IF NOT EXISTS checkpoints (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        project_key TEXT NOT NULL,
                        role TEXT NOT NULL,
                        revision INTEGER NOT NULL,
                        summary TEXT NOT NULL,
                        next_step TEXT NOT NULL DEFAULT '',
                        actor TEXT NOT NULL,
                        file_hashes_json TEXT NOT NULL,
                        created_at REAL NOT NULL
                    );

                    CREATE INDEX IF NOT EXISTS idx_checkpoints_project_role_rev
                        ON checkpoints (project_key, role, revision DESC, id DESC);

                    CREATE TABLE IF NOT EXISTS action_journal (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        project_key TEXT NOT NULL,
                        role TEXT NOT NULL,
                        action TEXT NOT NULL,
                        status TEXT NOT NULL,
                        detail TEXT NOT NULL DEFAULT '',
                        run_id TEXT NOT NULL DEFAULT '',
                        evidence_json TEXT NOT NULL,
                        created_at REAL NOT NULL
                    );

                    CREATE INDEX IF NOT EXISTS idx_action_journal_project_created
                        ON action_journal (project_key, created_at DESC, id DESC);

                    CREATE TABLE IF NOT EXISTS messages (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        project_key TEXT NOT NULL,
                        from_role TEXT NOT NULL,
                        to_role TEXT NOT NULL,
                        body TEXT NOT NULL,
                        created_at REAL NOT NULL
                    );

                    CREATE INDEX IF NOT EXISTS idx_messages_project_id
                        ON messages (project_key, id ASC);

                    CREATE TABLE IF NOT EXISTS board_messages (
                        seq INTEGER PRIMARY KEY AUTOINCREMENT,
                        run_id TEXT NOT NULL,
                        from_agent TEXT NOT NULL,
                        to_agent TEXT NOT NULL DEFAULT 'all',
                        kind TEXT NOT NULL,
                        body TEXT NOT NULL,
                        meta_json TEXT NOT NULL DEFAULT '{}',
                        resolved INTEGER NOT NULL DEFAULT 0,
                        resolution TEXT NOT NULL DEFAULT '',
                        resolved_by TEXT NOT NULL DEFAULT '',
                        resolved_at REAL,
                        created_at REAL NOT NULL
                    );

                    CREATE INDEX IF NOT EXISTS idx_board_messages_run_seq
                        ON board_messages (run_id, seq ASC);

                    CREATE TABLE IF NOT EXISTS file_leases (
                        run_id TEXT NOT NULL,
                        rel_path TEXT NOT NULL,
                        owner_agent TEXT NOT NULL,
                        acquired_at REAL NOT NULL,
                        expires_at REAL NOT NULL,
                        PRIMARY KEY (run_id, rel_path)
                    );
                    """
                )
        finally:
            conn.close()

    @staticmethod
    def verify_context_guard(
        repo: Path, expected_hashes: dict[str, str | None]
    ) -> dict[str, Any]:
        """Verify current file hashes in *repo* match *expected_hashes*."""
        actual_hashes = compute_file_hashes(repo, list(expected_hashes.keys()))
        conflicts: list[dict[str, Any]] = []
        for path, expected in expected_hashes.items():
            if path not in actual_hashes:
                conflicts.append(
                    {"path": path, "expected": expected, "actual": None}
                )
                continue
            actual = actual_hashes[path]
            if actual != expected:
                conflicts.append(
                    {"path": path, "expected": expected, "actual": actual}
                )
        return {
            "ok": len(conflicts) == 0,
            "conflicts": conflicts,
            "checked_count": len(expected_hashes),
        }

    @staticmethod
    def _row_to_checkpoint(row: sqlite3.Row) -> dict[str, Any]:
        file_hashes = json.loads(row["file_hashes_json"])
        return {
            "id": row["id"],
            "project_key": row["project_key"],
            "role": row["role"],
            "revision": row["revision"],
            "summary": row["summary"],
            "next_step": row["next_step"],
            "actor": row["actor"],
            "file_hashes": file_hashes,
            "context_guard": _compute_context_guard(
                row["project_key"],
                row["role"],
                row["revision"],
                file_hashes,
            ),
            "created_at": row["created_at"],
        }

    def publish_checkpoint(
        self,
        project_key: str,
        role: str,
        summary: str,
        next_step: str = "",
        actor: str = "open-swarm",
        expected_revision: int | None = None,
        file_hashes: dict[str, str | None] | None = None,
    ) -> dict[str, Any]:
        """Publish a new checkpoint revision with optional optimistic revision check."""
        normalized_hashes: dict[str, str | None] = (
            dict(file_hashes) if file_hashes is not None else {}
        )
        file_hashes_json = json.dumps(normalized_hashes, sort_keys=True)
        created_at = time.time()

        conn = self._connect()
        try:
            conn.isolation_level = None
            conn.execute("BEGIN IMMEDIATE")
            try:
                row = conn.execute(
                    """
                    SELECT COALESCE(MAX(revision), 0) AS latest_rev
                    FROM checkpoints
                    WHERE project_key = ? AND role = ?
                    """,
                    (project_key, role),
                ).fetchone()
                latest_rev = int(row["latest_rev"]) if row is not None else 0

                if expected_revision is not None and expected_revision != latest_rev:
                    conflict = {
                        "project_key": project_key,
                        "role": role,
                        "expected_revision": expected_revision,
                        "actual_revision": latest_rev,
                    }
                    raise ContextConflictError(
                        f"Checkpoint revision conflict for {project_key}/{role}: "
                        f"expected {expected_revision}, got {latest_rev}",
                        conflicts=[conflict],
                        detail=conflict,
                    )

                revision = latest_rev + 1
                cursor = conn.execute(
                    """
                    INSERT INTO checkpoints (
                        project_key, role, revision, summary, next_step,
                        actor, file_hashes_json, created_at
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        project_key,
                        role,
                        revision,
                        summary,
                        next_step,
                        actor,
                        file_hashes_json,
                        created_at,
                    ),
                )
                row_id = int(cursor.lastrowid)
                conn.execute("COMMIT")
            except Exception:
                conn.execute("ROLLBACK")
                raise
        finally:
            conn.close()

        return {
            "id": row_id,
            "project_key": project_key,
            "role": role,
            "revision": revision,
            "summary": summary,
            "next_step": next_step,
            "actor": actor,
            "file_hashes": normalized_hashes,
            "context_guard": _compute_context_guard(
                project_key, role, revision, normalized_hashes
            ),
            "created_at": created_at,
        }

    def get_latest_checkpoints(self, project_key: str) -> list[dict[str, Any]]:
        """Return the latest checkpoint per role for *project_key*, newest first."""
        conn = self._connect()
        try:
            rows = conn.execute(
                """
                SELECT c.*
                FROM checkpoints c
                INNER JOIN (
                    SELECT role, MAX(revision) AS max_rev
                    FROM checkpoints
                    WHERE project_key = ?
                    GROUP BY role
                ) latest
                    ON c.project_key = ?
                   AND c.role = latest.role
                   AND c.revision = latest.max_rev
                WHERE c.id = (
                    SELECT MAX(c2.id)
                    FROM checkpoints c2
                    WHERE c2.project_key = c.project_key
                      AND c2.role = c.role
                      AND c2.revision = c.revision
                )
                ORDER BY c.created_at DESC, c.id DESC
                """,
                (project_key, project_key),
            ).fetchall()
            return [self._row_to_checkpoint(r) for r in rows]
        finally:
            conn.close()

    def record_action(
        self,
        project_key: str,
        role: str,
        action: str,
        status: str,
        detail: str = "",
        run_id: str = "",
        evidence: list[dict[str, Any]] | None = None,
        repo: Path | None = None,
    ) -> dict[str, Any]:
        """Record an action journal entry, optionally hashing file evidence."""
        if status not in VALID_ACTION_STATUSES:
            raise ValueError(
                f"Invalid action status {status!r}; must be one of {VALID_ACTION_STATUSES}"
            )

        normalized_evidence: list[dict[str, Any]] = []
        for item in evidence or []:
            entry = dict(item)
            if repo is not None and entry.get("kind") == "file" and entry.get("path"):
                rel_path = str(entry["path"])
                file_hashes = compute_file_hashes(repo, [rel_path])
                entry["sha256"] = file_hashes.get(rel_path)
            normalized_evidence.append(entry)

        evidence_json = json.dumps(normalized_evidence)
        created_at = time.time()

        conn = self._connect()
        try:
            with conn:
                cursor = conn.execute(
                    """
                    INSERT INTO action_journal (
                        project_key, role, action, status, detail,
                        run_id, evidence_json, created_at
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        project_key,
                        role,
                        action,
                        status,
                        detail,
                        run_id,
                        evidence_json,
                        created_at,
                    ),
                )
                row_id = int(cursor.lastrowid)
        finally:
            conn.close()

        return {
            "id": row_id,
            "project_key": project_key,
            "role": role,
            "action": action,
            "status": status,
            "detail": detail,
            "run_id": run_id,
            "evidence": normalized_evidence,
            "created_at": created_at,
        }

    def list_actions(self, project_key: str, limit: int = 30) -> list[dict[str, Any]]:
        """List actions for *project_key*, newest first."""
        conn = self._connect()
        try:
            rows = conn.execute(
                """
                SELECT *
                FROM action_journal
                WHERE project_key = ?
                ORDER BY created_at DESC, id DESC
                LIMIT ?
                """,
                (project_key, limit),
            ).fetchall()
            return [
                {
                    "id": row["id"],
                    "project_key": row["project_key"],
                    "role": row["role"],
                    "action": row["action"],
                    "status": row["status"],
                    "detail": row["detail"],
                    "run_id": row["run_id"],
                    "evidence": json.loads(row["evidence_json"]),
                    "created_at": row["created_at"],
                }
                for row in rows
            ]
        finally:
            conn.close()

    def send_message(
        self,
        project_key: str,
        from_role: str,
        to_role: str,
        body: str,
    ) -> dict[str, Any]:
        """Store an advisory message from *from_role* to *to_role*."""
        created_at = time.time()
        conn = self._connect()
        try:
            with conn:
                cursor = conn.execute(
                    """
                    INSERT INTO messages (
                        project_key, from_role, to_role, body, created_at
                    ) VALUES (?, ?, ?, ?, ?)
                    """,
                    (project_key, from_role, to_role, body, created_at),
                )
                row_id = int(cursor.lastrowid)
        finally:
            conn.close()

        return {
            "id": row_id,
            "project_key": project_key,
            "from_role": from_role,
            "to_role": to_role,
            "body": body,
            "created_at": created_at,
        }

    def read_messages(
        self,
        project_key: str,
        to_role: str | None = None,
        after_id: int = 0,
        limit: int = 50,
    ) -> list[dict[str, Any]]:
        """Read advisory messages with id > *after_id*, ordered oldest first."""
        conn = self._connect()
        try:
            if to_role is not None:
                rows = conn.execute(
                    """
                    SELECT *
                    FROM messages
                    WHERE project_key = ?
                      AND id > ?
                      AND to_role IN (?, 'all', '*')
                    ORDER BY id ASC
                    LIMIT ?
                    """,
                    (project_key, after_id, to_role, limit),
                ).fetchall()
            else:
                rows = conn.execute(
                    """
                    SELECT *
                    FROM messages
                    WHERE project_key = ?
                      AND id > ?
                    ORDER BY id ASC
                    LIMIT ?
                    """,
                    (project_key, after_id, limit),
                ).fetchall()
            return [
                {
                    "id": row["id"],
                    "project_key": row["project_key"],
                    "from_role": row["from_role"],
                    "to_role": row["to_role"],
                    "body": row["body"],
                    "created_at": row["created_at"],
                }
                for row in rows
            ]
        finally:
            conn.close()

    def project_status(self, project_key: str) -> dict[str, Any]:
        """Return a combined continuity snapshot for *project_key*."""
        return {
            "project_key": project_key,
            "checkpoints": self.get_latest_checkpoints(project_key),
            "actions": self.list_actions(project_key),
            "messages": self.read_messages(project_key),
        }

    @staticmethod
    def _row_to_board_message(row: sqlite3.Row) -> dict[str, Any]:
        return {
            "seq": int(row["seq"]),
            "run_id": row["run_id"],
            "from_agent": row["from_agent"],
            "to_agent": row["to_agent"],
            "kind": row["kind"],
            "body": row["body"],
            "meta": json.loads(row["meta_json"]),
            "resolved": bool(row["resolved"]),
            "resolution": row["resolution"],
            "resolved_by": row["resolved_by"],
            "resolved_at": row["resolved_at"],
            "created_at": row["created_at"],
        }

    def post_board_message(
        self,
        run_id: str,
        from_agent: str,
        kind: str,
        body: str,
        to_agent: str = "all",
        meta: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Post a structured message onto the shared run board."""
        _require_non_empty(run_id, "run_id")
        _require_non_empty(from_agent, "from_agent")
        _require_non_empty(to_agent, "to_agent")
        if kind not in VALID_BOARD_KINDS:
            raise ValueError(
                f"Invalid board kind {kind!r}; must be one of {VALID_BOARD_KINDS}"
            )

        normalized_meta = dict(meta) if meta is not None else {}
        meta_json = json.dumps(normalized_meta, sort_keys=True)
        created_at = time.time()

        conn = self._connect()
        try:
            with conn:
                cursor = conn.execute(
                    """
                    INSERT INTO board_messages (
                        run_id, from_agent, to_agent, kind, body,
                        meta_json, resolved, resolution, resolved_by,
                        resolved_at, created_at
                    ) VALUES (?, ?, ?, ?, ?, ?, 0, '', '', NULL, ?)
                    """,
                    (run_id, from_agent, to_agent, kind, body, meta_json, created_at),
                )
                seq = int(cursor.lastrowid)
        finally:
            conn.close()

        return {
            "seq": seq,
            "run_id": run_id,
            "from_agent": from_agent,
            "to_agent": to_agent,
            "kind": kind,
            "body": body,
            "meta": normalized_meta,
            "resolved": False,
            "resolution": "",
            "resolved_by": "",
            "resolved_at": None,
            "created_at": created_at,
        }

    def read_board_messages(
        self,
        run_id: str,
        since_seq: int = 0,
        to_agent: str | None = None,
        kind: str | None = None,
        limit: int = 100,
    ) -> list[dict[str, Any]]:
        """Read board messages for *run_id* with seq > *since_seq*, oldest first."""
        _require_non_empty(run_id, "run_id")
        if kind is not None and kind not in VALID_BOARD_KINDS:
            raise ValueError(
                f"Invalid board kind {kind!r}; must be one of {VALID_BOARD_KINDS}"
            )

        clauses = ["run_id = ?", "seq > ?"]
        params: list[Any] = [run_id, since_seq]

        if to_agent is not None and to_agent != "all":
            clauses.append("to_agent IN (?, 'all')")
            params.append(to_agent)
        if kind is not None:
            clauses.append("kind = ?")
            params.append(kind)

        params.append(limit)
        where_sql = " AND ".join(clauses)

        conn = self._connect()
        try:
            rows = conn.execute(
                f"""
                SELECT *
                FROM board_messages
                WHERE {where_sql}
                ORDER BY seq ASC
                LIMIT ?
                """,
                tuple(params),
            ).fetchall()
            return [self._row_to_board_message(row) for row in rows]
        finally:
            conn.close()

    def resolve_board_decision(
        self,
        seq: int,
        resolution: str,
        resolver: str = "human",
    ) -> dict[str, Any]:
        """Resolve a board message and post a DECISION reply to its sender."""
        _require_non_empty(resolver, "resolver")
        now = time.time()
        reply_meta = {"reply_to_seq": seq}
        reply_meta_json = json.dumps(reply_meta, sort_keys=True)

        conn = self._connect()
        try:
            conn.isolation_level = None
            conn.execute("BEGIN IMMEDIATE")
            try:
                row = conn.execute(
                    "SELECT * FROM board_messages WHERE seq = ?",
                    (seq,),
                ).fetchone()
                if row is None:
                    raise ValueError(f"Board message seq={seq} not found")

                conn.execute(
                    """
                    UPDATE board_messages
                    SET resolved = 1,
                        resolution = ?,
                        resolved_by = ?,
                        resolved_at = ?
                    WHERE seq = ?
                    """,
                    (resolution, resolver, now, seq),
                )

                reply_cursor = conn.execute(
                    """
                    INSERT INTO board_messages (
                        run_id, from_agent, to_agent, kind, body,
                        meta_json, resolved, resolution, resolved_by,
                        resolved_at, created_at
                    ) VALUES (?, ?, ?, 'DECISION', ?, ?, 0, '', '', NULL, ?)
                    """,
                    (
                        row["run_id"],
                        resolver,
                        row["from_agent"],
                        resolution,
                        reply_meta_json,
                        now,
                    ),
                )
                decision_seq = int(reply_cursor.lastrowid)
                updated_row = conn.execute(
                    "SELECT * FROM board_messages WHERE seq = ?",
                    (seq,),
                ).fetchone()
                conn.execute("COMMIT")
            except Exception:
                conn.execute("ROLLBACK")
                raise
        finally:
            conn.close()

        result = self._row_to_board_message(updated_row)
        result["decision_seq"] = decision_seq
        return result

    def claim_file_lease(
        self,
        run_id: str,
        rel_path: str,
        owner_agent: str,
        ttl_s: float = 600.0,
    ) -> dict[str, Any]:
        """Atomically claim or renew an exclusive file lease for *owner_agent*."""
        _require_non_empty(run_id, "run_id")
        _require_non_empty(owner_agent, "owner_agent")
        if ttl_s <= 0:
            raise ValueError("ttl_s must be positive")
        norm_path = _normalize_rel_path(rel_path)

        now = time.time()
        new_expires_at = now + float(ttl_s)

        conn = self._connect()
        try:
            conn.isolation_level = None
            conn.execute("BEGIN IMMEDIATE")
            try:
                row = conn.execute(
                    """
                    SELECT owner_agent, acquired_at, expires_at
                    FROM file_leases
                    WHERE run_id = ? AND rel_path = ?
                    """,
                    (run_id, norm_path),
                ).fetchone()

                if row is None:
                    conn.execute(
                        """
                        INSERT INTO file_leases (
                            run_id, rel_path, owner_agent, acquired_at, expires_at
                        ) VALUES (?, ?, ?, ?, ?)
                        """,
                        (run_id, norm_path, owner_agent, now, new_expires_at),
                    )
                    fresh_claim = True
                    claimed = True
                    owner = owner_agent
                    expires_at = new_expires_at
                elif float(row["expires_at"]) <= now:
                    conn.execute(
                        """
                        UPDATE file_leases
                        SET owner_agent = ?, acquired_at = ?, expires_at = ?
                        WHERE run_id = ? AND rel_path = ?
                        """,
                        (owner_agent, now, new_expires_at, run_id, norm_path),
                    )
                    fresh_claim = True
                    claimed = True
                    owner = owner_agent
                    expires_at = new_expires_at
                elif row["owner_agent"] == owner_agent:
                    conn.execute(
                        """
                        UPDATE file_leases
                        SET expires_at = ?
                        WHERE run_id = ? AND rel_path = ?
                        """,
                        (new_expires_at, run_id, norm_path),
                    )
                    fresh_claim = False
                    claimed = True
                    owner = owner_agent
                    expires_at = new_expires_at
                else:
                    fresh_claim = False
                    claimed = False
                    owner = row["owner_agent"]
                    expires_at = float(row["expires_at"])

                if fresh_claim:
                    claim_meta = json.dumps(
                        {"rel_path": norm_path, "expires_at": expires_at},
                        sort_keys=True,
                    )
                    conn.execute(
                        """
                        INSERT INTO board_messages (
                            run_id, from_agent, to_agent, kind, body,
                            meta_json, resolved, resolution, resolved_by,
                            resolved_at, created_at
                        ) VALUES (?, ?, 'all', 'CLAIM_FILE', ?, ?, 0, '', '', NULL, ?)
                        """,
                        (
                            run_id,
                            owner_agent,
                            f"Claimed {norm_path}",
                            claim_meta,
                            now,
                        ),
                    )

                conn.execute("COMMIT")
            except Exception:
                conn.execute("ROLLBACK")
                raise
        finally:
            conn.close()

        return {
            "claimed": claimed,
            "rel_path": norm_path,
            "owner": owner,
            "expires_at": expires_at,
        }

    def release_file_lease(
        self,
        run_id: str,
        rel_path: str,
        owner_agent: str | None = None,
    ) -> bool:
        """Release a file lease, optionally restricted to *owner_agent*."""
        _require_non_empty(run_id, "run_id")
        if owner_agent is not None:
            _require_non_empty(owner_agent, "owner_agent")
        norm_path = _normalize_rel_path(rel_path)

        conn = self._connect()
        try:
            with conn:
                if owner_agent is not None:
                    cursor = conn.execute(
                        """
                        DELETE FROM file_leases
                        WHERE run_id = ? AND rel_path = ? AND owner_agent = ?
                        """,
                        (run_id, norm_path, owner_agent),
                    )
                else:
                    cursor = conn.execute(
                        """
                        DELETE FROM file_leases
                        WHERE run_id = ? AND rel_path = ?
                        """,
                        (run_id, norm_path),
                    )
                return cursor.rowcount > 0
        finally:
            conn.close()

    def list_file_leases(self, run_id: str) -> list[dict[str, Any]]:
        """Prune expired leases for *run_id* and return active ones ordered by rel_path."""
        _require_non_empty(run_id, "run_id")
        now = time.time()

        conn = self._connect()
        try:
            conn.isolation_level = None
            conn.execute("BEGIN IMMEDIATE")
            try:
                conn.execute(
                    """
                    DELETE FROM file_leases
                    WHERE run_id = ? AND expires_at <= ?
                    """,
                    (run_id, now),
                )
                rows = conn.execute(
                    """
                    SELECT run_id, rel_path, owner_agent, acquired_at, expires_at
                    FROM file_leases
                    WHERE run_id = ?
                    ORDER BY rel_path ASC
                    """,
                    (run_id,),
                ).fetchall()
                conn.execute("COMMIT")
            except Exception:
                conn.execute("ROLLBACK")
                raise
            return [
                {
                    "run_id": row["run_id"],
                    "rel_path": row["rel_path"],
                    "owner_agent": row["owner_agent"],
                    "acquired_at": float(row["acquired_at"]),
                    "expires_at": float(row["expires_at"]),
                }
                for row in rows
            ]
        finally:
            conn.close()
