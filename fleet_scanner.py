"""Read-only Cross-App Fleet Scanner for Open Swarm.

Discovers recent sessions across Claude Code (~/.claude/projects),
Codex (~/.codex), and Open Swarm (~/.open_swarm/runs), normalizes status
and attention signals, and groups sessions by project root.

Strictly read-only against Claude and Codex directories: never creates,
modifies, or locks files in source directories (SQLite catalogs are copied
with any -wal/-shm sidecars into a temporary directory before querying).
"""
from __future__ import annotations

from datetime import datetime, timezone
import json
import os
from pathlib import Path
import re
import shutil
import sqlite3
import tempfile
import time
from typing import Any

_BOUNDED_BYTES = 65536  # 64 KB head + 64 KB tail
_RECENT_WINDOW_S = 600.0
_SEVEN_DAYS_S = 7 * 86400.0
_UUID_TAIL_RE = re.compile(r"([0-9a-fA-F]{8}-[0-9a-fA-F-]{20,50})$")
_WHITESPACE_RE = re.compile(r"\s+")


def _now_ts() -> float:
    return time.time()


def _parse_ts(value: Any) -> float | None:
    """Parse an ISO-8601 string or epoch seconds/milliseconds value into epoch seconds."""
    if value is None or isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        num = float(value)
        if num <= 0:
            return None
        if abs(num) >= 100_000_000_000:
            return num / 1000.0
        return num
    if isinstance(value, str):
        text = value.strip()
        if not text:
            return None
        if text.replace(".", "", 1).isdigit():
            try:
                return _parse_ts(float(text))
            except ValueError:
                return None
        try:
            iso = text[:-1] + "+00:00" if text.endswith("Z") else text
            dt = datetime.fromisoformat(iso)
            if dt.tzinfo is None:
                dt = dt.replace(tzinfo=timezone.utc)
            return dt.timestamp()
        except ValueError:
            return None
    return None


def _normalize_project_root(raw: Any) -> str:
    if not isinstance(raw, str):
        return "unknown"
    stripped = raw.strip()
    if not stripped:
        return "unknown"
    expanded = os.path.expanduser(stripped)
    norm = os.path.normpath(expanded)
    return norm if norm else "unknown"


def _project_name(root: str) -> str:
    if not root or root == "unknown":
        return "unknown"
    name = Path(root).name
    return name or root


def _clean_title(raw: Any, fallback: str, max_len: int = 120) -> str:
    if not isinstance(raw, str):
        return fallback
    cleaned = _WHITESPACE_RE.sub(" ", raw).strip()
    if not cleaned:
        return fallback
    if len(cleaned) > max_len:
        return cleaned[: max_len - 1].rstrip() + "…"
    return cleaned


def _extract_text_content(content: Any) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: list[str] = []
        for block in content:
            if not isinstance(block, dict):
                continue
            btype = block.get("type")
            if btype in ("text", "input_text", "output_text") and isinstance(block.get("text"), str):
                parts.append(block["text"])
        return " ".join(parts)
    return ""


def _read_bounded_jsonl(path: Path, max_chunk: int = _BOUNDED_BYTES) -> tuple[list[dict[str, Any]], float]:
    """Read at most max_chunk head + max_chunk tail bytes and decode valid JSON objects."""
    with path.open("rb") as fh:
        st = os.fstat(fh.fileno())
        size = st.st_size
        mtime = float(st.st_mtime)
        if size == 0:
            return [], mtime
        if size <= max_chunk * 2:
            raw_chunks = [fh.read(size)]
        else:
            head = fh.read(max_chunk)
            if not head.endswith(b"\n"):
                last_nl = head.rfind(b"\n")
                head = head[: last_nl + 1] if last_nl >= 0 else b""
            fh.seek(size - max_chunk)
            tail = fh.read(max_chunk)
            first_nl = tail.find(b"\n")
            tail = tail[first_nl + 1 :] if first_nl >= 0 else b""
            raw_chunks = [head, tail]

    records: list[dict[str, Any]] = []
    for chunk in raw_chunks:
        for line in chunk.splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line.decode("utf-8", errors="replace"))
            except (ValueError, UnicodeError):
                continue
            if isinstance(obj, dict):
                records.append(obj)
    return records, mtime


def _walk_files(root: Path, suffix: str) -> list[tuple[Path, float]]:
    """Recursively list regular non-symlink files ending with suffix, sorted newest first."""
    found: list[tuple[Path, float]] = []
    for dirpath, dirnames, filenames in os.walk(root, followlinks=False):
        dirnames[:] = [
            d for d in dirnames if not (Path(dirpath) / d).is_symlink()
        ]
        for fname in filenames:
            if not fname.endswith(suffix):
                continue
            fpath = Path(dirpath) / fname
            if fpath.is_symlink():
                continue
            try:
                st = fpath.stat()
            except OSError:
                continue
            found.append((fpath, float(st.st_mtime)))
    found.sort(key=lambda item: item[1], reverse=True)
    return found


def _is_rate_limit_error(err: Any) -> bool:
    if isinstance(err, dict):
        etype = str(err.get("type") or err.get("code") or "").lower()
        status = err.get("status")
        msg = str(err.get("message") or "").lower()
        if status == 429 or "rate_limit" in etype or "usage_limit" in etype:
            return True
        return "rate limit" in msg or "usage limit" in msg or "rate_limit" in msg
    if isinstance(err, str):
        low = err.lower()
        return any(tok in low for tok in ("rate_limit", "usage_limit", "rate limit", "usage limit", "429"))
    return False


def _parse_claude_session(path: Path, file_mtime: float, now: float) -> dict[str, Any] | None:
    records, mtime = _read_bounded_jsonl(path)
    if not records:
        return None

    session_id = path.stem
    cwd: str | None = None
    branch: str | None = None
    custom_title: str | None = None
    ai_title: str | None = None
    first_user_prompt: str | None = None
    record_ts: float | None = None
    last_seen_type: str | None = None
    status: str | None = None
    last_event: str | None = None

    for obj in records:
        rec_type = obj.get("type")
        if isinstance(rec_type, str) and rec_type:
            last_seen_type = rec_type

        sid = obj.get("sessionId")
        if isinstance(sid, str) and sid.strip():
            session_id = sid.strip()

        rec_cwd = obj.get("cwd")
        if isinstance(rec_cwd, str) and rec_cwd.strip():
            cwd = rec_cwd.strip()

        rec_branch = obj.get("gitBranch")
        if isinstance(rec_branch, str) and rec_branch.strip():
            branch = rec_branch.strip()

        ts = _parse_ts(obj.get("timestamp"))
        if ts is not None:
            record_ts = ts if record_ts is None else max(record_ts, ts)

        if rec_type == "custom-title" and isinstance(obj.get("customTitle"), str) and obj["customTitle"].strip():
            custom_title = obj["customTitle"].strip()
        elif rec_type == "ai-title" and isinstance(obj.get("aiTitle"), str) and obj["aiTitle"].strip():
            ai_title = obj["aiTitle"].strip()

        if obj.get("isMeta") is True or obj.get("isSidechain") is True:
            continue

        if rec_type == "user":
            msg = obj.get("message") if isinstance(obj.get("message"), dict) else {}
            content = msg.get("content")
            if first_user_prompt is None:
                text = _extract_text_content(content).strip()
                if text:
                    first_user_prompt = text

            tool_use_result = obj.get("toolUseResult")
            has_tool_result = isinstance(content, list) and any(
                isinstance(b, dict) and b.get("type") == "tool_result" for b in content
            )
            has_tool_error = (
                isinstance(tool_use_result, dict)
                and (tool_use_result.get("is_error") is True or bool(tool_use_result.get("error")))
            ) or (
                isinstance(content, list)
                and any(
                    isinstance(b, dict)
                    and b.get("type") == "tool_result"
                    and b.get("is_error") is True
                    for b in content
                )
            )
            if has_tool_error:
                status = "errored"
                last_event = "user:tool_error"
            elif isinstance(tool_use_result, dict) and tool_use_result.get("interrupted") is True:
                status = "cut_off"
                last_event = "user:interrupted"
            elif has_tool_result:
                status = "cut_off"
                last_event = "user:tool_result"
            else:
                status = "cut_off"
                last_event = "user:prompt"

        elif rec_type == "assistant":
            msg = obj.get("message") if isinstance(obj.get("message"), dict) else {}
            err = obj.get("error")
            if obj.get("isApiErrorMessage") is True or err:
                if _is_rate_limit_error(err) or _is_rate_limit_error(_extract_text_content(msg.get("content"))):
                    status = "usage_limit"
                    last_event = "assistant:rate_limit"
                else:
                    status = "errored"
                    last_event = f"assistant:{err}" if isinstance(err, str) and err else "assistant:api_error"
            else:
                stop_reason = msg.get("stop_reason")
                content = msg.get("content")
                content_kinds = (
                    {b.get("type") for b in content if isinstance(b, dict)}
                    if isinstance(content, list)
                    else set()
                )
                if stop_reason in ("end_turn", "stop_sequence") and "tool_use" not in content_kinds:
                    status = "finished"
                    last_event = f"assistant:{stop_reason}"
                elif stop_reason == "tool_use" or "tool_use" in content_kinds:
                    status = "cut_off"
                    last_event = "assistant:tool_use"
                else:
                    status = "cut_off"
                    last_event = f"assistant:{stop_reason or 'incomplete'}"

        elif rec_type == "system":
            subtype = obj.get("subtype")
            err = obj.get("error")
            if subtype == "api_error" or obj.get("isApiErrorMessage") is True or err:
                if _is_rate_limit_error(err):
                    status = "usage_limit"
                    last_event = "system:rate_limit"
                else:
                    status = "errored"
                    last_event = f"system:{subtype or 'api_error'}"

    last_activity_ts = record_ts if record_ts is not None else (mtime or file_mtime)

    if status is None:
        age = now - last_activity_ts
        if -60.0 <= age < _RECENT_WINDOW_S:
            status = "recent"
        else:
            status = "finished"
        last_event = last_seen_type or "metadata_only"

    title_raw = custom_title or ai_title or first_user_prompt
    title = _clean_title(title_raw, f"Claude session {session_id[:8]}")
    needs_attention = status in ("usage_limit", "errored", "cut_off")
    project_root = _normalize_project_root(cwd)

    return {
        "id": f"claude:{session_id}",
        "source": "claude",
        "session_id": session_id,
        "title": title,
        "project_root": project_root,
        "branch": branch,
        "status": status,
        "needs_attention": needs_attention,
        "last_event": last_event or status,
        "last_activity_ts": float(last_activity_ts),
        "resume_cmd": f"claude --resume {session_id}",
    }


def _scan_claude(claude_dir: Path, max_sessions: int, now: float, warnings: list[str]) -> list[dict[str, Any]]:
    if not claude_dir.exists():
        warnings.append(f"claude_dir missing: {claude_dir}")
        return []
    if not claude_dir.is_dir():
        warnings.append(f"claude_dir is not a directory: {claude_dir}")
        return []

    files = _walk_files(claude_dir, ".jsonl")
    if max_sessions > 0:
        files = files[:max_sessions]

    sessions: list[dict[str, Any]] = []
    for path, mtime in files:
        try:
            sess = _parse_claude_session(path, mtime, now)
            if sess is not None:
                sessions.append(sess)
        except OSError as exc:
            warnings.append(f"claude read error ({path.name}): {exc}")
    return sessions


def _select_codex_sqlite(codex_dir: Path) -> Path | None:
    preferred = codex_dir / "state_5.sqlite"
    if preferred.is_file() and not preferred.is_symlink():
        return preferred
    candidates: list[tuple[int, Path]] = []
    try:
        for item in codex_dir.iterdir():
            if item.is_symlink() or not item.is_file():
                continue
            m = re.fullmatch(r"state_(\d+)\.sqlite", item.name)
            if m:
                candidates.append((int(m.group(1)), item))
    except OSError:
        return None
    if not candidates:
        return None
    candidates.sort(key=lambda x: x[0], reverse=True)
    return candidates[0][1]


def _read_codex_catalog_ro(db_path: Path, warnings: list[str]) -> dict[str, dict[str, Any]]:
    """Copy SQLite db (+ -wal/-shm if present) to a temp dir and read threads read-only."""
    threads: dict[str, dict[str, Any]] = {}
    with tempfile.TemporaryDirectory(prefix="open_swarm_codex_") as tmpdir:
        tmp_db = Path(tmpdir) / db_path.name
        shutil.copyfile(db_path, tmp_db)
        for suffix in ("-wal", "-shm"):
            sidecar = Path(str(db_path) + suffix)
            if sidecar.is_file() and not sidecar.is_symlink():
                shutil.copyfile(sidecar, Path(str(tmp_db) + suffix))

        conn = sqlite3.connect(str(tmp_db))
        conn.row_factory = sqlite3.Row
        try:
            cols = {row[1] for row in conn.execute("PRAGMA table_info(threads)").fetchall()}
            if not cols or "id" not in cols:
                return threads
            desired = (
                "id",
                "rollout_path",
                "cwd",
                "name",
                "title",
                "git_branch",
                "branch",
                "updated_at_ms",
                "updated_at",
                "archived",
                "thread_source",
                "agent_role",
            )
            select_expr = ", ".join(c if c in cols else f"NULL AS {c}" for c in desired)
            rows = conn.execute(f"SELECT {select_expr} FROM threads").fetchall()
            for row in rows:
                tsrc = row["thread_source"]
                arole = row["agent_role"]
                if tsrc in ("subagent", "guardian_review") or (isinstance(arole, str) and arole.strip()):
                    continue
                sid = row["id"]
                if not isinstance(sid, str) or not sid.strip():
                    continue
                sid = sid.strip()
                raw_title = (
                    row["name"]
                    if isinstance(row["name"], str) and row["name"].strip()
                    else row["title"]
                    if isinstance(row["title"], str) and row["title"].strip()
                    else None
                )
                updated_ts = _parse_ts(row["updated_at_ms"]) or _parse_ts(row["updated_at"])
                branch = (
                    row["git_branch"]
                    if isinstance(row["git_branch"], str) and row["git_branch"].strip()
                    else row["branch"]
                    if isinstance(row["branch"], str) and row["branch"].strip()
                    else None
                )
                threads[sid] = {
                    "session_id": sid,
                    "rollout_path": row["rollout_path"] if isinstance(row["rollout_path"], str) else None,
                    "cwd": row["cwd"] if isinstance(row["cwd"], str) else None,
                    "title": raw_title,
                    "branch": branch,
                    "updated_ts": updated_ts,
                }
        except sqlite3.Error as exc:
            warnings.append(f"codex catalog read error: {exc}")
        finally:
            conn.close()
    return threads


def _infer_codex_id_from_path(path: Path) -> str:
    m = _UUID_TAIL_RE.search(path.stem)
    return m.group(1) if m else path.stem


def _parse_codex_session(
    path: Path | None,
    file_mtime: float,
    meta: dict[str, Any] | None,
    now: float,
) -> dict[str, Any] | None:
    records: list[dict[str, Any]] = []
    mtime = file_mtime
    if path is not None and path.is_file():
        records, mtime = _read_bounded_jsonl(path)

    if not records and not meta:
        return None

    session_id = (meta or {}).get("session_id") or (_infer_codex_id_from_path(path) if path else "unknown")
    cwd: str | None = (meta or {}).get("cwd")
    branch: str | None = (meta or {}).get("branch")
    catalog_title: str | None = (meta or {}).get("title")
    first_user_prompt: str | None = None
    catalog_ts: float | None = (meta or {}).get("updated_ts")
    record_ts: float | None = None
    last_seen_type: str | None = None
    status: str | None = None
    last_event: str | None = None
    rate_limit_exhausted = False

    for obj in records:
        rec_type = obj.get("type")
        if isinstance(rec_type, str) and rec_type:
            last_seen_type = rec_type

        ts = _parse_ts(obj.get("timestamp"))
        if ts is not None:
            record_ts = ts if record_ts is None else max(record_ts, ts)

        payload = obj.get("payload") if isinstance(obj.get("payload"), dict) else {}

        # Support top-level metadata if present in lightweight transcripts
        if isinstance(obj.get("cwd"), str) and obj["cwd"].strip() and not cwd:
            cwd = obj["cwd"].strip()
        if isinstance(obj.get("gitBranch"), str) and obj["gitBranch"].strip() and not branch:
            branch = obj["gitBranch"].strip()

        if rec_type == "session_meta":
            pid = payload.get("id")
            if isinstance(pid, str) and pid.strip():
                session_id = pid.strip()
            pcwd = payload.get("cwd")
            if isinstance(pcwd, str) and pcwd.strip():
                cwd = pcwd.strip()
            pgit = payload.get("git") if isinstance(payload.get("git"), dict) else {}
            pbranch = pgit.get("branch") or payload.get("branch")
            if isinstance(pbranch, str) and pbranch.strip():
                branch = pbranch.strip()
            continue

        if rec_type == "event_msg":
            ev = payload.get("type")
            if ev == "user_message":
                msg_text = payload.get("message") or payload.get("text")
                if first_user_prompt is None and isinstance(msg_text, str) and msg_text.strip():
                    first_user_prompt = msg_text.strip()
                status = "cut_off"
                last_event = "user_message"
            elif ev == "task_started":
                rate_limit_exhausted = False
                status = "cut_off"
                last_event = "task_started"
            elif ev == "task_complete":
                err = payload.get("error")
                codex_err_info = (
                    err.get("codex_error_info")
                    if isinstance(err, dict)
                    else payload.get("codex_error_info")
                )
                if codex_err_info == "usage_limit_exceeded" or _is_rate_limit_error(err) or _is_rate_limit_error(codex_err_info):
                    status = "usage_limit"
                    last_event = "usage_limit_exceeded"
                elif isinstance(err, dict) or (isinstance(err, str) and err.strip()):
                    status = "errored"
                    last_event = "task_error"
                else:
                    status = "finished"
                    last_event = "task_complete"
                    rate_limit_exhausted = False
            elif ev == "turn_aborted":
                status = "cut_off"
                last_event = "turn_aborted"
            elif ev in ("error", "stream_error"):
                if _is_rate_limit_error(payload) or _is_rate_limit_error(payload.get("message")):
                    status = "usage_limit"
                    last_event = f"{ev}:rate_limit"
                else:
                    status = "errored"
                    last_event = str(ev)
            elif ev == "token_count":
                rates = payload.get("rate_limits") if isinstance(payload.get("rate_limits"), dict) else {}
                for window in ("primary", "secondary"):
                    w = rates.get(window)
                    if isinstance(w, dict):
                        pct = w.get("used_percent")
                        if isinstance(pct, (int, float)) and not isinstance(pct, bool) and pct >= 100:
                            rate_limit_exhausted = True
                            status = "usage_limit"
                            last_event = "rate_limit_exhausted"
            elif ev == "item_completed":
                item = payload.get("item") if isinstance(payload.get("item"), dict) else {}
                item_type = item.get("type")
                item_phase = item.get("phase")
                if item_type in ("CommandExecution", "McpToolCall", "FileChange"):
                    status = "cut_off"
                    last_event = f"item_completed:{item_type}"
                elif item_type in ("AgentMessage", "message"):
                    if item_phase == "final_answer":
                        status = "finished"
                        last_event = "item_completed:final_answer"
                    elif item_phase == "commentary":
                        status = "cut_off"
                        last_event = "item_completed:commentary"

        elif rec_type == "response_item":
            ev = payload.get("type")
            if ev in ("function_call", "custom_tool_call", "function_call_output", "custom_tool_call_output"):
                status = "cut_off"
                last_event = str(ev)
            elif ev == "message":
                role = payload.get("role")
                phase = payload.get("phase")
                if role == "user":
                    if first_user_prompt is None:
                        txt = _extract_text_content(payload.get("content")).strip()
                        if txt:
                            first_user_prompt = txt
                    status = "cut_off"
                    last_event = "message:user"
                elif role == "assistant":
                    if phase == "commentary":
                        status = "cut_off"
                        last_event = "message:commentary"
                    else:
                        status = "finished"
                        last_event = f"message:{phase or 'assistant'}"

    explicit_ts = [t for t in (record_ts, catalog_ts) if t is not None]
    last_activity_ts = max(explicit_ts) if explicit_ts else (mtime or file_mtime or now)

    if status is None:
        if rate_limit_exhausted:
            status = "usage_limit"
            last_event = "rate_limit_exhausted"
        else:
            age = now - last_activity_ts
            status = "recent" if -60.0 <= age < _RECENT_WINDOW_S else "finished"
            last_event = last_seen_type or "catalog_only"

    title = _clean_title(catalog_title or first_user_prompt, f"Codex session {session_id[:8]}")
    needs_attention = status in ("usage_limit", "errored", "cut_off")
    project_root = _normalize_project_root(cwd)

    return {
        "id": f"codex:{session_id}",
        "source": "codex",
        "session_id": session_id,
        "title": title,
        "project_root": project_root,
        "branch": branch,
        "status": status,
        "needs_attention": needs_attention,
        "last_event": last_event or status,
        "last_activity_ts": float(last_activity_ts),
        "resume_cmd": f"codex resume {session_id}",
    }


def _scan_codex(codex_dir: Path, max_sessions: int, now: float, warnings: list[str]) -> list[dict[str, Any]]:
    if not codex_dir.exists():
        warnings.append(f"codex_dir missing: {codex_dir}")
        return []
    if not codex_dir.is_dir():
        warnings.append(f"codex_dir is not a directory: {codex_dir}")
        return []

    catalog_by_id: dict[str, dict[str, Any]] = {}
    catalog_by_path: dict[str, dict[str, Any]] = {}
    db_path = _select_codex_sqlite(codex_dir)
    if db_path is not None:
        try:
            catalog_by_id = _read_codex_catalog_ro(db_path, warnings)
            for meta in catalog_by_id.values():
                rpath = meta.get("rollout_path")
                if rpath:
                    catalog_by_path[os.path.normpath(rpath)] = meta
        except OSError as exc:
            warnings.append(f"codex catalog copy error: {exc}")

    sessions_dir = codex_dir / "sessions"
    files: list[tuple[Path, float]] = []
    if sessions_dir.is_dir() and not sessions_dir.is_symlink():
        files = _walk_files(sessions_dir, ".jsonl")
    elif not sessions_dir.exists() and not catalog_by_id:
        warnings.append(f"codex sessions_dir missing: {sessions_dir}")

    # Combine discovered rollout files and catalog-only threads, sorted by recency.
    seen_ids: set[str] = set()
    candidates: list[tuple[float, Path | None, dict[str, Any] | None]] = []

    for fpath, mtime in files:
        norm_p = os.path.normpath(str(fpath))
        meta = catalog_by_path.get(norm_p)
        sid = meta["session_id"] if meta else _infer_codex_id_from_path(fpath)
        if sid in catalog_by_id and meta is None:
            meta = catalog_by_id[sid]
        seen_ids.add(sid)
        ts = max(mtime, (meta or {}).get("updated_ts") or 0.0)
        candidates.append((ts, fpath, meta))

    for sid, meta in catalog_by_id.items():
        if sid in seen_ids:
            continue
        rpath_str = meta.get("rollout_path")
        rpath: Path | None = None
        mtime = 0.0
        if isinstance(rpath_str, str) and rpath_str:
            candidate_path = Path(rpath_str)
            if not candidate_path.is_absolute():
                candidate_path = codex_dir / candidate_path
            try:
                resolved_root = codex_dir.resolve()
                resolved_cand = candidate_path.resolve()
                if (
                    str(resolved_cand).startswith(str(resolved_root) + os.sep)
                    and candidate_path.is_file()
                    and not candidate_path.is_symlink()
                ):
                    rpath = candidate_path
                    mtime = float(candidate_path.stat().st_mtime)
            except OSError:
                rpath = None
        ts = max(mtime, meta.get("updated_ts") or 0.0)
        candidates.append((ts, rpath, meta))

    candidates.sort(key=lambda item: item[0], reverse=True)
    if max_sessions > 0:
        candidates = candidates[:max_sessions]

    sessions: list[dict[str, Any]] = []
    for ts, fpath, meta in candidates:
        try:
            sess = _parse_codex_session(fpath, ts, meta, now)
            if sess is not None:
                sessions.append(sess)
        except OSError as exc:
            warnings.append(f"codex read error: {exc}")
    return sessions


def _parse_open_swarm_run(path: Path, file_mtime: float) -> dict[str, Any] | None:
    raw = path.read_text(encoding="utf-8", errors="replace")
    rec = json.loads(raw)
    if not isinstance(rec, dict):
        return None

    run_id = str(rec.get("run_id") or path.stem).strip() or path.stem
    repo = rec.get("repo")
    project_root = _normalize_project_root(repo)

    raw_status = str(rec.get("status") or "").strip().lower()
    err = rec.get("error")
    has_err = bool(err)

    passed_count = rec.get("passed_count")
    agents_count = rec.get("agents_count")
    model = rec.get("model")

    if raw_status == "error" or has_err:
        status = "errored"
        last_event = str(err or "error")
    elif raw_status == "running":
        status = "running"
        last_event = "running"
    else:
        status = "completed"
        if isinstance(passed_count, int) and isinstance(agents_count, int):
            last_event = f"completed ({passed_count}/{agents_count} passed)"
        else:
            last_event = "completed"

    needs_attention = status in ("usage_limit", "errored", "cut_off")

    branch: str | None = rec.get("branch") if isinstance(rec.get("branch"), str) else None
    if not branch and isinstance(rec.get("results"), list):
        for res in rec["results"]:
            if isinstance(res, dict) and isinstance(res.get("branch"), str) and res["branch"].strip():
                branch = res["branch"].strip()
                break

    ts_candidates = [
        _parse_ts(rec.get("finished_at")),
        _parse_ts(rec.get("created_at")),
    ]
    valid_ts = [t for t in ts_candidates if t is not None]
    last_activity_ts = max(valid_ts) if valid_ts else file_mtime

    title_parts: list[str] = []
    if isinstance(model, str) and model.strip():
        title_parts.append(model.strip().split("/")[-1])
    else:
        title_parts.append("Open Swarm")
    if isinstance(passed_count, int) and isinstance(agents_count, int):
        title_parts.append(f"({passed_count}/{agents_count} passed)")
    title = _clean_title(" ".join(title_parts), f"Open Swarm {run_id}")

    return {
        "id": f"open_swarm:{run_id}",
        "source": "open_swarm",
        "session_id": run_id,
        "title": title,
        "project_root": project_root,
        "branch": branch,
        "status": status,
        "needs_attention": needs_attention,
        "last_event": last_event,
        "last_activity_ts": float(last_activity_ts),
        "resume_cmd": f"open-swarm run {run_id}",
    }


def _scan_open_swarm(runs_dir: Path, max_sessions: int, warnings: list[str]) -> list[dict[str, Any]]:
    if not runs_dir.exists():
        warnings.append(f"open_swarm_runs_dir missing: {runs_dir}")
        return []
    if not runs_dir.is_dir():
        warnings.append(f"open_swarm_runs_dir is not a directory: {runs_dir}")
        return []

    files: list[tuple[Path, float]] = []
    try:
        for item in runs_dir.iterdir():
            if item.is_symlink() or not item.is_file() or not item.name.endswith(".json"):
                continue
            try:
                files.append((item, float(item.stat().st_mtime)))
            except OSError:
                continue
    except OSError as exc:
        warnings.append(f"open_swarm_runs_dir read error: {exc}")
        return []

    files.sort(key=lambda item: item[1], reverse=True)
    if max_sessions > 0:
        files = files[:max_sessions]

    sessions: list[dict[str, Any]] = []
    for path, mtime in files:
        try:
            sess = _parse_open_swarm_run(path, mtime)
            if sess is not None:
                sessions.append(sess)
        except (OSError, ValueError, UnicodeError) as exc:
            warnings.append(f"open_swarm receipt error ({path.name}): {exc}")
    return sessions


def scan_fleet(
    claude_dir: Path | None = None,
    codex_dir: Path | None = None,
    open_swarm_runs_dir: Path | None = None,
    max_sessions_per_source: int = 40,
) -> dict[str, Any]:
    """Scan Claude Code, Codex, and Open Swarm run histories (strictly read-only)."""
    now = _now_ts()
    c_dir = Path(claude_dir) if claude_dir is not None else Path.home() / ".claude" / "projects"
    x_dir = Path(codex_dir) if codex_dir is not None else Path.home() / ".codex"
    s_dir = Path(open_swarm_runs_dir) if open_swarm_runs_dir is not None else Path.home() / ".open_swarm" / "runs"

    warnings: list[str] = []
    all_sessions: list[dict[str, Any]] = []
    all_sessions.extend(_scan_claude(c_dir, max_sessions_per_source, now, warnings))
    all_sessions.extend(_scan_codex(x_dir, max_sessions_per_source, now, warnings))
    all_sessions.extend(_scan_open_swarm(s_dir, max_sessions_per_source, warnings))

    grouped: dict[str, list[dict[str, Any]]] = {}
    for sess in all_sessions:
        root = sess.get("project_root") or "unknown"
        grouped.setdefault(root, []).append(sess)

    projects: list[dict[str, Any]] = []
    for root, sess_list in grouped.items():
        sess_list.sort(key=lambda s: s["last_activity_ts"], reverse=True)
        needs_attn_count = sum(1 for s in sess_list if s["needs_attention"])
        last_ts = max(s["last_activity_ts"] for s in sess_list)
        p_counts = {
            "total": len(sess_list),
            "claude": sum(1 for s in sess_list if s["source"] == "claude"),
            "codex": sum(1 for s in sess_list if s["source"] == "codex"),
            "open_swarm": sum(1 for s in sess_list if s["source"] == "open_swarm"),
            "needs_attention": needs_attn_count,
        }
        projects.append(
            {
                "key": root,
                "name": _project_name(root),
                "root": root,
                "needs_attention": needs_attn_count > 0,
                "sessions": sess_list,
                "counts": p_counts,
                "last_activity_ts": last_ts,
            }
        )

    projects.sort(key=lambda p: (not p["needs_attention"], -p["last_activity_ts"]))

    counts = {
        "total": len(all_sessions),
        "claude": sum(1 for s in all_sessions if s["source"] == "claude"),
        "codex": sum(1 for s in all_sessions if s["source"] == "codex"),
        "open_swarm": sum(1 for s in all_sessions if s["source"] == "open_swarm"),
        "active_7d": sum(1 for s in all_sessions if -60.0 <= (now - s["last_activity_ts"]) <= _SEVEN_DAYS_S),
        "needs_attention": sum(1 for s in all_sessions if s["needs_attention"]),
    }

    return {
        "generated_at": now,
        "counts": counts,
        "projects": projects,
        "warnings": warnings,
    }
