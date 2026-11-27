#!/usr/bin/env python3
"""OutrCore Open Swarm — Real Agent Tool Loop ACI (Phase 2B).

Pure Python 3 stdlib (+ sibling `swarm_run` and `metering`) module providing:
- `scrub_secrets`: regex-based secret redaction for command outputs, diffs, notes, and errors.
- `WorktreeToolkit`: strictly worktree-confined file, directory, search, diff, command (with
  macOS Seatbelt / Linux bwrap OS-level sandboxing), background process, board, note, and finish tools
  with harness-enforced `readonly_paths`, symlink/traversal protection, inline syntax diagnostics,
  closest-match edit hints, and `__pycache__` purging.
- `TOOLS_SCHEMA`: OpenAI chat-completion tool schema list for all 15 tools.
- `run_agent_loop`: multi-turn agent loop supporting native `tool_calls`, HTTP 400/422 fallback
  to fenced ```` ```tool ```` / ```` ```json ```` blocks, legacy ```` ```file:rel/path``` ```` blocks,
  KV-cache-preserving context compaction, `RunMeter` telemetry, and progress callbacks.
"""
from __future__ import annotations

from dataclasses import dataclass, field
import difflib
import fnmatch
import inspect
import json
import logging
import os
from pathlib import Path, PurePosixPath
import platform
import posixpath
import py_compile
import re
import shutil
import signal
import subprocess
import sys
import tempfile
import threading
import time
from typing import Any, Callable, Sequence
import urllib.error
import urllib.request

SWARM_DIR = Path(__file__).resolve().parent
if str(SWARM_DIR) not in sys.path:
    sys.path.insert(0, str(SWARM_DIR))

from metering import SpendCapExceeded  # noqa: E402
from swarm_run import _CLOSE_FENCE_RE, _FILE_HEADER_RE, _parse_sse_chunk  # noqa: E402

_LOG = logging.getLogger(__name__)

_READ_MAX_CHARS = 12_000
_CMD_MAX_CHARS = 6_000
_BG_BUFFER_MAX_CHARS = 200_000
_SEARCH_MAX_MATCHES = 40
_SEARCH_MAX_FILE_BYTES = 1_000_000
_SKIP_DIRS = frozenset({".git", "__pycache__"})

_TOOL_FENCE_RE = re.compile(
    r"^[ \t]*```(tool|json)[ \t]*\r?\n(.*?)^[ \t]*```[ \t]*$",
    re.DOTALL | re.MULTILINE | re.IGNORECASE,
)

_SECRET_PATTERNS: tuple[tuple[re.Pattern[str], str], ...] = (
    (
        re.compile(r"-----BEGIN [A-Z0-9 ]*PRIVATE KEY-----[\s\S]*?-----END [A-Z0-9 ]*PRIVATE KEY-----"),
        "[REDACTED]",
    ),
    (re.compile(r"Bearer\s+[A-Za-z0-9._-]{24,}"), "Bearer [REDACTED]"),
    (re.compile(r"sk-(?:proj-|live-|test-)?[A-Za-z0-9_-]{16,}"), "[REDACTED]"),
    (re.compile(r"(?:pk|rk)_(?:live|test)_[A-Za-z0-9]{12,}"), "[REDACTED]"),
    (re.compile(r"oc_(?:live|sess|proxy)_[A-Za-z0-9_-]{8,}"), "[REDACTED]"),
    (re.compile(r"(?:ghp|gho|ghu|ghs)_[A-Za-z0-9]{20,}"), "[REDACTED]"),
    (re.compile(r"github_pat_[A-Za-z0-9_]{20,}"), "[REDACTED]"),
    (re.compile(r"AKIA[0-9A-Z]{16}"), "[REDACTED]"),
)


def scrub_secrets(text: str) -> str:
    """Redact API keys, bearer tokens, and private key blocks from text."""
    if text is None:
        return ""
    out = text if isinstance(text, str) else str(text)
    for pattern, repl in _SECRET_PATTERNS:
        out = pattern.sub(repl, out)
    return out


_FALLBACK_TOOL_INSTRUCTIONS = (
    "\n\n## Tool Protocol\n"
    "Native function calling is unavailable on this backend. Invoke tools by emitting one or more "
    "fenced ```tool JSON blocks:\n"
    "```tool\n"
    '{"name": "<tool_name>", "arguments": {...}}\n'
    "```\n"
    "Available tools: read_file, search_code, list_dir, edit_file, write_file, run_command, "
    "start_bg_cmd, poll_bg_cmd, stop_bg_cmd, git_diff, post_board, read_board, claim_file, "
    "write_note, finish.\n"
    "Call `finish` with a concise summary when done."
)


TOOLS_SCHEMA: list[dict[str, Any]] = [
    {
        "type": "function",
        "function": {
            "name": "read_file",
            "description": "Read a UTF-8 file inside the worktree with 1-indexed line numbers (bounded to 12,000 chars).",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {"type": "string", "description": "Relative path inside the worktree."},
                    "start_line": {"type": "integer", "description": "1-indexed start line (default 1).", "default": 1},
                    "end_line": {"type": "integer", "description": "1-indexed inclusive end line (default 200).", "default": 200},
                },
                "required": ["path"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "search_code",
            "description": "Search worktree files (skipping .git and __pycache__) by regex or substring, up to 40 matches.",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "Regex pattern or literal substring to search for."},
                    "glob": {"type": "string", "description": "Optional glob filter such as '*.py' or 'src/*.py'.", "default": ""},
                },
                "required": ["query"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "list_dir",
            "description": "List entries of a directory inside the worktree (skipping .git and __pycache__).",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {"type": "string", "description": "Relative directory path inside worktree (default '' for root).", "default": ""},
                    "max_entries": {"type": "integer", "description": "Maximum entries to return (default 100).", "default": 100},
                },
                "required": [],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "edit_file",
            "description": "Surgically replace an exact, unique occurrence of old_text with new_text in an existing file.",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {"type": "string", "description": "Relative path inside the worktree."},
                    "old_text": {"type": "string", "description": "Exact unique text to replace."},
                    "new_text": {"type": "string", "description": "Replacement text."},
                },
                "required": ["path", "old_text", "new_text"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "write_file",
            "description": "Create or overwrite a file inside the worktree, creating parent directories as needed.",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {"type": "string", "description": "Relative path inside the worktree."},
                    "content": {"type": "string", "description": "Full file contents to write."},
                },
                "required": ["path", "content"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "run_command",
            "description": "Run a shell command inside the worktree (OS-sandboxed when available) with __pycache__ purged and PYTHONDONTWRITEBYTECODE=1.",
            "parameters": {
                "type": "object",
                "properties": {
                    "cmd": {"type": "string", "description": "Shell command to execute inside the worktree."},
                    "timeout_s": {"type": "integer", "description": "Optional timeout in seconds."},
                },
                "required": ["cmd"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "start_bg_cmd",
            "description": "Start a background command inside the worktree and return a handle_id for polling or stopping.",
            "parameters": {
                "type": "object",
                "properties": {
                    "cmd": {"type": "string", "description": "Shell command to run in the background."},
                    "name": {"type": "string", "description": "Optional short label for the background task.", "default": ""},
                },
                "required": ["cmd"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "poll_bg_cmd",
            "description": "Poll buffered output and status of a background command started with start_bg_cmd.",
            "parameters": {
                "type": "object",
                "properties": {
                    "handle_id": {"type": "string", "description": "Handle ID returned by start_bg_cmd."},
                    "max_chars": {"type": "integer", "description": "Maximum characters to return (default 4000).", "default": 4000},
                },
                "required": ["handle_id"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "stop_bg_cmd",
            "description": "Terminate a background command started with start_bg_cmd.",
            "parameters": {
                "type": "object",
                "properties": {
                    "handle_id": {"type": "string", "description": "Handle ID returned by start_bg_cmd."},
                },
                "required": ["handle_id"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "git_diff",
            "description": "Inspect uncommitted git diff inside the worktree.",
            "parameters": {
                "type": "object",
                "properties": {
                    "stat_only": {"type": "boolean", "description": "Return only git diff --stat summary when true.", "default": False},
                },
                "required": [],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "post_board",
            "description": "Post a coordination message to the shared swarm board.",
            "parameters": {
                "type": "object",
                "properties": {
                    "kind": {"type": "string", "description": "Message kind (e.g. 'status', 'finding', 'question', 'handoff')."},
                    "body": {"type": "string", "description": "Message body."},
                    "to_agent": {"type": "string", "description": "Recipient agent ID or 'all' (default 'all').", "default": "all"},
                },
                "required": ["kind", "body"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "read_board",
            "description": "Read messages from the shared swarm board since sequence number since_seq.",
            "parameters": {
                "type": "object",
                "properties": {
                    "since_seq": {"type": "integer", "description": "Return events with seq > since_seq (default 0).", "default": 0},
                },
                "required": [],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "claim_file",
            "description": "Claim ownership of a worktree-relative file on the shared swarm board to avoid collisions.",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {"type": "string", "description": "Relative path inside the worktree to claim."},
                },
                "required": ["path"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "write_note",
            "description": "Append a timestamped Markdown section to the agent's continuity note file.",
            "parameters": {
                "type": "object",
                "properties": {
                    "title": {"type": "string", "description": "Section title."},
                    "body": {"type": "string", "description": "Markdown note body."},
                },
                "required": ["title", "body"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "finish",
            "description": "Mark the task complete with a concise summary.",
            "parameters": {
                "type": "object",
                "properties": {
                    "summary": {"type": "string", "description": "Summary of changes and verification results."},
                },
                "required": ["summary"],
            },
        },
    },
]

_KNOWN_TOOL_NAMES = frozenset(t["function"]["name"] for t in TOOLS_SCHEMA)


def _normalize_readonly_entry(raw: str) -> str:
    cleaned = str(raw or "").replace("\\", "/").strip().lstrip("/")
    if not cleaned:
        return ""
    norm = posixpath.normpath(cleaned).rstrip("/")
    if norm in ("", "."):
        return ""
    return norm


def _sanitize_agent_id(agent_id: str) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9._-]+", "-", str(agent_id or "agent-a")).strip(".-")
    if not cleaned or cleaned in (".", ".."):
        return "agent-a"
    return cleaned


def _bound_streams(stdout: str, stderr: str, limit: int = _CMD_MAX_CHARS) -> tuple[str, str]:
    """Bound stdout and stderr so len(stdout) + len(stderr) <= limit."""
    if len(stdout) + len(stderr) <= limit:
        return stdout, stderr
    marker = "... [truncated]\n"
    err_budget = min(len(stderr), max(limit // 2, limit - len(stdout)))
    out_budget = limit - err_budget

    def _tail_cap(text: str, budget: int) -> str:
        if len(text) <= budget:
            return text
        if budget <= len(marker):
            return text[-budget:] if budget > 0 else ""
        return marker + text[-(budget - len(marker)) :]

    return _tail_cap(stdout, out_budget), _tail_cap(stderr, err_budget)


def _sb_quote(val: str) -> str:
    return '"' + str(val).replace("\\", "\\\\").replace('"', '\\"') + '"'


def _find_closest_match_snippet(
    content: str,
    old_text: str,
    context_lines: int = 4,
) -> tuple[int, int, float, str] | None:
    """Return (start_1idx, end_1idx, ratio, line_numbered_snippet) for the closest line window."""
    file_lines = content.splitlines()
    old_lines = old_text.splitlines()
    if not file_lines or not old_lines:
        return None
    win = min(len(file_lines), max(1, len(old_lines)))
    best_start = 0
    best_ratio = -1.0
    max_i = max(0, len(file_lines) - win)
    for i in range(max_i + 1):
        cand = "\n".join(file_lines[i : i + win])
        sm = difflib.SequenceMatcher(None, old_text, cand)
        if sm.quick_ratio() < best_ratio * 0.85:
            continue
        ratio = sm.ratio()
        if ratio > best_ratio:
            best_ratio = ratio
            best_start = i
    ctx_start = max(0, best_start - context_lines)
    ctx_end = min(len(file_lines), best_start + win + context_lines)
    snippet = "\n".join(f"{idx}: {file_lines[idx - 1]}" for idx in range(ctx_start + 1, ctx_end + 1))
    return ctx_start + 1, ctx_end, max(0.0, best_ratio), snippet


def _run_inline_diagnostics(target: Path) -> list[str]:
    """Run instant syntax/lint diagnostics for .py, .json, and .js/.mjs files without polluting worktree."""
    suffix = target.suffix.lower()
    if suffix == ".py":
        with tempfile.TemporaryDirectory() as td:
            cfile = Path(td) / "check.pyc"
            try:
                py_compile.compile(str(target), cfile=str(cfile), doraise=True)
            except py_compile.PyCompileError as exc:
                return [scrub_secrets(str(exc))]
        return []
    if suffix == ".json":
        try:
            json.loads(target.read_text(encoding="utf-8", errors="replace"))
        except json.JSONDecodeError as exc:
            return [f"JSONDecodeError at line {exc.lineno}, column {exc.colno}: {exc.msg}"]
        return []
    if suffix in (".js", ".mjs"):
        node_bin = shutil.which("node")
        if node_bin:
            try:
                proc = subprocess.run(
                    [node_bin, "--check", str(target)],
                    capture_output=True,
                    text=True,
                    timeout=10,
                )
                if proc.returncode != 0:
                    msg = (proc.stderr or proc.stdout or f"node --check exited {proc.returncode}").strip()
                    return [scrub_secrets(msg)]
            except Exception as exc:
                return [scrub_secrets(f"node --check failed: {exc}")]
        return []
    return []


def _call_with_accepted_kwargs(fn: Callable[..., Any], candidate_kwargs: dict[str, Any]) -> Any:
    """Invoke `fn` passing only the keyword parameters it accepts (or all if **kwargs)."""
    try:
        sig = inspect.signature(fn)
    except (TypeError, ValueError):
        return fn(**candidate_kwargs)
    params = sig.parameters
    if any(p.kind == inspect.Parameter.VAR_KEYWORD for p in params.values()):
        return fn(**candidate_kwargs)
    filtered = {k: v for k, v in candidate_kwargs.items() if k in params}
    return fn(**filtered)


@dataclass
class _BgProcess:
    handle_id: str
    name: str
    cmd: str
    proc: subprocess.Popen[bytes]
    sandbox: str
    started_at: float
    sb_profile_path: Path | None = None
    lock: threading.Lock = field(default_factory=threading.Lock)
    buffer: bytearray = field(default_factory=bytearray)
    read_offset: int = 0
    dropped_prefix: bool = False
    reader_thread: threading.Thread | None = None
    stopped: bool = False


class WorktreeToolkit:
    """Worktree-confined agent toolkit with harness-enforced read-only path protection."""

    def __init__(
        self,
        worktree: Path,
        readonly_paths: Sequence[str] = (),
        agent_id: str = "agent-a",
        max_cmd_timeout_s: int = 120,
        sandbox_mode: str = "auto",
        board_store: Any = None,
        run_id: str = "",
        board_callback: Callable[..., Any] | None = None,
    ) -> None:
        self.worktree = Path(worktree).resolve()
        self.agent_id = _sanitize_agent_id(agent_id)
        self.max_cmd_timeout_s = max(1, int(max_cmd_timeout_s))
        norm_mode = str(sandbox_mode or "auto").strip().lower()
        if norm_mode not in ("auto", "strict", "off"):
            raise ValueError(f"Invalid sandbox_mode: {sandbox_mode!r}")
        self.sandbox_mode = norm_mode
        self.board_store = board_store
        self.run_id = str(run_id or "")
        self.board_callback = board_callback
        norms: list[str] = []
        for entry in readonly_paths:
            norm = _normalize_readonly_entry(entry)
            if norm and norm not in norms:
                norms.append(norm)
        self.readonly_paths: tuple[str, ...] = tuple(norms)
        self.written_files: list[str] = []
        self.readonly_violations: list[str] = []
        self.notes_path: str | None = None
        self.finished: bool = False
        self.finish_summary: str = ""
        self._bg_lock = threading.Lock()
        self._bg_handles: dict[str, _BgProcess] = {}
        self._bg_seq = 0
        self._seatbelt_nested_blocked: bool | None = None
        self._bwrap_nested_blocked: bool | None = None

    def _resolve_path(self, raw_path: str) -> tuple[str, Path] | tuple[None, str]:
        """Confine `raw_path` strictly inside `self.worktree`, rejecting .., .git/, and symlink escapes."""
        if not isinstance(raw_path, str) or "\x00" in raw_path:
            return None, "Invalid path"
        cleaned = raw_path.replace("\\", "/").strip()
        if not cleaned:
            return None, "Empty path"
        if cleaned.startswith("/") or os.path.isabs(cleaned):
            return None, f"Path must be relative to worktree: {raw_path}"
        raw_parts = [p for p in cleaned.split("/") if p not in ("", ".")]
        if not raw_parts:
            return None, f"Invalid path: {raw_path}"
        if ".." in raw_parts:
            return None, f"Path traversal rejected: {raw_path}"
        if ".git" in raw_parts:
            return None, f"Protected .git path rejected: {raw_path}"

        candidate = self.worktree.joinpath(*raw_parts)
        try:
            resolved = candidate.resolve()
            rel_resolved = resolved.relative_to(self.worktree)
        except (OSError, ValueError):
            return None, f"Path escapes worktree: {raw_path}"

        if resolved == self.worktree or ".git" in rel_resolved.parts:
            return None, f"Protected path rejected: {raw_path}"

        rel_posix = PurePosixPath(*raw_parts).as_posix()
        return rel_posix, resolved

    def _resolve_dir_path(self, raw_path: str) -> tuple[str, Path] | tuple[None, str]:
        """Confine a directory `raw_path` (including '' or '.') inside `self.worktree`."""
        if raw_path is None:
            raw_path = ""
        if not isinstance(raw_path, str) or "\x00" in raw_path:
            return None, "Invalid path"
        cleaned = raw_path.replace("\\", "/").strip()
        if cleaned in ("", ".", "./"):
            return ".", self.worktree
        if cleaned.startswith("/") or os.path.isabs(cleaned):
            return None, f"Path must be relative to worktree: {raw_path}"
        raw_parts = [p for p in cleaned.split("/") if p not in ("", ".")]
        if not raw_parts:
            return ".", self.worktree
        if ".." in raw_parts:
            return None, f"Path traversal rejected: {raw_path}"
        if ".git" in raw_parts:
            return None, f"Protected .git path rejected: {raw_path}"

        candidate = self.worktree.joinpath(*raw_parts)
        try:
            resolved = candidate.resolve()
            if resolved != self.worktree:
                rel_resolved = resolved.relative_to(self.worktree)
                if ".git" in rel_resolved.parts:
                    return None, f"Protected path rejected: {raw_path}"
        except (OSError, ValueError):
            return None, f"Path escapes worktree: {raw_path}"

        rel_posix = PurePosixPath(*raw_parts).as_posix()
        return rel_posix, resolved

    def _is_readonly(self, rel_posix: str) -> bool:
        for ro in self.readonly_paths:
            if rel_posix == ro or rel_posix.startswith(ro + "/"):
                return True
        return False

    def _record_written(self, rel_posix: str) -> None:
        if rel_posix not in self.written_files:
            self.written_files.append(rel_posix)

    def _record_readonly_violation(self, rel_posix: str) -> None:
        if rel_posix not in self.readonly_violations:
            self.readonly_violations.append(rel_posix)

    def _purge_pycache(self) -> None:
        for pc_dir in self.worktree.rglob("__pycache__"):
            if pc_dir.is_dir():
                shutil.rmtree(pc_dir, ignore_errors=True)

    def _build_seatbelt_profile(self) -> str:
        home_raw = Path.home()
        home_res = home_raw.resolve()
        home_roots: list[Path] = [home_res]
        if home_raw != home_res:
            home_roots.append(home_raw)

        lines = [
            "(version 1)",
            "(deny default)",
            "(allow process*)",
            "(allow sysctl-read)",
            "(allow mach*)",
            "(allow ipc*)",
            "(allow signal)",
            "(allow system*)",
            "(allow file-ioctl)",
            "(allow file-read*)",
        ]
        for hroot in home_roots:
            for hidden_name in (".ssh", ".aws", ".gnupg", ".open_swarm"):
                lines.append(f"(deny file-read* (subpath {_sb_quote(str(hroot / hidden_name))}))")
        # Keep worktree itself readable even if it sits under ~/.open_swarm
        lines.append(f"(allow file-read* (subpath {_sb_quote(str(self.worktree))}))")
        lines.append(f"(allow file-write* (subpath {_sb_quote(str(self.worktree))}))")
        lines.append('(allow file-write* (subpath "/tmp") (subpath "/private/tmp"))')
        lines.append('(allow file-write* (literal "/dev/null") (literal "/dev/zero") (literal "/dev/tty") (literal "/dev/urandom"))')
        lines.append("(allow network*)")
        return "\n".join(lines) + "\n"

    def _build_bwrap_argv(self, cmd: str, env: dict[str, str]) -> list[str]:
        home_dir = str(Path(env.get("HOME") or Path.home()).resolve())
        return [
            "bwrap",
            "--die-with-parent",
            "--ro-bind",
            "/",
            "/",
            "--dev",
            "/dev",
            "--proc",
            "/proc",
            "--tmpfs",
            "/tmp",
            "--tmpfs",
            home_dir,
            "--bind",
            str(self.worktree),
            str(self.worktree),
            "--chdir",
            str(self.worktree),
            "/bin/sh",
            "-lc",
            cmd,
        ]

    @staticmethod
    def _is_seatbelt_init_error(returncode: int, stderr: str) -> bool:
        if returncode == 0:
            return False
        low = (stderr or "").lower()
        return "sandbox-exec:" in low or "sandbox_apply" in low or "sandbox_init" in low

    @staticmethod
    def _is_bwrap_init_error(returncode: int, stderr: str) -> bool:
        if returncode == 0:
            return False
        low = (stderr or "").lower()
        return (
            "bwrap:" in low
            or "creating new namespace failed" in low
            or "no permissions to create a new namespace" in low
            or "setting up uid map" in low
        )

    def read_file(self, path: str, start_line: int = 1, end_line: int = 200) -> dict[str, Any]:
        resolved = self._resolve_path(path)
        if resolved[0] is None:
            return {"ok": False, "error": resolved[1]}
        rel, target = resolved
        if not target.exists() or not target.is_file():
            return {"ok": False, "error": f"File not found: {rel}"}

        try:
            s_line = max(1, int(start_line))
            e_line = max(s_line, int(end_line))
        except (TypeError, ValueError):
            return {"ok": False, "error": "start_line and end_line must be integers"}

        text = target.read_text(encoding="utf-8", errors="replace")
        lines = text.splitlines()
        total_lines = len(lines)
        if total_lines == 0 or s_line > total_lines:
            return {
                "ok": True,
                "path": rel,
                "start_line": s_line,
                "end_line": min(e_line, total_lines),
                "total_lines": total_lines,
                "content": "",
            }

        e_line = min(e_line, total_lines)
        trunc_suffix = "\n... [truncated]"
        out_lines: list[str] = []
        used_chars = 0
        actual_end = s_line - 1
        truncated = False
        for idx in range(s_line, e_line + 1):
            formatted = f"{idx}: {lines[idx - 1]}"
            sep = 1 if out_lines else 0
            budget = _READ_MAX_CHARS - len(trunc_suffix)
            if used_chars + sep + len(formatted) > budget:
                if not out_lines:
                    max_first = max(0, budget)
                    out_lines.append(formatted[:max_first])
                    actual_end = idx
                truncated = True
                break
            out_lines.append(formatted)
            used_chars += sep + len(formatted)
            actual_end = idx

        content = "\n".join(out_lines)
        if truncated and len(content) + len(trunc_suffix) <= _READ_MAX_CHARS:
            content += trunc_suffix

        return {
            "ok": True,
            "path": rel,
            "start_line": s_line,
            "end_line": actual_end,
            "total_lines": total_lines,
            "content": content,
        }

    def list_dir(self, path: str = "", max_entries: int = 100) -> dict[str, Any]:
        resolved = self._resolve_dir_path(path)
        if resolved[0] is None:
            return {"ok": False, "error": resolved[1]}
        rel_dir, target = resolved
        if not target.exists() or not target.is_dir():
            return {"ok": False, "error": f"Directory not found: {rel_dir}"}

        try:
            limit = max(1, min(int(max_entries), 1000))
        except (TypeError, ValueError):
            return {"ok": False, "error": "max_entries must be an integer"}

        entries: list[dict[str, Any]] = []
        truncated = False
        try:
            children = sorted(target.iterdir(), key=lambda p: p.name)
        except OSError as exc:
            return {"ok": False, "error": f"Unable to list directory {rel_dir}: {exc}"}

        for child in children:
            name = child.name
            if name in _SKIP_DIRS or name.endswith(".pyc"):
                continue
            try:
                resolved_c = child.resolve()
                rel_res = resolved_c.relative_to(self.worktree)
                if ".git" in rel_res.parts:
                    continue
            except (OSError, ValueError):
                continue

            if len(entries) >= limit:
                truncated = True
                break
            is_dir = child.is_dir()
            rel_child = child.relative_to(self.worktree).as_posix()
            size = 0
            if not is_dir:
                try:
                    size = child.stat().st_size
                except OSError:
                    size = 0
            entries.append(
                {
                    "name": name,
                    "path": rel_child,
                    "type": "dir" if is_dir else "file",
                    "size": size,
                }
            )

        return {
            "ok": True,
            "path": rel_dir,
            "entries": entries,
            "truncated": truncated,
        }

    def search_code(self, query: str, glob: str = "") -> dict[str, Any]:
        if not isinstance(query, str) or not query:
            return {"ok": False, "error": "Empty query"}
        glob_str = str(glob or "").replace("\\", "/").strip()
        try:
            pattern = re.compile(query)
        except re.error:
            pattern = re.compile(re.escape(query))

        matches: list[dict[str, Any]] = []
        truncated = False

        for dirpath, dirnames, filenames in os.walk(self.worktree, followlinks=False):
            dirnames[:] = sorted(d for d in dirnames if d not in _SKIP_DIRS)
            cur_dir = Path(dirpath)
            for fname in sorted(filenames):
                if fname.endswith(".pyc"):
                    continue
                fpath = cur_dir / fname
                if fpath.is_symlink():
                    try:
                        resolved_f = fpath.resolve()
                        rel_res = resolved_f.relative_to(self.worktree)
                        if ".git" in rel_res.parts:
                            continue
                    except (OSError, ValueError):
                        continue
                rel = fpath.relative_to(self.worktree).as_posix()
                if glob_str:
                    matched_glob = fnmatch.fnmatch(rel, glob_str) or (
                        "/" not in glob_str and fnmatch.fnmatch(fname, glob_str)
                    )
                    if not matched_glob:
                        continue
                try:
                    if fpath.stat().st_size > _SEARCH_MAX_FILE_BYTES:
                        continue
                    raw = fpath.read_bytes()
                except OSError:
                    continue
                if b"\x00" in raw[:4096]:
                    continue
                text = raw.decode("utf-8", errors="replace")
                for lineno, line in enumerate(text.splitlines(), start=1):
                    if pattern.search(line):
                        matches.append({"path": rel, "line": lineno, "text": line[:300]})
                        if len(matches) >= _SEARCH_MAX_MATCHES:
                            truncated = True
                            break
                if truncated:
                    break
            if truncated:
                break

        return {
            "ok": True,
            "query": query,
            "glob": glob_str,
            "matches": matches,
            "truncated": truncated,
        }

    def edit_file(self, path: str, old_text: str, new_text: str) -> dict[str, Any]:
        resolved = self._resolve_path(path)
        if resolved[0] is None:
            return {"ok": False, "error": resolved[1]}
        rel, target = resolved
        if self._is_readonly(rel):
            self._record_readonly_violation(rel)
            return {"ok": False, "error": f"Read-only path: {rel}"}
        if not isinstance(old_text, str) or old_text == "":
            return {"ok": False, "error": "old_text must be non-empty"}
        if not isinstance(new_text, str):
            return {"ok": False, "error": "new_text must be a string"}
        if not target.exists() or not target.is_file():
            return {"ok": False, "error": f"File not found: {rel}"}

        content = target.read_text(encoding="utf-8", errors="replace")
        count = content.count(old_text)
        if count == 0:
            closest = _find_closest_match_snippet(content, old_text, context_lines=4)
            if closest is not None:
                c_start, c_end, ratio, snippet = closest
                return {
                    "ok": False,
                    "error": (
                        f"old_text not found in {rel}. "
                        f"Closest match (lines {c_start}-{c_end}, similarity {ratio:.2f}):\n{snippet}"
                    ),
                    "closest_match": snippet,
                }
            return {"ok": False, "error": f"old_text not found in {rel}"}
        if count > 1:
            return {"ok": False, "error": f"old_text matched {count} times in {rel}; must be unique"}

        updated = content.replace(old_text, new_text, 1)
        target.write_text(updated, encoding="utf-8")
        self._record_written(rel)
        diags = _run_inline_diagnostics(target)
        return {"ok": True, "path": rel, "diagnostics": diags}

    def write_file(self, path: str, content: str) -> dict[str, Any]:
        resolved = self._resolve_path(path)
        if resolved[0] is None:
            return {"ok": False, "error": resolved[1]}
        rel, target = resolved
        if self._is_readonly(rel):
            self._record_readonly_violation(rel)
            return {"ok": False, "error": f"Read-only path: {rel}"}
        if target.exists() and target.is_dir():
            return {"ok": False, "error": f"Target is a directory: {rel}"}
        if not isinstance(content, str):
            content = str(content)

        target.parent.mkdir(parents=True, exist_ok=True)
        body = content
        if body and not body.endswith("\n"):
            body += "\n"
        target.write_text(body, encoding="utf-8")
        self._record_written(rel)
        diags = _run_inline_diagnostics(target)
        return {
            "ok": True,
            "path": rel,
            "bytes": len(body.encode("utf-8")),
            "diagnostics": diags,
        }

    def _run_direct(self, cmd: str, env: dict[str, str], timeout_s: int) -> subprocess.CompletedProcess[str]:
        return subprocess.run(
            cmd,
            cwd=str(self.worktree),
            shell=True,
            env=env,
            capture_output=True,
            text=True,
            timeout=timeout_s,
        )

    def run_command(self, cmd: str, timeout_s: int | None = None) -> dict[str, Any]:
        if not isinstance(cmd, str) or not cmd.strip():
            return {
                "ok": False,
                "returncode": -1,
                "stdout": "",
                "stderr": "Empty command",
                "elapsed_s": 0.0,
                "sandbox": "unsandboxed",
            }

        self._purge_pycache()

        env = dict(os.environ)
        env["PYTHONDONTWRITEBYTECODE"] = "1"
        eff_timeout = (
            self.max_cmd_timeout_s
            if timeout_s is None
            else max(1, min(int(timeout_s), self.max_cmd_timeout_s))
        )

        sandbox_used = "unsandboxed"
        t0 = time.monotonic()
        try:
            if self.sandbox_mode == "off":
                proc = self._run_direct(cmd, env, eff_timeout)
                sandbox_used = "unsandboxed"
            elif sys.platform == "darwin" and shutil.which("sandbox-exec"):
                sb_env = dict(env)
                sb_env["TMPDIR"] = "/tmp"
                profile_txt = self._build_seatbelt_profile()
                sb_file: Path | None = None
                try:
                    with tempfile.NamedTemporaryFile(
                        mode="w", suffix=".sb", delete=False, encoding="utf-8"
                    ) as fh:
                        fh.write(profile_txt)
                        sb_file = Path(fh.name)
                    sandbox_used = "seatbelt"
                    proc = subprocess.run(
                        ["sandbox-exec", "-f", str(sb_file), "/bin/sh", "-lc", cmd],
                        cwd=str(self.worktree),
                        env=sb_env,
                        capture_output=True,
                        text=True,
                        timeout=eff_timeout,
                    )
                    if self._is_seatbelt_init_error(proc.returncode, proc.stderr or ""):
                        if self.sandbox_mode == "strict":
                            elapsed_s = round(time.monotonic() - t0, 3)
                            out, err = _bound_streams(
                                scrub_secrets(proc.stdout or ""),
                                scrub_secrets(proc.stderr or ""),
                            )
                            return {
                                "ok": False,
                                "returncode": proc.returncode,
                                "stdout": out,
                                "stderr": err,
                                "elapsed_s": elapsed_s,
                                "sandbox": "seatbelt",
                            }
                        self._seatbelt_nested_blocked = True
                        _LOG.warning(
                            "sandbox: unsandboxed-nested-fallback (sandbox-exec nested init failure)"
                        )
                        sandbox_used = "unsandboxed-nested-fallback"
                        proc = self._run_direct(cmd, env, eff_timeout)
                    else:
                        self._seatbelt_nested_blocked = False
                finally:
                    if sb_file is not None:
                        sb_file.unlink(missing_ok=True)
            elif sys.platform.startswith("linux") and shutil.which("bwrap"):
                sandbox_used = "bwrap"
                bwrap_argv = self._build_bwrap_argv(cmd, env)
                proc = subprocess.run(
                    bwrap_argv,
                    cwd=str(self.worktree),
                    env=env,
                    capture_output=True,
                    text=True,
                    timeout=eff_timeout,
                )
                if self._is_bwrap_init_error(proc.returncode, proc.stderr or ""):
                    if self.sandbox_mode == "strict":
                        elapsed_s = round(time.monotonic() - t0, 3)
                        out, err = _bound_streams(
                            scrub_secrets(proc.stdout or ""),
                            scrub_secrets(proc.stderr or ""),
                        )
                        return {
                            "ok": False,
                            "returncode": proc.returncode,
                            "stdout": out,
                            "stderr": err,
                            "elapsed_s": elapsed_s,
                            "sandbox": "bwrap",
                        }
                    self._bwrap_nested_blocked = True
                    _LOG.warning(
                        "sandbox: unsandboxed-nested-fallback (bwrap nested init failure)"
                    )
                    sandbox_used = "unsandboxed-nested-fallback"
                    proc = self._run_direct(cmd, env, eff_timeout)
                else:
                    self._bwrap_nested_blocked = False
            else:
                if self.sandbox_mode == "strict":
                    return {
                        "ok": False,
                        "returncode": -1,
                        "stdout": "",
                        "stderr": "Strict sandbox requested but no OS sandbox tool is available",
                        "elapsed_s": 0.0,
                        "sandbox": "strict-unavailable",
                    }
                sandbox_used = "unsandboxed"
                proc = self._run_direct(cmd, env, eff_timeout)

            elapsed_s = round(time.monotonic() - t0, 3)
            out, err = _bound_streams(
                scrub_secrets(proc.stdout or ""),
                scrub_secrets(proc.stderr or ""),
            )
            return {
                "ok": proc.returncode == 0,
                "returncode": proc.returncode,
                "stdout": out,
                "stderr": err,
                "elapsed_s": elapsed_s,
                "sandbox": sandbox_used,
            }
        except subprocess.TimeoutExpired as exc:
            elapsed_s = round(time.monotonic() - t0, 3)
            raw_out = exc.stdout.decode("utf-8", errors="replace") if isinstance(exc.stdout, bytes) else (exc.stdout or "")
            raw_err = exc.stderr.decode("utf-8", errors="replace") if isinstance(exc.stderr, bytes) else (exc.stderr or "")
            timeout_msg = f"Command timed out after {eff_timeout}s"
            err_full = (raw_err.rstrip() + "\n" + timeout_msg).strip() if raw_err else timeout_msg
            out, err = _bound_streams(scrub_secrets(raw_out), scrub_secrets(err_full))
            return {
                "ok": False,
                "returncode": -1,
                "stdout": out,
                "stderr": err,
                "elapsed_s": elapsed_s,
                "sandbox": sandbox_used,
            }

    def _can_use_seatbelt_bg(self) -> bool:
        if self._seatbelt_nested_blocked is not None:
            return not self._seatbelt_nested_blocked
        try:
            probe = subprocess.run(
                ["sandbox-exec", "-p", "(version 1)(allow default)", "/usr/bin/true"],
                capture_output=True,
                text=True,
                timeout=5,
            )
            blocked = self._is_seatbelt_init_error(probe.returncode, probe.stderr or "")
            self._seatbelt_nested_blocked = blocked
            return not blocked
        except Exception:
            self._seatbelt_nested_blocked = True
            return False

    def _can_use_bwrap_bg(self) -> bool:
        if self._bwrap_nested_blocked is not None:
            return not self._bwrap_nested_blocked
        try:
            probe = subprocess.run(
                ["bwrap", "--die-with-parent", "--ro-bind", "/", "/", "/bin/true"],
                capture_output=True,
                text=True,
                timeout=5,
            )
            blocked = self._is_bwrap_init_error(probe.returncode, probe.stderr or "")
            self._bwrap_nested_blocked = blocked
            return not blocked
        except Exception:
            self._bwrap_nested_blocked = True
            return False

    def start_bg_cmd(self, cmd: str, name: str = "") -> dict[str, Any]:
        if not isinstance(cmd, str) or not cmd.strip():
            return {"ok": False, "error": "Empty command"}

        self._purge_pycache()
        env = dict(os.environ)
        env["PYTHONDONTWRITEBYTECODE"] = "1"
        clean_name = re.sub(r"[^A-Za-z0-9._-]+", "-", str(name or "").strip()).strip("-")

        sb_file: Path | None = None
        argv: list[str]
        popen_env = env
        sandbox_used = "unsandboxed"

        if self.sandbox_mode == "off":
            argv = ["/bin/sh", "-lc", cmd]
        elif sys.platform == "darwin" and shutil.which("sandbox-exec"):
            if self._can_use_seatbelt_bg():
                popen_env = dict(env)
                popen_env["TMPDIR"] = "/tmp"
                with tempfile.NamedTemporaryFile(
                    mode="w", suffix=".sb", delete=False, encoding="utf-8"
                ) as fh:
                    fh.write(self._build_seatbelt_profile())
                    sb_file = Path(fh.name)
                argv = ["sandbox-exec", "-f", str(sb_file), "/bin/sh", "-lc", cmd]
                sandbox_used = "seatbelt"
            elif self.sandbox_mode == "strict":
                return {
                    "ok": False,
                    "error": "Strict sandbox requested but sandbox-exec failed in nested sandbox",
                }
            else:
                _LOG.warning("sandbox: unsandboxed-nested-fallback (start_bg_cmd)")
                argv = ["/bin/sh", "-lc", cmd]
                sandbox_used = "unsandboxed-nested-fallback"
        elif sys.platform.startswith("linux") and shutil.which("bwrap"):
            if self._can_use_bwrap_bg():
                argv = self._build_bwrap_argv(cmd, env)
                sandbox_used = "bwrap"
            elif self.sandbox_mode == "strict":
                return {
                    "ok": False,
                    "error": "Strict sandbox requested but bwrap failed in nested sandbox",
                }
            else:
                _LOG.warning("sandbox: unsandboxed-nested-fallback (start_bg_cmd bwrap)")
                argv = ["/bin/sh", "-lc", cmd]
                sandbox_used = "unsandboxed-nested-fallback"
        else:
            if self.sandbox_mode == "strict":
                return {
                    "ok": False,
                    "error": "Strict sandbox requested but no OS sandbox tool is available",
                }
            argv = ["/bin/sh", "-lc", cmd]

        try:
            proc = subprocess.Popen(
                argv,
                cwd=str(self.worktree),
                env=popen_env,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                stdin=subprocess.DEVNULL,
                start_new_session=True,
            )
        except Exception as exc:
            if sb_file is not None:
                sb_file.unlink(missing_ok=True)
            return {"ok": False, "error": scrub_secrets(f"Failed to start background command: {exc}")}

        with self._bg_lock:
            self._bg_seq += 1
            suffix = f"-{clean_name}" if clean_name else ""
            handle_id = f"bg-{self._bg_seq}{suffix}"
            handle = _BgProcess(
                handle_id=handle_id,
                name=clean_name or handle_id,
                cmd=cmd,
                proc=proc,
                sandbox=sandbox_used,
                started_at=time.monotonic(),
                sb_profile_path=sb_file,
            )
            self._bg_handles[handle_id] = handle

        def _drain() -> None:
            pipe = proc.stdout
            if pipe is None:
                return
            try:
                while True:
                    chunk = pipe.read(4096)
                    if not chunk:
                        break
                    with handle.lock:
                        handle.buffer.extend(chunk)
                        if len(handle.buffer) > _BG_BUFFER_MAX_CHARS:
                            excess = len(handle.buffer) - _BG_BUFFER_MAX_CHARS
                            del handle.buffer[:excess]
                            handle.read_offset = max(0, handle.read_offset - excess)
                            handle.dropped_prefix = True
            except Exception:
                pass
            finally:
                try:
                    pipe.close()
                except Exception:
                    pass

        t = threading.Thread(target=_drain, daemon=True)
        handle.reader_thread = t
        t.start()

        return {
            "ok": True,
            "handle_id": handle_id,
            "name": handle.name,
            "pid": proc.pid,
            "sandbox": sandbox_used,
        }

    def poll_bg_cmd(self, handle_id: str, max_chars: int = 4000) -> dict[str, Any]:
        with self._bg_lock:
            handle = self._bg_handles.get(str(handle_id or ""))
        if handle is None:
            return {"ok": False, "error": f"Unknown handle_id: {handle_id}"}

        try:
            limit = max(64, min(int(max_chars), 32_000))
        except (TypeError, ValueError):
            limit = 4000

        rc = handle.proc.poll()
        if rc is not None and handle.reader_thread is not None and handle.reader_thread.is_alive():
            handle.reader_thread.join(timeout=0.1)

        with handle.lock:
            new_bytes = bytes(handle.buffer[handle.read_offset :])
            handle.read_offset = len(handle.buffer)
            dropped = handle.dropped_prefix
            handle.dropped_prefix = False

        text = scrub_secrets(new_bytes.decode("utf-8", errors="replace"))
        truncated = False
        if len(text) > limit:
            marker = "... [truncated]\n"
            text = marker + text[-(limit - len(marker)) :]
            truncated = True
        elif dropped:
            truncated = True

        return {
            "ok": True,
            "handle_id": handle.handle_id,
            "name": handle.name,
            "running": rc is None,
            "returncode": rc,
            "output": text,
            "truncated": truncated,
        }

    def _terminate_bg_handle(self, handle: _BgProcess) -> int | None:
        proc = handle.proc
        if proc.poll() is None:
            try:
                os.killpg(proc.pid, signal.SIGTERM)
            except (ProcessLookupError, PermissionError, OSError):
                try:
                    proc.terminate()
                except Exception:
                    pass
            try:
                proc.wait(timeout=1.0)
            except subprocess.TimeoutExpired:
                try:
                    os.killpg(proc.pid, signal.SIGKILL)
                except (ProcessLookupError, PermissionError, OSError):
                    try:
                        proc.kill()
                    except Exception:
                        pass
                try:
                    proc.wait(timeout=1.0)
                except Exception:
                    pass
        if handle.reader_thread is not None and handle.reader_thread.is_alive():
            handle.reader_thread.join(timeout=1.0)
        if proc.stdout is not None:
            try:
                proc.stdout.close()
            except Exception:
                pass
        if handle.sb_profile_path is not None:
            handle.sb_profile_path.unlink(missing_ok=True)
            handle.sb_profile_path = None
        handle.stopped = True
        return proc.returncode

    def stop_bg_cmd(self, handle_id: str) -> dict[str, Any]:
        with self._bg_lock:
            handle = self._bg_handles.get(str(handle_id or ""))
        if handle is None:
            return {"ok": False, "error": f"Unknown handle_id: {handle_id}"}
        rc = self._terminate_bg_handle(handle)
        return {
            "ok": True,
            "handle_id": handle.handle_id,
            "stopped": True,
            "returncode": rc,
        }

    def close(self) -> None:
        """Terminate and clean up all active background command handles."""
        with self._bg_lock:
            handles = list(self._bg_handles.values())
        for handle in handles:
            try:
                self._terminate_bg_handle(handle)
            except Exception:
                pass

    def git_diff(self, stat_only: bool = False) -> dict[str, Any]:
        cmd = ["git", "-c", "core.hooksPath=/dev/null", "diff"]
        if stat_only:
            cmd.append("--stat")
        try:
            proc = subprocess.run(
                cmd,
                cwd=str(self.worktree),
                capture_output=True,
                text=True,
                timeout=15,
            )
        except Exception as exc:
            return {"ok": False, "error": scrub_secrets(f"git diff failed: {exc}")}

        diff_raw = scrub_secrets(proc.stdout or "")
        err_raw = scrub_secrets(proc.stderr or "")
        if proc.returncode != 0:
            return {
                "ok": False,
                "returncode": proc.returncode,
                "error": err_raw.strip() or f"git diff exited {proc.returncode}",
                "diff": "",
                "stat_only": bool(stat_only),
                "truncated": False,
            }

        truncated = False
        if len(diff_raw) > _READ_MAX_CHARS:
            marker = "\n... [truncated]"
            diff_raw = diff_raw[: _READ_MAX_CHARS - len(marker)] + marker
            truncated = True

        return {
            "ok": True,
            "diff": diff_raw,
            "stat_only": bool(stat_only),
            "truncated": truncated,
        }

    def post_board(self, kind: str, body: str, to_agent: str = "all") -> dict[str, Any]:
        raw_kind = str(kind or "FINDING").strip() or "FINDING"
        kind_alias = {
            "CLAIM": "CLAIM_FILE",
            "CLAIM_FILE": "CLAIM_FILE",
            "CONTRACT": "CONTRACT",
            "FINDING": "FINDING",
            "STATUS": "FINDING",
            "QUESTION": "QUESTION",
            "NEED_DECISION": "NEED_DECISION",
            "BLOCKED": "BLOCKED",
            "DONE": "DONE",
            "DECISION": "DECISION",
        }
        clean_kind = kind_alias.get(raw_kind.upper(), raw_kind.upper())
        clean_to = str(to_agent or "all").strip() or "all"
        clean_body = scrub_secrets(str(body or ""))

        if self.board_callback is not None:
            res = self.board_callback(
                "post_board",
                {
                    "run_id": self.run_id,
                    "agent_id": self.agent_id,
                    "kind": clean_kind,
                    "body": clean_body,
                    "to_agent": clean_to,
                },
            )
            return res if isinstance(res, dict) else {"ok": True, "result": res}

        if self.board_store is not None:
            for attr in ("post_board_message", "post_board", "post"):
                fn = getattr(self.board_store, attr, None)
                if callable(fn):
                    res = _call_with_accepted_kwargs(
                        fn,
                        {
                            "run_id": self.run_id or str(self.worktree),
                            "agent_id": self.agent_id,
                            "from_agent": self.agent_id,
                            "author": self.agent_id,
                            "kind": clean_kind,
                            "body": clean_body,
                            "text": clean_body,
                            "to_agent": clean_to,
                        },
                    )
                    return res if isinstance(res, dict) else {"ok": True, "event": res}
            send_fn = getattr(self.board_store, "send_message", None)
            if callable(send_fn):
                project_key = self.run_id or str(self.worktree)
                rec = send_fn(project_key, self.agent_id, clean_to, f"[{clean_kind}] {clean_body}")
                return {"ok": True, "event": rec}

        return {
            "ok": True,
            "standalone": True,
            "posted": False,
            "message": "Standalone mode: no shared board configured.",
        }

    def read_board(self, since_seq: int = 0) -> dict[str, Any]:
        try:
            seq = max(0, int(since_seq))
        except (TypeError, ValueError):
            return {"ok": False, "error": "since_seq must be an integer"}

        if self.board_callback is not None:
            res = self.board_callback(
                "read_board",
                {
                    "run_id": self.run_id,
                    "agent_id": self.agent_id,
                    "since_seq": seq,
                },
            )
            return res if isinstance(res, dict) else {"ok": True, "events": res}

        if self.board_store is not None:
            for attr in ("read_board_messages", "read_board", "read", "list_events"):
                fn = getattr(self.board_store, attr, None)
                if callable(fn):
                    res = _call_with_accepted_kwargs(
                        fn,
                        {
                            "run_id": self.run_id or str(self.worktree),
                            "agent_id": self.agent_id,
                            "to_agent": self.agent_id,
                            "since_seq": seq,
                            "after": seq,
                        },
                    )
                    if isinstance(res, dict):
                        return res
                    events = list(res or [])
                    last_seq = max((int(e.get("seq", e.get("id", 0))) for e in events if isinstance(e, dict)), default=seq)
                    return {"ok": True, "events": events, "last_seq": last_seq}
            read_fn = getattr(self.board_store, "read_messages", None)
            if callable(read_fn):
                project_key = self.run_id or str(self.worktree)
                msgs = read_fn(project_key, to_role=self.agent_id, after_id=seq)
                last_seq = max((int(m.get("id", 0)) for m in msgs), default=seq)
                return {"ok": True, "events": msgs, "last_seq": last_seq}

        return {
            "ok": True,
            "standalone": True,
            "events": [],
            "last_seq": seq,
            "message": "Standalone mode: no shared board configured.",
        }

    def claim_file(self, path: str) -> dict[str, Any]:
        resolved = self._resolve_path(path)
        if resolved[0] is None:
            return {"ok": False, "error": resolved[1]}
        rel, _target = resolved
        if self._is_readonly(rel):
            self._record_readonly_violation(rel)
            return {"ok": False, "error": f"Read-only path: {rel}"}

        if self.board_callback is not None:
            res = self.board_callback(
                "claim_file",
                {
                    "run_id": self.run_id,
                    "agent_id": self.agent_id,
                    "path": rel,
                },
            )
            return res if isinstance(res, dict) else {"ok": True, "claimed": True, "path": rel, "result": res}

        if self.board_store is not None:
            for attr in ("claim_file_lease", "claim_file", "claim"):
                fn = getattr(self.board_store, attr, None)
                if callable(fn):
                    res = _call_with_accepted_kwargs(
                        fn,
                        {
                            "run_id": self.run_id or str(self.worktree),
                            "agent_id": self.agent_id,
                            "owner_agent": self.agent_id,
                            "path": rel,
                            "rel_path": rel,
                        },
                    )
                    return res if isinstance(res, dict) else {"ok": True, "claimed": bool(res), "path": rel}
            post_fn = getattr(self.board_store, "post_board", None) or getattr(self.board_store, "post", None)
            if callable(post_fn):
                res = _call_with_accepted_kwargs(
                    post_fn,
                    {
                        "run_id": self.run_id,
                        "agent_id": self.agent_id,
                        "kind": "claim",
                        "body": rel,
                        "to_agent": "all",
                    },
                )
                return {"ok": True, "claimed": True, "path": rel, "event": res}
            send_fn = getattr(self.board_store, "send_message", None)
            if callable(send_fn):
                project_key = self.run_id or str(self.worktree)
                rec = send_fn(project_key, self.agent_id, "all", f"[claim] {rel}")
                return {"ok": True, "claimed": True, "path": rel, "event": rec}

        return {
            "ok": True,
            "standalone": True,
            "claimed": True,
            "path": rel,
            "message": "Standalone mode: no shared board configured; proceeding locally.",
        }

    def write_note(self, title: str, body: str) -> dict[str, Any]:
        notes_rel = f".open_swarm/notes/{self.agent_id}.md"
        resolved = self._resolve_path(notes_rel)
        if resolved[0] is None:
            return {"ok": False, "error": resolved[1]}
        rel, target = resolved

        target.parent.mkdir(parents=True, exist_ok=True)
        ts = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
        title_lines = scrub_secrets(str(title or "Note")).strip().splitlines()
        clean_title = title_lines[0].strip() if title_lines else "Note"
        clean_body = scrub_secrets(str(body or "")).rstrip()
        section = f"## {ts} — {clean_title}\n{clean_body}\n"

        prefix = ""
        if target.exists() and target.is_file():
            existing = target.read_text(encoding="utf-8", errors="replace")
            if existing and not existing.endswith("\n\n"):
                prefix = "\n" if existing.endswith("\n") else "\n\n"

        with target.open("a", encoding="utf-8") as fh:
            fh.write(prefix + section)

        self.notes_path = rel
        return {"ok": True, "path": rel, "timestamp": ts}

    def finish(self, summary: str) -> dict[str, Any]:
        self.finished = True
        self.finish_summary = scrub_secrets(str(summary or "").strip())
        return {"ok": True, "finished": True, "summary": self.finish_summary}

    def dispatch(self, name: str, arguments: dict[str, Any] | str | None) -> dict[str, Any]:
        if isinstance(arguments, str):
            try:
                parsed = json.loads(arguments) if arguments.strip() else {}
            except json.JSONDecodeError as exc:
                return {"ok": False, "error": f"Invalid JSON arguments for {name}: {exc}"}
        elif arguments is None:
            parsed = {}
        elif isinstance(arguments, dict):
            parsed = arguments
        else:
            return {"ok": False, "error": f"Arguments for {name} must be an object"}

        if not isinstance(parsed, dict):
            return {"ok": False, "error": f"Arguments for {name} must be a JSON object"}

        try:
            if name == "read_file":
                return self.read_file(
                    path=parsed["path"],
                    start_line=parsed.get("start_line", 1),
                    end_line=parsed.get("end_line", 200),
                )
            if name == "search_code":
                return self.search_code(
                    query=parsed["query"],
                    glob=parsed.get("glob", ""),
                )
            if name == "list_dir":
                return self.list_dir(
                    path=parsed.get("path", ""),
                    max_entries=parsed.get("max_entries", 100),
                )
            if name == "edit_file":
                return self.edit_file(
                    path=parsed["path"],
                    old_text=parsed["old_text"],
                    new_text=parsed["new_text"],
                )
            if name == "write_file":
                return self.write_file(
                    path=parsed["path"],
                    content=parsed["content"],
                )
            if name == "run_command":
                return self.run_command(
                    cmd=parsed["cmd"],
                    timeout_s=parsed.get("timeout_s"),
                )
            if name == "start_bg_cmd":
                return self.start_bg_cmd(
                    cmd=parsed["cmd"],
                    name=parsed.get("name", ""),
                )
            if name == "poll_bg_cmd":
                return self.poll_bg_cmd(
                    handle_id=parsed["handle_id"],
                    max_chars=parsed.get("max_chars", 4000),
                )
            if name == "stop_bg_cmd":
                return self.stop_bg_cmd(handle_id=parsed["handle_id"])
            if name == "git_diff":
                return self.git_diff(stat_only=bool(parsed.get("stat_only", False)))
            if name == "post_board":
                return self.post_board(
                    kind=parsed["kind"],
                    body=parsed["body"],
                    to_agent=parsed.get("to_agent", "all"),
                )
            if name == "read_board":
                return self.read_board(since_seq=parsed.get("since_seq", 0))
            if name == "claim_file":
                return self.claim_file(path=parsed["path"])
            if name == "write_note":
                return self.write_note(
                    title=parsed["title"],
                    body=parsed["body"],
                )
            if name == "finish":
                return self.finish(summary=parsed["summary"])
        except KeyError as exc:
            return {"ok": False, "error": f"Missing required argument {exc} for {name}"}
        except (TypeError, ValueError) as exc:
            return {"ok": False, "error": f"Invalid argument for {name}: {exc}"}

        return {"ok": False, "error": f"Unknown tool: {name}"}


def _chat_completions_url(base_url: str) -> str:
    url = base_url.rstrip("/") + "/chat/completions"
    if not url.endswith("/v1/chat/completions") and "/v1/" not in url:
        url = base_url.rstrip("/") + "/v1/chat/completions"
    return url


def _accumulate_sse_tool_calls(
    chunk: dict[str, Any],
    tool_call_map: dict[int, dict[str, Any]],
    state: dict[str, Any],
    now: float,
) -> None:
    choice = (chunk.get("choices") or [{}])[0]
    delta = choice.get("delta") or choice.get("message") or {}
    tc_list = delta.get("tool_calls")
    if not isinstance(tc_list, list) or not tc_list:
        return
    if state["t_first_tok"] is None:
        state["t_first_tok"] = now
    for pos, tc in enumerate(tc_list):
        if not isinstance(tc, dict):
            continue
        idx = int(tc.get("index", pos))
        entry = tool_call_map.setdefault(
            idx,
            {"id": "", "type": "function", "function": {"name": "", "arguments": ""}},
        )
        if tc.get("id"):
            entry["id"] = str(tc["id"])
        if tc.get("type"):
            entry["type"] = str(tc["type"])
        fn = tc.get("function") or {}
        if isinstance(fn, dict):
            if fn.get("name"):
                entry["function"]["name"] += str(fn["name"])
            if fn.get("arguments"):
                entry["function"]["arguments"] += str(fn["arguments"])


def _openai_chat_turn(
    *,
    base_url: str,
    api_key: str,
    model: str,
    messages: list[dict[str, Any]],
    tools: list[dict[str, Any]] | None = None,
    max_tokens: int = 2048,
    temperature: float = 0.0,
    timeout_s: int = 180,
    stream: bool = True,
    on_delta: Callable[[str, str], None] | None = None,
) -> dict[str, Any]:
    url = _chat_completions_url(base_url)
    payload: dict[str, Any] = {
        "model": model,
        "messages": messages,
        "max_tokens": max_tokens,
        "temperature": temperature,
        "stream": stream,
    }
    if tools is not None:
        payload["tools"] = tools
    if stream:
        payload["stream_options"] = {"include_usage": True}

    req = urllib.request.Request(
        url,
        data=json.dumps(payload).encode("utf-8"),
        headers={
            "Content-Type": "application/json",
            "Accept": "text/event-stream, application/json",
            "Authorization": f"Bearer {api_key}",
            "User-Agent": "OutrCore-OpenSwarm/0.1",
        },
        method="POST",
    )
    t0 = time.monotonic()
    content_chunks: list[str] = []
    reasoning_chunks: list[str] = []
    tool_call_map: dict[int, dict[str, Any]] = {}
    raw_tool_calls: list[dict[str, Any]] = []
    state: dict[str, Any] = {"usage": {}, "timings": {}, "t_first_tok": None}

    with urllib.request.urlopen(req, timeout=timeout_s) as resp:
        ctype = (resp.headers.get("Content-Type") or "").lower()
        if "text/event-stream" in ctype:
            for raw_line in resp:
                line = raw_line.decode("utf-8", errors="replace").strip()
                if not line or line.startswith(":") or not line.startswith("data:"):
                    continue
                data_str = line[5:].strip()
                if data_str == "[DONE]":
                    break
                try:
                    chunk = json.loads(data_str)
                except json.JSONDecodeError:
                    continue
                now = time.monotonic()
                _parse_sse_chunk(chunk, content_chunks, reasoning_chunks, state, now, on_delta=on_delta)
                _accumulate_sse_tool_calls(chunk, tool_call_map, state, now)
        else:
            raw_text = resp.read().decode("utf-8", errors="replace").strip()
            if raw_text.startswith("data:"):
                for line in raw_text.splitlines():
                    line = line.strip()
                    if not line.startswith("data:"):
                        continue
                    data_str = line[5:].strip()
                    if data_str == "[DONE]":
                        break
                    try:
                        chunk = json.loads(data_str)
                    except json.JSONDecodeError:
                        continue
                    now = time.monotonic()
                    _parse_sse_chunk(chunk, content_chunks, reasoning_chunks, state, now, on_delta=on_delta)
                    _accumulate_sse_tool_calls(chunk, tool_call_map, state, now)
            else:
                data = json.loads(raw_text)
                choice = (data.get("choices") or [{}])[0]
                msg = choice.get("message") or {}
                if msg.get("reasoning_content"):
                    reasoning_chunks.append(msg["reasoning_content"])
                if msg.get("content"):
                    content_chunks.append(msg["content"])
                if isinstance(msg.get("tool_calls"), list):
                    raw_tool_calls = msg["tool_calls"]
                state["usage"] = data.get("usage") or {}
                state["timings"] = data.get("timings") or {}

    if tool_call_map and not raw_tool_calls:
        raw_tool_calls = [tool_call_map[k] for k in sorted(tool_call_map)]

    t_end = time.monotonic()
    elapsed_s = max(0.001, t_end - t0)
    reasoning_content = "".join(reasoning_chunks)
    content = "".join(content_chunks)
    if not content and not raw_tool_calls:
        content = reasoning_content

    usage = state["usage"]
    prompt_tokens = int(usage.get("prompt_tokens") or 0)
    completion_tokens = int(usage.get("completion_tokens") or 0)
    cached_tokens = int((usage.get("prompt_tokens_details") or {}).get("cached_tokens") or 0)
    timings = state["timings"]
    t_first_tok = state["t_first_tok"]
    if timings.get("gen_ms"):
        gen_s = float(timings["gen_ms"]) / 1000.0
    elif t_first_tok is not None:
        gen_s = max(0.001, t_end - t_first_tok)
    else:
        gen_s = elapsed_s
    if timings.get("prompt_ms") is not None:
        ttft_ms: float | None = float(timings["prompt_ms"])
    elif t_first_tok is not None:
        ttft_ms = round((t_first_tok - t0) * 1000.0, 2)
    else:
        ttft_ms = None
    tok_s = round(completion_tokens / max(gen_s, 0.001), 2)

    return {
        "content": content,
        "reasoning_content": reasoning_content,
        "tool_calls": raw_tool_calls,
        "prompt_tokens": prompt_tokens,
        "cached_tokens": cached_tokens,
        "completion_tokens": completion_tokens,
        "gen_s": gen_s,
        "tok_s": tok_s,
        "ttft_ms": ttft_ms,
        "wall_s": elapsed_s,
    }


def _extract_fenced_tool_calls(content: str) -> list[tuple[str, Any]]:
    """Extract tool invocations from ```tool or ```json fenced blocks."""
    calls: list[tuple[str, Any]] = []
    for match in _TOOL_FENCE_RE.finditer(content or ""):
        lang = match.group(1).lower()
        body = match.group(2).strip()
        if not body:
            continue
        try:
            parsed = json.loads(body)
        except json.JSONDecodeError:
            continue
        items = parsed if isinstance(parsed, list) else [parsed]
        for item in items:
            if not isinstance(item, dict):
                continue
            name = item.get("name")
            if not isinstance(name, str) or not name.strip():
                continue
            if lang == "json":
                if "arguments" not in item or not isinstance(item.get("arguments"), dict):
                    continue
                calls.append((name.strip(), item["arguments"]))
            else:
                if "arguments" in item:
                    args = item["arguments"]
                elif "args" in item:
                    args = item["args"]
                else:
                    args = {}
                calls.append((name.strip(), args))
    return calls


def _extract_file_blocks(content: str) -> list[tuple[str, str]]:
    """Parse legacy ```file:rel/path``` blocks using swarm_run's header/close regexes."""
    blocks: list[tuple[str, str]] = []
    txt = content or ""
    headers = list(_FILE_HEADER_RE.finditer(txt))
    for i, hdr in enumerate(headers):
        end = headers[i + 1].start() if i + 1 < len(headers) else len(txt)
        segment = txt[hdr.end() : end]
        close = _CLOSE_FENCE_RE.search(segment)
        body = segment[: close.start()] if close else segment
        blocks.append((hdr.group(1).strip(), body))
    return blocks


def _messages_char_len(messages: Sequence[dict[str, Any]]) -> int:
    total = 0
    for msg in messages:
        total += len(str(msg.get("content") or ""))
        if msg.get("tool_calls"):
            total += len(json.dumps(msg["tool_calls"]))
    return total


def _tool_status_label(content: str) -> str:
    try:
        data = json.loads(content)
        if isinstance(data, dict) and "ok" in data:
            return "ok" if data.get("ok") else "err"
    except (json.JSONDecodeError, TypeError):
        pass
    return "err" if '"ok": false' in content.lower() or "error" in content.lower() else "ok"


def _compact_middle_message(msg: dict[str, Any]) -> dict[str, Any]:
    role = msg.get("role")
    content = str(msg.get("content") or "")

    if role == "tool":
        tname = str(msg.get("name") or "tool")
        status = _tool_status_label(content)
        compacted = dict(msg)
        compacted["content"] = f"[Compacted tool output: {tname} -> {status}]"
        return compacted

    if role == "user" and content.startswith("Tool `"):
        stubs: list[str] = []
        for m in re.finditer(r"Tool `([^`]+)` result:\s*```json\s*(.*?)\s*```", content, re.DOTALL):
            stubs.append(f"[Compacted tool output: {m.group(1)} -> {_tool_status_label(m.group(2))}]")
        if stubs:
            compacted = dict(msg)
            compacted["content"] = "\n".join(stubs)
            return compacted

    if role == "assistant":
        compacted = dict(msg)
        if len(content) > 160:
            first_line = content.strip().splitlines()[0][:120]
            compacted["content"] = f"[Compacted assistant turn: {first_line}]"
        if isinstance(msg.get("tool_calls"), list):
            slim_calls: list[dict[str, Any]] = []
            for tc in msg["tool_calls"]:
                tc_copy = dict(tc)
                fn = tc_copy.get("function")
                if isinstance(fn, dict) and len(str(fn.get("arguments") or "")) > 200:
                    fn_copy = dict(fn)
                    fn_copy["arguments"] = '{"_compacted": true}'
                    tc_copy["function"] = fn_copy
                slim_calls.append(tc_copy)
            compacted["tool_calls"] = slim_calls
        return compacted

    if len(content) > 160:
        first_line = content.strip().splitlines()[0][:120]
        compacted = dict(msg)
        compacted["content"] = f"[Compacted {role or 'message'}: {first_line}]"
        return compacted

    return dict(msg)


def _compact_messages_if_needed(
    messages: list[dict[str, Any]],
    compact_threshold_chars: int,
) -> list[dict[str, Any]]:
    """Keep messages[0] (system) and messages[1] (task) verbatim, compact middle, keep last 4 verbatim."""
    if compact_threshold_chars <= 0 or len(messages) <= 6:
        return messages
    if _messages_char_len(messages) <= compact_threshold_chars:
        return messages

    prefix = [messages[0], messages[1]]
    middle = [_compact_middle_message(m) for m in messages[2:-4]]
    tail = messages[-4:]
    return prefix + middle + tail


def _record_meter_turn(meter: Any, agent_id: str, resp: dict[str, Any]) -> None:
    if meter is None:
        return
    kwargs = {
        "prompt_tokens": resp["prompt_tokens"],
        "cached_tokens": resp["cached_tokens"],
        "completion_tokens": resp["completion_tokens"],
        "gen_s": resp["gen_s"],
        "ttft_ms": resp["ttft_ms"],
    }
    if hasattr(meter, "record_usage"):
        meter.record_usage(agent_id, **kwargs)
    elif hasattr(meter, "record_turn"):
        meter.record_turn(agent_id, **kwargs)


def _build_turn0_env_header(
    worktree: Path,
    device_name: str = "",
    sandbox_mode: str = "auto",
) -> str:
    os_name = sys.platform
    arch = platform.machine() or "unknown"
    py_ver = platform.python_version()
    dev = (device_name or "").strip() or platform.node() or "local"
    return (
        f"[Env] os={os_name} arch={arch} python={py_ver}\n"
        f"[Device] name={dev} sandbox={sandbox_mode}\n"
        f"[Worktree] {Path(worktree).name}"
    )


def run_agent_loop(
    *,
    worktree: Path,
    system_prompt: str,
    user_prompt: str,
    base_url: str,
    api_key: str,
    model: str,
    agent_id: str = "agent-a",
    readonly_paths: Sequence[str] = (),
    max_turns: int = 8,
    max_tokens: int = 2048,
    temperature: float = 0.0,
    timeout_s: int = 180,
    compact_threshold_chars: int = 18000,
    meter: Any = None,
    on_progress: Callable[[dict[str, Any]], None] | None = None,
    on_delta: Callable[[str, str], None] | None = None,
    sandbox_mode: str = "auto",
    max_cmd_timeout_s: int = 120,
    board_store: Any = None,
    run_id: str = "",
    board_callback: Callable[..., Any] | None = None,
    device_name: str = "",
) -> dict[str, Any]:
    """Execute a multi-turn tool-calling agent loop inside a confined git worktree."""
    toolkit = WorktreeToolkit(
        worktree=worktree,
        readonly_paths=readonly_paths,
        agent_id=agent_id,
        max_cmd_timeout_s=max_cmd_timeout_s,
        sandbox_mode=sandbox_mode,
        board_store=board_store,
        run_id=run_id,
        board_callback=board_callback,
    )
    env_header = _build_turn0_env_header(worktree, device_name=device_name, sandbox_mode=toolkit.sandbox_mode)
    turn0_user = f"{env_header}\n\n{user_prompt}" if user_prompt else env_header
    # Keep messages[0] strictly byte-for-byte unchanged across all turns for KV cache reuse.
    messages: list[dict[str, Any]] = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": turn0_user},
    ]

    turns = 0
    tool_calls_count = 0
    prompt_tokens = 0
    cached_tokens = 0
    completion_tokens = 0
    total_llm_s = 0.0
    use_native_tools = True
    fallback_hint_added = False
    ok = False
    summary = ""
    error: str | None = None

    def _emit(snippet: str) -> None:
        if on_progress is None:
            return
        try:
            on_progress(
                {
                    "kind": "agent_stage",
                    "agent_id": agent_id,
                    "stage": "running",
                    "snippet": str(snippet or "")[-140:],
                }
            )
        except Exception:
            pass

    try:
        for turn_idx in range(1, max(1, int(max_turns)) + 1):
            if meter is not None and hasattr(meter, "check_cap"):
                meter.check_cap()

            messages = _compact_messages_if_needed(messages, compact_threshold_chars)
            _emit(f"Turn {turn_idx}/{max_turns}")

            try:
                resp = _openai_chat_turn(
                    base_url=base_url,
                    api_key=api_key,
                    model=model,
                    messages=messages,
                    tools=TOOLS_SCHEMA if use_native_tools else None,
                    max_tokens=max_tokens,
                    temperature=temperature,
                    timeout_s=timeout_s,
                    stream=True,
                    on_delta=on_delta,
                )
            except urllib.error.HTTPError as exc:
                if use_native_tools and exc.code in (400, 422):
                    try:
                        exc.read()
                        exc.close()
                    except Exception:
                        pass
                    use_native_tools = False
                    if not fallback_hint_added:
                        fallback_hint_added = True
                        if len(messages) == 2:
                            messages[1] = {
                                "role": "user",
                                "content": str(messages[1]["content"]) + _FALLBACK_TOOL_INSTRUCTIONS,
                            }
                        else:
                            messages.append({"role": "user", "content": _FALLBACK_TOOL_INSTRUCTIONS.strip()})
                    resp = _openai_chat_turn(
                        base_url=base_url,
                        api_key=api_key,
                        model=model,
                        messages=messages,
                        tools=None,
                        max_tokens=max_tokens,
                        temperature=temperature,
                        timeout_s=timeout_s,
                        stream=True,
                        on_delta=on_delta,
                    )
                else:
                    raise

            turns += 1
            prompt_tokens += int(resp["prompt_tokens"])
            cached_tokens += int(resp["cached_tokens"])
            completion_tokens += int(resp["completion_tokens"])
            total_llm_s += float(resp["wall_s"])
            _record_meter_turn(meter, agent_id, resp)

            raw_tool_calls = resp.get("tool_calls") or []
            content = str(resp.get("content") or "")

            # Wire protocol 1: Native OpenAI tool_calls
            if raw_tool_calls:
                messages.append(
                    {
                        "role": "assistant",
                        "content": content,
                        "tool_calls": raw_tool_calls,
                    }
                )
                for tc in raw_tool_calls:
                    tc_id = str(tc.get("id") or f"call_{tool_calls_count + 1}")
                    fn = tc.get("function") or {}
                    tname = str(fn.get("name") or "")
                    targs = fn.get("arguments")
                    tool_calls_count += 1
                    _emit(f"Tool: {tname}")
                    result = toolkit.dispatch(tname, targs)
                    messages.append(
                        {
                            "role": "tool",
                            "tool_call_id": tc_id,
                            "name": tname,
                            "content": json.dumps(result),
                        }
                    )
                    if toolkit.finished:
                        break
                if toolkit.finished:
                    ok = True
                    summary = toolkit.finish_summary
                    break
                continue

            # Wire protocol 2: Fenced ```tool / ```json tool blocks
            fenced_calls = _extract_fenced_tool_calls(content)
            if fenced_calls:
                messages.append({"role": "assistant", "content": content})
                result_sections: list[str] = []
                for tname, targs in fenced_calls:
                    tool_calls_count += 1
                    _emit(f"Tool: {tname}")
                    result = toolkit.dispatch(tname, targs)
                    result_sections.append(
                        f"Tool `{tname}` result:\n```json\n{json.dumps(result)}\n```"
                    )
                    if toolkit.finished:
                        break
                messages.append({"role": "user", "content": "\n\n".join(result_sections)})
                if toolkit.finished:
                    ok = True
                    summary = toolkit.finish_summary
                    break
                continue

            # Wire protocol 3: Legacy ```file:rel/path``` blocks
            file_blocks = _extract_file_blocks(content)
            if file_blocks:
                messages.append({"role": "assistant", "content": content})
                applied: list[str] = []
                rejected: list[str] = []
                for rel_raw, body in file_blocks:
                    res = toolkit.write_file(rel_raw, body)
                    if res.get("ok"):
                        applied.append(str(res["path"]))
                    else:
                        rejected.append(str(res.get("error") or f"Rejected: {rel_raw}"))

                if rejected:
                    err_note = "Some ```file:...``` blocks were rejected:\n- " + "\n- ".join(rejected)
                    if applied:
                        err_note += f"\nAllowed files written: {', '.join(applied)}"
                    err_note += "\nFix the rejected paths or call `finish`."
                    messages.append({"role": "user", "content": err_note})
                    error = scrub_secrets("; ".join(rejected))
                    continue

                ok = True
                summary = f"Applied {len(applied)} file block(s): {', '.join(applied)}"
                error = None
                break

            # Plain text with no tool call or file block: prompt agent to act or finish
            messages.append({"role": "assistant", "content": content})
            if turn_idx < max_turns:
                messages.append(
                    {
                        "role": "user",
                        "content": (
                            "Continue using tools (`read_file`, `list_dir`, `search_code`, `edit_file`, "
                            "`write_file`, `run_command`, `git_diff`, `write_note`, `finish`) "
                            "or emit ```file:rel/path``` blocks."
                        ),
                    }
                )
        else:
            if not ok and error is None:
                error = f"Max turns ({max_turns}) reached without finish"
    except SpendCapExceeded:
        raise
    except Exception as exc:
        ok = False
        error = scrub_secrets(f"{type(exc).__name__}: {exc}")
    finally:
        toolkit.close()

    return {
        "ok": ok,
        "summary": scrub_secrets(summary),
        "turns": turns,
        "tool_calls": tool_calls_count,
        "written_files": list(toolkit.written_files),
        "readonly_violations": list(toolkit.readonly_violations),
        "notes_path": toolkit.notes_path,
        "prompt_tokens": prompt_tokens,
        "cached_tokens": cached_tokens,
        "completion_tokens": completion_tokens,
        "llm_s": round(total_llm_s, 3),
        "error": scrub_secrets(error) if error is not None else None,
    }
