#!/usr/bin/env python3
"""OutrCore Open Swarm — Real Agent Tool Loop ACI (M1).

Pure Python 3 stdlib (+ sibling `swarm_run` and `metering`) module providing:
- `WorktreeToolkit`: strictly worktree-confined file, search, command, note, and finish tools
  with harness-enforced `readonly_paths`, symlink/traversal protection, and `__pycache__` purging.
- `TOOLS_SCHEMA`: OpenAI chat-completion tool schema list for all 7 tools.
- `run_agent_loop`: multi-turn agent loop supporting native `tool_calls`, HTTP 400/422 fallback
  to fenced ```` ```tool ```` / ```` ```json ```` blocks, legacy ```` ```file:rel/path``` ```` blocks,
  KV-cache-preserving context compaction, `RunMeter` telemetry, and progress callbacks.
"""
from __future__ import annotations

import fnmatch
import json
import os
from pathlib import Path, PurePosixPath
import posixpath
import re
import shutil
import subprocess
import sys
import time
from typing import Any, Callable, Sequence
import urllib.error
import urllib.request

SWARM_DIR = Path(__file__).resolve().parent
if str(SWARM_DIR) not in sys.path:
    sys.path.insert(0, str(SWARM_DIR))

from metering import SpendCapExceeded  # noqa: E402
from swarm_run import _CLOSE_FENCE_RE, _FILE_HEADER_RE, _parse_sse_chunk  # noqa: E402

_READ_MAX_CHARS = 12_000
_CMD_MAX_CHARS = 6_000
_SEARCH_MAX_MATCHES = 40
_SEARCH_MAX_FILE_BYTES = 1_000_000
_SKIP_DIRS = frozenset({".git", "__pycache__"})

_TOOL_FENCE_RE = re.compile(
    r"^[ \t]*```(tool|json)[ \t]*\r?\n(.*?)^[ \t]*```[ \t]*$",
    re.DOTALL | re.MULTILINE | re.IGNORECASE,
)

_FALLBACK_TOOL_INSTRUCTIONS = (
    "\n\n## Tool Protocol\n"
    "Native function calling is unavailable on this backend. Invoke tools by emitting one or more "
    "fenced ```tool JSON blocks:\n"
    "```tool\n"
    '{"name": "<tool_name>", "arguments": {...}}\n'
    "```\n"
    "Available tools: read_file, search_code, edit_file, write_file, run_command, write_note, finish.\n"
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
            "description": "Run a shell command inside the worktree with __pycache__ purged and PYTHONDONTWRITEBYTECODE=1.",
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


class WorktreeToolkit:
    """Worktree-confined agent toolkit with harness-enforced read-only path protection."""

    def __init__(
        self,
        worktree: Path,
        readonly_paths: Sequence[str] = (),
        agent_id: str = "agent-a",
        max_cmd_timeout_s: int = 30,
    ) -> None:
        self.worktree = Path(worktree).resolve()
        self.agent_id = _sanitize_agent_id(agent_id)
        self.max_cmd_timeout_s = max(1, int(max_cmd_timeout_s))
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
            return {"ok": False, "error": f"old_text not found in {rel}"}
        if count > 1:
            return {"ok": False, "error": f"old_text matched {count} times in {rel}; must be unique"}

        updated = content.replace(old_text, new_text, 1)
        target.write_text(updated, encoding="utf-8")
        self._record_written(rel)
        return {"ok": True, "path": rel}

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
        return {"ok": True, "path": rel, "bytes": len(body.encode("utf-8"))}

    def run_command(self, cmd: str, timeout_s: int | None = None) -> dict[str, Any]:
        if not isinstance(cmd, str) or not cmd.strip():
            return {
                "ok": False,
                "returncode": -1,
                "stdout": "",
                "stderr": "Empty command",
                "elapsed_s": 0.0,
            }

        for pc_dir in self.worktree.rglob("__pycache__"):
            if pc_dir.is_dir():
                shutil.rmtree(pc_dir, ignore_errors=True)

        env = dict(os.environ)
        env["PYTHONDONTWRITEBYTECODE"] = "1"
        eff_timeout = (
            self.max_cmd_timeout_s
            if timeout_s is None
            else max(1, min(int(timeout_s), self.max_cmd_timeout_s))
        )

        t0 = time.monotonic()
        try:
            proc = subprocess.run(
                cmd,
                cwd=str(self.worktree),
                shell=True,
                env=env,
                capture_output=True,
                text=True,
                timeout=eff_timeout,
            )
            elapsed_s = round(time.monotonic() - t0, 3)
            out, err = _bound_streams(proc.stdout or "", proc.stderr or "")
            return {
                "ok": proc.returncode == 0,
                "returncode": proc.returncode,
                "stdout": out,
                "stderr": err,
                "elapsed_s": elapsed_s,
            }
        except subprocess.TimeoutExpired as exc:
            elapsed_s = round(time.monotonic() - t0, 3)
            raw_out = exc.stdout.decode("utf-8", errors="replace") if isinstance(exc.stdout, bytes) else (exc.stdout or "")
            raw_err = exc.stderr.decode("utf-8", errors="replace") if isinstance(exc.stderr, bytes) else (exc.stderr or "")
            timeout_msg = f"Command timed out after {eff_timeout}s"
            err_full = (raw_err.rstrip() + "\n" + timeout_msg).strip() if raw_err else timeout_msg
            out, err = _bound_streams(raw_out, err_full)
            return {
                "ok": False,
                "returncode": -1,
                "stdout": out,
                "stderr": err,
                "elapsed_s": elapsed_s,
            }

    def write_note(self, title: str, body: str) -> dict[str, Any]:
        notes_rel = f".open_swarm/notes/{self.agent_id}.md"
        resolved = self._resolve_path(notes_rel)
        if resolved[0] is None:
            return {"ok": False, "error": resolved[1]}
        rel, target = resolved

        target.parent.mkdir(parents=True, exist_ok=True)
        ts = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
        title_lines = str(title or "Note").strip().splitlines()
        clean_title = title_lines[0].strip() if title_lines else "Note"
        clean_body = str(body or "").rstrip()
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
        self.finish_summary = str(summary or "").strip()
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
) -> dict[str, Any]:
    """Execute a multi-turn tool-calling agent loop inside a confined git worktree."""
    toolkit = WorktreeToolkit(
        worktree=worktree,
        readonly_paths=readonly_paths,
        agent_id=agent_id,
    )
    # Keep messages[0] strictly byte-for-byte unchanged across all turns for KV cache reuse.
    messages: list[dict[str, Any]] = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
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
                    error = "; ".join(rejected)
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
                        "content": "Continue using tools (`read_file`, `edit_file`, `write_file`, `run_command`, `write_note`, `finish`) or emit ```file:rel/path``` blocks.",
                    }
                )
        else:
            if not ok and error is None:
                error = f"Max turns ({max_turns}) reached without finish"
    except SpendCapExceeded:
        raise
    except Exception as exc:
        ok = False
        error = f"{type(exc).__name__}: {exc}"

    return {
        "ok": ok,
        "summary": summary,
        "turns": turns,
        "tool_calls": tool_calls_count,
        "written_files": list(toolkit.written_files),
        "readonly_violations": list(toolkit.readonly_violations),
        "notes_path": toolkit.notes_path,
        "prompt_tokens": prompt_tokens,
        "cached_tokens": cached_tokens,
        "completion_tokens": completion_tokens,
        "llm_s": round(total_llm_s, 3),
        "error": error,
    }
