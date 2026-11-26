#!/usr/bin/env python3
"""Built-In Agent Tools & Goal Decomposition Optimizer for Open Swarm.

Pure Python 3 stdlib (+ sibling `continuity_store`, `fleet_scanner`, `swarm_run`)
module providing:
- `index_repo_symbols`: deterministic AST + regex repository symbol indexer.
- `build_agent_context_brief`: concise continuity + fleet-attention brief.
- `decompose_goal_to_tasks`: Architect LLM task decomposition with safe fallback.
- `detect_file_collisions`: cross-agent shared-write collision detector.
"""
from __future__ import annotations

import ast
import json
import os
from pathlib import Path, PurePosixPath
import re
import sys
from typing import Any, Callable

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

from continuity_store import ContinuityStore  # noqa: E402
import fleet_scanner  # noqa: E402
import swarm_run  # noqa: E402

_SKIP_PATH_PARTS = frozenset({".git", "__pycache__", "dist", "node_modules", ".venv"})
_SUPPORTED_EXTENSIONS = frozenset({".py", ".js", ".html", ".swift"})
_VALID_LANES = ("coder", "mechanical", "deep-reason")
_DEFAULT_ACCEPTANCE_CMD = "python3 -m unittest discover -s tests -v"
_MAX_METHODS_PER_CLASS = 12

_FENCE_BLOCK_RE = re.compile(r"```(?:json)?\s*\r?\n?(.*?)\r?\n?```", re.DOTALL | re.IGNORECASE)
_JS_CLASS_RE = re.compile(
    r"\b(?:export\s+(?:default\s+)?)?(?:abstract\s+|final\s+|public\s+|private\s+|internal\s+|open\s+)*"
    r"class\s+([A-Za-z_$][\w$]*)(?=\s*(?:[:{<]|extends\b|implements\b))"
)
_JS_FUNC_RE = re.compile(r"\b(?:(async)\s+)?function\s*\*?\s*([A-Za-z_$][\w$]*)\s*\(([^)]*)\)")
_SWIFT_FUNC_RE = re.compile(r"\bfunc\s+([A-Za-z_]\w*(?:<[^>\n]+>)?)\s*\(([^)]*)\)")


def _should_skip_path(rel_parts: tuple[str, ...], name: str) -> bool:
    if any(part in _SKIP_PATH_PARTS for part in rel_parts):
        return True
    if name.endswith(".pyc"):
        return True
    return False


def _iter_repo_files(repo: Path) -> list[Path]:
    root = Path(repo)
    if not root.exists() or not root.is_dir():
        return []
    files: list[Path] = []
    for path in sorted(root.rglob("*")):
        try:
            if not path.is_file():
                continue
        except OSError:
            continue
        rel = path.relative_to(root)
        if _should_skip_path(rel.parts, path.name):
            continue
        files.append(path)
    return files


def _truncate_text(text: str, limit: int) -> str:
    cleaned = " ".join(text.split())
    if len(cleaned) <= limit:
        return cleaned
    if limit <= 3:
        return cleaned[:limit]
    return cleaned[: limit - 3].rstrip(", ") + "..."


def _format_py_args(args: ast.arguments) -> str:
    parts: list[str] = []
    pos_args = list(args.posonlyargs) + list(args.args)
    default_start = len(pos_args) - len(args.defaults)
    for idx, arg in enumerate(pos_args):
        if idx >= default_start:
            parts.append(f"{arg.arg}=...")
        else:
            parts.append(arg.arg)
    if args.vararg is not None:
        parts.append(f"*{args.vararg.arg}")
    elif args.kwonlyargs:
        parts.append("*")
    for idx, arg in enumerate(args.kwonlyargs):
        if idx < len(args.kw_defaults) and args.kw_defaults[idx] is not None:
            parts.append(f"{arg.arg}=...")
        else:
            parts.append(arg.arg)
    if args.kwarg is not None:
        parts.append(f"**{args.kwarg.arg}")
    return _truncate_text(", ".join(parts), 60)


def _extract_python_symbols(source: str) -> list[str]:
    try:
        tree = ast.parse(source)
    except (SyntaxError, ValueError, TypeError):
        return []

    lines: list[str] = []
    for node in tree.body:
        if isinstance(node, ast.ClassDef):
            lines.append(f"- class {node.name}")
            methods = [
                item
                for item in node.body
                if isinstance(item, (ast.FunctionDef, ast.AsyncFunctionDef))
            ]
            for method in methods[:_MAX_METHODS_PER_CLASS]:
                prefix = "async def " if isinstance(method, ast.AsyncFunctionDef) else "def "
                lines.append(f"  - {prefix}{method.name}({_format_py_args(method.args)})")
            if len(methods) > _MAX_METHODS_PER_CLASS:
                lines.append(f"  - (+{len(methods) - _MAX_METHODS_PER_CLASS} more methods)")
        elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            prefix = "async def" if isinstance(node, ast.AsyncFunctionDef) else "def"
            lines.append(f"- {prefix} {node.name}({_format_py_args(node.args)})")
    return lines


def _extract_regex_symbols(source: str, suffix: str) -> list[str]:
    entries: list[tuple[int, str]] = []
    for match in _JS_CLASS_RE.finditer(source):
        entries.append((match.start(), f"- class {match.group(1)}"))
    if suffix in (".js", ".html"):
        for match in _JS_FUNC_RE.finditer(source):
            async_prefix = "async " if match.group(1) else ""
            args = _truncate_text(match.group(3), 60)
            entries.append((match.start(), f"- {async_prefix}function {match.group(2)}({args})"))
    if suffix == ".swift":
        for match in _SWIFT_FUNC_RE.finditer(source):
            args = _truncate_text(match.group(2), 60)
            entries.append((match.start(), f"- func {match.group(1)}({args})"))
    entries.sort(key=lambda item: item[0])
    return [line for _, line in entries]


def _cap_lines(lines: list[str], max_chars: int) -> str:
    if max_chars <= 0 or not lines:
        return ""
    kept: list[str] = []
    total = 0
    for line in lines:
        cost = len(line) if not kept else 1 + len(line)
        if total + cost <= max_chars:
            kept.append(line)
            total += cost
        else:
            break
    return "\n".join(kept)


def index_repo_symbols(repo: Path, max_files: int = 80, max_chars: int = 2400) -> str:
    """Build a deterministic markdown symbol index for a repository."""
    if max_files <= 0 or max_chars <= 0:
        return ""

    root = Path(repo)
    blocks: list[list[str]] = []
    matched_files = 0

    for path in _iter_repo_files(root):
        suffix = path.suffix.lower()
        if suffix not in _SUPPORTED_EXTENSIONS:
            continue
        matched_files += 1
        if matched_files > max_files:
            break
        try:
            source = path.read_text(encoding="utf-8", errors="replace")
        except OSError:
            continue

        if suffix == ".py":
            sym_lines = _extract_python_symbols(source)
        else:
            sym_lines = _extract_regex_symbols(source, suffix)

        if not sym_lines:
            continue
        rel_posix = path.relative_to(root).as_posix()
        blocks.append([f"### {rel_posix}", *sym_lines])

    if not blocks:
        return ""

    output_lines: list[str] = []
    total = 0
    for block in blocks:
        header = block[0]
        first_sym = block[1]
        min_needed = len(header) + 1 + len(first_sym)
        if output_lines:
            min_needed += 1  # separating newline before header
        if total + min_needed > max_chars:
            break

        output_lines.append(header)
        total += len(header) if len(output_lines) == 1 else 1 + len(header)

        for sym_line in block[1:]:
            cost = 1 + len(sym_line)
            if total + cost > max_chars:
                return "\n".join(output_lines)
            output_lines.append(sym_line)
            total += cost

    return "\n".join(output_lines)


def _normalize_compare_root(raw: Any) -> str:
    if not isinstance(raw, str) or not raw.strip():
        return ""
    expanded = os.path.expanduser(raw.strip())
    try:
        return str(Path(expanded).resolve())
    except OSError:
        return os.path.normpath(expanded)


def build_agent_context_brief(
    repo: Path,
    continuity: ContinuityStore | None = None,
    runs_dir: Path | None = None,
    max_chars: int = 900,
) -> str:
    """Build a concise markdown continuity & fleet-attention brief for *repo*."""
    if max_chars <= 0:
        return ""

    project_key = str(Path(repo).resolve())
    norm_project_key = os.path.normpath(project_key)
    sections: list[str] = []

    if continuity is not None:
        checkpoints: list[dict[str, Any]] = []
        try:
            checkpoints = list(continuity.get_latest_checkpoints(project_key) or [])
        except Exception:
            checkpoints = []

        if checkpoints:
            cp_lines = ["### Checkpoints"]
            for cp in checkpoints:
                role = str(cp.get("role") or "unknown").strip()
                rev = cp.get("revision")
                rev_txt = f" r{rev}" if rev is not None else ""
                summary = _truncate_text(str(cp.get("summary") or ""), 160)
                next_step = _truncate_text(str(cp.get("next_step") or ""), 120)
                line = f"- **{role}{rev_txt}**: {summary}" if summary else f"- **{role}{rev_txt}**"
                if next_step:
                    line += f" (Next: {next_step})"
                cp_lines.append(line)
            sections.append("\n".join(cp_lines))

        messages: list[dict[str, Any]] = []
        try:
            messages = list((continuity.read_messages(project_key, limit=20) or [])[-5:])
        except Exception:
            messages = []

        if messages:
            msg_lines = ["### Recent Messages"]
            for msg in messages:
                from_role = str(msg.get("from_role") or "unknown").strip()
                to_role = str(msg.get("to_role") or "all").strip()
                body = _truncate_text(str(msg.get("body") or ""), 160)
                msg_lines.append(f"- {from_role} -> {to_role}: {body}")
            sections.append("\n".join(msg_lines))

    attention_sessions: list[dict[str, Any]] = []
    sessions_to_check: list[dict[str, Any]] = []
    try:
        fleet = fleet_scanner.scan_fleet(open_swarm_runs_dir=runs_dir)
        for proj in fleet.get("projects") or []:
            sessions_to_check.extend(proj.get("sessions") or [])
    except Exception:
        if runs_dir is not None:
            try:
                sessions_to_check = fleet_scanner._scan_open_swarm(Path(runs_dir), 40, [])
            except Exception:
                sessions_to_check = []

    for sess in sessions_to_check:
        if not sess.get("needs_attention"):
            continue
        raw_root = sess.get("project_root")
        resolved_root = _normalize_compare_root(raw_root)
        norm_raw = (
            os.path.normpath(os.path.expanduser(str(raw_root)))
            if isinstance(raw_root, str) and raw_root.strip()
            else ""
        )
        if resolved_root == project_key or norm_raw == norm_project_key:
            attention_sessions.append(sess)

    if attention_sessions:
        attn_lines = ["### Attention Sessions"]
        for sess in attention_sessions[:5]:
            source = str(sess.get("source") or "session")
            title = _truncate_text(str(sess.get("title") or sess.get("session_id") or ""), 80)
            status = str(sess.get("status") or "attention")
            event = _truncate_text(str(sess.get("last_event") or ""), 100)
            detail = f"{status}: {event}" if event and event != status else status
            attn_lines.append(f"- [{source}] {title} ({detail})")
        sections.append("\n".join(attn_lines))

    if not sections:
        return ""

    raw_lines = "\n".join(sections).splitlines()
    capped = _cap_lines(raw_lines, max_chars)
    if capped:
        return capped
    # Single line exceeded max_chars
    return raw_lines[0][:max_chars] if raw_lines else ""


def _extract_json_array(raw_text: str) -> list[Any] | None:
    txt = (raw_text or "").strip()
    if not txt:
        return None

    candidates: list[str] = [txt]
    for match in _FENCE_BLOCK_RE.finditer(txt):
        inner = match.group(1).strip()
        if inner:
            candidates.insert(0, inner)

    if txt.startswith("```"):
        stripped = re.sub(r"^\s*```(?:json)?\s*\r?\n?", "", txt, count=1, flags=re.IGNORECASE)
        stripped = re.sub(r"\r?\n?\s*```\s*$", "", stripped, count=1)
        if stripped.strip():
            candidates.append(stripped.strip())

    first_bracket = txt.find("[")
    last_bracket = txt.rfind("]")
    if first_bracket != -1 and last_bracket > first_bracket:
        candidates.append(txt[first_bracket : last_bracket + 1])

    for cand in candidates:
        try:
            parsed = json.loads(cand)
        except (json.JSONDecodeError, TypeError, ValueError):
            continue
        if isinstance(parsed, list):
            return parsed
    return None


def _fallback_tickets(goal: str) -> list[dict[str, Any]]:
    cleaned = (goal or "").strip()
    return [
        {
            "title": (cleaned or "Implement goal")[:60],
            "spec": cleaned or "Implement the requested change cleanly with unit tests.",
            "acceptance_cmd": _DEFAULT_ACCEPTANCE_CMD,
            "lane": "coder",
            "owned_files": [],
        }
    ]


def _normalize_lane(raw_lane: Any) -> str:
    val = str(raw_lane or "").strip().lower()
    if val in _VALID_LANES:
        return val
    return "coder"


def _normalize_ticket(item: Any, fallback_goal: str, index: int) -> dict[str, Any] | None:
    if not isinstance(item, dict):
        return None
    raw_title = str(item.get("title") or "").strip()
    raw_spec = str(item.get("spec") or "").strip()
    if not raw_title and not raw_spec:
        return None

    title = (raw_title or raw_spec or fallback_goal or f"Task {index + 1}")[:120]
    spec = raw_spec or raw_title or fallback_goal or "Implement the requested change cleanly with unit tests."
    acceptance_cmd = str(item.get("acceptance_cmd") or "").strip() or _DEFAULT_ACCEPTANCE_CMD
    lane = _normalize_lane(item.get("lane"))

    raw_owned = item.get("owned_files")
    owned_files: list[str] = []
    if isinstance(raw_owned, list):
        for entry in raw_owned:
            s = str(entry).strip()
            if s:
                owned_files.append(s)

    return {
        "title": title,
        "spec": spec,
        "acceptance_cmd": acceptance_cmd,
        "lane": lane,
        "owned_files": owned_files,
    }


def decompose_goal_to_tasks(
    *,
    repo: Path,
    goal: str,
    base_url: str,
    model: str,
    api_key: str = "outrcore-swarm",
    continuity_brief: str = "",
    max_agents: int = 3,
    chat_fn: Callable[..., dict[str, Any]] | None = None,
) -> list[dict[str, Any]]:
    """Decompose a high-level goal into 1..max_agents non-overlapping parallel tickets."""
    root = Path(repo)
    limit = max(1, int(max_agents))
    fn = chat_fn or swarm_run.openai_chat

    try:
        tree_files = [p.relative_to(root).as_posix() for p in _iter_repo_files(root)[:60]]
        tree_txt = "\n".join(f"- {rel}" for rel in tree_files) or "(empty repository)"
        symbols_txt = index_repo_symbols(root, max_chars=1800) or "(no symbols indexed)"
        brief_section = f"\n\n## Continuity Brief\n{continuity_brief.strip()}" if (continuity_brief or "").strip() else ""

        system_prompt = (
            "You are a principal software architect decomposing an engineering goal into "
            f"1 to {limit} non-overlapping parallel git-worktree tickets.\n"
            "Assign disjoint `owned_files` per ticket whenever possible to prevent merge collisions.\n"
            "For Python repositories, always write stdlib unittest.TestCase tests and use "
            "'python3 -m unittest <test_file_or_discover>' as acceptance_cmd — never assume pytest or bare 'python' is installed.\n"
            "Respond ONLY with a valid JSON array matching this exact schema:\n"
            '[{"title": "<short title>", "spec": "<clear implementation spec with tests>", '
            '"acceptance_cmd": "<shell test command>", '
            '"lane": "coder" | "mechanical" | "deep-reason", '
            '"owned_files": ["relative/path.py"]}]'
        )
        user_prompt = (
            f"## Goal\n{(goal or '').strip() or 'Implement requested changes.'}\n\n"
            f"## Repository Files (up to 60)\n{tree_txt}\n\n"
            f"## Repository Symbols\n{symbols_txt}"
            f"{brief_section}"
        )

        resp = fn(
            base_url=base_url,
            api_key=api_key,
            model=model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            max_tokens=2048,
            temperature=0.0,
            stream=False,
        )
        raw_content = str((resp or {}).get("content") or "")
        parsed = _extract_json_array(raw_content)
        if not parsed:
            return _fallback_tickets(goal)

        tickets: list[dict[str, Any]] = []
        for idx, item in enumerate(parsed):
            norm = _normalize_ticket(item, goal, idx)
            if norm is not None:
                tickets.append(norm)
            if len(tickets) >= limit:
                break

        return tickets if tickets else _fallback_tickets(goal)
    except Exception:
        return _fallback_tickets(goal)


def detect_file_collisions(results: list[dict[str, Any]]) -> dict[str, list[str]]:
    """Return `{rel_path: sorted(agent_ids)}` for files written by >= 2 passing agents."""
    file_agents: dict[str, set[str]] = {}
    for idx, res in enumerate(results or []):
        if not isinstance(res, dict):
            continue
        if not res.get("tests_ok"):
            continue
        agent_id = str(res.get("agent_id") or f"agent-{idx}")
        written = res.get("written_files") or []
        if not isinstance(written, (list, tuple, set)):
            continue
        for raw_path in written:
            raw_str = str(raw_path or "").strip()
            if not raw_str:
                continue
            rel_path = PurePosixPath(raw_str.replace("\\", "/")).as_posix()
            if rel_path == ".":
                continue
            file_agents.setdefault(rel_path, set()).add(agent_id)

    return {
        rel_path: sorted(agent_ids)
        for rel_path, agent_ids in sorted(file_agents.items())
        if len(agent_ids) >= 2
    }
