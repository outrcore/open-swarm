#!/usr/bin/env python3
"""Open Swarm Orchestrator: Lane routing, DAG scheduling, critic gate, and blind tournament engine.

Pure Python 3 stdlib (+ sibling `swarm_run`, `agent_loop`) module providing:
- `LANES`: routing presets (`mechanical`, `coder`, `deep-reason`, `reviewer`).
- `resolve_vendor`: model name to vendor family mapping.
- `check_review_eligibility`: author-never-reviews-own-work policy checks.
- `plan_task_dag` / `run_dag_waves`: deterministic topological wave scheduling with cancellation.
- `review_agent_diff`: fresh-context (and optional read-only tool-enabled) critic gate.
- `review_and_repair_agent`: automated critic-repair feedback loop.
- `run_tournament`: multi-stance / multi-model parallel fan-out with auto-disqualification
  and anonymous blind judging.
"""
from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
import inspect
import json
from pathlib import Path
import re
import sys
from typing import Any, Callable

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

import agent_loop  # noqa: E402
import swarm_run  # noqa: E402

LANES: dict[str, dict[str, Any]] = {
    "mechanical": {
        "label": "Mechanical",
        "description": "Fast deterministic edits, formatting, renames, and simple boilerplate.",
        "default_max_tokens": 1536,
        "temperature": 0.0,
    },
    "coder": {
        "label": "Coder",
        "description": "Standard feature implementation, bug fixes, and unit test authoring.",
        "default_max_tokens": 4096,
        "temperature": 0.0,
    },
    "deep-reason": {
        "label": "Deep Reason",
        "description": "Complex architecture, multi-file debugging, and careful tradeoff reasoning.",
        "default_max_tokens": 6144,
        "temperature": 0.2,
    },
    "reviewer": {
        "label": "Reviewer",
        "description": "Independent critic gate and blind tournament judge.",
        "default_max_tokens": 2048,
        "temperature": 0.0,
    },
}

_VENDOR_KEYWORDS: tuple[tuple[str, tuple[str, ...]], ...] = (
    ("anthropic", ("claude", "anthropic", "sonnet", "opus", "haiku", "fable")),
    ("xai", ("grok", "xai")),
    ("google", ("gemini", "google")),
    ("outrcore", ("inkling", "ornith", "qwen", "outrcore", "thinkingmachines")),
    ("openai", ("gpt", "codex", "openai", "o1", "o3", "o4")),
)

_DEFAULT_TOURNAMENT_STANCES: tuple[tuple[str, str], ...] = (
    (
        "minimal-patch",
        "Prioritize the smallest, most surgical diff that satisfies all requirements and tests without unrelated edits.",
    ),
    (
        "defensive-edge-cases",
        "Prioritize defensive validation, explicit error/boundary handling, and thorough edge-case test coverage.",
    ),
    (
        "clean-modular",
        "Prioritize clean modular structure, clear helper decomposition, and readable maintainable code.",
    ),
)

_FENCE_BLOCK_RE = re.compile(r"```(?:json)?\s*\n?(.*?)\n?```", re.DOTALL | re.IGNORECASE)
_REVIEW_TOOL_FENCE_RE = re.compile(
    r"^[ \t]*```(tool|json)[ \t]*\r?\n(.*?)^[ \t]*```[ \t]*$",
    re.DOTALL | re.MULTILINE | re.IGNORECASE,
)

_PHASE_ORDER: dict[str, int] = {
    "contract": 0,
    "impl": 1,
    "verify": 2,
}

_REVIEWER_ALLOWED_TOOLS = frozenset({"read_file", "search_code", "list_dir"})
_REVIEWER_TOOLS_SCHEMA: list[dict[str, Any]] = [
    {
        "type": "function",
        "function": {
            "name": "read_file",
            "description": "Read a UTF-8 file inside the worktree with 1-indexed line numbers.",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {"type": "string", "description": "Relative path inside the worktree."},
                    "start_line": {"type": "integer", "description": "1-indexed start line.", "default": 1},
                    "end_line": {"type": "integer", "description": "1-indexed inclusive end line.", "default": 200},
                },
                "required": ["path"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "search_code",
            "description": "Search worktree files (skipping .git and __pycache__) by regex or substring.",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "Regex or literal substring to search for."},
                    "glob": {"type": "string", "description": "Optional glob filter such as '*.py'.", "default": ""},
                },
                "required": ["query"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "list_dir",
            "description": "List files and subdirectories inside a relative worktree directory.",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {"type": "string", "description": "Relative directory path (default '.').", "default": "."},
                },
            },
        },
    },
]


def resolve_vendor(model: str) -> str:
    """Map a model identifier string to a canonical vendor family via case-insensitive substrings."""
    low = (model or "").strip().lower()
    if not low:
        return "other"
    for vendor, keywords in _VENDOR_KEYWORDS:
        if any(kw in low for kw in keywords):
            return vendor
    return "other"


def check_review_eligibility(
    author_model: str,
    reviewer_model: str,
    allow_same_vendor_fresh: bool = True,
) -> dict[str, Any]:
    """Evaluate cross-vendor review eligibility between an author model and a reviewer model."""
    author_vendor = resolve_vendor(author_model)
    reviewer_vendor = resolve_vendor(reviewer_model)
    same_model = (author_model or "").strip().lower() == (reviewer_model or "").strip().lower()
    same_vendor = same_model or (author_vendor == reviewer_vendor and author_vendor != "other")
    cross_vendor = not same_vendor

    if same_vendor and not allow_same_vendor_fresh:
        return {
            "eligible": False,
            "author_vendor": author_vendor,
            "reviewer_vendor": reviewer_vendor,
            "same_model": same_model,
            "same_vendor": same_vendor,
            "cross_vendor": cross_vendor,
            "warning": (
                f"Blocked by author-never-reviews-own-work vendor policy: "
                f"same vendor ({author_vendor}) cannot review when allow_same_vendor_fresh is False."
            ),
        }

    if same_vendor and allow_same_vendor_fresh:
        return {
            "eligible": True,
            "author_vendor": author_vendor,
            "reviewer_vendor": reviewer_vendor,
            "same_model": same_model,
            "same_vendor": same_vendor,
            "cross_vendor": cross_vendor,
            "warning": (
                f"Fresh-context single-vendor review ({author_vendor}): "
                "allowed, but prefer cross-vendor review when available."
            ),
        }

    return {
        "eligible": True,
        "author_vendor": author_vendor,
        "reviewer_vendor": reviewer_vendor,
        "same_model": same_model,
        "same_vendor": same_vendor,
        "cross_vendor": cross_vendor,
        "warning": None,
    }


def _extract_json_dict(raw_text: str) -> dict[str, Any] | None:
    """Safely extract and parse a JSON object from raw model output, stripping markdown fences."""
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

    first_brace = txt.find("{")
    last_brace = txt.rfind("}")
    if first_brace != -1 and last_brace > first_brace:
        candidates.append(txt[first_brace : last_brace + 1])

    for cand in candidates:
        try:
            parsed = json.loads(cand)
        except json.JSONDecodeError:
            continue
        if isinstance(parsed, dict):
            return parsed
    return None


def _normalize_verdict(raw_verdict: Any) -> str:
    val = str(raw_verdict or "").strip().lower().replace("-", "_").replace(" ", "_")
    if val in ("approve", "approved", "pass", "lgtm"):
        return "approve"
    if val in ("block", "blocked", "reject", "rejected"):
        return "block"
    return "request_changes"


def _normalize_findings(raw_findings: Any) -> list[dict[str, str]]:
    if not isinstance(raw_findings, list):
        return []
    out: list[dict[str, str]] = []
    for item in raw_findings:
        if not isinstance(item, dict):
            continue
        sev = str(item.get("severity") or "medium").strip().lower()
        if sev not in ("high", "medium", "low"):
            sev = "medium"
        out.append(
            {
                "severity": sev,
                "file": str(item.get("file") or ""),
                "issue": str(item.get("issue") or ""),
                "fix": str(item.get("fix") or ""),
            }
        )
    return out


def _normalize_test_quality(raw_quality: Any) -> str:
    val = str(raw_quality or "").strip().lower()
    if val in ("strong", "adequate", "weak"):
        return val
    return "adequate"


def _format_wave_context(
    wave_idx: int,
    wave_tasks: list[dict[str, Any]],
    wave_results: list[dict[str, Any]],
) -> str:
    """Summarize a completed DAG wave (including any CONTRACT outputs) for downstream waves."""
    task_by_id = {str(t.get("id") or ""): t for t in wave_tasks}
    lines: list[str] = [f"## Wave {wave_idx} Results"]
    has_contract = False

    for idx, res in enumerate(wave_results):
        if not isinstance(res, dict):
            continue
        rid = str(res.get("id") or res.get("task_id") or res.get("agent_id") or "")
        task = task_by_id.get(rid) or (wave_tasks[idx] if idx < len(wave_tasks) else {})
        tid = rid or str(task.get("id") or f"task-{idx}")
        phase = str(res.get("phase") or task.get("phase") or "").strip().lower()
        title = str(res.get("title") or task.get("title") or "").strip()
        contract = str(res.get("contract") or "").strip()
        summary = str(res.get("summary") or res.get("output") or "").strip()
        written = res.get("written_files") or []
        tests_ok = res.get("tests_ok")

        is_contract = phase == "contract" or bool(contract)
        if is_contract:
            has_contract = True
        prefix = "[CONTRACT] " if is_contract else ""
        header = f"- {prefix}{tid}" + (f" ({title})" if title else "")
        details: list[str] = []
        if contract:
            details.append(f"CONTRACT: {contract}")
        if summary:
            details.append(f"summary: {summary}")
        if isinstance(written, (list, tuple)) and written:
            details.append("files: " + ", ".join(str(p) for p in written))
        if tests_ok is not None:
            details.append(f"tests_ok={bool(tests_ok)}")
        if not details and res.get("diff"):
            diff_snippet = str(res["diff"]).strip().splitlines()[:6]
            details.append("diff: " + " | ".join(diff_snippet))
        lines.append(f"{header}: " + ("; ".join(details) if details else "completed"))

    if has_contract:
        lines[0] = f"## Wave {wave_idx} CONTRACT & Results"
    return "\n".join(lines) if len(lines) > 1 else ""


def plan_task_dag(tasks: list[dict[str, Any]]) -> list[list[dict[str, Any]]]:
    """Plan deterministic topological execution waves from `depends_on` and optional `phase`."""
    if not tasks:
        return []

    prepared: list[dict[str, Any]] = []
    ordered_ids: list[str] = []
    tasks_by_id: dict[str, dict[str, Any]] = {}
    order_idx: dict[str, int] = {}
    explicit_deps_map: dict[str, list[str]] = {}
    phase_map: dict[str, str] = {}

    for idx, raw_task in enumerate(tasks):
        task = dict(raw_task)
        tid = str(task.get("id") or "").strip() or f"t{idx}"
        if tid in tasks_by_id:
            raise ValueError(f"Duplicate task id in DAG: {tid}")
        task["id"] = tid

        raw_deps = task.get("depends_on")
        if raw_deps is None:
            explicit_deps: list[str] = []
        elif isinstance(raw_deps, (list, tuple)):
            explicit_deps = [str(d).strip() for d in raw_deps if str(d or "").strip()]
        else:
            raise ValueError(f"Task '{tid}' depends_on must be a list of task IDs")

        phase = str(task.get("phase") or "").strip().lower()
        prepared.append(task)
        ordered_ids.append(tid)
        tasks_by_id[tid] = task
        order_idx[tid] = idx
        explicit_deps_map[tid] = explicit_deps
        phase_map[tid] = phase

    in_degree: dict[str, int] = {}
    dependents: dict[str, list[str]] = {tid: [] for tid in ordered_ids}

    for tid in ordered_ids:
        explicit_deps = explicit_deps_map[tid]
        phase = phase_map[tid]
        if explicit_deps:
            deps = list(dict.fromkeys(explicit_deps))
        elif phase in _PHASE_ORDER and _PHASE_ORDER[phase] > 0:
            my_rank = _PHASE_ORDER[phase]
            deps = [
                other_id
                for other_id in ordered_ids
                if phase_map[other_id] in _PHASE_ORDER and _PHASE_ORDER[phase_map[other_id]] < my_rank
            ]
        else:
            deps = []

        for dep in deps:
            if dep not in tasks_by_id:
                raise ValueError(f"Task '{tid}' depends on missing task ID '{dep}'")
            if dep == tid:
                raise ValueError(f"Dependency cycle detected involving task '{tid}'")
            dependents[dep].append(tid)

        tasks_by_id[tid]["depends_on"] = list(deps)
        in_degree[tid] = len(deps)

    ready = [tid for tid in ordered_ids if in_degree[tid] == 0]
    waves: list[list[dict[str, Any]]] = []
    placed = 0

    while ready:
        waves.append([tasks_by_id[tid] for tid in ready])
        placed += len(ready)
        next_ready: list[str] = []
        for tid in ready:
            for child in dependents[tid]:
                in_degree[child] -= 1
                if in_degree[child] == 0:
                    next_ready.append(child)
        next_ready.sort(key=lambda x: order_idx[x])
        ready = next_ready

    if placed != len(ordered_ids):
        raise ValueError("Dependency cycle detected in task DAG")

    return waves


def run_dag_waves(
    tasks: list[dict[str, Any]],
    execute_wave_fn: Callable[[int, list[dict[str, Any]], str], list[dict[str, Any]]],
    *,
    cancel_event: Any = None,
) -> dict[str, Any]:
    """Execute DAG waves sequentially, forwarding accumulated context and honoring cancellation."""
    waves = plan_task_dag(tasks)
    wave_records: list[dict[str, Any]] = []
    all_results: list[dict[str, Any]] = []
    context_parts: list[str] = []

    for wave_idx, wave_tasks in enumerate(waves):
        if cancel_event is not None and cancel_event.is_set():
            return {
                "cancelled": True,
                "waves": wave_records,
                "results": all_results,
                "context": "\n\n".join(context_parts),
            }

        context_str = "\n\n".join(context_parts)
        raw_results = execute_wave_fn(wave_idx, wave_tasks, context_str)
        wave_results = list(raw_results) if raw_results is not None else []
        all_results.extend(wave_results)

        wave_summary = _format_wave_context(wave_idx, wave_tasks, wave_results)
        if wave_summary:
            context_parts.append(wave_summary)

        wave_records.append(
            {
                "wave_index": wave_idx,
                "task_ids": [t["id"] for t in wave_tasks],
                "tasks": wave_tasks,
                "results": wave_results,
                "summary": wave_summary,
            }
        )

    return {
        "cancelled": False,
        "waves": wave_records,
        "results": all_results,
        "context": "\n\n".join(context_parts),
    }


class _ReviewerReadOnlyToolbox:
    """Strictly read-only worktree inspection helper wrapping `WorktreeToolkit`."""

    def __init__(self, worktree: Path) -> None:
        self.toolkit = agent_loop.WorktreeToolkit(
            worktree=worktree,
            readonly_paths=(".",),
            agent_id="reviewer",
        )
        self.inspected_files: list[str] = []

    def list_dir(self, path: str = ".") -> dict[str, Any]:
        cleaned = str(path or ".").replace("\\", "/").strip()
        if cleaned in ("", ".", "./"):
            rel = "."
            target = self.toolkit.worktree
        else:
            resolved = self.toolkit._resolve_path(cleaned)
            if resolved[0] is None:
                return {"ok": False, "error": resolved[1]}
            rel, target = resolved

        if not target.exists() or not target.is_dir():
            return {"ok": False, "error": f"Directory not found: {rel}"}

        entries: list[dict[str, str]] = []
        for entry in sorted(target.iterdir(), key=lambda p: p.name):
            if entry.name in (".git", "__pycache__") or entry.name.endswith(".pyc"):
                continue
            if entry.is_symlink():
                try:
                    entry_res = entry.resolve()
                    entry_res.relative_to(self.toolkit.worktree)
                except (OSError, ValueError):
                    continue
            entries.append(
                {
                    "name": entry.name,
                    "type": "dir" if entry.is_dir() else "file",
                }
            )
        return {"ok": True, "path": rel, "entries": entries}

    def dispatch(self, name: str, arguments: dict[str, Any] | str | None) -> dict[str, Any]:
        if name not in _REVIEWER_ALLOWED_TOOLS:
            return {"ok": False, "error": f"Tool '{name}' is disabled in read-only critic review mode"}

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

        if name == "read_file":
            res = self.toolkit.read_file(
                path=str(parsed.get("path") or ""),
                start_line=parsed.get("start_line", 1),
                end_line=parsed.get("end_line", 200),
            )
            if res.get("ok") and res.get("path"):
                rel_path = str(res["path"])
                if rel_path not in self.inspected_files:
                    self.inspected_files.append(rel_path)
            return res

        if name == "search_code":
            return self.toolkit.search_code(
                query=str(parsed.get("query") or ""),
                glob=str(parsed.get("glob") or ""),
            )

        return self.list_dir(path=str(parsed.get("path") or "."))


def _extract_reviewer_fenced_tool_calls(content: str) -> list[tuple[str, Any]]:
    """Extract read-only tool calls from ```tool or ```json blocks without confusing final verdict JSON."""
    calls: list[tuple[str, Any]] = []
    for match in _REVIEW_TOOL_FENCE_RE.finditer(content or ""):
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
            if lang == "tool" and (not isinstance(name, str) or not name.strip()):
                name = item.get("tool")
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


def _fn_accepts_tools(fn: Callable[..., Any]) -> bool:
    try:
        sig = inspect.signature(fn)
    except (TypeError, ValueError):
        return False
    for param in sig.parameters.values():
        if param.kind == inspect.Parameter.VAR_KEYWORD or param.name == "tools":
            return True
    return False


def review_agent_diff(
    *,
    ticket_title: str,
    ticket_spec: str,
    diff: str,
    test_output: str,
    author_model: str,
    reviewer_base_url: str,
    reviewer_model: str,
    api_key: str = "outrcore-swarm",
    allow_same_vendor_fresh: bool = True,
    worktree: Path | str | None = None,
    max_tool_turns: int = 4,
    chat_fn: Callable[..., dict[str, Any]] | None = None,
) -> dict[str, Any]:
    """Run an independent fresh-context critic gate over a diff and return a structured verdict."""
    eligibility = check_review_eligibility(
        author_model,
        reviewer_model,
        allow_same_vendor_fresh=allow_same_vendor_fresh,
    )
    review_mode = "cross_vendor" if eligibility["cross_vendor"] else "same_vendor_fresh"

    if not eligibility["eligible"]:
        return {
            "verdict": "block",
            "test_quality": "adequate",
            "review_mode": review_mode,
            "inspected_files": [],
            "summary": eligibility["warning"],
            "findings": [],
            "eligibility": eligibility,
            "error": eligibility["warning"],
        }

    if not (diff or "").strip():
        return {
            "verdict": "block",
            "test_quality": "adequate",
            "review_mode": review_mode,
            "inspected_files": [],
            "summary": "Empty diff — nothing to review",
            "findings": [],
            "eligibility": eligibility,
            "error": "empty diff",
        }

    worktree_path: Path | None = None
    if worktree is not None:
        cand_wt = Path(worktree)
        if cand_wt.exists() and cand_wt.is_dir():
            worktree_path = cand_wt.resolve()

    tool_instructions = ""
    if worktree_path is not None:
        tool_instructions = (
            "\nYou have read-only inspection tools (`read_file`, `search_code`, `list_dir`) to inspect "
            "callers, untouched tests, or imported modules in the worktree before giving your verdict. "
            "Invoke them via native `tool_calls` or fenced ```tool blocks "
            '(`{"name": "read_file", "arguments": {"path": "..."}}`). '
            "Once you have inspected enough context, respond ONLY with the final JSON verdict."
        )

    system_prompt = (
        "You are an independent senior code reviewer performing a strict critic gate.\n"
        "Evaluate correctness, edge cases, regressions, and adherence to the ticket specification.\n"
        "Explicitly inspect test quality: reject or flag tautological assertions (e.g., `assert True`), "
        "mocked-out assertions that bypass real behavior under test, or untested error paths."
        f"{tool_instructions}\n"
        "Respond ONLY with valid JSON matching this exact schema:\n"
        '{"verdict": "approve" | "request_changes" | "block", '
        '"test_quality": "strong" | "adequate" | "weak", '
        '"summary": "<concise summary>", '
        '"findings": [{"severity": "high" | "medium" | "low", "file": "<path>", '
        '"issue": "<description>", "fix": "<recommended fix>"}]}'
    )
    user_prompt = (
        f"## Ticket Title\n{ticket_title}\n\n"
        f"## Ticket Specification\n{ticket_spec}\n\n"
        f"## Unified Diff\n```diff\n{diff}\n```\n\n"
        f"## Acceptance Test Output\n```\n{test_output}\n```"
    )

    messages: list[dict[str, Any]] = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]

    if worktree_path is None:
        fn = chat_fn or swarm_run.openai_chat
        resp = fn(
            base_url=reviewer_base_url,
            api_key=api_key,
            model=reviewer_model,
            messages=messages,
            max_tokens=LANES["reviewer"]["default_max_tokens"],
            temperature=LANES["reviewer"]["temperature"],
            stream=False,
        )
        raw_content = str(resp.get("content") or "")
        total_completion_tokens = int(resp.get("completion_tokens") or 0)
        tok_s = resp.get("tok_s")
        inspected_files: list[str] = []
    else:
        toolbox = _ReviewerReadOnlyToolbox(worktree_path)
        fn = chat_fn or agent_loop._openai_chat_turn
        accepts_tools = _fn_accepts_tools(fn)
        turns_budget = max(0, int(max_tool_turns))
        total_completion_tokens = 0
        tok_s = None
        raw_content = ""
        tool_calls_counter = 0
        got_final = False

        for _ in range(turns_budget):
            call_kwargs: dict[str, Any] = {
                "base_url": reviewer_base_url,
                "api_key": api_key,
                "model": reviewer_model,
                "messages": messages,
                "max_tokens": LANES["reviewer"]["default_max_tokens"],
                "temperature": LANES["reviewer"]["temperature"],
                "stream": False,
            }
            if accepts_tools:
                call_kwargs["tools"] = _REVIEWER_TOOLS_SCHEMA

            resp = fn(**call_kwargs)
            total_completion_tokens += int(resp.get("completion_tokens") or 0)
            if resp.get("tok_s") is not None:
                tok_s = resp.get("tok_s")

            raw_tool_calls = resp.get("tool_calls") or []
            content = str(resp.get("content") or "")

            if raw_tool_calls:
                messages.append(
                    {
                        "role": "assistant",
                        "content": content,
                        "tool_calls": raw_tool_calls,
                    }
                )
                for tc in raw_tool_calls:
                    tool_calls_counter += 1
                    tc_id = str(tc.get("id") or f"review_call_{tool_calls_counter}")
                    fn_obj = tc.get("function") or {}
                    tname = str(fn_obj.get("name") or "")
                    targs = fn_obj.get("arguments")
                    result = toolbox.dispatch(tname, targs)
                    messages.append(
                        {
                            "role": "tool",
                            "tool_call_id": tc_id,
                            "name": tname,
                            "content": json.dumps(result),
                        }
                    )
                continue

            fenced_calls = _extract_reviewer_fenced_tool_calls(content)
            if fenced_calls:
                messages.append({"role": "assistant", "content": content})
                result_sections: list[str] = []
                for tname, targs in fenced_calls:
                    result = toolbox.dispatch(tname, targs)
                    result_sections.append(
                        f"Tool `{tname}` result:\n```json\n{json.dumps(result)}\n```"
                    )
                messages.append({"role": "user", "content": "\n\n".join(result_sections)})
                continue

            raw_content = content
            got_final = True
            break

        if not got_final:
            if turns_budget > 0:
                messages.append(
                    {
                        "role": "user",
                        "content": "Inspection turn limit reached. Do not call any tools; respond ONLY with the final JSON verdict now.",
                    }
                )
            final_kwargs: dict[str, Any] = {
                "base_url": reviewer_base_url,
                "api_key": api_key,
                "model": reviewer_model,
                "messages": messages,
                "max_tokens": LANES["reviewer"]["default_max_tokens"],
                "temperature": LANES["reviewer"]["temperature"],
                "stream": False,
            }
            if accepts_tools:
                final_kwargs["tools"] = None
            resp = fn(**final_kwargs)
            total_completion_tokens += int(resp.get("completion_tokens") or 0)
            if resp.get("tok_s") is not None:
                tok_s = resp.get("tok_s")
            raw_content = str(resp.get("content") or "")

        inspected_files = list(toolbox.inspected_files)

    parsed = _extract_json_dict(raw_content)

    if parsed is None:
        snippet = raw_content.strip()[:300]
        return {
            "verdict": "request_changes",
            "test_quality": "adequate",
            "review_mode": review_mode,
            "inspected_files": inspected_files,
            "summary": f"Malformed JSON from critic reviewer; raw output snippet: {snippet}",
            "findings": [],
            "eligibility": eligibility,
            "reviewer_model": reviewer_model,
            "tok_s": tok_s,
            "completion_tokens": total_completion_tokens,
        }

    verdict = _normalize_verdict(parsed.get("verdict"))
    test_quality = _normalize_test_quality(parsed.get("test_quality"))
    summary = str(parsed.get("summary") or "").strip() or f"Reviewer verdict: {verdict}"
    findings = _normalize_findings(parsed.get("findings"))
    return {
        "verdict": verdict,
        "test_quality": test_quality,
        "review_mode": review_mode,
        "inspected_files": inspected_files,
        "summary": summary,
        "findings": findings,
        "eligibility": eligibility,
        "reviewer_model": reviewer_model,
        "tok_s": tok_s,
        "completion_tokens": total_completion_tokens,
    }


def review_and_repair_agent(
    *,
    ticket_title: str,
    ticket_spec: str,
    diff: str,
    test_output: str,
    author_model: str,
    reviewer_base_url: str,
    reviewer_model: str,
    api_key: str = "outrcore-swarm",
    allow_same_vendor_fresh: bool = True,
    worktree: Path | str | None = None,
    max_repair_rounds: int = 1,
    repair_fn: Callable[[list[dict[str, str]], dict[str, Any]], dict[str, Any]] | None = None,
    chat_fn: Callable[..., dict[str, Any]] | None = None,
) -> dict[str, Any]:
    """Run critic review and, when non-approved and repairable, iterate repair + fresh review."""
    initial_review = review_agent_diff(
        ticket_title=ticket_title,
        ticket_spec=ticket_spec,
        diff=diff,
        test_output=test_output,
        author_model=author_model,
        reviewer_base_url=reviewer_base_url,
        reviewer_model=reviewer_model,
        api_key=api_key,
        allow_same_vendor_fresh=allow_same_vendor_fresh,
        worktree=worktree,
        chat_fn=chat_fn,
    )

    eligible = bool((initial_review.get("eligibility") or {}).get("eligible", False))
    if (
        initial_review.get("verdict") == "approve"
        or max_repair_rounds < 1
        or repair_fn is None
        or not eligible
    ):
        result = dict(initial_review)
        result["initial_review"] = initial_review
        result["repair_attempted"] = False
        result["repair_rounds"] = 0
        result["history"] = [initial_review]
        result["diff"] = diff
        result["test_output"] = test_output
        return result

    history: list[dict[str, Any]] = [initial_review]
    current_review = initial_review
    current_diff = diff
    current_test_output = test_output
    repair_rounds = 0

    for _ in range(max_repair_rounds):
        repair_out = repair_fn(list(current_review.get("findings") or []), current_review)
        repair_rounds += 1
        if isinstance(repair_out, dict):
            if "diff" in repair_out:
                current_diff = str(repair_out.get("diff") or "")
            if "test_output" in repair_out:
                current_test_output = str(repair_out.get("test_output") or "")

        current_review = review_agent_diff(
            ticket_title=ticket_title,
            ticket_spec=ticket_spec,
            diff=current_diff,
            test_output=current_test_output,
            author_model=author_model,
            reviewer_base_url=reviewer_base_url,
            reviewer_model=reviewer_model,
            api_key=api_key,
            allow_same_vendor_fresh=allow_same_vendor_fresh,
            worktree=worktree,
            chat_fn=chat_fn,
        )
        history.append(current_review)
        if current_review.get("verdict") == "approve":
            break

    result = dict(current_review)
    result["initial_review"] = initial_review
    result["repair_attempted"] = True
    result["repair_rounds"] = repair_rounds
    result["history"] = history
    result["diff"] = current_diff
    result["test_output"] = current_test_output
    return result


def _strip_identifying_strings(text: str, forbidden: list[str]) -> str:
    """Remove any author/model/stance identifiers before presenting text to a blind judge."""
    out = text or ""
    tokens = sorted({tok.strip() for tok in forbidden if tok and len(tok.strip()) >= 2}, key=len, reverse=True)
    for tok in tokens:
        escaped = re.escape(tok)
        if tok.isalnum() and len(tok) <= 3:
            pattern = re.compile(rf"\b{escaped}\b", re.IGNORECASE)
        else:
            pattern = re.compile(escaped, re.IGNORECASE)
        out = pattern.sub("[REDACTED]", out)
    return out


def _merge_receipts(
    sub_receipts: list[dict[str, Any]],
    *,
    repo: Path,
    base_url: str,
    model: str,
) -> dict[str, Any]:
    """Combine multiple `run_swarm` receipts (when candidates span distinct models/endpoints)."""
    if len(sub_receipts) == 1:
        return dict(sub_receipts[0])

    all_results: list[dict[str, Any]] = []
    for rec in sub_receipts:
        all_results.extend(rec.get("results") or [])
    all_results.sort(key=lambda r: str(r.get("agent_id") or ""))

    first = sub_receipts[0]
    merged_metering: dict[str, Any] = {}
    for rec in sub_receipts:
        m = rec.get("metering")
        if not isinstance(m, dict):
            continue
        for k, v in m.items():
            if isinstance(v, (int, float)) and not isinstance(v, bool):
                merged_metering[k] = merged_metering.get(k, 0) + v
            elif k not in merged_metering:
                merged_metering[k] = v

    return {
        "run_id": first.get("run_id") or "tournament",
        "repo": str(repo.resolve()),
        "backend_url": base_url,
        "model": model,
        "agents_count": len(all_results),
        "passed_count": sum(1 for r in all_results if r.get("tests_ok")),
        "metering": merged_metering,
        "results": all_results,
    }


def _disqualification_reason(res: dict[str, Any]) -> str:
    reasons: list[str] = []
    if not res.get("tests_ok"):
        err = str(res.get("error") or "").strip()
        reasons.append(err if err else "failing acceptance tests")
    if not (res.get("diff") or "").strip():
        reasons.append("empty diff")
    return "; ".join(reasons) or "disqualified"


def run_tournament(
    *,
    repo: Path,
    ticket: dict[str, str],
    base_url: str,
    model: str,
    candidates: list[dict[str, Any]] | None = None,
    judge_base_url: str | None = None,
    judge_model: str | None = None,
    api_key: str = "outrcore-swarm",
    spend_cap_usd: float | None = 1.0,
    max_tokens: int = 4096,
    keep_worktrees: bool = True,
    on_progress: Callable[[dict[str, Any]], None] | None = None,
    runner_fn: Callable[..., dict[str, Any]] | None = None,
    chat_fn: Callable[..., dict[str, Any]] | None = None,
) -> dict[str, Any]:
    """Run competing candidate implementations in parallel, disqualify failures, and blind-judge survivors."""
    cand_specs = candidates if candidates else [
        {"agent_id": f"cand-{chr(ord('a') + idx)}", "stance": stance, "guidance": guidance}
        for idx, (stance, guidance) in enumerate(_DEFAULT_TOURNAMENT_STANCES)
    ]

    prepared: list[tuple[dict[str, str], str, str, str]] = []
    forbidden_strings: list[str] = [model]
    base_title = str(ticket.get("title") or "Tournament Ticket")
    base_spec = str(ticket.get("spec") or "")
    base_id = str(ticket.get("id") or "T-001")
    base_cmd = str(ticket.get("acceptance_cmd") or "")

    for idx, cand in enumerate(cand_specs):
        agent_id = str(cand.get("agent_id") or f"cand-{chr(ord('a') + idx)}")
        default_stance, default_guidance = _DEFAULT_TOURNAMENT_STANCES[idx % len(_DEFAULT_TOURNAMENT_STANCES)]
        stance = str(cand.get("stance") or default_stance)
        guidance = str(cand.get("guidance") or cand.get("instructions") or default_guidance)
        cand_model = str(cand.get("model") or model)
        cand_base_url = str(cand.get("base_url") or base_url)

        task = dict(ticket)
        task["id"] = base_id
        task["title"] = base_title
        task["acceptance_cmd"] = base_cmd
        task["agent_id"] = agent_id
        task["stance"] = stance
        task["spec"] = f"{base_spec}\n\n## Approach Stance: {stance}\n{guidance}".strip()

        prepared.append((task, cand_base_url, cand_model, stance))
        forbidden_strings.extend([agent_id, stance, cand_model])

    runner = runner_fn or swarm_run.run_swarm
    groups: dict[tuple[str, str], list[dict[str, str]]] = {}
    for task, c_url, c_model, _ in prepared:
        groups.setdefault((c_url, c_model), []).append(task)

    if len(groups) == 1:
        (only_url, only_model), group_tasks = next(iter(groups.items()))
        raw_receipt = runner(
            repo=repo,
            tasks=group_tasks,
            base_url=only_url,
            model=only_model,
            api_key=api_key,
            spend_cap_usd=spend_cap_usd,
            max_tokens=max_tokens,
            keep_worktrees=keep_worktrees,
            on_progress=on_progress,
        )
        receipt = dict(raw_receipt)
    else:
        sub_receipts: list[dict[str, Any]] = []
        with ThreadPoolExecutor(max_workers=len(groups)) as pool:
            futs = [
                pool.submit(
                    runner,
                    repo=repo,
                    tasks=group_tasks,
                    base_url=g_url,
                    model=g_model,
                    api_key=api_key,
                    spend_cap_usd=spend_cap_usd,
                    max_tokens=max_tokens,
                    keep_worktrees=keep_worktrees,
                    on_progress=on_progress,
                )
                for (g_url, g_model), group_tasks in groups.items()
            ]
            for fut in futs:
                sub_receipts.append(fut.result())
        receipt = _merge_receipts(sub_receipts, repo=repo, base_url=base_url, model=model)

    results: list[dict[str, Any]] = list(receipt.get("results") or [])
    for r in results:
        if r.get("agent_id"):
            forbidden_strings.append(str(r["agent_id"]))
        if r.get("model"):
            forbidden_strings.append(str(r["model"]))
        if r.get("stance"):
            forbidden_strings.append(str(r["stance"]))

    survivors: list[dict[str, Any]] = []
    disqualified: list[dict[str, str]] = []
    for r in results:
        if not r.get("tests_ok") or not (r.get("diff") or "").strip():
            disqualified.append(
                {
                    "agent_id": str(r.get("agent_id") or ""),
                    "reason": _disqualification_reason(r),
                }
            )
        else:
            survivors.append(r)

    effective_judge_base_url = judge_base_url or base_url
    effective_judge_model = judge_model or model

    if not survivors:
        receipt["mode"] = "tournament"
        receipt["tournament"] = {
            "winner_agent_id": None,
            "winner_label": None,
            "rationale": "All candidates were disqualified (failing tests or empty diff); judge skipped.",
            "disqualified": disqualified,
            "rankings": [],
            "judge_model": effective_judge_model,
        }
        return receipt

    label_to_agent: dict[str, str] = {}
    candidate_blocks: list[str] = []
    for idx, surv in enumerate(survivors):
        label = f"Candidate {chr(ord('A') + idx)}"
        label_to_agent[label] = str(surv.get("agent_id") or "")
        clean_diff = _strip_identifying_strings(str(surv.get("diff") or ""), forbidden_strings)
        clean_test_out = _strip_identifying_strings(str(surv.get("test_output") or ""), forbidden_strings)
        candidate_blocks.append(
            f"### {label}\n"
            f"Diff:\n```diff\n{clean_diff}\n```\n"
            f"Test Output:\n```\n{clean_test_out}\n```"
        )

    clean_title = _strip_identifying_strings(base_title, forbidden_strings)
    clean_spec = _strip_identifying_strings(base_spec, forbidden_strings)
    judge_system_prompt = (
        "You are an impartial blind code judge evaluating anonymous candidate patches.\n"
        "All candidates passed acceptance tests. Compare correctness, simplicity, edge-case safety, "
        "and code clarity without bias.\n"
        "Respond ONLY with valid JSON matching this schema:\n"
        '{"winner": "Candidate A", "rationale": "<why the winner is strongest>", '
        '"rankings": [{"label": "Candidate A", "score": 9, "strengths": "<strengths>", "objection": "<weakness>"}]}'
    )
    judge_user_prompt = (
        f"## Ticket: {clean_title}\n{clean_spec}\n\n"
        "## Anonymous Candidates\n\n" + "\n\n".join(candidate_blocks)
    )

    chat = chat_fn or swarm_run.openai_chat
    judge_resp = chat(
        base_url=effective_judge_base_url,
        api_key=api_key,
        model=effective_judge_model,
        messages=[
            {"role": "system", "content": judge_system_prompt},
            {"role": "user", "content": judge_user_prompt},
        ],
        max_tokens=LANES["reviewer"]["default_max_tokens"],
        temperature=LANES["reviewer"]["temperature"],
        stream=False,
    )
    raw_judge = str(judge_resp.get("content") or "")
    parsed_judge = _extract_json_dict(raw_judge)

    first_label = "Candidate A"
    first_agent_id = label_to_agent[first_label]

    if parsed_judge is None:
        winner_label = first_label
        winner_agent_id = first_agent_id
        rationale = f"Malformed judge JSON; defaulted to {first_label}. Raw snippet: {raw_judge.strip()[:250]}"
        rankings: list[dict[str, Any]] = []
    else:
        raw_winner = str(parsed_judge.get("winner") or "").strip()
        if raw_winner in label_to_agent:
            winner_label = raw_winner
        else:
            matched = next(
                (lbl for lbl in label_to_agent if lbl.lower() == raw_winner.lower()),
                None,
            )
            winner_label = matched or first_label
        winner_agent_id = label_to_agent[winner_label]
        rationale = str(parsed_judge.get("rationale") or "").strip() or f"Selected {winner_label}."
        raw_rankings = parsed_judge.get("rankings")
        rankings = raw_rankings if isinstance(raw_rankings, list) else []

    receipt["mode"] = "tournament"
    receipt["tournament"] = {
        "winner_agent_id": winner_agent_id,
        "winner_label": winner_label,
        "rationale": rationale,
        "disqualified": disqualified,
        "rankings": rankings,
        "judge_model": effective_judge_model,
    }
    return receipt
