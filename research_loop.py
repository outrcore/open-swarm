#!/usr/bin/env python3
"""M2 — Karpathy Autoresearch Swarm Engine for OutrCore Open Swarm.

Runs barrier-synchronized parallel research islands in isolated git worktrees
with:
- Anti-tamper verification (`readonly_paths` SHA-256 + `editable_paths` allowlist)
- Isolated `health_cmd` correctness gating (timed separately from benchmark eval)
- Isolated `eval_cmd` benchmarking (`PYTHONDONTWRITEBYTECODE=1`, `__pycache__` purged)
- Monotonic git ratchet (`keep` commits improvements; `revert` resets regressions)
- Shepherd cross-pollination of global best findings and diffs across islands
- `.open_swarm/` artifact logging (`program.md`, `results.tsv`, `findings.md`)
"""
from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
import inspect
import os
from pathlib import Path, PurePosixPath
import re
import shutil
import subprocess
import sys
import threading
import time
from typing import Any, Callable

SWARM_DIR = Path(__file__).resolve().parent
if str(SWARM_DIR) not in sys.path:
    sys.path.insert(0, str(SWARM_DIR))

from continuity_store import compute_file_hashes, extract_diff_paths  # noqa: E402
from metering import PricingSchedule, RunMeter, SpendCapExceeded  # noqa: E402
from swarm_run import _git, apply_file_blocks, openai_chat  # noqa: E402

_OPEN_SWARM_DIR = ".open_swarm"
_EXCLUDE_LOCK = threading.Lock()
_GIT_IDENTITY_ARGS = ("-c", "user.name=OpenSwarm", "-c", "user.email=swarm@outrcore.local")


@dataclass
class ResearchConfig:
    """Configuration for a Karpathy-style autoresearch swarm run."""

    goal: str
    eval_cmd: str
    health_cmd: str = ""
    metric_regex: str = r"(?:score|metric|val_bpb|tok_s|ms)\s*[:=]\s*([-+]?\d+(?:\.\d+)?)"
    direction: str = "max"
    editable_paths: list[str] = field(default_factory=list)
    readonly_paths: list[str] = field(default_factory=list)
    islands: int = 2
    rounds: int = 3
    cross_pollinate_every: int = 2
    spend_cap_usd: float = 0.50
    eval_timeout_s: int = 60
    health_timeout_s: int = 30
    use_tool_loop: bool = True
    max_tool_turns: int = 6
    max_tokens: int = 2048


@dataclass
class _IslandState:
    island_id: str
    branch: str
    worktree: Path
    best_score: float
    best_commit: str
    kept_count: int = 0
    findings: list[str] = field(default_factory=list)
    tsv_rows: list[str] = field(default_factory=list)
    completion_tokens: int = 0
    prompt_tokens: int = 0
    cached_tokens: int = 0
    last_error: str | None = None
    last_output_preview: str = ""
    last_diff_summary: str = ""


def _island_name(idx: int) -> str:
    if idx < 26:
        return f"island-{chr(ord('a') + idx)}"
    return f"island-{idx + 1}"


def _is_better(score: float, best: float, direction: str) -> bool:
    if direction == "max":
        return score > best
    return score < best


def _norm_rel(path_str: str) -> str:
    raw = path_str.strip().replace("\\", "/")
    while raw.startswith("./"):
        raw = raw[2:]
    return raw.rstrip("/")


def _path_matches_rule(rel_path: str, rule: str) -> bool:
    norm_path = _norm_rel(rel_path)
    norm_rule = _norm_rel(rule)
    if not norm_rule:
        return False
    if norm_path == norm_rule:
        return True
    if norm_path.startswith(norm_rule + "/"):
        return True
    return PurePosixPath(norm_path).match(norm_rule)


def _expand_readonly_paths(repo: Path, readonly_paths: list[str]) -> list[str]:
    root = repo.resolve()
    expanded: list[str] = []
    seen: set[str] = set()
    for raw in readonly_paths:
        rel = _norm_rel(raw)
        if not rel or ".." in Path(rel).parts:
            continue
        candidate = (root / rel).resolve()
        if not (candidate == root or candidate.is_relative_to(root)):
            continue
        if candidate.is_dir():
            for fp in sorted(candidate.rglob("*")):
                if not fp.is_file():
                    continue
                fp_rel = fp.relative_to(root).as_posix()
                if fp_rel.startswith(".git/") or "__pycache__" in fp_rel:
                    continue
                if fp_rel not in seen:
                    seen.add(fp_rel)
                    expanded.append(fp_rel)
        else:
            if rel not in seen:
                seen.add(rel)
                expanded.append(rel)
    return expanded


def _ensure_open_swarm_excluded(wt: Path) -> None:
    """Ignore `.open_swarm/` via git info/exclude so metadata never leaks into code diffs."""
    proc = _git(wt, "rev-parse", "--git-path", "info/exclude", check=False)
    rel_or_abs = (proc.stdout or "").strip()
    if not rel_or_abs:
        return
    exclude_file = Path(rel_or_abs) if Path(rel_or_abs).is_absolute() else (wt / rel_or_abs)
    with _EXCLUDE_LOCK:
        exclude_file.parent.mkdir(parents=True, exist_ok=True)
        existing = exclude_file.read_text(encoding="utf-8", errors="replace") if exclude_file.exists() else ""
        lines = {ln.strip() for ln in existing.splitlines()}
        if f"{_OPEN_SWARM_DIR}/" in lines or _OPEN_SWARM_DIR in lines:
            return
        prefix = "" if (not existing or existing.endswith("\n")) else "\n"
        with exclude_file.open("a", encoding="utf-8") as fh:
            fh.write(f"{prefix}{_OPEN_SWARM_DIR}/\n")


def _purge_pycache(root: Path) -> None:
    for pc_dir in root.rglob("__pycache__"):
        if ".git" in pc_dir.parts:
            continue
        if pc_dir.is_dir():
            shutil.rmtree(pc_dir, ignore_errors=True)


def _run_isolated_cmd(cwd: Path, cmd: str, timeout_s: int) -> tuple[int, str, float]:
    """Run a command with `__pycache__` purged and `PYTHONDONTWRITEBYTECODE=1`, timed via perf_counter."""
    _purge_pycache(cwd)
    env = dict(os.environ)
    env["PYTHONDONTWRITEBYTECODE"] = "1"
    t0 = time.perf_counter()
    try:
        proc = subprocess.run(
            cmd,
            cwd=str(cwd),
            shell=True,
            env=env,
            capture_output=True,
            text=True,
            timeout=max(1, int(timeout_s)),
        )
        elapsed_s = time.perf_counter() - t0
        out = (proc.stdout + "\n" + proc.stderr).strip()
        return proc.returncode, out, elapsed_s
    except subprocess.TimeoutExpired as exc:
        elapsed_s = time.perf_counter() - t0
        stdout_txt = exc.stdout.decode("utf-8", errors="replace") if isinstance(exc.stdout, bytes) else (exc.stdout or "")
        stderr_txt = exc.stderr.decode("utf-8", errors="replace") if isinstance(exc.stderr, bytes) else (exc.stderr or "")
        out = (stdout_txt + "\n" + stderr_txt + f"\nCommand timed out after {timeout_s}s").strip()
        return 124, out, elapsed_s


def _parse_metric(output: str, metric_regex: str) -> float | None:
    matches = list(re.finditer(metric_regex, output))
    if not matches:
        return None
    last = matches[-1]
    raw_val = last.group(1) if last.lastindex and last.lastindex >= 1 else last.group(0)
    try:
        return float(raw_val)
    except (TypeError, ValueError):
        return None


def _git_dirty_paths(wt: Path) -> list[str]:
    proc = _git(wt, "status", "--porcelain", "--untracked-files=all", "-z", check=False)
    tokens = (proc.stdout or "").split("\0")
    paths: list[str] = []
    seen: set[str] = set()
    i = 0
    while i < len(tokens):
        entry = tokens[i]
        if not entry or len(entry) < 4:
            i += 1
            continue
        xy = entry[:2]
        path = _norm_rel(entry[3:])
        if path and path != _OPEN_SWARM_DIR and not path.startswith(f"{_OPEN_SWARM_DIR}/"):
            if path not in seen:
                seen.add(path)
                paths.append(path)
        if "R" in xy or "C" in xy:
            if i + 1 < len(tokens):
                old_p = _norm_rel(tokens[i + 1])
                if old_p and old_p != _OPEN_SWARM_DIR and not old_p.startswith(f"{_OPEN_SWARM_DIR}/"):
                    if old_p not in seen:
                        seen.add(old_p)
                        paths.append(old_p)
            i += 2
        else:
            i += 1
    return paths


def _check_tamper(
    wt: Path,
    readonly_hashes: dict[str, str | None],
    editable_paths: list[str],
) -> str | None:
    """Return a non-empty tamper reason if readonly files or non-editable paths were touched."""
    if readonly_hashes:
        actual = compute_file_hashes(wt, list(readonly_hashes.keys()))
        for rel_path, expected in readonly_hashes.items():
            if rel_path not in actual:
                return f"readonly path missing or escaped: {rel_path}"
            if actual[rel_path] != expected:
                return f"readonly path modified or deleted: {rel_path}"

    dirty = _git_dirty_paths(wt)
    if editable_paths:
        for path in dirty:
            if not any(_path_matches_rule(path, rule) for rule in editable_paths):
                return f"modified path outside editable_paths: {path}"
    return None


def _revert_worktree(wt: Path) -> None:
    _git(wt, "reset", "--hard", "HEAD", check=False)
    _git(wt, "clean", "-fd", check=False)


def _apply_safe_file_blocks(wt: Path, content: str) -> list[str]:
    """Filter out `.open_swarm/` file blocks before calling `apply_file_blocks`."""
    safe_chunks: list[str] = []
    skipping = False
    for line in (content or "").splitlines(keepends=True):
        stripped = line.strip()
        if not skipping and stripped.startswith("```file:"):
            target = _norm_rel(stripped[len("```file:") :])
            if target == _OPEN_SWARM_DIR or target.startswith(f"{_OPEN_SWARM_DIR}/"):
                skipping = True
                continue
        elif skipping:
            if stripped == "```":
                skipping = False
            continue
        safe_chunks.append(line)
    return apply_file_blocks(wt, "".join(safe_chunks))


def _sanitize_tsv_text(text: str, limit: int = 160) -> str:
    cleaned = " ".join((text or "").replace("\t", " ").split())
    if len(cleaned) <= limit:
        return cleaned
    return cleaned[: limit - 3].rstrip() + "..."


def _extract_description(content: str, written: list[str], round_idx: int) -> str:
    for line in (content or "").splitlines():
        s = line.strip()
        if not s or s.startswith("```"):
            continue
        return _sanitize_tsv_text(s, limit=140)
    if written:
        return _sanitize_tsv_text(f"updated {', '.join(written)}", limit=140)
    return f"round {round_idx} edit"


def _build_program_md(config: ResearchConfig) -> str:
    dir_word = "maximize" if config.direction == "max" else "minimize"
    editable_txt = "\n".join(f"- `{p}`" for p in config.editable_paths) if config.editable_paths else "- (any non-readonly repository file)"
    readonly_txt = "\n".join(f"- `{p}`" for p in config.readonly_paths) if config.readonly_paths else "- (none)"
    health_txt = f"`{config.health_cmd}`" if config.health_cmd.strip() else "(none)"
    return (
        "# Autoresearch Program\n\n"
        f"## Goal\n{config.goal}\n\n"
        f"## Optimization Target\n- Direction: **{config.direction}** ({dir_word} the metric)\n"
        f"- Metric Regex: `{config.metric_regex}`\n\n"
        f"## Evaluation & Health Commands\n"
        f"- Health Command: {health_txt}\n"
        f"- Eval Command: `{config.eval_cmd}`\n\n"
        f"## Editable Paths\n{editable_txt}\n\n"
        f"## Readonly Paths (Anti-Tamper Protected)\n{readonly_txt}\n\n"
        "## Rules\n"
        "1. Only edit allowed files (`editable_paths` if non-empty).\n"
        "2. NEVER modify, delete, or bypass `readonly_paths` or `.open_swarm/`.\n"
        "3. Keep edits minimal and hypothesis-driven; only strictly improved scores are kept.\n"
        "4. `health_cmd` (if set) must exit 0 before `eval_cmd` is accepted.\n"
    )


def _build_system_prefix(repo: Path, config: ResearchConfig) -> str:
    """Deterministic system prefix shared across islands/rounds for KV cache reuse."""
    files: list[str] = []
    for p in sorted(repo.rglob("*")):
        if not p.is_file():
            continue
        rel = p.relative_to(repo).as_posix()
        if rel.startswith(".git/") or rel.startswith(f"{_OPEN_SWARM_DIR}/") or "__pycache__" in rel or rel.endswith(".pyc"):
            continue
        files.append(rel)
        if len(files) >= 60:
            break
    tree_txt = "\n".join(f"- {f}" for f in files)
    mode_instr = (
        "You may inspect and edit files via tools or emit ```file:relative/path``` blocks."
        if config.use_tool_loop
        else "Respond with complete ```file:relative/path``` blocks for the files you update."
    )
    return (
        "You are an autonomous research engineer inside an OutrCore Open Swarm git worktree.\n"
        f"{mode_instr}\n\n"
        f"{_build_program_md(config)}\n"
        f"## Base Repository Tree ({repo.name})\n{tree_txt}"
    )


def _snapshot_editable_files(wt: Path, editable_paths: list[str], max_total_chars: int = 8000) -> str:
    if not editable_paths:
        return ""
    snippets: list[str] = []
    total = 0
    for p in sorted(wt.rglob("*")):
        if not p.is_file():
            continue
        rel = p.relative_to(wt).as_posix()
        if rel.startswith(".git/") or rel.startswith(f"{_OPEN_SWARM_DIR}/") or "__pycache__" in rel:
            continue
        if not any(_path_matches_rule(rel, rule) for rule in editable_paths):
            continue
        if p.stat().st_size > 16_384:
            continue
        txt = p.read_text(encoding="utf-8", errors="replace")
        blk = f"### CURRENT FILE: {rel}\n```\n{txt}\n```"
        if snippets and total + len(blk) > max_total_chars:
            break
        snippets.append(blk[:max_total_chars] if not snippets and len(blk) > max_total_chars else blk)
        total += len(snippets[-1])
    return "\n\n".join(snippets)


def _invoke_tool_loop(
    *,
    run_agent_loop: Callable[..., Any],
    wt: Path,
    base_url: str,
    api_key: str,
    model: str,
    system_prompt: str,
    user_prompt: str,
    config: ResearchConfig,
    meter: RunMeter,
    agent_id: str,
    on_delta: Callable[[str, str], None] | None,
    on_progress: Callable[[dict[str, Any]], None] | None,
) -> tuple[dict[str, Any], bool]:
    """Adapt arguments to `agent_loop.run_agent_loop` via signature inspection."""
    sig = inspect.signature(run_agent_loop)
    params = sig.parameters
    accepts_var_kw = any(p.kind == inspect.Parameter.VAR_KEYWORD for p in params.values())

    candidate_kwargs: dict[str, Any] = {
        "worktree": wt,
        "repo": wt,
        "cwd": wt,
        "root": wt,
        "base_url": base_url,
        "api_key": api_key,
        "model": model,
        "system_prompt": system_prompt,
        "user_prompt": user_prompt,
        "prompt": user_prompt,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        "max_turns": config.max_tool_turns,
        "max_tool_turns": config.max_tool_turns,
        "max_tokens": config.max_tokens,
        "temperature": 0.2,
        "editable_paths": list(config.editable_paths),
        "readonly_paths": list(config.readonly_paths),
        "meter": meter,
        "agent_id": agent_id,
        "on_delta": on_delta,
        "on_progress": on_progress,
    }
    call_kwargs = (
        dict(candidate_kwargs)
        if accepts_var_kw
        else {k: v for k, v in candidate_kwargs.items() if k in params}
    )
    meter_passed = "meter" in call_kwargs
    raw_res = run_agent_loop(**call_kwargs)
    if isinstance(raw_res, dict):
        return raw_res, meter_passed
    return {"content": str(raw_res or "")}, meter_passed


def run_research_swarm(
    *,
    repo: Path,
    config: ResearchConfig,
    base_url: str,
    model: str,
    api_key: str = "outrcore-swarm",
    keep_worktrees: bool = True,
    on_progress: Callable[[dict[str, Any]], None] | None = None,
    continuity_store: Any = None,
    meter: Any = None,
) -> dict[str, Any]:
    """Execute a multi-island Karpathy autoresearch swarm on `repo`."""
    if config.direction not in ("max", "min"):
        raise ValueError(f"config.direction must be 'max' or 'min', got {config.direction!r}")

    repo = Path(repo).resolve()
    run_meter: RunMeter = meter if meter is not None else RunMeter(spend_cap_usd=config.spend_cap_usd)
    t_start = time.monotonic()

    def _emit(payload: dict[str, Any]) -> None:
        if on_progress is None:
            return
        try:
            on_progress(payload)
        except Exception:
            pass

    # 1. Base commit
    base_commit = _git(repo, "rev-parse", "HEAD").stdout.strip()

    # 2. Hash readonly_paths on base HEAD
    expanded_readonly = _expand_readonly_paths(repo, config.readonly_paths)
    readonly_hashes = compute_file_hashes(repo, expanded_readonly)

    # 3. Create isolated git worktrees
    num_islands = max(1, int(config.islands))
    ts = f"{time.strftime('%Y%m%d_%H%M%S')}_{os.getpid()}_{time.monotonic_ns() % 1_000_000:06d}"
    run_id = f"research_{ts}"
    wt_base = repo.parent / ".swarm_worktrees" / run_id
    wt_base.mkdir(parents=True, exist_ok=True)

    program_md = _build_program_md(config)
    shared_sys = _build_system_prefix(repo, config)
    tsv_header = "round\tisland\tcommit\tscore\tbest_score\teval_s\thealth_s\tllm_s\tstatus\tdescription\n"

    island_states: list[_IslandState] = []
    try:
        for idx in range(num_islands):
            isl_id = _island_name(idx)
            branch = f"swarm/{run_id}/{isl_id}"
            wt = wt_base / isl_id
            _git(repo, "worktree", "add", "-B", branch, str(wt), base_commit)
            _ensure_open_swarm_excluded(wt)
            meta_dir = wt / _OPEN_SWARM_DIR
            meta_dir.mkdir(parents=True, exist_ok=True)
            (meta_dir / "program.md").write_text(program_md, encoding="utf-8")
            (meta_dir / "results.tsv").write_text(tsv_header, encoding="utf-8")
            (meta_dir / "findings.md").write_text(f"# Findings — {isl_id}\n\n", encoding="utf-8")
            island_states.append(
                _IslandState(
                    island_id=isl_id,
                    branch=branch,
                    worktree=wt,
                    best_score=0.0,
                    best_commit=base_commit,
                )
            )

        # 5. Baseline health + baseline eval in first island worktree
        baseline_wt = island_states[0].worktree
        baseline_health_s = 0.0
        if config.health_cmd.strip():
            h_rc, h_out, baseline_health_s = _run_isolated_cmd(
                baseline_wt, config.health_cmd, config.health_timeout_s
            )
            if h_rc != 0:
                raise RuntimeError(f"Baseline health_cmd failed (rc={h_rc}): {h_out[-1000:]}")

        e_rc, e_out, baseline_eval_s = _run_isolated_cmd(
            baseline_wt, config.eval_cmd, config.eval_timeout_s
        )
        if e_rc != 0:
            raise RuntimeError(f"Baseline eval_cmd failed (rc={e_rc}): {e_out[-1000:]}")
        baseline_score = _parse_metric(e_out, config.metric_regex)
        if baseline_score is None:
            raise RuntimeError(
                f"Baseline eval_cmd output did not match metric_regex {config.metric_regex!r}: {e_out[-1000:]}"
            )

        short_base = base_commit[:8]
        baseline_finding = f"## Round 0 (baseline)\n- Score: `{baseline_score}` (eval={baseline_eval_s:.4f}s, health={baseline_health_s:.4f}s)\n"
        for isl in island_states:
            isl.best_score = baseline_score
            base_row = (
                f"0\t{isl.island_id}\t{short_base}\t{baseline_score}\t{baseline_score}\t"
                f"{baseline_eval_s:.4f}\t{baseline_health_s:.4f}\t0.0000\tbaseline\tinitial baseline\n"
            )
            isl.tsv_rows.append(base_row)
            with (isl.worktree / _OPEN_SWARM_DIR / "results.tsv").open("a", encoding="utf-8") as fh:
                fh.write(base_row)
            with (isl.worktree / _OPEN_SWARM_DIR / "findings.md").open("a", encoding="utf-8") as fh:
                fh.write(baseline_finding + "\n")

        experiments: list[dict[str, Any]] = []
        total_rounds = max(1, int(config.rounds))
        cross_every = max(1, int(config.cross_pollinate_every))

        # 6. Barrier-synchronized rounds
        for round_idx in range(1, total_rounds + 1):
            # Shepherd Cross-Pollination check before launching round wave
            cross_briefs: dict[str, str] = {}
            if round_idx > 1 and (round_idx - 1) % cross_every == 0 and len(island_states) > 1:
                sorted_islands = sorted(
                    island_states,
                    key=lambda s: (
                        -s.best_score if config.direction == "max" else s.best_score,
                        -s.kept_count,
                        s.island_id,
                    ),
                )
                global_best = sorted_islands[0]
                if _is_better(global_best.best_score, baseline_score, config.direction) or global_best.kept_count > 0:
                    best_diff_txt = _git(
                        global_best.worktree,
                        "diff",
                        base_commit,
                        global_best.best_commit,
                        "--",
                        ".",
                        f":(exclude){_OPEN_SWARM_DIR}",
                        check=False,
                    ).stdout
                    if len(best_diff_txt) > 2500:
                        best_diff_txt = best_diff_txt[:2500] + "\n... (truncated)"
                    findings_tail = "\n".join(global_best.findings[-5:]) if global_best.findings else "(no findings yet)"
                    for isl in island_states:
                        if isl.island_id == global_best.island_id:
                            continue
                        cross_briefs[isl.island_id] = (
                            f"## Shepherd Cross-Pollination from {global_best.island_id}\n"
                            f"- Winning Island: {global_best.island_id}\n"
                            f"- Best Score: {global_best.best_score} (baseline: {baseline_score}, direction: {config.direction})\n\n"
                            f"### Winning Island Findings\n{findings_tail}\n\n"
                            f"### Winning Git Diff Summary\n```diff\n{best_diff_txt.strip()}\n```"
                        )
                        _emit(
                            {
                                "kind": "cross_pollination",
                                "round": round_idx,
                                "source_island": global_best.island_id,
                                "target_island": isl.island_id,
                                "best_score": global_best.best_score,
                            }
                        )
                        if continuity_store is not None:
                            try:
                                continuity_store.send_message(
                                    str(repo),
                                    global_best.island_id,
                                    isl.island_id,
                                    f"Shepherd R{round_idx}: {global_best.island_id} best_score={global_best.best_score}",
                                )
                            except Exception:
                                pass

            def _run_island_round(isl: _IslandState) -> dict[str, Any]:
                wt = isl.worktree
                isl_id = isl.island_id
                ticket_lbl = f"R{round_idx}"
                prog_state = {"stage": "thinking", "reasoning_chars": 0, "content_chars": 0, "snippet": "", "last_emit": 0.0}

                def _emit_stage(stage: str, *, force: bool = False) -> None:
                    now = time.monotonic()
                    changed = stage != prog_state["stage"]
                    prog_state["stage"] = stage
                    if not force and not changed and (now - prog_state["last_emit"]) < 0.4:
                        return
                    prog_state["last_emit"] = now
                    _emit(
                        {
                            "kind": "agent_stage",
                            "agent_id": isl_id,
                            "ticket": ticket_lbl,
                            "title": f"{isl_id} round {round_idx}",
                            "stage": stage,
                            "reasoning_chars": prog_state["reasoning_chars"],
                            "content_chars": prog_state["content_chars"],
                            "snippet": prog_state["snippet"],
                        }
                    )

                def _on_delta(kind: str, piece: str) -> None:
                    if kind == "reasoning":
                        prog_state["reasoning_chars"] += len(piece)
                    else:
                        prog_state["content_chars"] += len(piece)
                        prog_state["stage"] = "coding"
                    prog_state["snippet"] = (prog_state["snippet"] + piece)[-140:]
                    _emit_stage(str(prog_state["stage"]))

                _emit_stage("thinking", force=True)

                llm_s = 0.0
                health_s = 0.0
                eval_s = 0.0
                score: float | None = None
                status = "eval_failed"
                desc = f"round {round_idx}"
                step_error: str | None = None
                written: list[str] = []

                try:
                    run_meter.check_cap()
                    recent_findings = "\n".join(isl.findings[-6:]) if isl.findings else "(none yet)"
                    current_files_txt = _snapshot_editable_files(wt, config.editable_paths)
                    user_parts = [
                        f"## Research Round {round_idx}/{total_rounds} ({isl_id})",
                        f"- Current Island Best Score: `{isl.best_score}` (baseline: `{baseline_score}`, direction: `{config.direction}`)",
                        f"### Recent Findings\n{recent_findings}",
                    ]
                    if cross_briefs.get(isl_id):
                        user_parts.append(cross_briefs[isl_id])
                    if current_files_txt:
                        user_parts.append(f"### Current Editable Files\n{current_files_txt}")
                    user_parts.append(
                        "Propose and apply a single targeted improvement for this round. "
                        "Only edit allowed files. Do NOT touch readonly files or `.open_swarm/`."
                    )
                    user_prompt = "\n\n".join(user_parts)

                    used_tool_loop = False
                    resp: dict[str, Any] = {}
                    meter_already_recorded = False

                    if config.use_tool_loop:
                        try:
                            from agent_loop import run_agent_loop  # type: ignore[import-not-found]
                        except ImportError:
                            run_agent_loop = None
                        if run_agent_loop is not None:
                            used_tool_loop = True
                            t_llm0 = time.perf_counter()
                            resp, meter_already_recorded = _invoke_tool_loop(
                                run_agent_loop=run_agent_loop,
                                wt=wt,
                                base_url=base_url,
                                api_key=api_key,
                                model=model,
                                system_prompt=shared_sys,
                                user_prompt=user_prompt,
                                config=config,
                                meter=run_meter,
                                agent_id=isl_id,
                                on_delta=_on_delta,
                                on_progress=on_progress,
                            )
                            llm_s = time.perf_counter() - t_llm0

                    if not used_tool_loop:
                        messages = [
                            {"role": "system", "content": shared_sys},
                            {"role": "user", "content": user_prompt},
                        ]
                        t_llm0 = time.perf_counter()
                        resp = openai_chat(
                            base_url=base_url,
                            api_key=api_key,
                            model=model,
                            messages=messages,
                            max_tokens=config.max_tokens,
                            temperature=0.2,
                            stream=True,
                            on_delta=_on_delta,
                        )
                        llm_s = time.perf_counter() - t_llm0

                    content_txt = str(resp.get("content") or resp.get("final_text") or resp.get("summary") or "")
                    isl.last_output_preview = content_txt[-1200:]
                    p_toks = int(resp.get("prompt_tokens") or 0)
                    c_toks = int(resp.get("cached_tokens") or 0)
                    o_toks = int(resp.get("completion_tokens") or 0)
                    isl.prompt_tokens += p_toks
                    isl.cached_tokens += c_toks
                    isl.completion_tokens += o_toks
                    if not meter_already_recorded:
                        run_meter.record_usage(
                            isl_id,
                            prompt_tokens=p_toks,
                            cached_tokens=c_toks,
                            completion_tokens=o_toks,
                            gen_s=float(resp.get("gen_s") or llm_s),
                            ttft_ms=resp.get("ttft_ms"),
                        )

                    if used_tool_loop:
                        written = list(resp.get("written_files") or [])
                        if not written and "```file:" in content_txt:
                            written = _apply_safe_file_blocks(wt, content_txt)
                    else:
                        written = _apply_safe_file_blocks(wt, content_txt)

                    desc = _extract_description(content_txt, written, round_idx)

                    # Anti-Tamper check
                    tamper_reason = _check_tamper(wt, readonly_hashes, config.editable_paths)
                    if tamper_reason is None and resp.get("readonly_violations"):
                        tamper_reason = f"readonly tool violation: {resp['readonly_violations']}"
                    if tamper_reason is not None:
                        _revert_worktree(wt)
                        status = "tamper_reverted"
                        step_error = tamper_reason
                    else:
                        # Separate Health Check (isolated from eval_s)
                        health_ok = True
                        if config.health_cmd.strip():
                            _emit_stage("testing", force=True)
                            h_rc, h_out, health_s = _run_isolated_cmd(
                                wt, config.health_cmd, config.health_timeout_s
                            )
                            if h_rc != 0:
                                _revert_worktree(wt)
                                status = "health_failed"
                                step_error = h_out[-800:] or f"health_cmd exited {h_rc}"
                                health_ok = False

                        # Separate Benchmark Eval (isolated timing)
                        if health_ok:
                            _emit_stage("evaluating", force=True)
                            e_rc, e_out, eval_s = _run_isolated_cmd(
                                wt, config.eval_cmd, config.eval_timeout_s
                            )
                            parsed = _parse_metric(e_out, config.metric_regex) if e_rc == 0 else None
                            if e_rc != 0 or parsed is None:
                                _revert_worktree(wt)
                                status = "eval_failed"
                                step_error = (
                                    e_out[-800:]
                                    if e_rc != 0
                                    else f"eval_cmd output did not match metric_regex {config.metric_regex!r}"
                                )
                            else:
                                score = parsed
                                if _is_better(score, isl.best_score, config.direction):
                                    _git(wt, "add", "-A", check=False)
                                    has_staged = _git(wt, "diff", "--cached", "--quiet", check=False).returncode != 0
                                    if has_staged:
                                        _git(
                                            wt,
                                            *_GIT_IDENTITY_ARGS,
                                            "commit",
                                            "-m",
                                            f"research({isl_id} r{round_idx}): score={score}",
                                        )
                                        isl.best_commit = _git(wt, "rev-parse", "HEAD").stdout.strip()
                                    isl.best_score = score
                                    isl.kept_count += 1
                                    status = "keep"
                                else:
                                    _revert_worktree(wt)
                                    status = "revert"
                except SpendCapExceeded as exc:
                    _revert_worktree(wt)
                    status = "spend_cap"
                    step_error = str(exc)
                    isl.last_error = step_error
                except Exception as exc:
                    _revert_worktree(wt)
                    status = "eval_failed"
                    step_error = f"{type(exc).__name__}: {exc}"
                    isl.last_error = step_error

                commit_short = isl.best_commit[:8]
                score_str = f"{score}" if score is not None else "NA"
                tsv_row = (
                    f"{round_idx}\t{isl_id}\t{commit_short}\t{score_str}\t{isl.best_score}\t"
                    f"{eval_s:.4f}\t{health_s:.4f}\t{llm_s:.4f}\t{status}\t{desc}\n"
                )
                isl.tsv_rows.append(tsv_row)
                with (wt / _OPEN_SWARM_DIR / "results.tsv").open("a", encoding="utf-8") as fh:
                    fh.write(tsv_row)

                finding_entry = (
                    f"## Round {round_idx} — `{status}`\n"
                    f"- Score: `{score_str}` (island best: `{isl.best_score}`)\n"
                    f"- Timings: eval={eval_s:.4f}s, health={health_s:.4f}s, llm={llm_s:.4f}s\n"
                    f"- Summary: {desc}\n"
                )
                if step_error:
                    finding_entry += f"- Detail: {_sanitize_tsv_text(step_error, limit=240)}\n"
                isl.findings.append(finding_entry.strip())
                with (wt / _OPEN_SWARM_DIR / "findings.md").open("a", encoding="utf-8") as fh:
                    fh.write(finding_entry + "\n")

                step_record = {
                    "round": round_idx,
                    "island": isl_id,
                    "commit": commit_short,
                    "score": score,
                    "best_score": isl.best_score,
                    "eval_s": round(eval_s, 6),
                    "health_s": round(health_s, 6),
                    "llm_s": round(llm_s, 6),
                    "status": status,
                    "description": desc,
                    "error": step_error,
                }
                _emit({"kind": "research_step", **step_record})
                _emit_stage("passed" if status == "keep" else status, force=True)
                return step_record

            round_records: list[dict[str, Any]] = []
            with ThreadPoolExecutor(max_workers=num_islands) as pool:
                futs = [pool.submit(_run_island_round, isl) for isl in island_states]
                for fut in as_completed(futs):
                    round_records.append(fut.result())
            round_records.sort(key=lambda r: r["island"])
            experiments.extend(round_records)

        # 7. Compute per-island diffs & results matching swarm_run.py / DesktopAppState.apply_agent_diff
        results: list[dict[str, Any]] = []
        findings_sections: list[str] = []
        merged_tsv_lines: list[str] = [tsv_header.rstrip("\n")]

        for idx, isl in enumerate(island_states):
            wt = isl.worktree
            diff_txt = _git(
                wt,
                "diff",
                base_commit,
                "HEAD",
                "--",
                ".",
                f":(exclude){_OPEN_SWARM_DIR}",
                check=False,
            ).stdout
            names_raw = _git(
                wt,
                "diff",
                "--name-only",
                base_commit,
                "HEAD",
                "--",
                ".",
                f":(exclude){_OPEN_SWARM_DIR}",
                check=False,
            ).stdout
            written_files = [
                ln.strip()
                for ln in names_raw.splitlines()
                if ln.strip() and not ln.strip().startswith(f"{_OPEN_SWARM_DIR}/")
            ]
            if not written_files and diff_txt.strip():
                written_files = [
                    p for p in extract_diff_paths(diff_txt) if not p.startswith(f"{_OPEN_SWARM_DIR}/")
                ]
            base_file_hashes = compute_file_hashes(repo, written_files) if written_files else {}
            tests_ok = bool(isl.kept_count > 0 and diff_txt.strip())

            results.append(
                {
                    "agent_id": isl.island_id,
                    "ticket": f"R-{idx + 1:03d}",
                    "title": f"{isl.island_id} (best={isl.best_score})",
                    "branch": isl.branch,
                    "worktree": str(wt),
                    "stage": "passed" if tests_ok else "failed",
                    "tests_ok": tests_ok,
                    "diff": diff_txt,
                    "written_files": written_files,
                    "base_file_hashes": base_file_hashes,
                    "best_score": isl.best_score,
                    "kept_count": isl.kept_count,
                    "completion_tokens": isl.completion_tokens,
                    "prompt_tokens": isl.prompt_tokens,
                    "cached_tokens": isl.cached_tokens,
                    "error": isl.last_error,
                    "output_preview": isl.last_output_preview,
                }
            )
            findings_md_path = wt / _OPEN_SWARM_DIR / "findings.md"
            if findings_md_path.exists():
                findings_sections.append(findings_md_path.read_text(encoding="utf-8", errors="replace").strip())
            for row in isl.tsv_rows:
                merged_tsv_lines.append(row.rstrip("\n"))
            run_meter.set_agent_wall(isl.island_id, time.monotonic() - t_start)

        global_best = sorted(
            island_states,
            key=lambda s: (
                -s.best_score if config.direction == "max" else s.best_score,
                -s.kept_count,
                s.island_id,
            ),
        )[0]

        combined_results_tsv = "\n".join(merged_tsv_lines) + "\n"
        combined_findings_md = "\n\n".join(findings_sections) + ("\n" if findings_sections else "")
    finally:
        # 8. Clean up worktrees if not keep_worktrees
        if not keep_worktrees:
            for isl in island_states:
                if isl.worktree.exists():
                    _git(repo, "worktree", "remove", "--force", str(isl.worktree), check=False)
            shutil.rmtree(wt_base, ignore_errors=True)
            _git(repo, "worktree", "prune", check=False)

    total_wall = time.monotonic() - t_start
    return {
        "run_id": run_id,
        "repo": str(repo),
        "mode": "research",
        "base_commit": base_commit,
        "baseline_score": baseline_score,
        "baseline_eval_s": round(baseline_eval_s, 6),
        "baseline_health_s": round(baseline_health_s, 6),
        "best_score": global_best.best_score,
        "best_island": global_best.island_id,
        "direction": config.direction,
        "experiments": experiments,
        "artifacts": {
            "program_md": program_md,
            "results_tsv": combined_results_tsv,
            "findings_md": combined_findings_md,
        },
        "results": results,
        "passed_count": sum(1 for r in results if r["tests_ok"]),
        "agents_count": len(results),
        "metering": run_meter.summary(total_wall),
    }
