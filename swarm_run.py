#!/usr/bin/env python3
"""Open Swarm v0.1 — Parallel worktree coding orchestrator with OutrCore prefix-cache & spend-cap metering.

Can run against any OpenAI-compatible `--backend-url` (e.g. A5000 Pod, inference.outrcore.com,
AntSeed local proxy http://127.0.0.1:8377/v1, or local Spark), fanning out N independent
git-worktree coding agents that share a deterministic repo-tree system prefix (for 90%+
k5cuda prefix cache reuse), running per-worktree acceptance tests, collecting unified diffs
for human approval, and emitting a JSON receipt with exact token/cache/cost telemetry.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
import time
from typing import Any, Callable
import urllib.error
import urllib.request

SWARM_DIR = Path(__file__).resolve().parent
if str(SWARM_DIR) not in sys.path:
    sys.path.insert(0, str(SWARM_DIR))

from continuity_store import compute_file_hashes  # noqa: E402
from metering import PricingSchedule, RunMeter, SpendCapExceeded  # noqa: E402


def build_shared_repo_prefix(
    repo: Path,
    max_files: int = 60,
    max_total_chars: int = 6000,
    context_brief: str = "",
) -> str:
    """Deterministic repository context shared byte-for-byte across all parallel workers for prefix caching."""
    from agent_tools import index_repo_symbols

    files: list[str] = []
    for p in sorted(repo.rglob("*")):
        if not p.is_file():
            continue
        rel = p.relative_to(repo).as_posix()
        if rel.startswith(".git/") or "__pycache__" in rel or rel.endswith(".pyc"):
            continue
        files.append(rel)
        if len(files) >= max_files:
            break
    prioritized = sorted(files, key=lambda r: (0 if Path(r).name.lower() == "readme.md" else 1, r))
    snippets: list[str] = []
    total_chars = 0
    for rel in prioritized:
        fp = repo / rel
        if fp.stat().st_size > 3072:
            continue
        txt = fp.read_text(encoding="utf-8", errors="replace")
        block = f"### FILE: {rel}\n```\n{txt}\n```"
        if snippets and total_chars + len(block) > max_total_chars:
            continue
        snippets.append(block[:max_total_chars] if not snippets and len(block) > max_total_chars else block)
        total_chars += len(snippets[-1])
        if total_chars >= max_total_chars:
            break
    tree_txt = "\n".join(f"- {f}" for f in files)
    symbols_txt = index_repo_symbols(repo, max_chars=1800)
    parts = [
        "You are an expert Python engineer operating inside an OutrCore Open Swarm git worktree.\n"
        "Output ONLY clean, complete file updates using explicit fenced file blocks:\n"
        "```file:relative/path.py\n<full file content>\n```\n"
        "Keep internal reasoning brief (<250 words) so plenty of tokens remain for complete ```file:relative/path.py``` blocks.\n"
        "Keep edits minimal, preserve existing behavior/tests, and write thorough unittest cases.\n\n"
        f"## Repository Tree ({repo.name})\n{tree_txt}"
    ]
    if symbols_txt:
        parts.append(f"## Symbol Index\n{symbols_txt}")
    if context_brief.strip():
        parts.append(f"## Prior Continuity & Notes\n{context_brief.strip()}")
    parts.append("## Existing Files\n" + "\n\n".join(snippets))
    return "\n\n".join(parts)


FILE_BLOCK_RE = re.compile(r"^[ \t]*```file:([^\n`]+)\r?\n(.*?)^[ \t]*```[ \t]*$", re.DOTALL | re.MULTILINE)
_FILE_HEADER_RE = re.compile(r"^[ \t]*```file:([^\n`]+)\r?\n", re.MULTILINE)
_CLOSE_FENCE_RE = re.compile(r"^[ \t]*```[ \t]*$", re.MULTILINE)


def _write_confined_block(root: Path, rel_raw: str, body: str) -> str | None:
    rel = rel_raw.strip().lstrip("/")
    if not rel or ".." in Path(rel).parts or rel.startswith(".git"):
        return None
    target = (root / rel).resolve()
    if not (target == root or str(target).startswith(str(root) + os.sep)):
        return None
    target.parent.mkdir(parents=True, exist_ok=True)
    if body and not body.endswith("\n"):
        body += "\n"
    target.write_text(body, encoding="utf-8")
    return rel


def apply_file_blocks(worktree: Path, content: str) -> list[str]:
    """Safely write ```file:rel/path``` blocks strictly inside `worktree`."""
    written: list[str] = []
    root = worktree.resolve()
    txt = content or ""
    headers = list(_FILE_HEADER_RE.finditer(txt))
    for i, hdr in enumerate(headers):
        end = headers[i + 1].start() if i + 1 < len(headers) else len(txt)
        segment = txt[hdr.end() : end]
        close = _CLOSE_FENCE_RE.search(segment)
        body = segment[: close.start()] if close else segment
        rel = _write_confined_block(root, hdr.group(1), body)
        if rel:
            written.append(rel)
    return written


def _parse_sse_chunk(
    chunk: dict[str, Any],
    content_chunks: list[str],
    reasoning_chunks: list[str],
    state: dict[str, Any],
    now: float,
    on_delta: Callable[[str, str], None] | None = None,
) -> None:
    if isinstance(chunk.get("usage"), dict):
        state["usage"] = chunk["usage"]
    if isinstance(chunk.get("timings"), dict):
        state["timings"] = chunk["timings"]
    choice = (chunk.get("choices") or [{}])[0]
    delta = choice.get("delta") or choice.get("message") or {}
    r_piece = delta.get("reasoning_content") or ""
    c_piece = delta.get("content") or ""
    if (r_piece or c_piece) and state["t_first_tok"] is None:
        state["t_first_tok"] = now
    if r_piece:
        reasoning_chunks.append(r_piece)
        if on_delta is not None:
            on_delta("reasoning", r_piece)
    if c_piece:
        content_chunks.append(c_piece)
        if on_delta is not None:
            on_delta("content", c_piece)


def openai_chat(
    *,
    base_url: str,
    api_key: str,
    model: str,
    messages: list[dict[str, str]],
    max_tokens: int = 4096,
    temperature: float = 0.0,
    timeout_s: int = 180,
    stream: bool = False,
    on_delta: Callable[[str, str], None] | None = None,
) -> dict[str, Any]:
    url = base_url.rstrip("/") + "/chat/completions"
    if not url.endswith("/v1/chat/completions") and "/v1/" not in url:
        url = base_url.rstrip("/") + "/v1/chat/completions"
    payload: dict[str, Any] = {
        "model": model,
        "messages": messages,
        "max_tokens": max_tokens,
        "temperature": temperature,
        "stream": stream,
    }
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
    state: dict[str, Any] = {"usage": {}, "timings": {}, "t_first_tok": None}
    with urllib.request.urlopen(req, timeout=timeout_s) as resp:
        ctype = (resp.headers.get("Content-Type") or "").lower()
        if "text/event-stream" in ctype:
            for raw_line in resp:
                line = raw_line.decode("utf-8", errors="replace").strip()
                if not line or line.startswith(":"):
                    continue
                if not line.startswith("data:"):
                    continue
                data_str = line[5:].strip()
                if data_str == "[DONE]":
                    break
                try:
                    chunk = json.loads(data_str)
                except json.JSONDecodeError:
                    continue
                _parse_sse_chunk(chunk, content_chunks, reasoning_chunks, state, time.monotonic(), on_delta=on_delta)
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
                    _parse_sse_chunk(chunk, content_chunks, reasoning_chunks, state, time.monotonic(), on_delta=on_delta)
            else:
                data = json.loads(raw_text)
                choice = (data.get("choices") or [{}])[0]
                msg = choice.get("message") or {}
                if msg.get("reasoning_content"):
                    reasoning_chunks.append(msg["reasoning_content"])
                if msg.get("content"):
                    content_chunks.append(msg["content"])
                state["usage"] = data.get("usage") or {}
                state["timings"] = data.get("timings") or {}
    t_end = time.monotonic()
    elapsed_s = max(0.001, t_end - t0)
    reasoning_content = "".join(reasoning_chunks)
    content = "".join(content_chunks) or reasoning_content or ""
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
        "prompt_tokens": prompt_tokens,
        "cached_tokens": cached_tokens,
        "completion_tokens": completion_tokens,
        "gen_s": gen_s,
        "tok_s": tok_s,
        "ttft_ms": ttft_ms,
        "wall_s": elapsed_s,
    }


def _git(cwd: Path, *args: str, check: bool = True) -> subprocess.CompletedProcess:
    return subprocess.run(["git", *args], cwd=str(cwd), capture_output=True, text=True, check=check)


def run_swarm(
    *,
    repo: Path,
    tasks: list[dict[str, str]],
    base_url: str,
    model: str,
    api_key: str = "outrcore-swarm",
    pricing: PricingSchedule | None = None,
    spend_cap_usd: float | None = 1.0,
    max_tokens: int = 4096,
    keep_worktrees: bool = False,
    on_progress: Callable[[dict[str, Any]], None] | None = None,
    max_repair_turns: int = 2,
    auto_review: bool = False,
    continuity_store: Any = None,
    runs_dir: Path | None = None,
    use_tool_loop: bool = False,
    max_tool_turns: int = 8,
    readonly_paths: list[str] | None = None,
) -> dict[str, Any]:
    """Execute `tasks` in parallel isolated git worktrees and return metered diffs + test verdicts."""
    from agent_tools import build_agent_context_brief, detect_file_collisions
    from orchestrator import review_agent_diff

    repo = repo.resolve()
    meter = RunMeter(pricing=pricing, spend_cap_usd=spend_cap_usd)
    context_brief = (
        build_agent_context_brief(repo, continuity=continuity_store, runs_dir=runs_dir)
        if continuity_store is not None
        else ""
    )
    shared_sys = build_shared_repo_prefix(repo, context_brief=context_brief)

    def _emit_progress(payload: dict[str, Any]) -> None:
        if on_progress is None:
            return
        try:
            on_progress(payload)
        except Exception:
            pass

    if len(tasks) > 1:
        # Best-effort 1-shot KV prefix pre-warm before parallel fan-out so workers hit cache instead of cold prefill.
        _emit_progress({"kind": "prewarm_start"})
        t_pw0 = time.monotonic()
        pw_ok = False
        try:
            openai_chat(
                base_url=base_url,
                api_key=api_key,
                model=model,
                messages=[
                    {"role": "system", "content": shared_sys},
                    {"role": "user", "content": "Ready."},
                ],
                max_tokens=1,
                temperature=0.0,
                timeout_s=45,
            )
            pw_ok = True
        except Exception:
            pass
        _emit_progress({"kind": "prewarm_done", "ok": pw_ok, "wall_s": round(time.monotonic() - t_pw0, 2)})
    run_id = time.strftime("swarm-%Y%m%d-%H%M%S")
    wt_base = repo.parent / f".open_swarm_{run_id}"
    wt_base.mkdir(parents=True, exist_ok=True)
    t_start = time.monotonic()

    def worker(idx: int, item: dict[str, Any]) -> dict[str, Any]:
        agent_id = str(item.get("agent_id") or f"coder-{chr(ord('a') + idx)}")
        ticket_id = str(item.get("id") or f"T-{idx+1:03d}")
        title = str(item.get("title") or ticket_id)
        item_model = str(item.get("model") or model)
        item_base_url = str(item.get("base_url") or base_url)
        item_lane = str(item.get("lane") or "coder")
        item_max_tokens = int(item.get("max_tokens") or max_tokens)
        item_temp = float(item.get("temperature") if item.get("temperature") is not None else 0.0)
        branch = f"swarm/{run_id}/{agent_id}"
        wt = wt_base / agent_id
        t_ag0 = time.monotonic()
        _git(repo, "worktree", "add", "-B", branch, str(wt), "HEAD")
        res_info: dict[str, Any] = {
            "agent_id": agent_id,
            "ticket": ticket_id,
            "title": title,
            "model": item_model,
            "lane": item_lane,
            "branch": branch,
            "worktree": str(wt),
            "stage": "worktree",
            "written_files": [],
            "base_file_hashes": {},
            "tests_ok": False,
            "test_output": "",
            "diff": "",
            "repair_attempts": 0,
            "review": None,
            "reasoning_preview": "",
            "output_preview": "",
            "completion_tokens": 0,
            "prompt_tokens": 0,
            "cached_tokens": 0,
            "tok_s": None,
            "ttft_ms": None,
            "error": None,
        }
        prog_state: dict[str, Any] = {
            "stage": "worktree",
            "reasoning_chars": 0,
            "content_chars": 0,
            "snippet": "",
            "last_emit": 0.0,
        }

        def _emit_agent(stage: str, *, force: bool = False) -> None:
            now = time.monotonic()
            changed = stage != prog_state["stage"]
            prog_state["stage"] = stage
            if not force and not changed and (now - prog_state["last_emit"]) < 0.4:
                return
            prog_state["last_emit"] = now
            _emit_progress(
                {
                    "kind": "agent_stage",
                    "agent_id": agent_id,
                    "ticket": ticket_id,
                    "title": title,
                    "stage": stage,
                    "reasoning_chars": prog_state["reasoning_chars"],
                    "content_chars": prog_state["content_chars"],
                    "snippet": prog_state["snippet"],
                }
            )

        def _on_delta(kind: str, piece: str) -> None:
            if kind == "reasoning":
                prog_state["reasoning_chars"] += len(piece)
                next_stage = "thinking" if prog_state["stage"] in ("worktree", "thinking") else prog_state["stage"]
            else:
                prog_state["content_chars"] += len(piece)
                next_stage = "coding"
            prog_state["snippet"] = (prog_state["snippet"] + piece)[-140:]
            _emit_agent(next_stage)

        _emit_agent("worktree", force=True)
        try:
            meter.check_cap()
            item_readonly = list(item.get("readonly_paths") or readonly_paths or [])
            item_use_tools = bool(item.get("use_tool_loop") if item.get("use_tool_loop") is not None else use_tool_loop)
            user_prompt = (
                f"## Ticket {res_info['ticket']}: {title}\n"
                f"{item['spec']}\n\n"
                f"Acceptance command that will run in this worktree: `{item['acceptance_cmd']}`\n"
                "Keep internal reasoning brief (<250 words) so plenty of tokens remain for complete ```file:...``` blocks.\n"
                "Respond ONLY with ```file:...``` blocks for the files you create or update."
            )
            convo: list[dict[str, str]] = [
                {"role": "system", "content": shared_sys},
                {"role": "user", "content": user_prompt},
            ]
            written: list[str] = []

            def _run_turn() -> dict[str, Any]:
                resp_turn = openai_chat(
                    base_url=item_base_url,
                    api_key=api_key,
                    model=item_model,
                    messages=convo,
                    max_tokens=item_max_tokens,
                    temperature=item_temp,
                    stream=True,
                    on_delta=_on_delta,
                )
                meter.record_usage(
                    agent_id,
                    prompt_tokens=resp_turn["prompt_tokens"],
                    cached_tokens=resp_turn["cached_tokens"],
                    completion_tokens=resp_turn["completion_tokens"],
                    gen_s=resp_turn["gen_s"],
                    ttft_ms=resp_turn["ttft_ms"],
                )
                res_info["reasoning_preview"] = (resp_turn.get("reasoning_content") or "")[-1200:]
                res_info["output_preview"] = (resp_turn.get("content") or "")[-1200:]
                res_info["completion_tokens"] += int(resp_turn["completion_tokens"])
                res_info["prompt_tokens"] += int(resp_turn["prompt_tokens"])
                res_info["cached_tokens"] += int(resp_turn["cached_tokens"])
                res_info["tok_s"] = resp_turn["tok_s"]
                res_info["ttft_ms"] = resp_turn["ttft_ms"]
                return resp_turn

            def _run_accept() -> None:
                _emit_agent("testing", force=True)
                for pc_dir in wt.rglob("__pycache__"):
                    if pc_dir.is_dir():
                        shutil.rmtree(pc_dir, ignore_errors=True)
                accept_env = dict(os.environ)
                accept_env["PYTHONDONTWRITEBYTECODE"] = "1"
                tproc = subprocess.run(
                    item["acceptance_cmd"],
                    cwd=str(wt),
                    shell=True,
                    env=accept_env,
                    capture_output=True,
                    text=True,
                    timeout=60,
                )
                res_info["tests_ok"] = tproc.returncode == 0
                res_info["test_output"] = (tproc.stdout + "\n" + tproc.stderr).strip()[-2000:]

            resp: dict[str, Any] = {"content": "", "reasoning_content": ""}
            if item_use_tools:
                from agent_loop import run_agent_loop

                tool_prompt = (
                    f"## Ticket {res_info['ticket']}: {title}\n"
                    f"{item['spec']}\n\n"
                    f"Acceptance command: `{item['acceptance_cmd']}`\n"
                    "Use `read_file` / `search_code` to inspect files first, make surgical edits via `edit_file` or `write_file`, "
                    "verify using `run_command`, optionally log insights with `write_note`, and call `finish` when verified."
                )
                loop_res = run_agent_loop(
                    worktree=wt,
                    system_prompt=shared_sys,
                    user_prompt=tool_prompt,
                    base_url=item_base_url,
                    api_key=api_key,
                    model=item_model,
                    agent_id=agent_id,
                    readonly_paths=item_readonly,
                    max_turns=int(item.get("max_tool_turns") or max_tool_turns),
                    max_tokens=item_max_tokens,
                    temperature=item_temp,
                    meter=meter,
                    on_progress=_emit_progress,
                )
                res_info["turns"] = int(loop_res.get("turns") or 0)
                res_info["tool_calls"] = int(loop_res.get("tool_calls") or 0)
                res_info["notes_path"] = loop_res.get("notes_path")
                res_info["completion_tokens"] += int(loop_res.get("completion_tokens") or 0)
                res_info["prompt_tokens"] += int(loop_res.get("prompt_tokens") or 0)
                res_info["cached_tokens"] += int(loop_res.get("cached_tokens") or 0)
                res_info["output_preview"] = str(loop_res.get("summary") or loop_res.get("error") or "")[-1200:]
                for rel_w in loop_res.get("written_files") or []:
                    if rel_w not in written:
                        written.append(rel_w)
                if item.get("acceptance_cmd"):
                    _run_accept()
                resp = {"content": res_info["output_preview"], "reasoning_content": ""}
            else:
                resp = _run_turn()
                for rel_w in apply_file_blocks(wt, resp["content"]):
                    if rel_w not in written:
                        written.append(rel_w)
                if written and item.get("acceptance_cmd"):
                    _run_accept()

            repairs_allowed = max(0, int(max_repair_turns))
            while repairs_allowed > 0 and res_info["repair_attempts"] < repairs_allowed and (
                not written or (item.get("acceptance_cmd") and not res_info["tests_ok"])
            ):
                meter.check_cap()
                res_info["repair_attempts"] += 1
                _emit_agent("repairing", force=True)
                fail_reason = (
                    res_info["test_output"][-1200:]
                    if written and res_info.get("test_output")
                    else "No ```file:...``` blocks were emitted. Keep reasoning brief and output complete ```file:...``` blocks now."
                )
                convo.append({"role": "assistant", "content": resp.get("content") or ""})
                convo.append(
                    {
                        "role": "user",
                        "content": (
                            f"Repair attempt {res_info['repair_attempts']}/{repairs_allowed}. Verification failed:\n"
                            f"```\n{fail_reason}\n```\n"
                            "Respond ONLY with complete ```file:...``` blocks fixing the failure."
                        ),
                    }
                )
                resp = _run_turn()
                for rel_w in apply_file_blocks(wt, resp["content"]):
                    if rel_w not in written:
                        written.append(rel_w)
                if written and item.get("acceptance_cmd"):
                    _run_accept()

            res_info["written_files"] = written
            if written:
                res_info["base_file_hashes"] = compute_file_hashes(repo, written)
                res_info["error"] = None
            else:
                res_info["error"] = (
                    f"No ```file:...``` blocks emitted "
                    f"(completion_tokens={res_info['completion_tokens']}/{item_max_tokens}; "
                    f"reasoning_chars={len(resp.get('reasoning_content') or '')}, "
                    f"content_chars={len(resp.get('content') or '')})"
                )

            _git(wt, "add", "-A", check=False)
            diff_proc = _git(wt, "diff", "--cached", "HEAD", check=False)
            res_info["diff"] = diff_proc.stdout

            if auto_review and res_info["tests_ok"] and res_info["diff"].strip():
                _emit_agent("reviewing", force=True)
                try:
                    res_info["review"] = review_agent_diff(
                        ticket_title=title,
                        ticket_spec=str(item.get("spec") or ""),
                        diff=res_info["diff"],
                        test_output=res_info["test_output"],
                        author_model=item_model,
                        reviewer_base_url=item_base_url,
                        reviewer_model=item_model,
                        api_key=api_key,
                        allow_same_vendor_fresh=True,
                    )
                except Exception:
                    pass

            if continuity_store is not None and res_info["written_files"]:
                try:
                    verdict_lbl = "PASS" if res_info["tests_ok"] else "FAIL"
                    continuity_store.send_message(
                        str(repo),
                        agent_id,
                        "all",
                        f"{verdict_lbl} {title}: {', '.join(res_info['written_files'])}",
                    )
                except Exception:
                    pass

            final_stage = "passed" if res_info["tests_ok"] else "failed"
            res_info["stage"] = final_stage
            _emit_agent(final_stage, force=True)
        except SpendCapExceeded as exc:
            res_info["error"] = str(exc)
            res_info["stage"] = "failed"
            _emit_agent("failed", force=True)
        except Exception as exc:
            res_info["error"] = f"{type(exc).__name__}: {exc}"
            res_info["stage"] = "failed"
            _emit_agent("failed", force=True)
        finally:
            meter.set_agent_wall(agent_id, time.monotonic() - t_ag0)
        return res_info

    agent_results: list[dict[str, Any]] = []
    try:
        with ThreadPoolExecutor(max_workers=max(1, len(tasks))) as pool:
            futs = [pool.submit(worker, i, t) for i, t in enumerate(tasks)]
            for f in as_completed(futs):
                agent_results.append(f.result())
        agent_results.sort(key=lambda r: r["agent_id"])
    finally:
        if not keep_worktrees:
            for r in agent_results:
                wt = Path(r["worktree"])
                if wt.exists():
                    _git(repo, "worktree", "remove", "--force", str(wt), check=False)
            shutil.rmtree(wt_base, ignore_errors=True)
            _git(repo, "worktree", "prune", check=False)

    total_wall = time.monotonic() - t_start
    return {
        "run_id": run_id,
        "repo": str(repo),
        "backend_url": base_url,
        "model": model,
        "agents_count": len(tasks),
        "passed_count": sum(1 for r in agent_results if r["tests_ok"]),
        "collisions": detect_file_collisions(agent_results),
        "metering": meter.summary(total_wall),
        "results": agent_results,
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", required=True)
    ap.add_argument("--backend-url", required=True)
    ap.add_argument("--model", default="thinkingmachines/Inkling-2-Mini")
    ap.add_argument("--api-key", default="outrcore-swarm")
    ap.add_argument("--spend-cap-usd", type=float, default=1.0)
    ap.add_argument("--tasks-json", required=True, help="JSON list of {title, spec, acceptance_cmd}")
    ap.add_argument("--out", default="")
    args = ap.parse_args()
    tasks = json.loads(Path(args.tasks_json).read_text(encoding="utf-8"))
    rec = run_swarm(
        repo=Path(args.repo),
        tasks=tasks,
        base_url=args.backend_url,
        model=args.model,
        api_key=args.api_key,
        spend_cap_usd=args.spend_cap_usd,
    )
    txt = json.dumps(rec, indent=2)
    if args.out:
        Path(args.out).write_text(txt + "\n", encoding="utf-8")
    print(txt)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
