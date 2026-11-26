#!/usr/bin/env python3
"""Open Swarm Orchestrator: Lane routing, cross-vendor critic gate, and blind tournament engine.

Pure Python 3 stdlib (+ sibling `swarm_run`) module providing:
- `LANES`: routing presets (`mechanical`, `coder`, `deep-reason`, `reviewer`).
- `resolve_vendor`: model name to vendor family mapping.
- `check_review_eligibility`: author-never-reviews-own-work policy checks.
- `review_agent_diff`: fresh-context critic gate producing structured JSON verdicts.
- `run_tournament`: multi-stance / multi-model parallel fan-out with auto-disqualification
  and anonymous blind judging.
"""
from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path
import re
import sys
from typing import Any, Callable

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

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
    chat_fn: Callable[..., dict[str, Any]] | None = None,
) -> dict[str, Any]:
    """Run an independent fresh-context critic gate over a diff and return a structured verdict."""
    eligibility = check_review_eligibility(
        author_model,
        reviewer_model,
        allow_same_vendor_fresh=allow_same_vendor_fresh,
    )
    if not eligibility["eligible"]:
        return {
            "verdict": "block",
            "summary": eligibility["warning"],
            "findings": [],
            "eligibility": eligibility,
            "error": eligibility["warning"],
        }

    if not (diff or "").strip():
        return {
            "verdict": "block",
            "summary": "Empty diff — nothing to review",
            "findings": [],
            "eligibility": eligibility,
            "error": "empty diff",
        }

    system_prompt = (
        "You are an independent senior code reviewer performing a strict critic gate.\n"
        "Evaluate correctness, edge cases, regressions, and adherence to the ticket specification.\n"
        "Respond ONLY with valid JSON matching this exact schema:\n"
        '{"verdict": "approve" | "request_changes" | "block", '
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

    fn = chat_fn or swarm_run.openai_chat
    resp = fn(
        base_url=reviewer_base_url,
        api_key=api_key,
        model=reviewer_model,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        max_tokens=LANES["reviewer"]["default_max_tokens"],
        temperature=LANES["reviewer"]["temperature"],
        stream=False,
    )
    raw_content = str(resp.get("content") or "")
    parsed = _extract_json_dict(raw_content)

    if parsed is None:
        snippet = raw_content.strip()[:300]
        return {
            "verdict": "request_changes",
            "summary": f"Malformed JSON from critic reviewer; raw output snippet: {snippet}",
            "findings": [],
            "eligibility": eligibility,
            "reviewer_model": reviewer_model,
            "tok_s": resp.get("tok_s"),
            "completion_tokens": int(resp.get("completion_tokens") or 0),
        }

    verdict = _normalize_verdict(parsed.get("verdict"))
    summary = str(parsed.get("summary") or "").strip() or f"Reviewer verdict: {verdict}"
    findings = _normalize_findings(parsed.get("findings"))
    return {
        "verdict": verdict,
        "summary": summary,
        "findings": findings,
        "eligibility": eligibility,
        "reviewer_model": reviewer_model,
        "tok_s": resp.get("tok_s"),
        "completion_tokens": int(resp.get("completion_tokens") or 0),
    }


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
