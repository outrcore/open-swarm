#!/usr/bin/env python3
"""Unit tests for open_swarm/orchestrator.py (offline, deterministic)."""
from __future__ import annotations

import json
from pathlib import Path
import sys
import tempfile
import threading
import unittest

SWARM_DIR = Path(__file__).resolve().parents[1]
if str(SWARM_DIR) not in sys.path:
    sys.path.insert(0, str(SWARM_DIR))

from orchestrator import (  # noqa: E402
    LANES,
    check_review_eligibility,
    plan_task_dag,
    resolve_vendor,
    review_agent_diff,
    review_and_repair_agent,
    run_dag_waves,
    run_tournament,
)


class LanesTests(unittest.TestCase):
    def test_lanes_structure_and_defaults(self) -> None:
        self.assertEqual(set(LANES.keys()), {"mechanical", "coder", "deep-reason", "reviewer"})
        expected = {
            "mechanical": (1536, 0.0),
            "coder": (4096, 0.0),
            "deep-reason": (6144, 0.2),
            "reviewer": (2048, 0.0),
        }
        for key, (max_tok, temp) in expected.items():
            lane = LANES[key]
            self.assertIn("label", lane)
            self.assertIn("description", lane)
            self.assertEqual(lane["default_max_tokens"], max_tok)
            self.assertEqual(lane["temperature"], temp)


class ResolveVendorTests(unittest.TestCase):
    def test_all_vendors_case_insensitive(self) -> None:
        cases = [
            ("anthropic/claude-3.7-sonnet", "anthropic"),
            ("Claude-Opus-4", "anthropic"),
            ("Haiku-3.5", "anthropic"),
            ("fable-preview", "anthropic"),
            ("Anthropic-Model", "anthropic"),
            ("gpt-4o", "openai"),
            ("GPT-5-Codex", "openai"),
            ("openai/o1-preview", "openai"),
            ("o3-mini", "openai"),
            ("o4-mini-high", "openai"),
            ("xai/grok-3", "xai"),
            ("GROK-2", "xai"),
            ("google/gemini-2.5-pro", "google"),
            ("GEMINI-flash", "google"),
            ("thinkingmachines/Inkling-2-Mini", "outrcore"),
            ("ornith-ai/Ornith-1.5-35B-A3B", "outrcore"),
            ("Qwen/Qwen2.5-Coder-32B", "outrcore"),
            ("outrcore-local", "outrcore"),
            ("meta-llama/Llama-3.3-70B", "other"),
            ("", "other"),
        ]
        for model_name, expected_vendor in cases:
            with self.subTest(model=model_name):
                self.assertEqual(resolve_vendor(model_name), expected_vendor)


class ReviewEligibilityTests(unittest.TestCase):
    def test_cross_vendor_eligible_no_warning(self) -> None:
        res = check_review_eligibility("claude-sonnet-4", "gpt-4o", allow_same_vendor_fresh=False)
        self.assertTrue(res["eligible"])
        self.assertEqual(res["author_vendor"], "anthropic")
        self.assertEqual(res["reviewer_vendor"], "openai")
        self.assertFalse(res["same_model"])
        self.assertFalse(res["same_vendor"])
        self.assertTrue(res["cross_vendor"])
        self.assertIsNone(res["warning"])

    def test_same_vendor_blocked_when_strict(self) -> None:
        res = check_review_eligibility("claude-sonnet-4", "claude-opus-4", allow_same_vendor_fresh=False)
        self.assertFalse(res["eligible"])
        self.assertTrue(res["same_vendor"])
        self.assertFalse(res["same_model"])
        self.assertFalse(res["cross_vendor"])
        self.assertIn("author-never-reviews-own-work", res["warning"])

    def test_same_vendor_fresh_context_warning_when_allowed(self) -> None:
        res = check_review_eligibility(
            "thinkingmachines/Inkling-2-Mini",
            "thinkingmachines/Inkling-2-Mini",
            allow_same_vendor_fresh=True,
        )
        self.assertTrue(res["eligible"])
        self.assertTrue(res["same_vendor"])
        self.assertTrue(res["same_model"])
        self.assertFalse(res["cross_vendor"])
        self.assertIsNotNone(res["warning"])
        self.assertIn("prefer cross-vendor", res["warning"].lower())


class ReviewAgentDiffTests(unittest.TestCase):
    def test_plain_json_response(self) -> None:
        captured: list[dict] = []

        def fake_chat(**kwargs):
            captured.append(kwargs)
            return {
                "content": json.dumps(
                    {
                        "verdict": "approve",
                        "summary": "Clean fix with tests.",
                        "findings": [
                            {
                                "severity": "low",
                                "file": "app.py",
                                "issue": "Minor docstring typo",
                                "fix": "Fix typo",
                            }
                        ],
                    }
                ),
                "tok_s": 84.5,
                "completion_tokens": 42,
            }

        res = review_agent_diff(
            ticket_title="Fix parser",
            ticket_spec="Handle empty input cleanly",
            diff="--- a/app.py\n+++ b/app.py\n@@ -1 +1 @@\n-x\n+y\n",
            test_output="Ran 2 tests in 0.01s\nOK",
            author_model="thinkingmachines/Inkling-2-Mini",
            reviewer_base_url="http://127.0.0.1:9999/v1",
            reviewer_model="claude-sonnet-4",
            chat_fn=fake_chat,
        )
        self.assertEqual(len(captured), 1)
        prompt_text = "\n".join(m["content"] for m in captured[0]["messages"])
        self.assertNotIn("Inkling-2-Mini", prompt_text)
        self.assertNotIn("thinkingmachines", prompt_text)
        self.assertIn("Fix parser", prompt_text)
        self.assertIn("Handle empty input cleanly", prompt_text)

        self.assertEqual(res["verdict"], "approve")
        self.assertEqual(res["summary"], "Clean fix with tests.")
        self.assertEqual(len(res["findings"]), 1)
        self.assertTrue(res["eligibility"]["cross_vendor"])
        self.assertEqual(res["reviewer_model"], "claude-sonnet-4")
        self.assertEqual(res["tok_s"], 84.5)
        self.assertEqual(res["completion_tokens"], 42)

    def test_fenced_markdown_json_extraction(self) -> None:
        def fake_chat(**_kwargs):
            return {
                "content": (
                    "Here is my review:\n"
                    "```json\n"
                    "{\n"
                    '  "verdict": "Request-Changes",\n'
                    '  "summary": "Missing null guard.",\n'
                    '  "findings": [{"severity": "high", "file": "core.py", "issue": "NoneType deref", "fix": "Check None"}]\n'
                    "}\n"
                    "```\n"
                ),
                "tok_s": 50.0,
                "completion_tokens": 30,
            }

        res = review_agent_diff(
            ticket_title="Guard nulls",
            ticket_spec="Must not crash on None",
            diff="--- a/core.py\n+++ b/core.py\n+pass\n",
            test_output="OK",
            author_model="gpt-4o",
            reviewer_base_url="http://127.0.0.1:9999/v1",
            reviewer_model="gemini-2.5-pro",
            chat_fn=fake_chat,
        )
        self.assertEqual(res["verdict"], "request_changes")
        self.assertEqual(res["summary"], "Missing null guard.")
        self.assertEqual(res["findings"][0]["severity"], "high")

    def test_blocked_same_vendor_skips_chat(self) -> None:
        called = False

        def fake_chat(**_kwargs):
            nonlocal called
            called = True
            return {"content": "{}"}

        res = review_agent_diff(
            ticket_title="T1",
            ticket_spec="Spec",
            diff="+line",
            test_output="OK",
            author_model="gpt-4o",
            reviewer_base_url="http://127.0.0.1:9999/v1",
            reviewer_model="o3-mini",
            allow_same_vendor_fresh=False,
            chat_fn=fake_chat,
        )
        self.assertFalse(called)
        self.assertEqual(res["verdict"], "block")
        self.assertIn("error", res)
        self.assertFalse(res["eligibility"]["eligible"])

    def test_empty_diff_blocked_without_chat(self) -> None:
        called = False

        def fake_chat(**_kwargs):
            nonlocal called
            called = True
            return {"content": "{}"}

        res = review_agent_diff(
            ticket_title="T1",
            ticket_spec="Spec",
            diff="   \n",
            test_output="OK",
            author_model="gpt-4o",
            reviewer_base_url="http://127.0.0.1:9999/v1",
            reviewer_model="claude-sonnet-4",
            chat_fn=fake_chat,
        )
        self.assertFalse(called)
        self.assertEqual(res["verdict"], "block")
        self.assertEqual(res["error"], "empty diff")

    def test_malformed_json_fallback(self) -> None:
        def fake_chat(**_kwargs):
            return {
                "content": "Sorry, I could not format JSON: broken output {not valid",
                "tok_s": 12.0,
                "completion_tokens": 11,
            }

        res = review_agent_diff(
            ticket_title="T1",
            ticket_spec="Spec",
            diff="+x = 1",
            test_output="OK",
            author_model="gpt-4o",
            reviewer_base_url="http://127.0.0.1:9999/v1",
            reviewer_model="grok-3",
            chat_fn=fake_chat,
        )
        self.assertEqual(res["verdict"], "request_changes")
        self.assertIn("broken output {not valid", res["summary"])
        self.assertEqual(res["findings"], [])


class RunTournamentTests(unittest.TestCase):
    def test_disqualification_blind_prompt_and_winner_mapping(self) -> None:
        runner_calls: list[dict] = []
        judge_calls: list[dict] = []

        def fake_runner(**kwargs):
            runner_calls.append(kwargs)
            tasks = kwargs["tasks"]
            self.assertEqual([t["agent_id"] for t in tasks], ["cand-a", "cand-b", "cand-c"])
            self.assertEqual([t["stance"] for t in tasks], ["minimal-patch", "defensive-edge-cases", "clean-modular"])
            return {
                "run_id": "swarm-test-001",
                "repo": str(kwargs["repo"]),
                "backend_url": kwargs["base_url"],
                "model": kwargs["model"],
                "agents_count": 3,
                "passed_count": 2,
                "metering": {"total_cost_usd": 0.02},
                "results": [
                    {
                        "agent_id": "cand-a",
                        "tests_ok": False,
                        "diff": "--- a/x.py\n+++ b/x.py\n+bad()\n",
                        "test_output": "FAIL: assertion failed",
                        "error": "AssertionError",
                    },
                    {
                        "agent_id": "cand-b",
                        "tests_ok": True,
                        "diff": "--- a/x.py\n+++ b/x.py\n+# cand-b defensive-edge-cases gpt-4o\n+def safe(): return 1\n",
                        "test_output": "OK",
                    },
                    {
                        "agent_id": "cand-c",
                        "tests_ok": True,
                        "diff": "--- a/x.py\n+++ b/x.py\n+def clean(): return 2\n",
                        "test_output": "OK (clean-modular)",
                    },
                ],
            }

        def fake_judge(**kwargs):
            judge_calls.append(kwargs)
            return {
                "content": (
                    "```json\n"
                    '{"winner": "Candidate B", "rationale": "Cleaner implementation.", '
                    '"rankings": [{"label": "Candidate B", "score": 9, "strengths": "Simple", "objection": "None"}, '
                    '{"label": "Candidate A", "score": 7, "strengths": "Safe", "objection": "Verbose"}]}\n'
                    "```"
                ),
                "completion_tokens": 55,
            }

        rec = run_tournament(
            repo=Path("/tmp/fake-repo"),
            ticket={"id": "T-42", "title": "Add safe helper", "spec": "Implement helper", "acceptance_cmd": "pytest"},
            base_url="http://127.0.0.1:8000/v1",
            model="gpt-4o",
            judge_base_url="http://127.0.0.1:8001/v1",
            judge_model="claude-sonnet-4",
            runner_fn=fake_runner,
            chat_fn=fake_judge,
        )

        self.assertEqual(len(runner_calls), 1)
        self.assertEqual(len(judge_calls), 1)
        self.assertEqual(judge_calls[0]["base_url"], "http://127.0.0.1:8001/v1")
        self.assertEqual(judge_calls[0]["model"], "claude-sonnet-4")

        judge_prompt = "\n".join(m["content"] for m in judge_calls[0]["messages"])
        # Verify authorship / model / stance strings are stripped from blind judge prompt
        for forbidden in ("cand-a", "cand-b", "cand-c", "gpt-4o", "minimal-patch", "defensive-edge-cases", "clean-modular"):
            self.assertNotIn(forbidden, judge_prompt)
        self.assertIn("Candidate A", judge_prompt)
        self.assertIn("Candidate B", judge_prompt)
        self.assertNotIn("Candidate C", judge_prompt)

        self.assertEqual(rec["mode"], "tournament")
        t = rec["tournament"]
        # Candidate A -> cand-b, Candidate B -> cand-c (since cand-a failed tests)
        self.assertEqual(t["winner_label"], "Candidate B")
        self.assertEqual(t["winner_agent_id"], "cand-c")
        self.assertEqual(t["judge_model"], "claude-sonnet-4")
        self.assertEqual(len(t["disqualified"]), 1)
        self.assertEqual(t["disqualified"][0]["agent_id"], "cand-a")
        self.assertEqual(t["rationale"], "Cleaner implementation.")
        self.assertEqual(len(t["rankings"]), 2)

    def test_empty_diff_disqualified_and_zero_survivors_skips_judge(self) -> None:
        judge_called = False

        def fake_runner(**kwargs):
            return {
                "run_id": "swarm-test-002",
                "repo": str(kwargs["repo"]),
                "backend_url": kwargs["base_url"],
                "model": kwargs["model"],
                "agents_count": 2,
                "passed_count": 1,
                "metering": {},
                "results": [
                    {"agent_id": "cand-a", "tests_ok": False, "diff": "+x", "test_output": "ERR"},
                    {"agent_id": "cand-b", "tests_ok": True, "diff": "   ", "test_output": "OK"},
                ],
            }

        def fake_judge(**_kwargs):
            nonlocal judge_called
            judge_called = True
            return {"content": "{}"}

        rec = run_tournament(
            repo=Path("/tmp/fake-repo"),
            ticket={"id": "T-99", "title": "Nothing", "spec": "Do work", "acceptance_cmd": "pytest"},
            base_url="http://127.0.0.1:8000/v1",
            model="thinkingmachines/Inkling-2-Mini",
            candidates=[{"stance": "minimal-patch"}, {"stance": "clean-modular"}],
            runner_fn=fake_runner,
            chat_fn=fake_judge,
        )

        self.assertFalse(judge_called)
        self.assertEqual(rec["mode"], "tournament")
        t = rec["tournament"]
        self.assertIsNone(t["winner_agent_id"])
        self.assertIsNone(t["winner_label"])
        self.assertEqual(len(t["disqualified"]), 2)
        self.assertEqual(t["rankings"], [])

    def test_malformed_judge_output_falls_back_to_first_survivor(self) -> None:
        def fake_runner(**kwargs):
            return {
                "run_id": "swarm-test-003",
                "repo": str(kwargs["repo"]),
                "backend_url": kwargs["base_url"],
                "model": kwargs["model"],
                "agents_count": 2,
                "passed_count": 2,
                "metering": {},
                "results": [
                    {"agent_id": "cand-a", "tests_ok": True, "diff": "+a", "test_output": "OK"},
                    {"agent_id": "cand-b", "tests_ok": True, "diff": "+b", "test_output": "OK"},
                ],
            }

        def fake_judge(**_kwargs):
            return {"content": "not valid json at all"}

        rec = run_tournament(
            repo=Path("/tmp/fake-repo"),
            ticket={"id": "T-100", "title": "Fallback", "spec": "Test", "acceptance_cmd": "pytest"},
            base_url="http://127.0.0.1:8000/v1",
            model="gpt-4o",
            runner_fn=fake_runner,
            chat_fn=fake_judge,
        )
        t = rec["tournament"]
        self.assertEqual(t["winner_label"], "Candidate A")
        self.assertEqual(t["winner_agent_id"], "cand-a")
        self.assertIn("not valid json", t["rationale"])

    def test_heterogeneous_candidate_models_override(self) -> None:
        seen_models: list[str] = []

        def fake_runner(**kwargs):
            seen_models.append(kwargs["model"])
            res = []
            for task in kwargs["tasks"]:
                res.append(
                    {
                        "agent_id": task["agent_id"],
                        "tests_ok": True,
                        "diff": f"+from {task['agent_id']}",
                        "test_output": "OK",
                    }
                )
            return {
                "run_id": "swarm-multi",
                "repo": str(kwargs["repo"]),
                "backend_url": kwargs["base_url"],
                "model": kwargs["model"],
                "agents_count": len(res),
                "passed_count": len(res),
                "metering": {"completion_tokens": 10},
                "results": res,
            }

        def fake_judge(**_kwargs):
            return {"content": json.dumps({"winner": "Candidate A", "rationale": "Good", "rankings": []})}

        rec = run_tournament(
            repo=Path("/tmp/fake-repo"),
            ticket={"id": "T-101", "title": "Multi", "spec": "Spec", "acceptance_cmd": "pytest"},
            base_url="http://127.0.0.1:8000/v1",
            model="gpt-4o",
            candidates=[
                {"agent_id": "cand-a", "stance": "minimal-patch", "model": "claude-sonnet-4"},
                {"agent_id": "cand-b", "stance": "clean-modular", "model": "gemini-2.5-pro"},
            ],
            runner_fn=fake_runner,
            chat_fn=fake_judge,
        )
        self.assertEqual(set(seen_models), {"claude-sonnet-4", "gemini-2.5-pro"})
        self.assertEqual(rec["agents_count"], 2)
        self.assertEqual(rec["passed_count"], 2)
        self.assertEqual(rec["tournament"]["winner_agent_id"], "cand-a")


class DagSchedulerTests(unittest.TestCase):
    def test_plan_task_dag_default_ids_and_phases(self) -> None:
        tasks = [
            {"title": "Define API schema", "phase": "contract"},
            {"title": "Implement service", "phase": "impl"},
            {"title": "Implement client", "phase": "impl"},
            {"title": "Integration check", "phase": "verify"},
        ]
        waves = plan_task_dag(tasks)
        self.assertEqual(len(waves), 3)
        self.assertEqual([t["id"] for t in waves[0]], ["t0"])
        self.assertEqual([t["id"] for t in waves[1]], ["t1", "t2"])
        self.assertEqual([t["id"] for t in waves[2]], ["t3"])
        self.assertEqual(waves[1][0]["depends_on"], ["t0"])
        self.assertEqual(waves[2][0]["depends_on"], ["t0", "t1", "t2"])

    def test_plan_task_dag_explicit_depends_on_and_mixed_phases(self) -> None:
        tasks = [
            {"id": "c1", "phase": "contract"},
            {"id": "i1", "phase": "impl"},
            {"id": "i2", "phase": "impl", "depends_on": ["i1"]},
            {"id": "v1", "phase": "verify"},
        ]
        waves = plan_task_dag(tasks)
        self.assertEqual(
            [[t["id"] for t in w] for w in waves],
            [["c1"], ["i1"], ["i2"], ["v1"]],
        )

    def test_plan_task_dag_missing_dependency_raises(self) -> None:
        with self.assertRaises(ValueError):
            plan_task_dag([{"id": "a", "depends_on": ["nonexistent"]}])

    def test_plan_task_dag_cycle_raises(self) -> None:
        with self.assertRaises(ValueError):
            plan_task_dag(
                [
                    {"id": "a", "depends_on": ["b"]},
                    {"id": "b", "depends_on": ["a"]},
                ]
            )
        with self.assertRaises(ValueError):
            plan_task_dag([{"id": "self", "depends_on": ["self"]}])

    def test_run_dag_waves_forwards_contract_context(self) -> None:
        tasks = [
            {"id": "c0", "title": "Schema", "phase": "contract"},
            {"id": "i0", "title": "Server", "phase": "impl"},
            {"id": "v0", "title": "Verify", "phase": "verify"},
        ]
        seen_contexts: list[tuple[int, str]] = []

        def fake_exec(wave_idx: int, wave_tasks: list[dict], context: str) -> list[dict]:
            seen_contexts.append((wave_idx, context))
            if wave_idx == 0:
                return [
                    {
                        "id": "c0",
                        "phase": "contract",
                        "contract": "POST /v1/items -> 201 {id: str}",
                        "summary": "Defined HTTP contract",
                        "tests_ok": True,
                    }
                ]
            if wave_idx == 1:
                return [
                    {
                        "id": "i0",
                        "summary": "Implemented POST /v1/items handler",
                        "written_files": ["server.py"],
                        "tests_ok": True,
                    }
                ]
            return [{"id": "v0", "summary": "All E2E checks passed", "tests_ok": True}]

        res = run_dag_waves(tasks, fake_exec)
        self.assertFalse(res["cancelled"])
        self.assertEqual(len(res["waves"]), 3)
        self.assertEqual(len(res["results"]), 3)
        self.assertEqual(seen_contexts[0], (0, ""))
        self.assertIn("CONTRACT", seen_contexts[1][1])
        self.assertIn("POST /v1/items -> 201 {id: str}", seen_contexts[1][1])
        self.assertIn("POST /v1/items -> 201 {id: str}", seen_contexts[2][1])
        self.assertIn("Implemented POST /v1/items handler", seen_contexts[2][1])

    def test_run_dag_waves_cancellation_between_waves(self) -> None:
        cancel = threading.Event()
        tasks = [
            {"id": "w0", "phase": "contract"},
            {"id": "w1", "phase": "impl"},
        ]

        def fake_exec(wave_idx: int, wave_tasks: list[dict], _context: str) -> list[dict]:
            cancel.set()
            return [{"id": wave_tasks[0]["id"], "summary": "done"}]

        res = run_dag_waves(tasks, fake_exec, cancel_event=cancel)
        self.assertTrue(res["cancelled"])
        self.assertEqual(len(res["waves"]), 1)
        self.assertEqual(len(res["results"]), 1)
        self.assertEqual(res["results"][0]["id"], "w0")

        res_immediate = run_dag_waves(tasks, fake_exec, cancel_event=cancel)
        self.assertTrue(res_immediate["cancelled"])
        self.assertEqual(res_immediate["waves"], [])
        self.assertEqual(res_immediate["results"], [])


class ReviewAgentDiffPhase2CTests(unittest.TestCase):
    def test_test_quality_rubric_normalization_and_review_mode(self) -> None:
        captured: list[dict] = []

        def strict_openai_chat_sig(*, base_url, api_key, model, messages, max_tokens=4096, temperature=0.0, stream=False):
            captured.append({"messages": messages})
            return {
                "content": json.dumps(
                    {
                        "verdict": "request_changes",
                        "test_quality": "WEAK",
                        "summary": "Tautological assertion.",
                        "findings": [
                            {
                                "severity": "high",
                                "file": "tests/test_app.py",
                                "issue": "Only asserts True",
                                "fix": "Assert actual return value",
                            }
                        ],
                    }
                ),
                "completion_tokens": 20,
            }

        res = review_agent_diff(
            ticket_title="Add parser",
            ticket_spec="Parse JSON safely",
            diff="--- a/app.py\n+++ b/app.py\n+assert True\n",
            test_output="OK",
            author_model="gpt-4o",
            reviewer_base_url="http://127.0.0.1:9999/v1",
            reviewer_model="gpt-4o",
            allow_same_vendor_fresh=True,
            worktree=None,
            chat_fn=strict_openai_chat_sig,
        )
        sys_msg = captured[0]["messages"][0]["content"]
        self.assertIn("assert True", sys_msg)
        self.assertIn("test_quality", sys_msg)
        self.assertEqual(res["test_quality"], "weak")
        self.assertEqual(res["review_mode"], "same_vendor_fresh")
        self.assertEqual(res["inspected_files"], [])

        # Legacy mock omitting test_quality defaults to "adequate"
        res_legacy = review_agent_diff(
            ticket_title="Add parser",
            ticket_spec="Parse JSON safely",
            diff="+x = 1",
            test_output="OK",
            author_model="gpt-4o",
            reviewer_base_url="http://127.0.0.1:9999/v1",
            reviewer_model="claude-sonnet-4",
            chat_fn=lambda **_kw: {"content": json.dumps({"verdict": "approve", "summary": "OK", "findings": []})},
        )
        self.assertEqual(res_legacy["test_quality"], "adequate")
        self.assertEqual(res_legacy["review_mode"], "cross_vendor")

    def test_tool_enabled_critic_native_tool_calls_reads_untouched_files(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            wt = Path(tmp)
            (wt / "helpers.py").write_text("def normalize_token(s):\n    return s.strip().lower()\n", encoding="utf-8")
            (wt / "tests").mkdir()
            (wt / "tests" / "test_helpers.py").write_text("def test_norm():\n    assert True\n", encoding="utf-8")

            turns: list[dict] = []

            def fake_chat(**kwargs):
                turns.append(kwargs)
                idx = len(turns)
                if idx == 1:
                    self.assertIn("tools", kwargs)
                    return {
                        "content": "",
                        "tool_calls": [
                            {"id": "c1", "function": {"name": "list_dir", "arguments": {"path": "."}}},
                            {"id": "c2", "function": {"name": "search_code", "arguments": {"query": "normalize_token"}}},
                            {"id": "c3", "function": {"name": "read_file", "arguments": {"path": "helpers.py"}}},
                            {"id": "c4", "function": {"name": "edit_file", "arguments": {"path": "helpers.py", "old_text": "a", "new_text": "b"}}},
                        ],
                        "completion_tokens": 15,
                    }
                # Verify tool outputs were fed back to the model
                tool_msgs = [m for m in kwargs["messages"] if m.get("role") == "tool"]
                self.assertEqual(len(tool_msgs), 4)
                self.assertIn("normalize_token", tool_msgs[2]["content"])
                self.assertIn("disabled in read-only", tool_msgs[3]["content"])
                return {
                    "content": json.dumps(
                        {
                            "verdict": "approve",
                            "test_quality": "strong",
                            "summary": "Verified untouched helper signature.",
                            "findings": [],
                        }
                    ),
                    "completion_tokens": 25,
                    "tok_s": 70.0,
                }

            res = review_agent_diff(
                ticket_title="Use helper",
                ticket_spec="Call normalize_token",
                diff="--- a/app.py\n+++ b/app.py\n+from helpers import normalize_token\n",
                test_output="Ran 3 tests in 0.01s\nOK",
                author_model="gpt-4o",
                reviewer_base_url="http://127.0.0.1:9999/v1",
                reviewer_model="claude-sonnet-4",
                worktree=wt,
                max_tool_turns=4,
                chat_fn=fake_chat,
            )

            self.assertEqual(len(turns), 2)
            self.assertEqual(res["verdict"], "approve")
            self.assertEqual(res["test_quality"], "strong")
            self.assertEqual(res["review_mode"], "cross_vendor")
            self.assertEqual(res["inspected_files"], ["helpers.py"])
            self.assertEqual(res["completion_tokens"], 40)
            self.assertEqual(res["tok_s"], 70.0)
            self.assertEqual((wt / "helpers.py").read_text(encoding="utf-8"), "def normalize_token(s):\n    return s.strip().lower()\n")

    def test_tool_enabled_critic_fenced_tool_blocks_and_turn_cap(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            wt = Path(tmp)
            (wt / "untouched.py").write_text("VALUE = 42\n", encoding="utf-8")

            calls = 0

            def fake_chat(**kwargs):
                nonlocal calls
                calls += 1
                if calls == 1:
                    return {
                        "content": 'Let me read untouched.py.\n```tool\n{"name": "read_file", "arguments": {"path": "untouched.py"}}\n```',
                        "completion_tokens": 10,
                    }
                # Turn 2 after max_tool_turns=1 reached: tools should be None
                self.assertIsNone(kwargs.get("tools"))
                user_msgs = [m["content"] for m in kwargs["messages"] if m.get("role") == "user"]
                self.assertTrue(any("VALUE = 42" in msg for msg in user_msgs))
                return {
                    "content": json.dumps(
                        {
                            "verdict": "request_changes",
                            "test_quality": "weak",
                            "summary": "Needs real assertions.",
                            "findings": [{"severity": "medium", "file": "app.py", "issue": "Missing test", "fix": "Add test"}],
                        }
                    ),
                    "completion_tokens": 20,
                }

            res = review_agent_diff(
                ticket_title="Read value",
                ticket_spec="Use VALUE",
                diff="+x = 1",
                test_output="OK",
                author_model="gpt-4o",
                reviewer_base_url="http://127.0.0.1:9999/v1",
                reviewer_model="gemini-2.5-pro",
                worktree=wt,
                max_tool_turns=1,
                chat_fn=fake_chat,
            )

            self.assertEqual(calls, 2)
            self.assertEqual(res["verdict"], "request_changes")
            self.assertEqual(res["test_quality"], "weak")
            self.assertEqual(res["inspected_files"], ["untouched.py"])


class ReviewAndRepairAgentTests(unittest.TestCase):
    def test_approve_immediately_skips_repair(self) -> None:
        repair_called = False

        def fake_repair(_findings, _review):
            nonlocal repair_called
            repair_called = True
            return {"diff": "+repaired", "test_output": "OK"}

        res = review_and_repair_agent(
            ticket_title="Clean change",
            ticket_spec="Do work",
            diff="+good = 1",
            test_output="Ran 2 tests\nOK",
            author_model="gpt-4o",
            reviewer_base_url="http://127.0.0.1:9999/v1",
            reviewer_model="claude-sonnet-4",
            max_repair_rounds=2,
            repair_fn=fake_repair,
            chat_fn=lambda **_kw: {
                "content": json.dumps({"verdict": "approve", "test_quality": "strong", "summary": "LGTM", "findings": []})
            },
        )
        self.assertFalse(repair_called)
        self.assertFalse(res["repair_attempted"])
        self.assertEqual(res["repair_rounds"], 0)
        self.assertEqual(len(res["history"]), 1)
        self.assertEqual(res["verdict"], "approve")

    def test_ineligible_or_zero_rounds_skips_repair(self) -> None:
        res_blocked = review_and_repair_agent(
            ticket_title="Blocked",
            ticket_spec="Spec",
            diff="+x = 1",
            test_output="OK",
            author_model="gpt-4o",
            reviewer_base_url="http://127.0.0.1:9999/v1",
            reviewer_model="o3-mini",
            allow_same_vendor_fresh=False,
            max_repair_rounds=2,
            repair_fn=lambda _f, _r: {"diff": "+y", "test_output": "OK"},
            chat_fn=lambda **_kw: {"content": "{}"},
        )
        self.assertFalse(res_blocked["repair_attempted"])
        self.assertEqual(res_blocked["repair_rounds"], 0)
        self.assertEqual(len(res_blocked["history"]), 1)
        self.assertEqual(res_blocked["verdict"], "block")

    def test_approve_after_one_repair_stops_early(self) -> None:
        review_calls = 0

        def fake_chat(**kwargs):
            nonlocal review_calls
            review_calls += 1
            prompt_txt = "\n".join(m["content"] for m in kwargs["messages"])
            if review_calls == 1:
                self.assertIn("+buggy()", prompt_txt)
                return {
                    "content": json.dumps(
                        {
                            "verdict": "request_changes",
                            "test_quality": "weak",
                            "summary": "Missing null check",
                            "findings": [
                                {
                                    "severity": "high",
                                    "file": "app.py",
                                    "issue": "Unhandled None",
                                    "fix": "Guard against None",
                                }
                            ],
                        }
                    )
                }
            self.assertIn("+safe_guard()", prompt_txt)
            return {
                "content": json.dumps(
                    {
                        "verdict": "approve",
                        "test_quality": "strong",
                        "summary": "Fixed and well tested.",
                        "findings": [],
                    }
                )
            }

        repair_inputs: list[tuple[list[dict], str]] = []

        def fake_repair(findings, review):
            repair_inputs.append((findings, review["verdict"]))
            return {"diff": "+safe_guard()", "test_output": "Ran 4 tests\nOK"}

        res = review_and_repair_agent(
            ticket_title="Fix bug",
            ticket_spec="Handle None",
            diff="+buggy()",
            test_output="Ran 1 test\nOK",
            author_model="gpt-4o",
            reviewer_base_url="http://127.0.0.1:9999/v1",
            reviewer_model="claude-sonnet-4",
            max_repair_rounds=3,
            repair_fn=fake_repair,
            chat_fn=fake_chat,
        )

        self.assertEqual(review_calls, 2)
        self.assertEqual(len(repair_inputs), 1)
        self.assertEqual(repair_inputs[0][0][0]["issue"], "Unhandled None")
        self.assertTrue(res["repair_attempted"])
        self.assertEqual(res["repair_rounds"], 1)
        self.assertEqual(len(res["history"]), 2)
        self.assertEqual(res["initial_review"]["verdict"], "request_changes")
        self.assertEqual(res["verdict"], "approve")
        self.assertEqual(res["test_quality"], "strong")
        self.assertEqual(res["diff"], "+safe_guard()")
        self.assertEqual(res["test_output"], "Ran 4 tests\nOK")

    def test_exhaust_max_repair_rounds(self) -> None:
        repair_rounds_seen = 0

        def fake_repair(_findings, _review):
            nonlocal repair_rounds_seen
            repair_rounds_seen += 1
            return {"diff": f"+attempt_{repair_rounds_seen}", "test_output": "OK"}

        res = review_and_repair_agent(
            ticket_title="Hard fix",
            ticket_spec="Handle edge cases",
            diff="+initial",
            test_output="OK",
            author_model="gpt-4o",
            reviewer_base_url="http://127.0.0.1:9999/v1",
            reviewer_model="gemini-2.5-pro",
            max_repair_rounds=2,
            repair_fn=fake_repair,
            chat_fn=lambda **_kw: {
                "content": json.dumps(
                    {
                        "verdict": "request_changes",
                        "test_quality": "adequate",
                        "summary": "Still missing edge case",
                        "findings": [{"severity": "medium", "file": "app.py", "issue": "Edge case", "fix": "Handle it"}],
                    }
                )
            },
        )

        self.assertEqual(repair_rounds_seen, 2)
        self.assertTrue(res["repair_attempted"])
        self.assertEqual(res["repair_rounds"], 2)
        self.assertEqual(len(res["history"]), 3)
        self.assertEqual(res["verdict"], "request_changes")
        self.assertEqual(res["diff"], "+attempt_2")


if __name__ == "__main__":
    unittest.main()
