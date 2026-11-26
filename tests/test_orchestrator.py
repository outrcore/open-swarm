#!/usr/bin/env python3
"""Unit tests for open_swarm/orchestrator.py (offline, deterministic)."""
from __future__ import annotations

import json
from pathlib import Path
import sys
import unittest

SWARM_DIR = Path(__file__).resolve().parents[1]
if str(SWARM_DIR) not in sys.path:
    sys.path.insert(0, str(SWARM_DIR))

from orchestrator import (  # noqa: E402
    LANES,
    check_review_eligibility,
    resolve_vendor,
    review_agent_diff,
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


if __name__ == "__main__":
    unittest.main()
