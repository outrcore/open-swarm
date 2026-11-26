#!/usr/bin/env python3
"""Unit tests for open_swarm/agent_tools.py (offline, deterministic)."""
from __future__ import annotations

import json
from pathlib import Path
import sys
import tempfile
import unittest

SWARM_DIR = Path(__file__).resolve().parents[1]
if str(SWARM_DIR) not in sys.path:
    sys.path.insert(0, str(SWARM_DIR))

from agent_tools import (  # noqa: E402
    build_agent_context_brief,
    decompose_goal_to_tasks,
    detect_file_collisions,
    index_repo_symbols,
)
from continuity_store import ContinuityStore  # noqa: E402


class IndexRepoSymbolsTests(unittest.TestCase):
    def test_python_ast_and_skip_dirs_and_malformed(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            repo = Path(tmp)
            (repo / "src").mkdir()
            (repo / "src" / "engine.py").write_text(
                "class Engine:\n"
                "    def __init__(self, name: str, limit: int = 10):\n"
                "        self.name = name\n"
                "    async def run(self, *args, **kwargs):\n"
                "        return True\n\n"
                "def helper(x, y=1):\n"
                "    return x + y\n\n"
                "async def fetch_data(url: str, timeout: float = 5.0):\n"
                "    return url\n",
                encoding="utf-8",
            )
            (repo / "src" / "broken.py").write_text("def oops(:\n    pass\n", encoding="utf-8")

            # Directories/files that must be skipped
            for skipped in (
                repo / ".git" / "hook.py",
                repo / "__pycache__" / "cached.py",
                repo / "dist" / "bundle.py",
                repo / "node_modules" / "pkg" / "index.js",
                repo / ".venv" / "lib.py",
                repo / "src" / "compiled.pyc",
            ):
                skipped.parent.mkdir(parents=True, exist_ok=True)
                skipped.write_text("def should_not_appear():\n    pass\n", encoding="utf-8")

            idx = index_repo_symbols(repo)
            self.assertIn("### src/engine.py", idx)
            self.assertIn("class Engine", idx)
            self.assertIn("__init__(self, name, limit=...)", idx)
            self.assertIn("async def run(self, *args, **kwargs)", idx)
            self.assertIn("def helper(x, y=...)", idx)
            self.assertIn("async def fetch_data(url, timeout=...)", idx)
            self.assertNotIn("broken.py", idx)
            self.assertNotIn("should_not_appear", idx)
            self.assertNotIn(".git", idx)
            self.assertNotIn("dist/", idx)
            self.assertNotIn("__pycache__", idx)
            self.assertNotIn("node_modules", idx)
            self.assertNotIn(".venv", idx)

            # Deterministic output across calls
            self.assertEqual(idx, index_repo_symbols(repo))

    def test_js_html_and_swift_regex_extraction(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            repo = Path(tmp)
            (repo / "web.js").write_text(
                "export class Store extends Base {\n}\n"
                "async function loadItems(limit, offset) {\n  return [];\n}\n",
                encoding="utf-8",
            )
            (repo / "ui.html").write_text(
                '<div class="card">Hello</div>\n'
                "<script>\n"
                "class Widget {\n}\n"
                "function renderCard(id) {}\n"
                "</script>\n",
                encoding="utf-8",
            )
            (repo / "App.swift").write_text(
                "public final class ModelRunner: Sendable {\n"
                "    func generate(prompt: String, maxTokens: Int) -> String { return \"\" }\n"
                "}\n",
                encoding="utf-8",
            )

            idx = index_repo_symbols(repo)
            self.assertIn("### App.swift", idx)
            self.assertIn("class ModelRunner", idx)
            self.assertIn("func generate(prompt: String, maxTokens: Int)", idx)
            self.assertIn("### ui.html", idx)
            self.assertIn("class Widget", idx)
            self.assertIn("function renderCard(id)", idx)
            self.assertNotIn("class card", idx)
            self.assertIn("### web.js", idx)
            self.assertIn("class Store", idx)
            self.assertIn("async function loadItems(limit, offset)", idx)

    def test_max_chars_line_boundary_cap_and_max_files(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            repo = Path(tmp)
            for i in range(6):
                (repo / f"mod_{i}.py").write_text(
                    f"def alpha_{i}(a, b):\n    return a\n\n"
                    f"def beta_{i}(c, d):\n    return c\n",
                    encoding="utf-8",
                )

            capped = index_repo_symbols(repo, max_chars=95)
            self.assertLessEqual(len(capped), 95)
            self.assertTrue(capped)
            # Clean line boundaries: every non-empty line is complete
            for line in capped.splitlines():
                if line.startswith("- def "):
                    self.assertTrue(line.endswith(")"))

            limited_files = index_repo_symbols(repo, max_files=2, max_chars=2400)
            self.assertIn("mod_0.py", limited_files)
            self.assertIn("mod_1.py", limited_files)
            self.assertNotIn("mod_2.py", limited_files)


class BuildAgentContextBriefTests(unittest.TestCase):
    def test_pulls_checkpoints_messages_and_respects_max_chars(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            repo = tmp_path / "repo"
            repo.mkdir()
            project_key = str(repo.resolve())

            store = ContinuityStore(tmp_path / "continuity.db")
            store.publish_checkpoint(
                project_key=project_key,
                role="coder",
                summary="Implemented parser and unit tests",
                next_step="Wire CLI entrypoint",
                actor="coder-a",
                expected_revision=0,
            )
            for idx in range(7):
                store.send_message(
                    project_key=project_key,
                    from_role="architect",
                    to_role="coder",
                    body=f"MSG_{idx}: keep changes scoped",
                )

            runs_dir = tmp_path / "runs"
            runs_dir.mkdir()
            (runs_dir / "run-err.json").write_text(
                json.dumps(
                    {
                        "run_id": "run-err",
                        "repo": str(repo),
                        "status": "error",
                        "error": "spend cap exceeded",
                        "model": "outrcore/Inkling-2",
                    }
                ),
                encoding="utf-8",
            )

            brief = build_agent_context_brief(
                repo,
                continuity=store,
                runs_dir=runs_dir,
                max_chars=900,
            )

            self.assertIn("Implemented parser and unit tests", brief)
            self.assertIn("Wire CLI entrypoint", brief)
            # Only last 5 of the first 20 messages should appear
            self.assertNotIn("MSG_0:", brief)
            self.assertNotIn("MSG_1:", brief)
            self.assertIn("MSG_2:", brief)
            self.assertIn("MSG_6:", brief)
            self.assertIn("spend cap exceeded", brief)
            self.assertLessEqual(len(brief), 900)

            tight = build_agent_context_brief(
                repo,
                continuity=store,
                runs_dir=tmp_path / "missing_runs",
                max_chars=80,
            )
            self.assertLessEqual(len(tight), 80)
            self.assertTrue(tight)

    def test_empty_when_nothing_relevant_exists(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            repo = Path(tmp) / "empty_repo"
            repo.mkdir()
            brief = build_agent_context_brief(repo, continuity=None, runs_dir=Path(tmp) / "none")
            self.assertEqual(brief, "")


class DecomposeGoalToTasksTests(unittest.TestCase):
    def test_valid_and_fenced_json_clamping_and_lane_normalization(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            repo = Path(tmp)
            (repo / "app.py").write_text("def run():\n    return 1\n", encoding="utf-8")

            captured_prompts: list[list[dict[str, str]]] = []

            def fake_chat(**kwargs):
                captured_prompts.append(kwargs["messages"])
                payload = [
                    {
                        "title": "Add config loader",
                        "spec": "Create config.py and tests",
                        "acceptance_cmd": "python3 -m unittest tests/test_config.py",
                        "lane": "mechanical",
                        "owned_files": ["config.py", "tests/test_config.py"],
                    },
                    {
                        "title": "Refactor engine",
                        "spec": "Update app.py to use config",
                        "acceptance_cmd": "python3 -m unittest tests/test_app.py",
                        "lane": "UNKNOWN_LANE",
                        "owned_files": ["app.py"],
                    },
                    {
                        "title": "Should be clamped out",
                        "spec": "Extra ticket beyond max_agents",
                        "acceptance_cmd": "true",
                        "lane": "deep-reason",
                        "owned_files": ["extra.py"],
                    },
                ]
                return {"content": f"Here is the plan:\n```json\n{json.dumps(payload)}\n```"}

            tasks = decompose_goal_to_tasks(
                repo=repo,
                goal="Add config support",
                base_url="http://127.0.0.1:8000/v1",
                model="test-model",
                continuity_brief="### Checkpoints\n- coder r1: ready",
                max_agents=2,
                chat_fn=fake_chat,
            )

            self.assertEqual(len(tasks), 2)
            self.assertEqual(tasks[0]["title"], "Add config loader")
            self.assertEqual(tasks[0]["lane"], "mechanical")
            self.assertEqual(tasks[0]["owned_files"], ["config.py", "tests/test_config.py"])
            self.assertEqual(tasks[1]["lane"], "coder")  # invalid lane normalized
            self.assertEqual(tasks[1]["title"], "Refactor engine")

            # Verify prompt included repo tree, symbols, continuity brief, and goal
            user_msg = captured_prompts[0][1]["content"]
            self.assertIn("Add config support", user_msg)
            self.assertIn("app.py", user_msg)
            self.assertIn("def run()", user_msg)
            self.assertIn("coder r1: ready", user_msg)

    def test_plain_json_array_and_fallback_on_malformed_or_exception(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            repo = Path(tmp)
            (repo / "lib.py").write_text("def f():\n    pass\n", encoding="utf-8")

            def plain_json_chat(**_kwargs):
                return {
                    "content": json.dumps(
                        [
                            {
                                "title": "Deep analysis",
                                "spec": "Analyze concurrency",
                                "acceptance_cmd": "pytest",
                                "lane": "deep-reason",
                                "owned_files": ["lib.py"],
                            }
                        ]
                    )
                }

            tasks = decompose_goal_to_tasks(
                repo=repo,
                goal="Fix race condition",
                base_url="http://localhost:8000/v1",
                model="test-model",
                max_agents=3,
                chat_fn=plain_json_chat,
            )
            self.assertEqual(len(tasks), 1)
            self.assertEqual(tasks[0]["lane"], "deep-reason")

            def bad_json_chat(**_kwargs):
                return {"content": "Not valid JSON at all {oops"}

            fb1 = decompose_goal_to_tasks(
                repo=repo,
                goal="Build feature X",
                base_url="http://localhost:8000/v1",
                model="test-model",
                chat_fn=bad_json_chat,
            )
            self.assertEqual(
                fb1,
                [
                    {
                        "title": "Build feature X",
                        "spec": "Build feature X",
                        "acceptance_cmd": "python3 -m unittest discover -s tests -v",
                        "lane": "coder",
                        "owned_files": [],
                    }
                ],
            )

            def raising_chat(**_kwargs):
                raise RuntimeError("backend unreachable")

            fb2 = decompose_goal_to_tasks(
                repo=repo,
                goal="   ",
                base_url="http://localhost:8000/v1",
                model="test-model",
                chat_fn=raising_chat,
            )
            self.assertEqual(
                fb2,
                [
                    {
                        "title": "Implement goal",
                        "spec": "Implement the requested change cleanly with unit tests.",
                        "acceptance_cmd": "python3 -m unittest discover -s tests -v",
                        "lane": "coder",
                        "owned_files": [],
                    }
                ],
            )


class DetectFileCollisionsTests(unittest.TestCase):
    def test_ignores_failing_agents_and_detects_shared_passing_writes(self) -> None:
        results = [
            {
                "agent_id": "coder-b",
                "tests_ok": True,
                "written_files": ["open_swarm/shared.py", "open_swarm/b_only.py"],
            },
            {
                "agent_id": "coder-a",
                "tests_ok": True,
                "written_files": ["open_swarm/shared.py", "open_swarm/shared.py", "open_swarm/a_only.py"],
            },
            {
                "agent_id": "coder-c",
                "tests_ok": False,
                "written_files": ["open_swarm/a_only.py", "open_swarm/shared.py"],
            },
        ]

        collisions = detect_file_collisions(results)
        self.assertEqual(collisions, {"open_swarm/shared.py": ["coder-a", "coder-b"]})


if __name__ == "__main__":
    unittest.main()
