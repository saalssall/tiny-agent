"""Unit tests for the workspace and tools. Run with: python -m unittest"""

import tempfile
import unittest
from pathlib import Path

from tiny_agent.tools import (
    EditFileTool,
    ListFilesTool,
    ReadFileTool,
    RunCommandTool,
    ToolRegistry,
    WriteFileTool,
)
from tiny_agent.workspace import Workspace, WorkspaceError


class WorkspaceTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.ws = Workspace(Path(self.tmp.name))

    def tearDown(self):
        self.tmp.cleanup()

    def test_relative_paths_stay_inside_root(self):
        self.assertEqual(self.ws.resolve("a/b.txt"), self.ws.root / "a" / "b.txt")
        self.assertEqual(self.ws.resolve("."), self.ws.root)

    def test_escaping_paths_are_refused(self):
        for bad in ("../secret", "/etc/passwd", "a/../../x"):
            with self.assertRaises(WorkspaceError, msg=bad):
                self.ws.resolve(bad)

    def test_write_read_roundtrip(self):
        self.assertFalse(self.ws.write("src/hello.py", "print('hi')\n"))
        self.assertEqual(self.ws.read("src/hello.py"), "print('hi')\n")
        self.assertTrue(self.ws.write("src/hello.py", "x"))  # existed now

    def test_replace_once_requires_unique_match(self):
        self.ws.write("f.txt", "a\na\n")
        with self.assertRaises(WorkspaceError):
            self.ws.replace_once("f.txt", "a", "b")  # ambiguous
        with self.assertRaises(WorkspaceError):
            self.ws.replace_once("f.txt", "zzz", "b")  # missing
        self.ws.replace_once("f.txt", "a\na", "b\nc")
        self.assertEqual(self.ws.read("f.txt"), "b\nc\n")

    def test_list_skips_noise_directories(self):
        self.ws.write("keep.txt", "")
        self.ws.write("node_modules/junk.js", "")
        self.ws.write("pkg/mod.py", "")
        self.assertEqual(self.ws.list(), ["pkg/", "keep.txt"])
        self.assertEqual(self.ws.list(recursive=True), ["keep.txt", "pkg/mod.py"])


class ToolTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.ws = Workspace(Path(self.tmp.name))
        self.approved: list[str] = []
        self.registry = ToolRegistry(
            [
                ListFilesTool(self.ws),
                ReadFileTool(self.ws),
                WriteFileTool(self.ws),
                EditFileTool(self.ws),
                RunCommandTool(self.ws, self._approve),
            ],
            max_output=50,
        )
        self.allow_commands = True

    def tearDown(self):
        self.tmp.cleanup()

    def _approve(self, command: str) -> bool:
        self.approved.append(command)
        return self.allow_commands

    def test_validation_rejects_bad_input(self):
        r = self.registry.execute("read_file", {})
        self.assertTrue(r.is_error)
        self.assertIn("missing required field 'path'", r.content)

        r = self.registry.execute("list_files", {"recursive": "yes"})
        self.assertTrue(r.is_error)
        self.assertIn("should be boolean", r.content)

        r = self.registry.execute("write_file", {"path": "a", "content": "b", "extra": 1})
        self.assertTrue(r.is_error)
        self.assertIn("unexpected field", r.content)

    def test_unknown_tool_is_an_error(self):
        self.assertTrue(self.registry.execute("nope", {}).is_error)

    def test_read_file_numbers_lines(self):
        self.ws.write("a.txt", "one\ntwo\n")
        r = self.registry.execute("read_file", {"path": "a.txt"})
        self.assertFalse(r.is_error)
        self.assertEqual(r.content.splitlines(), ["    1\tone", "    2\ttwo"])

    def test_workspace_errors_become_tool_errors(self):
        r = self.registry.execute("read_file", {"path": "../outside"})
        self.assertTrue(r.is_error)
        self.assertIn("outside the workspace", r.content)

    def test_long_output_is_truncated(self):
        self.ws.write("big.txt", "x" * 500)
        r = self.registry.execute("read_file", {"path": "big.txt"})
        self.assertIn("truncated", r.content)
        self.assertLess(len(r.content), 200)

    def test_run_command_asks_for_approval(self):
        r = self.registry.execute("run_command", {"command": "echo hi; exit 3"})
        self.assertEqual(self.approved, ["echo hi; exit 3"])
        self.assertFalse(r.is_error)
        self.assertIn("hi", r.content)
        self.assertIn("[exit code 3]", r.content)

    def test_declined_command_does_not_run(self):
        self.allow_commands = False
        r = self.registry.execute("run_command", {"command": "touch should_not_exist"})
        self.assertIn("declined", r.content)
        self.assertFalse((self.ws.root / "should_not_exist").exists())

    def test_schemas_are_well_formed(self):
        for schema in self.registry.schemas():
            self.assertIn("name", schema)
            self.assertEqual(schema["input_schema"]["type"], "object")
            self.assertFalse(schema["input_schema"]["additionalProperties"])
            self.assertTrue(schema["eager_input_streaming"])


if __name__ == "__main__":
    unittest.main()
