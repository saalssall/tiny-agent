"""Tests for the Console: formatting, colour handling, and injected input."""

import io
import unittest
from pathlib import Path

from tiny_agent.config import Settings
from tiny_agent.console import Console, _first_line, _summarise


def make_console(*answers: str, color: bool = False) -> tuple[Console, io.StringIO, list[str]]:
    out = io.StringIO()
    prompts: list[str] = []
    script = iter(answers)

    def fake_input(prompt: str) -> str:
        prompts.append(prompt)
        try:
            return next(script)
        except StopIteration as exc:
            raise EOFError from exc

    return Console(out, color=color, input_fn=fake_input), out, prompts


class ConsoleOutputTests(unittest.TestCase):
    def test_plain_text_when_colour_is_off(self):
        console, out, _ = make_console()
        console.info("hello")
        console.warn("careful")
        console.error("bad")
        self.assertEqual(out.getvalue(), "  hello\n  careful\n  bad\n")

    def test_ansi_codes_when_colour_is_on(self):
        console, out, _ = make_console(color=True)
        console.error("bad")
        self.assertIn("\033[31m", out.getvalue())
        self.assertIn("\033[0m", out.getvalue())

    def test_colour_auto_detection_respects_non_tty(self):
        console = Console(io.StringIO())  # StringIO is not a TTY
        self.assertFalse(console.color)

    def test_table_aligns_labels_and_right_aligns_values(self):
        console, out, _ = make_console()
        console.table([("a", "1"), ("longer label", "2,000")])
        lines = out.getvalue().splitlines()
        self.assertEqual(len(lines[0]), len(lines[1]))  # values are right-aligned to the same column
        self.assertTrue(lines[0].startswith("  a ") and lines[0].endswith(" 1"))
        self.assertTrue(lines[1].startswith("  longer label  ") and lines[1].endswith(" 2,000"))

    def test_banner_mentions_model_workspace_and_yolo(self):
        console, out, _ = make_console()
        settings = Settings(workspace=Path("/tmp/w"), api_key="k", model="claude-opus-5", auto_approve=True)
        console.banner(settings)
        text = out.getvalue()
        self.assertIn("claude-opus-5", text)
        self.assertIn("/tmp/w", text)
        self.assertIn("--yolo", text)

    def test_tool_call_and_result_rendering(self):
        console, out, _ = make_console()
        console.tool_call("write_file", {"path": "a.py", "content": "line1\nline2"})
        console.tool_result("Created a.py (2 lines)\nmore", is_error=False)
        console.tool_result("boom", is_error=True)
        text = out.getvalue()
        self.assertIn("▸ write_file path=a.py content=line1⏎line2", text)
        self.assertIn("✓ Created a.py (2 lines)", text)
        self.assertIn("✗ boom", text)
        self.assertNotIn("more", text)  # only the first line of a result is shown

    def test_streaming_helpers_do_not_add_newlines(self):
        console, out, _ = make_console()
        console.assistant_text("Hel")
        console.assistant_text("lo")
        console.tool_pending("read_file")
        console.tool_progress()
        self.assertEqual(out.getvalue(), "Hello\n  … preparing read_file.")


class ConsoleInputTests(unittest.TestCase):
    def test_ask_strips_and_uses_prompt(self):
        console, _, prompts = make_console("  hi there  ")
        self.assertEqual(console.ask("you › "), "hi there")
        self.assertEqual(prompts, ["you › "])

    def test_ask_propagates_eof(self):
        console, _, _ = make_console()
        with self.assertRaises(EOFError):
            console.ask("> ")

    def test_confirm_accepts_y_and_yes_only(self):
        for answer, expected in [("y", True), ("YES", True), ("n", False), ("", False), ("maybe", False)]:
            console, _, _ = make_console(answer)
            self.assertEqual(console.confirm("run?"), expected, answer)

    def test_confirm_treats_eof_as_no(self):
        console, _, _ = make_console()
        self.assertFalse(console.confirm("run?"))


class HelperTests(unittest.TestCase):
    def test_summarise_truncates_long_values_and_lines(self):
        self.assertEqual(_summarise({"a": 1, "b": "x" * 50}), "a=1 b=" + "x" * 37 + "...")
        self.assertEqual(_summarise("not a dict"), "")
        long = _summarise({f"k{i}": "v" * 20 for i in range(10)})
        self.assertLessEqual(len(long), 70)
        self.assertTrue(long.endswith("..."))

    def test_first_line(self):
        self.assertEqual(_first_line("\n\n first\nsecond"), "first")
        self.assertEqual(_first_line("   "), "")
        self.assertEqual(_first_line("x" * 100), "x" * 87 + "...")


if __name__ == "__main__":
    unittest.main()
