"""Tests for the chat loop, slash commands, error handling, and command approval."""

import io
import logging
import os
import tempfile
import unittest
import unittest.mock
from pathlib import Path

import anthropic

from tiny_agent.agent import Conversation, UsageTracker
from tiny_agent.cli import Approver, ChatApp, build_app, configure_logging, describe_error, main
from tiny_agent.config import Settings
from tiny_agent.console import Console
from tiny_agent.risk import CommandGate, Verdict


class FakeAgent:
    """Stands in for Agent: records questions, optionally raises, keeps real history/usage objects."""

    def __init__(self, raise_on: dict[str, BaseException] | None = None):
        self.conversation = Conversation()
        self.usage = UsageTracker()
        self.asked: list[str] = []
        self.raise_on = raise_on or {}

    def ask(self, text: str) -> None:
        self.asked.append(text)
        self.conversation.add_user(text)
        if text in self.raise_on:
            raise self.raise_on[text]
        self.conversation.add_assistant("ok")


def scripted_console(*lines: str) -> tuple[Console, io.StringIO]:
    out = io.StringIO()
    script = iter(lines)

    def fake_input(_prompt: str) -> str:
        try:
            return next(script)
        except StopIteration as exc:
            raise EOFError from exc

    return Console(out, color=False, input_fn=fake_input), out


SETTINGS = Settings(workspace=Path("/tmp"), api_key="k")


class ChatAppTests(unittest.TestCase):
    def test_commands_and_chat_are_dispatched(self):
        console, out = scripted_console("", "/help", "/bogus", "hello", "/cost", "/clear", "/quit")
        agent = FakeAgent()
        ChatApp(SETTINGS, console, agent, gate_enabled=True).run()  # type: ignore[arg-type]

        self.assertEqual(agent.asked, ["hello"])
        self.assertEqual(agent.conversation.messages, [])  # /clear emptied it
        text = out.getvalue()
        self.assertIn("/cost    show token usage", text)
        self.assertIn("unknown command '/bogus'", text)
        self.assertIn("conversation cleared", text)
        self.assertIn("safe-looking ones run automatically", text)
        self.assertIn("session usage", text)

    def test_eof_at_prompt_exits_cleanly(self):
        console, out = scripted_console()  # immediate EOF
        ChatApp(SETTINGS, console, FakeAgent(), gate_enabled=False).run()  # type: ignore[arg-type]
        self.assertIn("every command asks you first", out.getvalue())
        self.assertIn("session usage", out.getvalue())

    def test_failed_turn_rolls_back_history_and_explains(self):
        agent = FakeAgent(raise_on={"boom": KeyboardInterrupt()})
        console, out = scripted_console("fine", "boom", "/quit")
        ChatApp(SETTINGS, console, agent).run()  # type: ignore[arg-type]
        roles = [m["role"] for m in agent.conversation.messages]
        self.assertEqual(roles, ["user", "assistant"])  # the 'boom' turn was removed
        self.assertIn("[interrupted]", out.getvalue())

    def test_api_error_is_explained(self):
        exc = anthropic.RateLimitError.__new__(anthropic.RateLimitError)
        agent = FakeAgent(raise_on={"x": exc})
        console, out = scripted_console("x", "/quit")
        ChatApp(SETTINGS, console, agent).run()  # type: ignore[arg-type]
        self.assertIn("Rate limited", out.getvalue())
        self.assertEqual(agent.conversation.messages, [])


class DescribeErrorTests(unittest.TestCase):
    def test_messages(self):
        self.assertEqual(describe_error(KeyboardInterrupt()), "[interrupted]")
        self.assertIn(
            "API key", describe_error(anthropic.AuthenticationError.__new__(anthropic.AuthenticationError))
        )
        self.assertIn(
            "Rate limited", describe_error(anthropic.RateLimitError.__new__(anthropic.RateLimitError))
        )
        status = anthropic.APIStatusError.__new__(anthropic.APIStatusError)
        status.status_code, status.message = 503, "overloaded"
        self.assertEqual(describe_error(status), "API error 503: overloaded")
        self.assertIn(
            "Network error",
            describe_error(anthropic.APIConnectionError.__new__(anthropic.APIConnectionError)),
        )
        self.assertIn("Unexpected error", describe_error(RuntimeError("odd")))


class FakeGate:
    def __init__(self, verdict: Verdict):
        self.verdict = verdict
        self.seen: list[str] = []

    def assess(self, command: str) -> Verdict:
        self.seen.append(command)
        return self.verdict


class ApproverTests(unittest.TestCase):
    def test_yolo_skips_the_gate_entirely(self):
        console, out = scripted_console()
        gate = FakeGate(Verdict(risky=True, reason="asking"))
        self.assertTrue(Approver(console, gate, auto_approve=True)("rm -rf x"))  # type: ignore[arg-type]
        self.assertEqual(gate.seen, [])
        self.assertIn("$ rm -rf x", out.getvalue())

    def test_safe_verdict_runs_without_asking(self):
        console, out = scripted_console()  # any input() call would raise EOFError -> False
        gate = FakeGate(Verdict(risky=False, reason="auto-approved, looks safe (97%)"))
        self.assertTrue(Approver(console, gate, auto_approve=False)("ls"))  # type: ignore[arg-type]
        self.assertIn("auto-approved", out.getvalue())

    def test_risky_verdict_asks_and_respects_answer(self):
        for answer, expected in (("y", True), ("n", False)):
            console, out = scripted_console(answer)
            gate = FakeGate(Verdict(risky=True, reason="asking because it deletes files"))
            self.assertEqual(Approver(console, gate, auto_approve=False)("rm x"), expected)  # type: ignore[arg-type]
            self.assertIn("deletes files", out.getvalue())


class WiringTests(unittest.TestCase):
    def test_build_app_wires_gate_and_tools(self):
        with tempfile.TemporaryDirectory() as tmp:
            settings = Settings(workspace=Path(tmp), api_key="k")
            app = build_app(settings, Console(io.StringIO(), color=False))
            self.assertTrue(app.gate_enabled)  # the shipped model loads
            self.assertEqual(
                sorted(t["name"] for t in app.agent.tools.schemas()),
                ["edit_file", "list_files", "read_file", "run_command", "write_file"],
            )
            always_ask = build_app(
                Settings(workspace=Path(tmp), api_key="k", always_ask=True), Console(io.StringIO())
            )
            self.assertFalse(always_ask.gate_enabled)
            self.assertIsInstance(always_ask.agent.tools.execute, object)
            self.assertIsInstance(CommandGate(None), CommandGate)

    def test_main_reports_config_errors(self):
        with unittest.mock.patch("sys.stdout", new_callable=io.StringIO) as out:
            self.assertEqual(main(["/definitely/not/a/dir"]), 1)
        self.assertIn("not a directory", out.getvalue())

    def test_main_runs_the_app(self):
        with (
            tempfile.TemporaryDirectory() as tmp,
            unittest.mock.patch.dict(os.environ, {"ANTHROPIC_API_KEY": "k"}),
        ):
            with (
                unittest.mock.patch("builtins.input", side_effect=EOFError),
                unittest.mock.patch("sys.stdout", new_callable=io.StringIO) as out,
            ):
                self.assertEqual(main([tmp, "--verbose"]), 0)
        self.assertIn("session usage", out.getvalue())

    def test_configure_logging_levels(self):
        configure_logging(verbose=False)
        self.assertEqual(logging.getLogger("tiny_agent").level, logging.WARNING)
        logging.getLogger().handlers.clear()
        configure_logging(verbose=True)
        self.assertEqual(logging.getLogger("tiny_agent").level, logging.DEBUG)
        self.assertEqual(logging.getLogger().level, logging.WARNING)  # third-party loggers stay quiet
        logging.getLogger().handlers.clear()


if __name__ == "__main__":
    unittest.main()
