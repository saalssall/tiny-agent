"""Unit tests for the agent loop, using stubbed API responses (no network)."""

import io
import tempfile
import unittest
from pathlib import Path

from anthropic.types.beta import BetaMessage, BetaTextBlock, BetaToolUseBlock, BetaUsage

from tiny_agent.agent import Agent, Conversation, UsageTracker
from tiny_agent.config import Settings
from tiny_agent.console import Console
from tiny_agent.tools import ReadFileTool, ToolRegistry, WriteFileTool
from tiny_agent.workspace import Workspace


def message(*blocks, stop_reason="end_turn") -> BetaMessage:
    """Build a realistic BetaMessage the way the SDK would."""
    return BetaMessage(
        id="msg_test",
        type="message",
        role="assistant",
        model="claude-opus-5",
        content=list(blocks),
        stop_reason=stop_reason,
        stop_sequence=None,
        usage=BetaUsage(input_tokens=10, output_tokens=5),
    )


def text(s: str) -> BetaTextBlock:
    return BetaTextBlock(type="text", text=s)


def tool_use(name: str, args: dict, call_id: str = "toolu_1") -> BetaToolUseBlock:
    return BetaToolUseBlock(type="tool_use", id=call_id, name=name, input=args)


class ScriptedAgent(Agent):
    """An Agent whose _request() returns pre-scripted responses instead of calling the API."""

    def __init__(self, responses, **kwargs):
        super().__init__(client=None, **kwargs)
        self.responses = list(responses)
        self.requests_made = 0

    def _request(self) -> BetaMessage:
        self.requests_made += 1
        return self.responses.pop(0)


class AgentLoopTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        root = Path(self.tmp.name)
        self.ws = Workspace(root)
        self.out = io.StringIO()
        self.settings = Settings(workspace=root, api_key="test")
        self.tools = ToolRegistry([ReadFileTool(self.ws), WriteFileTool(self.ws)])

    def tearDown(self):
        self.tmp.cleanup()

    def make_agent(self, *responses) -> ScriptedAgent:
        return ScriptedAgent(
            responses, settings=self.settings, tools=self.tools, console=Console(self.out, color=False)
        )

    def test_plain_answer_completes_in_one_round(self):
        agent = self.make_agent(message(text("Hello!")))
        agent.ask("hi")
        self.assertEqual(agent.requests_made, 1)
        roles = [m["role"] for m in agent.conversation.messages]
        self.assertEqual(roles, ["user", "assistant"])
        self.assertIn("Hello!", str(agent.conversation.messages[1]["content"]))

    def test_tool_call_round_trip(self):
        agent = self.make_agent(
            message(tool_use("write_file", {"path": "a.txt", "content": "hi"}), stop_reason="tool_use"),
            message(text("Done.")),
        )
        agent.ask("make a.txt")

        self.assertEqual(agent.requests_made, 2)
        self.assertEqual((self.ws.root / "a.txt").read_text(), "hi")
        roles = [m["role"] for m in agent.conversation.messages]
        self.assertEqual(roles, ["user", "assistant", "user", "assistant"])
        result_block = agent.conversation.messages[2]["content"][0]
        self.assertEqual(result_block["type"], "tool_result")
        self.assertEqual(result_block["tool_use_id"], "toolu_1")
        self.assertNotIn("is_error", result_block)

    def test_failed_tool_is_reported_as_error_and_loop_continues(self):
        agent = self.make_agent(
            message(tool_use("read_file", {"path": "missing.txt"}), stop_reason="tool_use"),
            message(text("That file does not exist.")),
        )
        agent.ask("read it")
        result_block = agent.conversation.messages[2]["content"][0]
        self.assertTrue(result_block["is_error"])
        self.assertIn("does not exist", result_block["content"])

    def test_refusal_is_not_added_to_history(self):
        agent = self.make_agent(message(stop_reason="refusal"))
        agent.ask("something")
        self.assertEqual([m["role"] for m in agent.conversation.messages], ["user"])
        self.assertIn("declined", self.out.getvalue())

    def test_truncated_tool_call_is_not_added_to_history(self):
        agent = self.make_agent(
            message(tool_use("write_file", {"path": "a.txt", "content": "partial"}), stop_reason="max_tokens")
        )
        agent.ask("write a lot")
        self.assertEqual([m["role"] for m in agent.conversation.messages], ["user"])
        self.assertFalse((self.ws.root / "a.txt").exists(), "a truncated tool call must not run")

    def test_pause_turn_resends(self):
        agent = self.make_agent(message(text("working..."), stop_reason="pause_turn"), message(text("done")))
        agent.ask("go")
        self.assertEqual(agent.requests_made, 2)

    def test_malformed_json_is_retried_then_raised(self):
        class Flaky(ScriptedAgent):
            def _request(self):
                self.requests_made += 1
                raise ValueError("bad json")

        agent = Flaky([], settings=self.settings, tools=self.tools, console=Console(self.out, color=False))
        with self.assertRaises(ValueError):
            agent.ask("x")
        self.assertEqual(agent.requests_made, Agent.MAX_JSON_RETRIES + 1)

    def test_usage_is_accumulated(self):
        agent = self.make_agent(
            message(tool_use("read_file", {"path": "x"}), stop_reason="tool_use"), message(text("ok"))
        )
        agent.ask("x")
        self.assertEqual(agent.usage.input, 20)
        self.assertEqual(agent.usage.output, 10)


class ConversationTests(unittest.TestCase):
    def test_checkpoint_and_rollback(self):
        conv = Conversation()
        conv.add_user("a")
        mark = conv.checkpoint()
        conv.add_user("b")
        conv.add_assistant("c")
        conv.rollback(mark)
        self.assertEqual(len(conv.messages), 1)
        conv.clear()
        self.assertEqual(conv.messages, [])


class UsageTrackerTests(unittest.TestCase):
    def test_cost_for_known_and_unknown_models(self):
        tracker = UsageTracker()
        tracker.add(BetaUsage(input_tokens=1_000_000, output_tokens=0))
        self.assertAlmostEqual(tracker.cost_usd("claude-opus-5"), 5.00)
        self.assertIsNone(tracker.cost_usd("some-other-model"))
        self.assertEqual(tracker.rows("some-other-model")[-1], ("estimated cost", "unknown model"))


if __name__ == "__main__":
    unittest.main()
