"""The agent loop: send the conversation, stream the reply, run tools, repeat."""

from __future__ import annotations

from typing import Any

import anthropic
from anthropic.types.beta import BetaMessage, BetaToolUseBlock

from .config import PRICES, SYSTEM_PROMPT, Settings
from .console import Console
from .tools import ToolRegistry

# Retry the same request server-side on a fallback model if a safety classifier declines.
FALLBACK_BETA = "server-side-fallback-2026-07-01"

# How many streamed tool-input fragments to receive between progress dots.
PROGRESS_EVERY = 25


class Conversation:
    """The message history sent to the API, with checkpoint/rollback for error recovery."""

    def __init__(self) -> None:
        self.messages: list[dict[str, Any]] = []

    def add_user(self, content: Any) -> None:
        self.messages.append({"role": "user", "content": content})

    def add_assistant(self, content: Any) -> None:
        self.messages.append({"role": "assistant", "content": content})

    def checkpoint(self) -> int:
        return len(self.messages)

    def rollback(self, checkpoint: int) -> None:
        del self.messages[checkpoint:]

    def clear(self) -> None:
        self.messages.clear()


class UsageTracker:
    """Accumulates token counts across requests and estimates cost."""

    def __init__(self) -> None:
        self.input = self.output = self.cache_write = self.cache_read = 0

    def add(self, usage: Any) -> None:
        self.input += usage.input_tokens or 0
        self.output += usage.output_tokens or 0
        self.cache_write += getattr(usage, "cache_creation_input_tokens", 0) or 0
        self.cache_read += getattr(usage, "cache_read_input_tokens", 0) or 0

    def cost_usd(self, model: str) -> float | None:
        prices = PRICES.get(model)
        if prices is None:
            return None
        p_in, p_out, p_write, p_read = prices
        total = (
            self.input * p_in + self.output * p_out + self.cache_write * p_write + self.cache_read * p_read
        )
        return total / 1_000_000

    def rows(self, model: str) -> list[tuple[str, str]]:
        """Label/value pairs for display."""
        cost = self.cost_usd(model)
        return [
            ("input tokens", f"{self.input:,}"),
            ("output tokens", f"{self.output:,}"),
            ("cache writes", f"{self.cache_write:,}"),
            ("cache reads", f"{self.cache_read:,}"),
            ("estimated cost", f"${cost:.4f}" if cost is not None else "unknown model"),
        ]


class Agent:
    """Drives one Claude conversation with tool use."""

    MAX_JSON_RETRIES = 2

    def __init__(
        self, client: anthropic.Anthropic, settings: Settings, tools: ToolRegistry, console: Console
    ):
        self.client = client
        self.settings = settings
        self.tools = tools
        self.console = console
        self.conversation = Conversation()
        self.usage = UsageTracker()
        # Identical bytes on every request, so the API serves it from cache.
        self._system = [{"type": "text", "text": SYSTEM_PROMPT, "cache_control": {"type": "ephemeral"}}]

    # -- public API -------------------------------------------------------

    def ask(self, user_text: str) -> None:
        """Handle one user message, including any number of tool rounds."""
        self.conversation.add_user(user_text)
        while self._step():
            pass

    # -- the loop ---------------------------------------------------------

    def _step(self) -> bool:
        """Make one request and act on it. Returns True if another round is needed."""
        response = self._request_with_retry()
        self.usage.add(response.usage)
        tool_calls = [block for block in response.content if isinstance(block, BetaToolUseBlock)]

        # Replies that must not enter the history: they may hold a half-finished tool call.
        if response.stop_reason == "refusal":
            self._report_refusal(response)
            return False
        if response.stop_reason == "max_tokens" and tool_calls:
            self.console.error("\n[reply was cut off before the tool call completed]")
            return False

        self.conversation.add_assistant(response.content)

        if response.stop_reason == "pause_turn":
            return True  # a server-side pause; send the turn again as-is
        if not tool_calls:
            return False  # a plain text answer: the turn is complete

        self.conversation.add_user(self._execute(tool_calls))
        return True

    def _request_with_retry(self) -> BetaMessage:
        """Retry when a streamed tool input was unparseable JSON (nothing was appended yet)."""
        for attempt in range(self.MAX_JSON_RETRIES + 1):
            try:
                return self._request()
            except ValueError:
                if attempt == self.MAX_JSON_RETRIES:
                    raise
                self.console.info("(retrying a malformed tool call)")
        raise AssertionError("unreachable")

    def _request(self) -> BetaMessage:
        """Stream one response, printing text as it arrives. Returns the final message."""
        started_text = False
        json_chunks = 0
        with self.client.beta.messages.stream(
            model=self.settings.model,
            max_tokens=self.settings.max_tokens,
            system=self._system,
            tools=self.tools.schemas(),
            messages=self.conversation.messages,
            thinking={"type": "adaptive"},
            output_config={"effort": self.settings.effort},
            betas=[FALLBACK_BETA],
            fallbacks="default",
        ) as stream:
            for event in stream:
                if event.type == "text":
                    if not started_text:
                        self.console.write()
                        started_text = True
                    self.console.assistant_text(event.text)
                elif event.type == "content_block_start" and event.content_block.type == "tool_use":
                    self.console.tool_pending(event.content_block.name)
                    json_chunks = 0
                elif event.type == "input_json":
                    json_chunks += 1
                    if json_chunks % PROGRESS_EVERY == 0:
                        self.console.tool_progress()
            return stream.get_final_message()

    def _execute(self, tool_calls: list[BetaToolUseBlock]) -> list[dict[str, Any]]:
        """Run every requested tool and build the tool_result blocks to send back."""
        results = []
        for call in tool_calls:
            self.console.tool_call(call.name, call.input)
            result = self.tools.execute(call.name, call.input)
            self.console.tool_result(result.content, result.is_error)
            block: dict[str, Any] = {"type": "tool_result", "tool_use_id": call.id, "content": result.content}
            if result.is_error:
                block["is_error"] = True
            results.append(block)
        return results

    def _report_refusal(self, response: BetaMessage) -> None:
        details = response.stop_details
        category = f" ({details.category})" if details and details.category else ""
        self.console.error(f"\n[the model declined this request{category}]")
