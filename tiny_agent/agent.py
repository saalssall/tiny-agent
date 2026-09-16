"""The agent loop: send the conversation, stream the reply, run tools, repeat."""

from __future__ import annotations

from typing import Any

import anthropic

from .config import PRICES, SYSTEM_PROMPT, Settings
from .console import Console
from .tools import ToolRegistry

# Retry the same request server-side on a fallback model if a safety classifier declines.
FALLBACK_BETA = "server-side-fallback-2026-07-01"


class Conversation:
    """The message history sent to the API, with checkpoint/rollback for error recovery."""

    def __init__(self) -> None:
        self.messages: list[dict[str, Any]] = []

    def add_user(self, content: Any) -> None:
        self.messages.append({"role": "user", "content": content})

    def add_assistant(self, content: Any) -> None:
        self.messages.append({"role": "assistant", "content": content})

    def drop_last(self) -> None:
        if self.messages:
            self.messages.pop()

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
        total = (self.input * p_in + self.output * p_out
                 + self.cache_write * p_write + self.cache_read * p_read)
        return total / 1_000_000

    def rows(self, model: str) -> list[tuple[str, str]]:
        rows = [
            ("input tokens", f"{self.input:,}"),
            ("output tokens", f"{self.output:,}"),
            ("cache writes", f"{self.cache_write:,}"),
            ("cache reads", f"{self.cache_read:,}"),
        ]
        cost = self.cost_usd(model)
        rows.append(("estimated cost", f"${cost:.4f}" if cost is not None else "unknown model"))
        return rows


class Agent:
    """Drives one Claude conversation with tool use."""

    MAX_JSON_RETRIES = 2

    def __init__(self, client: anthropic.Anthropic, settings: Settings,
                 tools: ToolRegistry, console: Console):
        self.client = client
        self.settings = settings
        self.tools = tools
        self.console = console
        self.conversation = Conversation()
        self.usage = UsageTracker()
        # A cached system prompt: identical bytes every request, so it is served from cache.
        self._system = [{"type": "text", "text": SYSTEM_PROMPT, "cache_control": {"type": "ephemeral"}}]

    # -- public API -------------------------------------------------------

    def ask(self, user_text: str) -> None:
        """Handle one user message, including any number of tool rounds."""
        self.conversation.add_user(user_text)
        json_retries = 0

        while True:
            try:
                response = self._request()
            except ValueError:
                # Eager input streaming: a tool's JSON was unparseable. Nothing was
                # appended to the history, so simply re-issue the request (bounded).
                json_retries += 1
                if json_retries > self.MAX_JSON_RETRIES:
                    raise
                self.console.info("(retrying a malformed tool call)")
                continue
            json_retries = 0

            self.usage.add(response.usage)
            self.conversation.add_assistant(response.content)

            if response.stop_reason == "pause_turn":
                continue  # a server-side pause; just send the turn again

            if response.stop_reason == "refusal":
                self._report_refusal(response)
                self.conversation.drop_last()  # may hold a half-finished tool call
                return

            tool_calls = [block for block in response.content if block.type == "tool_use"]
            if not tool_calls:
                return  # a plain text answer: the turn is complete

            if response.stop_reason == "max_tokens":
                self.console.error("\n[reply was cut off before the tool call completed]")
                self.conversation.drop_last()
                return

            self.conversation.add_user(self._execute(tool_calls))

    # -- internals --------------------------------------------------------

    def _request(self) -> Any:
        """Stream one response, printing text as it arrives. Returns the final message."""
        started_text = False
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
            json_chunks = 0
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
                    if json_chunks % 25 == 0:
                        self.console.tool_progress()
            return stream.get_final_message()

    def _execute(self, tool_calls: list[Any]) -> list[dict[str, Any]]:
        """Run every requested tool and build the tool_result blocks to send back."""
        results = []
        for call in tool_calls:
            self.console.tool_call(call.name, call.input)
            result = self.tools.execute(call.name, call.input)
            self.console.tool_result(result.content, result.is_error)
            block: dict[str, Any] = {
                "type": "tool_result",
                "tool_use_id": call.id,
                "content": result.content,
            }
            if result.is_error:
                block["is_error"] = True
            results.append(block)
        return results

    def _report_refusal(self, response: Any) -> None:
        details = response.stop_details
        category = f" ({details.category})" if details and details.category else ""
        self.console.error(f"\n[the model declined this request{category}]")
