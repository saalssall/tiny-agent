"""The interactive chat loop and slash commands."""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

import anthropic

from .agent import Agent
from .config import APP_NAME, ConfigError, Settings, parse_settings
from .console import Console
from .tools import EditFileTool, ListFilesTool, ReadFileTool, RunCommandTool, ToolRegistry, WriteFileTool
from .workspace import Workspace

HELP_TEXT = f"""\
  {APP_NAME} works inside one directory. Ask it to read, write or edit files,
  or to run commands. Shell commands wait for your y/N approval.

  /help    show this message
  /cost    show token usage and estimated cost so far
  /clear   forget the conversation and start fresh
  /quit    exit (Ctrl-D or Ctrl-C at the prompt also works)
"""


class ChatApp:
    """Reads user input, dispatches slash commands, and hands everything else to the agent."""

    def __init__(self, settings: Settings, console: Console, agent: Agent):
        self.settings = settings
        self.console = console
        self.agent = agent
        self.running = False
        self.commands: dict[str, Callable[[], None]] = {
            "/help": self._cmd_help,
            "/cost": self._cmd_cost,
            "/clear": self._cmd_clear,
            "/quit": self._cmd_quit,
            "/exit": self._cmd_quit,
            "/q": self._cmd_quit,
        }

    def run(self) -> None:
        self.console.banner(self.settings)
        self.running = True
        while self.running:
            try:
                text = self.console.ask("\nyou › ")
            except (EOFError, KeyboardInterrupt):
                self.console.write()
                break
            self._handle(text)

        self.console.write(self.console.dim("\nsession usage"))
        self._cmd_cost()

    def _handle(self, text: str) -> None:
        if not text:
            return
        if text in self.commands:
            self.commands[text]()
        elif text.startswith("/"):
            self.console.warn(f"unknown command {text!r}; type /help for the list")
        else:
            self._chat(text)

    # -- one message to the agent, with friendly error handling ---------------

    def _chat(self, text: str) -> None:
        checkpoint = self.agent.conversation.checkpoint()
        try:
            self.agent.ask(text)
            self.console.write()
        except (KeyboardInterrupt, anthropic.APIError) as exc:
            # Roll back so a half-finished turn never lingers in the history.
            self.agent.conversation.rollback(checkpoint)
            self.console.error("\n" + describe_error(exc))

    # -- slash commands -------------------------------------------------------

    def _cmd_help(self) -> None:
        self.console.write(HELP_TEXT)

    def _cmd_cost(self) -> None:
        self.console.table(self.agent.usage.rows(self.settings.model))

    def _cmd_clear(self) -> None:
        self.agent.conversation.clear()
        self.console.info("conversation cleared")

    def _cmd_quit(self) -> None:
        self.running = False


def describe_error(exc: BaseException) -> str:
    """A one-line, human-friendly explanation of a failed turn."""
    if isinstance(exc, KeyboardInterrupt):
        return "[interrupted]"
    if isinstance(exc, anthropic.AuthenticationError):
        return "Authentication failed. Check your API key."
    if isinstance(exc, anthropic.RateLimitError):
        return "Rate limited. Wait a moment and try again."
    if isinstance(exc, anthropic.APIStatusError):
        return f"API error {exc.status_code}: {exc.message}"
    if isinstance(exc, anthropic.APIConnectionError):
        return "Network error. Check your connection and try again."
    return f"Unexpected error: {exc}"


def build_app(settings: Settings, console: Console) -> ChatApp:
    """Wire the pieces together. Kept separate so tests can build an app without a TTY."""
    workspace = Workspace(settings.workspace)

    def approve(command: str) -> bool:
        console.command_preview(command)
        return settings.auto_approve or console.confirm("run this?")

    tools = ToolRegistry(
        [
            ListFilesTool(workspace),
            ReadFileTool(workspace),
            WriteFileTool(workspace),
            EditFileTool(workspace),
            RunCommandTool(workspace, approve, timeout=settings.command_timeout),
        ],
        max_output=settings.max_tool_output,
    )
    client = anthropic.Anthropic(api_key=settings.api_key)
    agent = Agent(client, settings, tools, console)
    return ChatApp(settings, console, agent)


def main(argv: list[str] | None = None, key_search_dir: Path | None = None) -> int:
    """Program entry point. Returns the process exit code."""
    console = Console()
    try:
        settings = parse_settings(argv, key_search_dir or Path.cwd())
    except ConfigError as exc:
        console.error(str(exc))
        return 1
    build_app(settings, console).run()
    return 0
