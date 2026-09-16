"""The interactive chat loop and slash commands."""

from __future__ import annotations

from pathlib import Path
from typing import Callable

import anthropic

from .agent import Agent
from .config import APP_NAME, ConfigError, Settings, parse_settings
from .console import Console
from .tools import (EditFileTool, ListFilesTool, ReadFileTool, RunCommandTool,
                    ToolRegistry, WriteFileTool)
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
        self.commands: dict[str, Callable[[], bool]] = {
            "/help": self._cmd_help,
            "/cost": self._cmd_cost,
            "/clear": self._cmd_clear,
            "/quit": self._cmd_quit,
            "/exit": self._cmd_quit,
            "/q": self._cmd_quit,
        }

    def run(self) -> None:
        self.console.banner(self.settings)
        while True:
            try:
                text = self.console.ask(f"\nyou › ")
            except (EOFError, KeyboardInterrupt):
                self.console.write()
                break
            if not text:
                continue
            if text in self.commands:
                if not self.commands[text]():  # a command returns False to exit
                    break
                continue
            if text.startswith("/"):
                self.console.warn(f"unknown command {text!r}; type /help for the list")
                continue
            self._chat(text)

        self.console.write(self.console.dim("\nsession usage"))
        self.console.table(self.agent.usage.rows(self.settings.model))

    # -- one message to the agent, with friendly error handling ---------------

    def _chat(self, text: str) -> None:
        checkpoint = self.agent.conversation.checkpoint()
        try:
            self.agent.ask(text)
            self.console.write()
        except KeyboardInterrupt:
            self.console.warn("\n[interrupted]")
            self.agent.conversation.rollback(checkpoint)
        except anthropic.AuthenticationError:
            self.console.error("\nAuthentication failed. Check your API key.")
            self.agent.conversation.rollback(checkpoint)
        except anthropic.RateLimitError:
            self.console.error("\nRate limited. Wait a moment and try again.")
            self.agent.conversation.rollback(checkpoint)
        except anthropic.APIStatusError as exc:
            self.console.error(f"\nAPI error {exc.status_code}: {exc.message}")
            self.agent.conversation.rollback(checkpoint)
        except anthropic.APIConnectionError:
            self.console.error("\nNetwork error. Check your connection and try again.")
            self.agent.conversation.rollback(checkpoint)

    # -- slash commands -------------------------------------------------------

    def _cmd_help(self) -> bool:
        self.console.write(HELP_TEXT)
        return True

    def _cmd_cost(self) -> bool:
        self.console.table(self.agent.usage.rows(self.settings.model))
        return True

    def _cmd_clear(self) -> bool:
        self.agent.conversation.clear()
        self.console.info("conversation cleared")
        return True

    def _cmd_quit(self) -> bool:
        return False


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
