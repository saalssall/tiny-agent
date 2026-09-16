"""Settings, defaults and constants."""

from __future__ import annotations

import argparse
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Literal, cast, get_args

APP_NAME = "tiny-agent"
DEFAULT_MODEL = "claude-opus-5"
Effort = Literal["low", "medium", "high", "xhigh", "max"]
EFFORT_LEVELS: tuple[Effort, ...] = get_args(Effort)
DEFAULT_EFFORT: Effort = "high"
KEY_FILE_NAME = "API.KEY"

# USD per million tokens: (input, output, cache write, cache read)
PRICES: dict[str, tuple[float, float, float, float]] = {
    "claude-opus-5": (5.00, 25.00, 6.25, 0.50),
    "claude-fable-5-1": (10.00, 50.00, 12.50, 0.25),
    "claude-sonnet-5": (2.00, 10.00, 2.50, 0.20),
    "claude-haiku-4-5": (1.00, 5.00, 1.25, 0.10),
}

SYSTEM_PROMPT = f"""\
You are {APP_NAME}, a capable software engineering assistant running in the user's terminal.
You have tools to inspect and modify files and to run shell commands inside a single
workspace directory. Paths you pass to tools are relative to that workspace.

Guidelines:
- Look before you leap: read a file before editing it, list a directory before assuming its layout.
- Prefer edit_file (exact string replacement) over rewriting whole files with write_file.
- Shell commands are shown to the user for approval; keep them short and purposeful.
- When you finish a task, give a brief plain-language summary of what changed.
- If something is ambiguous in a way that changes the outcome, ask rather than guess.
"""


class ConfigError(Exception):
    """Raised when the program cannot start with the given configuration."""


@dataclass(frozen=True)
class Settings:
    """Everything the application needs to run, resolved once at startup."""

    workspace: Path
    api_key: str
    model: str = DEFAULT_MODEL
    effort: Effort = DEFAULT_EFFORT
    auto_approve: bool = False
    always_ask: bool = False
    verbose: bool = False
    max_rounds: int = 50  # tool rounds allowed per user message before the agent stops
    max_tokens: int = 64_000
    command_timeout: int = 120
    max_tool_output: int = 40_000


def load_api_key(search_dir: Path) -> str | None:
    """Return the API key from the environment, or from API.KEY in search_dir."""
    key = os.environ.get("ANTHROPIC_API_KEY", "").strip()
    if key:
        return key
    key_file = search_dir / KEY_FILE_NAME
    if key_file.is_file():
        return key_file.read_text().strip() or None
    return None


def build_parser(version: str) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog=APP_NAME,
        description="A small Claude-powered agent that works inside one directory.",
    )
    parser.add_argument("--version", action="version", version=f"{APP_NAME} {version}")
    parser.add_argument(
        "--verbose", action="store_true", help="log requests, stop reasons and tool calls to stderr"
    )
    parser.add_argument(
        "workspace",
        nargs="?",
        default=".",
        help="directory the agent may read, edit and run commands in (default: current)",
    )
    parser.add_argument(
        "--yolo", action="store_true", help="run shell commands without asking for confirmation"
    )
    parser.add_argument(
        "--always-ask",
        action="store_true",
        help="disable the risk classifier so every shell command asks for confirmation",
    )
    parser.add_argument(
        "--model",
        default=os.environ.get("AGENT_MODEL", DEFAULT_MODEL),
        help=f"Claude model ID (default: {DEFAULT_MODEL})",
    )
    parser.add_argument(
        "--effort",
        default=os.environ.get("AGENT_EFFORT", DEFAULT_EFFORT),
        choices=EFFORT_LEVELS,
        help=f"how hard the model thinks (default: {DEFAULT_EFFORT})",
    )
    return parser


def parse_settings(argv: list[str] | None, key_search_dir: Path, version: str = "unknown") -> Settings:
    """Turn command-line arguments into validated Settings. Raises ConfigError."""
    args = build_parser(version).parse_args(argv)

    workspace = Path(args.workspace).expanduser().resolve()
    if not workspace.is_dir():
        raise ConfigError(f"'{args.workspace}' is not a directory.")

    api_key = load_api_key(key_search_dir)
    if not api_key:
        raise ConfigError(
            "No API key found.\n"
            "  Either:  export ANTHROPIC_API_KEY=sk-ant-...\n"
            f"  or put the key in a file named {KEY_FILE_NAME} next to main.py"
        )

    return Settings(
        workspace=workspace,
        api_key=api_key,
        model=args.model,
        effort=cast("Effort", args.effort),
        auto_approve=args.yolo,
        always_ask=args.always_ask,
        verbose=args.verbose,
    )
