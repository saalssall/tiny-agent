"""All terminal input and output lives here, so the rest of the code never prints."""

from __future__ import annotations

import os
import sys
from collections.abc import Callable, Iterable
from typing import TextIO

from .config import APP_NAME, Settings


class Console:
    """A thin, colour-aware wrapper around print() and input()."""

    def __init__(
        self,
        out: TextIO | None = None,
        color: bool | None = None,
        input_fn: Callable[[str], str] | None = None,
    ):
        """out defaults to the current stdout; input_fn is injectable so tests need no terminal."""
        self.out = sys.stdout if out is None else out
        self.input_fn = input if input_fn is None else input_fn
        if color is None:
            color = self.out.isatty() and os.environ.get("NO_COLOR") is None
        self.color = color

    # -- styling ----------------------------------------------------------

    def _paint(self, code: str, text: str) -> str:
        return f"\033[{code}m{text}\033[0m" if self.color else text

    def dim(self, t: str) -> str:
        return self._paint("2", t)

    def bold(self, t: str) -> str:
        return self._paint("1", t)

    def cyan(self, t: str) -> str:
        return self._paint("36", t)

    def green(self, t: str) -> str:
        return self._paint("32", t)

    def yellow(self, t: str) -> str:
        return self._paint("33", t)

    def red(self, t: str) -> str:
        return self._paint("31", t)

    # -- basic output -----------------------------------------------------

    def write(self, text: str = "", end: str = "\n") -> None:
        self.out.write(text + end)
        self.out.flush()

    def info(self, text: str) -> None:
        self.write(self.dim(f"  {text}"))

    def warn(self, text: str) -> None:
        self.write(self.yellow(f"  {text}"))

    def error(self, text: str) -> None:
        self.write(self.red(f"  {text}"))

    def table(self, rows: Iterable[tuple[str, str]]) -> None:
        items = list(rows)
        width = max((len(label) for label, _ in items), default=0)
        for label, value in items:
            self.write(f"  {label.ljust(width)}  {value:>12}")

    # -- input ------------------------------------------------------------

    def ask(self, prompt: str) -> str:
        """Read a line from the user. EOFError / KeyboardInterrupt propagate."""
        return self.input_fn(self.bold(self.green(prompt))).strip()

    def confirm(self, question: str) -> bool:
        try:
            answer = self.input_fn(self.yellow(f"  {question} [y/N] ")).strip().lower()
        except EOFError:
            return False
        return answer in ("y", "yes")

    # -- agent-specific output --------------------------------------------

    def banner(self, settings: Settings) -> None:
        self.write(self.bold(APP_NAME) + self.dim(f"  ·  {settings.model}  ·  effort {settings.effort}"))
        self.write(self.dim(f"workspace: {settings.workspace}"))
        self.write(self.dim("type /help for commands"))
        if settings.auto_approve:
            self.warn("--yolo: shell commands run without asking")

    def assistant_text(self, chunk: str) -> None:
        """Print a streamed piece of the model's reply, no newline."""
        self.out.write(chunk)
        self.out.flush()

    def tool_pending(self, name: str) -> None:
        self.write(self.dim(f"\n  … preparing {name}"), end="")

    def tool_progress(self) -> None:
        self.write(self.dim("."), end="")

    def tool_call(self, name: str, args: object) -> None:
        self.write(f"\n  {self.cyan('▸ ' + name)} {self.dim(_summarise(args))}")

    def tool_result(self, content: str, is_error: bool) -> None:
        marker = self.red("  ✗ ") if is_error else self.green("  ✓ ")
        self.write(marker + self.dim(_first_line(content)))

    def command_preview(self, command: str) -> None:
        self.write("\n" + self.yellow("  $ ") + self.bold(command))


def _summarise(args: object, width: int = 70) -> str:
    if not isinstance(args, dict):
        return ""
    parts = []
    for key, value in args.items():
        text = value.replace("\n", "⏎") if isinstance(value, str) else repr(value)
        if len(text) > 40:
            text = text[:37] + "..."
        parts.append(f"{key}={text}")
    joined = " ".join(parts)
    return joined if len(joined) <= width else joined[: width - 3] + "..."


def _first_line(text: str, width: int = 90) -> str:
    stripped = text.strip()
    line = stripped.splitlines()[0] if stripped else ""
    return line if len(line) <= width else line[: width - 3] + "..."
