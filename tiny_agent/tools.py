"""Tools the model can call. Each is a small class; the registry ties them together."""

from __future__ import annotations

import subprocess
from abc import ABC, abstractmethod
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any, ClassVar

from .workspace import Workspace, WorkspaceError

# Map JSON-schema type names to the Python types we accept for them.
_JSON_TYPES: dict[str, type | tuple[type, ...]] = {
    "string": str,
    "boolean": bool,
    "integer": int,
    "number": (int, float),
}


@dataclass(frozen=True)
class ToolResult:
    content: str
    is_error: bool = False


class Tool(ABC):
    """Base class for a callable tool. Subclasses declare a schema and implement run()."""

    name: ClassVar[str]
    description: ClassVar[str]
    # property name -> {"type": ..., "description": ...}
    parameters: ClassVar[dict[str, dict[str, str]]]
    required: ClassVar[tuple[str, ...]] = ()

    def schema(self) -> dict[str, Any]:
        """The tool definition sent to the API."""
        return {
            "name": self.name,
            "description": self.description,
            "input_schema": {
                "type": "object",
                "properties": self.parameters,
                "required": list(self.required),
                "additionalProperties": False,
            },
            # Stream large inputs (file contents) as they are generated. In exchange
            # the server no longer validates them, so validate() must run first.
            "eager_input_streaming": True,
        }

    def validate(self, args: object) -> str | None:
        """Return a problem description if args don't fit the schema, else None."""
        if not isinstance(args, dict):
            return "input is not an object"
        for key in self.required:
            if key not in args:
                return f"missing required field '{key}'"
        for key, value in args.items():
            spec = self.parameters.get(key)
            if spec is None:
                return f"unexpected field '{key}'"
            expected = _JSON_TYPES[spec["type"]]
            if isinstance(value, bool) and expected is not bool:
                return f"field '{key}' should be {spec['type']}"
            if not isinstance(value, expected):
                return f"field '{key}' should be {spec['type']}"
        return None

    @abstractmethod
    def run(self, **kwargs: Any) -> str:
        """Execute the tool and return text for the model. Raise on failure."""


# --------------------------------------------------------------------------- #
# File tools
# --------------------------------------------------------------------------- #


class WorkspaceTool(Tool):
    """A tool whose only dependency is the workspace."""

    def __init__(self, workspace: Workspace):
        self.workspace = workspace


class ListFilesTool(WorkspaceTool):
    name = "list_files"
    description = (
        "List files and directories under a path in the workspace. "
        "Noise directories (.git, node_modules, .venv, __pycache__) are skipped."
    )
    parameters = {
        "path": {"type": "string", "description": "Directory relative to the workspace. Defaults to '.'."},
        "recursive": {"type": "boolean", "description": "Walk subdirectories too. Defaults to false."},
    }

    def run(self, path: str = ".", recursive: bool = False) -> str:
        return "\n".join(self.workspace.list(path, recursive)) or "(empty)"


class ReadFileTool(WorkspaceTool):
    name = "read_file"
    description = "Read a text file and return its contents with line numbers."
    parameters = {"path": {"type": "string", "description": "File path relative to the workspace."}}
    required = ("path",)

    def run(self, path: str) -> str:
        lines = self.workspace.read(path).splitlines()
        return "\n".join(f"{i:5d}\t{line}" for i, line in enumerate(lines, 1)) or "(empty file)"


class WriteFileTool(WorkspaceTool):
    name = "write_file"
    description = "Create or overwrite a text file. Parent directories are created as needed."
    parameters = {
        "path": {"type": "string", "description": "File path relative to the workspace."},
        "content": {"type": "string", "description": "Full new contents of the file."},
    }
    required = ("path", "content")

    def run(self, path: str, content: str) -> str:
        existed = self.workspace.write(path, content)
        verb = "Overwrote" if existed else "Created"
        return f"{verb} {path} ({len(content.splitlines())} lines)"


class EditFileTool(WorkspaceTool):
    name = "edit_file"
    description = (
        "Replace one exact occurrence of old_text with new_text in a file. "
        "old_text must appear exactly once; include enough context to make it unique."
    )
    parameters = {
        "path": {"type": "string", "description": "File path relative to the workspace."},
        "old_text": {"type": "string", "description": "Exact text to find (must be unique in the file)."},
        "new_text": {"type": "string", "description": "Replacement text."},
    }
    required = ("path", "old_text", "new_text")

    def run(self, path: str, old_text: str, new_text: str) -> str:
        self.workspace.replace_once(path, old_text, new_text)
        return f"Edited {path}"


# --------------------------------------------------------------------------- #
# Shell tool
# --------------------------------------------------------------------------- #


class RunCommandTool(Tool):
    name = "run_command"
    description = (
        "Run a shell command in the workspace root and return stdout, stderr and the "
        "exit code. The user must approve each command before it runs."
    )
    parameters = {"command": {"type": "string", "description": "The shell command to run."}}
    required = ("command",)

    def __init__(self, workspace: Workspace, approve: Callable[[str], bool], timeout: int = 120):
        """approve(command) is asked before every run; return False to refuse."""
        self.workspace = workspace
        self.approve = approve
        self.timeout = timeout

    def run(self, command: str) -> str:
        if not self.approve(command):
            return "The user declined to run this command."
        try:
            proc = subprocess.run(
                command,
                shell=True,
                cwd=self.workspace.root,
                capture_output=True,
                text=True,
                timeout=self.timeout,
                check=False,
            )
        except subprocess.TimeoutExpired:
            return f"Command timed out after {self.timeout} seconds."

        parts = []
        if proc.stdout:
            parts.append(proc.stdout.rstrip())
        if proc.stderr:
            parts.append("[stderr]\n" + proc.stderr.rstrip())
        parts.append(f"[exit code {proc.returncode}]")
        return "\n".join(parts)


# --------------------------------------------------------------------------- #
# Registry
# --------------------------------------------------------------------------- #


class ToolRegistry:
    """Holds the available tools, exposes their schemas, and executes calls safely."""

    def __init__(self, tools: list[Tool], max_output: int = 40_000):
        self._tools = {tool.name: tool for tool in tools}
        self.max_output = max_output

    def schemas(self) -> list[dict[str, Any]]:
        return [tool.schema() for tool in self._tools.values()]

    def execute(self, name: str, args: object) -> ToolResult:
        tool = self._tools.get(name)
        if tool is None:
            return ToolResult(f"unknown tool '{name}'", is_error=True)

        problem = tool.validate(args)
        if problem:
            return ToolResult(f"invalid input: {problem}", is_error=True)

        try:
            output = tool.run(**args)  # type: ignore[arg-type]  (validated above)
        except WorkspaceError as exc:
            return ToolResult(str(exc), is_error=True)
        except Exception as exc:
            return ToolResult(f"{type(exc).__name__}: {exc}", is_error=True)

        if len(output) > self.max_output:
            extra = len(output) - self.max_output
            output = output[: self.max_output] + f"\n... [truncated, {extra} more characters]"
        return ToolResult(output)
