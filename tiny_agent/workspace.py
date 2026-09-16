"""A sandboxed view of one directory: every path is checked before use."""

from __future__ import annotations

import os
from pathlib import Path

SKIP_DIRS = frozenset(
    {
        ".git",
        ".venv",
        "venv",
        "node_modules",
        "__pycache__",
        ".mypy_cache",
        ".pytest_cache",
    }
)


class WorkspaceError(Exception):
    """A file operation was refused or failed inside the workspace."""


class Workspace:
    """File-system access confined to a single root directory."""

    def __init__(self, root: Path):
        self.root = root.resolve()

    def resolve(self, relative: str) -> Path:
        """Turn a model-supplied path into an absolute one, refusing escapes."""
        target = (self.root / relative).resolve()
        if target != self.root and self.root not in target.parents:
            raise WorkspaceError(f"path '{relative}' is outside the workspace")
        return target

    def list(self, relative: str = ".", recursive: bool = False, limit: int = 400) -> list[str]:
        base = self.resolve(relative)
        if not base.is_dir():
            raise WorkspaceError(f"'{relative}' is not a directory")

        if not recursive:
            entries = sorted(base.iterdir(), key=lambda p: (not p.is_dir(), p.name.lower()))
            return [e.name + ("/" if e.is_dir() else "") for e in entries if e.name not in SKIP_DIRS]

        found: list[str] = []
        for dirpath, dirnames, filenames in os.walk(base):
            dirnames[:] = sorted(d for d in dirnames if d not in SKIP_DIRS)
            rel_dir = Path(dirpath).relative_to(self.root)
            for name in sorted(filenames):
                found.append(str(rel_dir / name))
                if len(found) >= limit:
                    found.append(f"... truncated at {limit} entries")
                    return found
        return found

    def read(self, relative: str) -> str:
        target = self.resolve(relative)
        if not target.is_file():
            raise WorkspaceError(f"'{relative}' does not exist or is not a file")
        return target.read_text(encoding="utf-8", errors="replace")

    def write(self, relative: str, content: str) -> bool:
        """Write the file, creating parents. Returns True if it already existed."""
        target = self.resolve(relative)
        target.parent.mkdir(parents=True, exist_ok=True)
        existed = target.exists()
        target.write_text(content, encoding="utf-8")
        return existed

    def replace_once(self, relative: str, old: str, new: str) -> None:
        """Replace exactly one occurrence of old with new, or raise."""
        text = self.read(relative)
        count = text.count(old)
        if count == 0:
            raise WorkspaceError("old_text was not found in the file")
        if count > 1:
            raise WorkspaceError(f"old_text appears {count} times; include more context to make it unique")
        self.resolve(relative).write_text(text.replace(old, new, 1), encoding="utf-8")
