#!/usr/bin/env python3
"""Launch tiny-agent. See README.md for usage."""

from pathlib import Path

from tiny_agent.cli import main

if __name__ == "__main__":
    raise SystemExit(main(key_search_dir=Path(__file__).resolve().parent))
