"""tiny-agent: a small Claude-powered agent for your terminal."""

from .agent import Agent
from .config import APP_NAME, Settings

__all__ = ["APP_NAME", "Agent", "Settings"]
