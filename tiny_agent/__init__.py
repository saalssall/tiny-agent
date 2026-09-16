"""tiny-agent: a small Claude-powered agent for your terminal."""

from .config import APP_NAME, Settings
from .agent import Agent

__all__ = ["APP_NAME", "Settings", "Agent"]
