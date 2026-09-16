"""tiny-agent: a small Claude-powered agent for your terminal."""

from importlib.metadata import PackageNotFoundError, version

from .agent import Agent
from .config import APP_NAME, Settings

try:
    __version__ = version("tiny-agent")
except PackageNotFoundError:  # running from a checkout without installing
    __version__ = "0.0.0+local"

__all__ = ["APP_NAME", "Agent", "Settings", "__version__"]
