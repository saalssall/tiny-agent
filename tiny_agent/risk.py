"""Decide whether a shell command can run without asking the user.

Two layers, in order:

1. Hard rules. A short list of patterns that always mean "ask": deleting,
   privilege escalation, network access, installing software, rewriting git
   state, redirecting into files, leaving the workspace, and so on. The rules
   are the floor; nothing below can override them.
2. A trained classifier (see ml/train.py). For commands no rule matched, the
   model estimates the probability that the command is safe. Only a confident
   "safe" verdict auto-approves; anything else asks.

If scikit-learn or the model file is missing, layer 2 is skipped and every
command that passes the rules still asks, which is the pre-classifier behaviour.
"""

from __future__ import annotations

import re
import warnings
from dataclasses import dataclass
from pathlib import Path

DEFAULT_MODEL_PATH = Path(__file__).with_name("risk_model.joblib")

# Matches the start of a command word: beginning of string, or after a separator.
_CMD = r"(?:^|[\s;&|(])"

# (pattern, reason shown to the user). Order does not matter; the first match wins.
RULES: list[tuple[re.Pattern[str], str]] = [
    (re.compile(r"\|\s*(ba|z|da|k|c|tc)?sh(?=\s|$)"), "pipes text into a shell"),
    (re.compile(_CMD + r"(sudo|su|doas)(?=\s|$)"), "runs with elevated privileges"),
    (re.compile(_CMD + r"(rm|rmdir|shred|unlink)(?=\s|$)"), "deletes files"),
    (re.compile(_CMD + r"(dd|mkfs(\.\w+)?|fdisk|parted|diskutil|wipefs)(?=\s|$)"), "can destroy disks"),
    (re.compile(_CMD + r"(shutdown|reboot|halt|poweroff)(?=\s|$)"), "affects the whole machine"),
    (
        re.compile(_CMD + r"(mv|cp|rsync|chmod|chown|chgrp|ln|truncate|tee)(?=\s|$)"),
        "moves, copies or changes files",
    ),
    (re.compile(_CMD + r"(sed|perl)\s+-\S*i"), "edits files in place"),
    (
        re.compile(_CMD + r"(curl|wget|nc|ncat|netcat|ssh|scp|sftp|ftp|telnet|ngrok)(?=\s|$)"),
        "reaches the network",
    ),
    (
        re.compile(
            _CMD + r"(pip3?|uv|pipx|npm|npx|yarn|pnpm|brew|apt|apt-get|yum|dnf|pacman|cargo|gem|go)\s+"
            r"(install|uninstall|remove|purge|add|i|get|update|upgrade)(?=\s|$)"
        ),
        "installs or removes software",
    ),
    (re.compile(_CMD + r"npx(?=\s|$)"), "downloads and runs a package"),
    (
        re.compile(
            _CMD + r"git\s+(push|reset|clean|checkout|switch|restore|rebase|merge|cherry-pick|revert|"
            r"commit|add|rm|mv|stash|filter-branch|gc|config|branch\s+-[dDmM]|tag\s+-d)(?=\s|$)"
        ),
        "changes git state",
    ),
    (
        re.compile(
            _CMD + r"gh\s+(pr\s+(merge|close|create)|repo\s+(delete|create|fork)|release\s+(create|delete)|"
            r"issue\s+(close|create|delete)|secret|api\s+.*-X)"
        ),
        "changes GitHub state",
    ),
    (re.compile(_CMD + r"(kill|killall|pkill)(?=\s|$)"), "kills processes"),
    (re.compile(_CMD + r"(eval|exec|source)(?=\s|$)"), "executes arbitrary code"),
    (re.compile(r"(?:^|[;&|(])\s*\.\s"), "sources a file into the shell"),
    (
        re.compile(_CMD + r"(python3?|node|ruby|perl|php|bash|sh|zsh|deno|bun)\s+(-c|-e|-r|-)(?=\s|$)"),
        "runs inline code",
    ),
    (
        re.compile(
            _CMD + r"(python3?|node|ruby|perl|php|bash|sh|zsh|deno|bun|osascript)\s+"
            r"\S+\.(py|js|ts|rb|pl|php|sh|zsh|scpt)(?=\s|$)"
        ),
        "runs a script",
    ),
    (
        re.compile(_CMD + r"(make|cmake|ninja|gradle|mvn|ant|rake|just)(?=\s|$)"),
        "runs a build tool with arbitrary targets",
    ),
    (
        re.compile(_CMD + r"find\b.*\s-(delete|exec|execdir|ok|okdir)(?=\s|$)"),
        "find with a destructive action",
    ),
    (re.compile(_CMD + r"xargs(?=\s|$)"), "feeds results into another command"),
    (
        re.compile(
            _CMD + r"(crontab|launchctl|systemctl|service|defaults|osascript|open|pbcopy|say)(?=\s|$)"
        ),
        "changes system or desktop state",
    ),
    (
        re.compile(
            _CMD
            + r"(docker|docker-compose|podman|kubectl|helm|terraform|aws|gcloud|az|heroku|fly|vercel|netlify)"
            r"\s+(compose\s+)?(rm|rmi|prune|delete|destroy|apply|run|exec|down|up|stop|kill|system|push|"
            r"deploy|create|update|scale|cp|set|sync|terminate-instances|--prod)(?=\s|$)"
        ),
        "changes infrastructure",
    ),
    (
        re.compile(_CMD + r"(env|printenv|export|unset|set|history)(?=\s|$)"),
        "reads or changes the environment",
    ),
    (re.compile(r"\$\{?[A-Za-z_]*(KEY|TOKEN|SECRET|PASS|PASSWORD|CREDENTIAL)"), "references a secret"),
    (
        re.compile(
            r"(\.env|API\.KEY|id_rsa|id_ed25519|\.aws/|\.netrc|\.pgpass|/etc/shadow|/etc/passwd|\.ssh/|find-generic-password)"
        ),
        "touches a secrets file",
    ),
    (
        re.compile(r"(^|[\s=])~(?=/|\s|$)|(^|[\s=])/(?!dev/null)|(^|[\s/=])\.\.(?=/|\s|$)"),
        "touches paths outside the workspace",
    ),
    (re.compile(r":\(\)\s*\{"), "fork bomb"),
    (re.compile(r"\$\(|`"), "uses command substitution"),
    (re.compile(_CMD + r"(base64\s+(-d|--decode)|xxd\s+-r)"), "decodes a hidden payload"),
]

# Redirections that only silence output and never write a file.
_HARMLESS_REDIRECTS = re.compile(r"[12&]?>\s*/dev/null|[12]>&[12]")
_SEPARATORS = re.compile(r"\s*(?:\|\||&&|;|\|)\s*")


@dataclass(frozen=True)
class Verdict:
    risky: bool
    reason: str
    p_safe: float | None = None  # model probability, when the model was consulted


def match_rule(command: str) -> str | None:
    """Return the reason from the first hard rule that matches, or None."""
    for pattern, reason in RULES:
        if pattern.search(command):
            return reason
    if ">" in _HARMLESS_REDIRECTS.sub("", command):
        return "redirects output into a file"
    return None


def split_segments(command: str) -> list[str]:
    """Split on ; && || | so every sub-command is judged on its own."""
    return [part for part in _SEPARATORS.split(command) if part]


class RiskModel:
    """A thin wrapper around the trained scikit-learn pipeline."""

    def __init__(self, pipeline, threshold: float):
        self.pipeline = pipeline
        self.threshold = threshold

    @classmethod
    def load(cls, path: Path = DEFAULT_MODEL_PATH) -> RiskModel | None:
        """Load the model, or return None if scikit-learn or the file is unavailable."""
        if not path.is_file():
            return None
        try:
            import joblib
        except ImportError:
            return None
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")  # tolerate minor scikit-learn version differences
                bundle = joblib.load(path)
            return cls(bundle["pipeline"], float(bundle["threshold"]))
        except Exception:  # any load problem simply disables the model
            return None

    def p_safe(self, command: str) -> float:
        """Probability that a single command segment is safe."""
        classes = list(self.pipeline.classes_)
        return float(self.pipeline.predict_proba([command])[0][classes.index(0)])


class CommandGate:
    """Combines the hard rules and the model into one verdict per command."""

    def __init__(self, model: RiskModel | None):
        self.model = model

    @property
    def enabled(self) -> bool:
        return self.model is not None

    def assess(self, command: str) -> Verdict:
        reason = match_rule(command)
        if reason:
            return Verdict(risky=True, reason=f"asking because it {reason}")
        if self.model is None:
            return Verdict(risky=True, reason="asking (no classifier available)")

        segments = split_segments(command) or [command]
        p_safe = min(self.model.p_safe(segment) for segment in segments)
        if p_safe >= self.model.threshold:
            return Verdict(risky=False, reason=f"auto-approved, looks safe ({p_safe:.0%})", p_safe=p_safe)
        return Verdict(risky=True, reason=f"asking, only {p_safe:.0%} sure it is safe", p_safe=p_safe)
