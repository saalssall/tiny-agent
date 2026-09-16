# tiny-agent

A small Claude-powered agent that lives in your terminal. It can list, read,
write and edit files, and run shell commands, all confined to one workspace
directory. Every shell command is shown to you and waits for a `y` before it
runs.

## Setup

```bash
uv venv .venv && uv pip install --python .venv/bin/python -r requirements.txt
# or: python3 -m venv .venv && .venv/bin/pip install -r requirements.txt
```

Provide an API key either way:

- `export ANTHROPIC_API_KEY=sk-ant-...`, or
- put the raw key in a file called `API.KEY` next to `main.py` (git-ignored).

## Run

```bash
.venv/bin/python main.py                 # work in the current directory
.venv/bin/python main.py ~/some/project  # work in another directory
.venv/bin/python main.py --yolo          # skip command confirmations
.venv/bin/python main.py --effort xhigh  # low | medium | high | xhigh | max
.venv/bin/python main.py --model claude-sonnet-5
```

Inside the chat: `/help` lists commands, `/cost` shows tokens and estimated
spend, `/clear` forgets the conversation, `/quit` exits.

## Project layout

```
main.py                 launcher
tiny_agent/
  config.py             Settings dataclass, CLI arguments, API-key lookup, prices
  console.py            Console: all colours, prompts and printing
  workspace.py          Workspace: file access confined to one root directory
  tools.py              Tool base class, the five tools, and ToolRegistry
  agent.py              Agent loop, Conversation history, UsageTracker
  cli.py                ChatApp: the read-eval loop and slash commands
tests/test_tools.py     unit tests for the workspace and tools
tests/test_agent.py     unit tests for the agent loop, using scripted responses
```

How a turn flows: `ChatApp` reads a line and calls `Agent.ask`. The agent
streams a response, printing text as it arrives. If the model asked for tools,
`ToolRegistry` validates each input against its schema, runs it, and the
results go back to the model. This repeats until the model answers in plain
text. If a request fails or you press Ctrl-C, the conversation rolls back to
the last complete turn so the history never gets out of sync.

## Tests

```bash
.venv/bin/python -m unittest
```

## Adding a tool

Subclass `WorkspaceTool` (or `Tool` if you need other dependencies) in
`tiny_agent/tools.py`, declare `name`, `description`, `parameters` and
`required`, implement `run(**kwargs)`, then add an instance to
the list in `build_app` in `tiny_agent/cli.py`. Validation, error reporting and
output truncation are handled for you.

## Development

Lint and formatting use [ruff](https://docs.astral.sh/ruff/); settings live in
`pyproject.toml`.

```bash
.venv/bin/pip install ruff
.venv/bin/ruff format .        # format
.venv/bin/ruff check .         # lint
.venv/bin/python -m unittest   # tests
```

## CI/CD

Two GitHub Actions workflows live in `.github/workflows/`:

- **CI** (`ci.yml`) runs on every push and pull request to `main`. It checks
  formatting and lint, then byte-compiles and runs the unit tests on Python
  3.10 through 3.13.
- **Release** (`release.yml`) runs when you push a tag like `v0.1.0`. It
  re-runs the tests, then publishes a GitHub Release with generated notes and
  a zip of the source.

```bash
git tag v0.1.0 && git push origin v0.1.0   # cut a release
```
