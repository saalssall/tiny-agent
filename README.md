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
.venv/bin/python main.py --yolo          # run every shell command without asking
.venv/bin/python main.py --always-ask    # ask for every command, ignore the classifier
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
  cli.py                ChatApp: the read-eval loop, slash commands, Approver
  risk.py               command risk gate: hard rules + trained classifier
  risk_model.json       the trained classifier as plain weights (from ml/train.py)
ml/make_dataset.py      generates the labelled command dataset from templates
ml/commands.csv         the dataset: command, label, template id
ml/train.py             trains, evaluates and saves the classifier
tests/test_tools.py     unit tests for the workspace and tools
tests/test_agent.py     unit tests for the agent loop, using scripted responses
tests/test_risk.py      unit tests for the rules, the gate, and the shipped model
```

How a turn flows: `ChatApp` reads a line and calls `Agent.ask`. The agent
streams a response, printing text as it arrives. If the model asked for tools,
`ToolRegistry` validates each input against its schema, runs it, and the
results go back to the model. This repeats until the model answers in plain
text. If a request fails or you press Ctrl-C, the conversation rolls back to
the last complete turn so the history never gets out of sync.

## Command risk classifier

Every shell command the model wants to run passes through a two-layer gate
before it can skip the y/N prompt.

1. **Hard rules** in `tiny_agent/risk.py`. About thirty patterns that always
   mean "ask": deleting, `sudo`, network tools, package installs, git commands
   that rewrite state, redirecting into files, in-place edits, running scripts
   or inline code, and any path that leaves the workspace. The rules are the
   floor. Nothing below can override them.
2. **A trained classifier.** For commands no rule matched, a model trained
   with scikit-learn (character n-gram and word TF-IDF into logistic
   regression) estimates the probability that the command is safe. Only a
   confident verdict, 85 percent or higher, auto-approves. Anything else asks,
   and the reason is shown either way.

The model ships as plain JSON weights in `tiny_agent/risk_model.json` and is
scored in pure Python, so the app needs no ML libraries at runtime and the file
loads on every Python version. With `--always-ask`, or if the file is missing,
the model layer is skipped and every command asks.

### How it was built

- `ml/make_dataset.py` expands about 390 command templates into roughly 800
  labelled commands. Each row keeps its template id.
- `ml/train.py` evaluates with 5-fold cross-validation **grouped by template**,
  so every held-out fold contains command shapes the model never saw. That is a
  deliberately hard test. It then reports what the deployed gate would do on
  those held-out commands and fails if any risky command would be auto-approved,
  which CI runs on every push. Finally it exports the weights to JSON and
  checks that the pure-Python scorer reproduces scikit-learn's probabilities
  on every training command before saving.
- Under that test the rules alone catch about 92 percent of risky commands, the
  gate auto-approves about a quarter of unfamiliar safe commands and zero risky
  ones. Familiar everyday commands such as `git status` or `pytest` are in the
  training data and auto-approve with high confidence.

To change the behaviour, edit the templates or the rules, then (scikit-learn
is needed only for this step):

```bash
uv pip install --python .venv/bin/python -r requirements-ml.txt
.venv/bin/python ml/make_dataset.py
.venv/bin/python ml/train.py
.venv/bin/python -m unittest
```

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
