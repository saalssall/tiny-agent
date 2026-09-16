# tiny-agent

A small Claude-powered agent that lives in your terminal. It can list, read,
write and edit files, and run shell commands, all confined to one workspace
directory. Every shell command is shown to you and waits for a `y` before it
runs.

## Install

The project uses [uv](https://docs.astral.sh/uv/). From a checkout:

```bash
uv sync                  # creates .venv and installs the app
uv run tiny-agent        # run it
```

Or install the command globally without cloning into a venv:

```bash
uv tool install git+https://github.com/saalssall/tiny-agent
tiny-agent --version
```

Provide an API key either way:

- `export ANTHROPIC_API_KEY=sk-ant-...`, or
- put the raw key in a file called `API.KEY` in the directory you launch from
  (it is git-ignored in this repo).

## Run

```bash
tiny-agent                       # work in the current directory
tiny-agent ~/some/project        # work in another directory
tiny-agent --yolo                # run every shell command without asking
tiny-agent --always-ask          # ask for every command, ignore the classifier
tiny-agent --effort xhigh        # low | medium | high | xhigh | max
tiny-agent --model claude-sonnet-5
tiny-agent --verbose             # log requests, stop reasons and tool calls to stderr
```

Prefix with `uv run` if you have not installed the tool globally. `python main.py`
and `python -m tiny_agent` also work from a checkout.

Inside the chat: `/help` lists commands, `/cost` shows tokens and estimated
spend, `/clear` forgets the conversation, `/quit` exits.

## Project layout

```
main.py                 convenience launcher for a checkout
tiny_agent/
  __main__.py           `python -m tiny_agent`
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
text or the per-turn round cap (50) is reached. If a request fails or you press
Ctrl-C, the conversation rolls back to the last complete turn so the history
never gets out of sync.

Shell commands run with credential-looking environment variables (`*KEY*`,
`*TOKEN*`, `*SECRET*`, `*PASSWORD*`, ...) removed, so a command cannot read the
API key even if it were approved.

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

To change the behaviour, edit the templates or the rules, then:

```bash
uv sync --group ml
uv run python ml/make_dataset.py
uv run python ml/train.py
uv run pytest
```

## Development

```bash
uv sync --all-groups            # app + dev tools + scikit-learn for retraining
uv run ruff format .            # format
uv run ruff check .             # lint
uv run mypy                     # strict type check
uv run coverage run -m pytest && uv run coverage report   # tests, 85% minimum
uv run pre-commit install       # optional: run the checks on every commit
```

CI runs the same commands on every push and pull request, tests on Python 3.10
through 3.13, re-evaluates the risk classifier, and smoke-tests the built wheel.
Dependencies are locked in `uv.lock`; Dependabot proposes updates weekly.

To cut a release, update `CHANGELOG.md` and the version in `pyproject.toml`, then:

```bash
git tag v0.3.0 && git push origin v0.3.0
```

The release workflow re-runs the checks, builds the wheel and source
distribution, and publishes a GitHub Release with generated notes.

## Adding a tool

Subclass `WorkspaceTool` (or `Tool` if you need other dependencies) in
`tiny_agent/tools.py`, declare `name`, `description`, `parameters` and
`required`, implement `run` with keyword parameters matching the schema, then add an instance to
the list in `build_app` in `tiny_agent/cli.py`. Validation, error reporting and
output truncation are handled for you.
