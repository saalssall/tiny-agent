# tiny-agent

A small Claude-powered coding agent that lives in your terminal. You chat with
it in plain language; it can list, read, write and edit files, and run shell
commands, all confined to one workspace directory. Shell commands pass through
a risk gate: safe-looking ones run on their own, everything else is shown to
you and waits for a `y`.

It is deliberately small and fully tested, so you can read the whole thing,
understand how an agent loop works, and bend it to your needs.

## Contents

- [Requirements](#requirements)
- [Install](#install)
- [Run](#run)
- [Using the chat](#using-the-chat)
- [What the agent can do](#what-the-agent-can-do)
- [Safety model](#safety-model)
- [Command risk classifier](#command-risk-classifier)
- [Configuration reference](#configuration-reference)
- [Troubleshooting](#troubleshooting)
- [Project layout](#project-layout)
- [How a turn flows](#how-a-turn-flows)
- [Development](#development)
- [Adding a tool](#adding-a-tool)

## Requirements

- Python 3.10 or newer
- An Anthropic API key (<https://console.anthropic.com/>)
- [uv](https://docs.astral.sh/uv/) is recommended but not required

## Install

From a checkout:

```bash
uv sync                  # creates .venv and installs the app
uv run tiny-agent        # run it
```

Or install the command globally without cloning:

```bash
uv tool install git+https://github.com/saalssall/tiny-agent
tiny-agent --version
```

Without uv, a plain virtual environment works too:

```bash
python3 -m venv .venv && .venv/bin/pip install -e .
.venv/bin/tiny-agent
```

Provide an API key in one of two ways:

- **Environment variable** (checked first):

  ```bash
  export ANTHROPIC_API_KEY=sk-ant-...
  ```

- **Key file:** put the raw key, and nothing else, in a file called `API.KEY`.
  The `tiny-agent` command looks in the directory you launch from; the
  `main.py` launcher looks next to itself. The file is git-ignored in this
  repo.

If neither is found the app exits immediately with a message telling you how
to fix it.

## Run

```bash
tiny-agent [workspace] [--yolo | --always-ask] [--model MODEL] [--effort LEVEL] [--verbose]
```

Prefix with `uv run` if you have not installed the tool globally. From a
checkout, `python main.py` and `python -m tiny_agent` also work.

| Argument / flag  | What it does                                                                              | Default           |
| ---------------- | ----------------------------------------------------------------------------------------- | ----------------- |
| `workspace`      | The directory the agent may read, edit and run commands in.                               | current directory |
| `--yolo`         | Run every shell command without asking. See [Safety model](#safety-model).                | off               |
| `--always-ask`   | Skip the risk classifier so every shell command asks for confirmation.                    | off               |
| `--model MODEL`  | Claude model ID to use. Can also be set with `AGENT_MODEL`.                               | `claude-opus-5`   |
| `--effort LEVEL` | How hard the model thinks: `low`, `medium`, `high`, `xhigh` or `max`. Can also be set with `AGENT_EFFORT`. | `high` |
| `--verbose`      | Log each request, stop reason and tool call to stderr.                                    | off               |
| `--version`      | Print the version and exit.                                                               |                   |
| `-h`, `--help`   | Show the built-in usage text.                                                             |                   |

Examples:

```bash
tiny-agent                            # work in the current directory
tiny-agent ~/code/my-project          # work in another directory
tiny-agent --effort low               # faster, cheaper answers
tiny-agent --model claude-sonnet-5    # a different model
tiny-agent --always-ask               # confirm every command yourself
tiny-agent --yolo                     # never ask (use with care)
AGENT_MODEL=claude-haiku-4-5 tiny-agent
```

Pick a cheap model and low effort for quick questions, and the default for
real coding tasks.

## Using the chat

When the app starts it prints a banner with the model, effort level and
workspace, a line explaining how shell commands will be approved, then a
`you ›` prompt. Type what you want done and press Enter. Anything that does
not start with `/` is sent to the model.

Things you might ask:

```
you › What does this project do? Start with the README and the entry point.
you › Add type hints to tiny_agent/workspace.py without changing behaviour.
you › Run the tests and fix whatever fails.
you › Find every place we call subprocess and list them with line numbers.
```

While the model works you will see:

- Its reply streamed as it is written.
- `… preparing <tool>` while it composes a tool call, with dots for progress.
- `▸ <tool> key=value ...` when a tool runs, then `✓` or `✗` and the first
  line of the result.
- `$ <command>` before any shell command, followed by the risk gate's reason.
  Safe-looking commands then run straight away. Anything else shows
  `run this? [y/N]`; answer `y` or `yes` to run it. Any other answer refuses,
  and the model is told you declined.

The model keeps calling tools until it has a plain-text answer for you, up to
50 tool rounds per message. The whole conversation is remembered until you
`/clear` or quit.

### Slash commands

| Command                | Effect                                                       |
| ---------------------- | ------------------------------------------------------------ |
| `/help`                | Show the command list.                                       |
| `/cost`                | Show token counts and estimated spend so far this session.   |
| `/clear`               | Forget the conversation and start fresh (usage totals stay). |
| `/quit`, `/exit`, `/q` | Exit. `Ctrl-D` or `Ctrl-C` at the prompt does the same.      |

On exit the app prints a final usage summary.

### Interrupting

Press `Ctrl-C` while the model is replying or running tools to stop it. The
conversation rolls back to the last complete turn, so you can rephrase and try
again without the history getting out of sync. The same rollback happens after
any API error.

### Cost estimates

`/cost` shows input, output, cache-write and cache-read tokens and an estimated
USD cost. Prices are known for `claude-opus-5`, `claude-fable-5-1`,
`claude-sonnet-5` and `claude-haiku-4-5`. For any other model the cost shows
as `unknown model`; add a row to `PRICES` in `tiny_agent/config.py` to fix
that. The system prompt is sent with prompt caching enabled, so after the
first request most of it is billed at the cheaper cache-read rate.

## What the agent can do

The model has exactly five tools. Every path is relative to the workspace.

| Tool          | What it does                                                                                              |
| ------------- | --------------------------------------------------------------------------------------------------------- |
| `list_files`  | List a directory, optionally recursively. Skips `.git`, `.venv`, `venv`, `node_modules`, `__pycache__`, `.mypy_cache` and `.pytest_cache`. Recursive listings stop after 400 entries. |
| `read_file`   | Return a text file with line numbers. Undecodable bytes are replaced rather than failing.                 |
| `write_file`  | Create or overwrite a file, creating parent directories as needed.                                        |
| `edit_file`   | Replace one exact occurrence of `old_text` with `new_text`. Fails if the text is missing or appears more than once, which prevents accidental edits. |
| `run_command` | Run a shell command in the workspace root and return stdout, stderr and the exit code. Goes through the approval gate. Credential-like environment variables are withheld. |

Tool inputs are validated against their schema before running, and any tool
error is returned to the model as text rather than crashing the app. Tool
output longer than 40,000 characters is truncated with a note.

## Safety model

- **Path confinement.** Every path the model supplies is resolved and checked
  to be inside the workspace. `../secrets` or an absolute path outside the
  root is refused with an error the model can see. Symlinks are resolved
  before the check.
- **Command approval.** Each shell command is printed in full. With the
  default settings it is then assessed by the
  [risk classifier](#command-risk-classifier): a confident "safe" verdict runs
  it, anything else asks you. `--always-ask` makes every command ask.
  `--yolo` makes none of them ask. Commands run with your user's permissions
  and are not sandboxed, so only use `--yolo` in a directory you would be
  happy to hand to an automated script.
- **Secrets withheld.** Shell commands run with credential-looking environment
  variables (`*KEY*`, `*TOKEN*`, `*SECRET*`, `*PASSWORD*`, `*PASSWD*`,
  `*CREDENTIAL*`) removed, so a command cannot read the API key even if it
  were approved.
- **Timeouts and caps.** Shell commands are killed after 120 seconds. A single
  message can trigger at most 50 tool rounds before the agent stops and asks
  you to say `continue`.
- **No hidden state.** The app writes nothing outside the workspace. It keeps
  no logs, history file or cache of its own.

The workspace check protects against mistakes in file tools. It does not
constrain what a shell command can touch, which is why the approval gate
exists.

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

## Configuration reference

Command-line flags and environment variables:

| Setting        | Flag           | Environment variable            | Default                      |
| -------------- | -------------- | ------------------------------- | ---------------------------- |
| Workspace      | positional     |                                 | `.`                          |
| Model          | `--model`      | `AGENT_MODEL`                   | `claude-opus-5`              |
| Effort         | `--effort`     | `AGENT_EFFORT`                  | `high`                       |
| Auto-approve   | `--yolo`       |                                 | off                          |
| Always ask     | `--always-ask` |                                 | off                          |
| Verbose logs   | `--verbose`    |                                 | off                          |
| API key        |                | `ANTHROPIC_API_KEY`             | `API.KEY` file               |
| Colour output  |                | `NO_COLOR` (any value disables) | on when stdout is a terminal |

Values without a flag live in the `Settings` dataclass in
`tiny_agent/config.py` and can be edited there:

| Field             | Meaning                                              | Default |
| ----------------- | ---------------------------------------------------- | ------- |
| `max_rounds`      | Tool rounds allowed per user message                 | 50      |
| `max_tokens`      | Maximum tokens per model response                    | 64,000  |
| `command_timeout` | Seconds before a shell command is killed             | 120     |
| `max_tool_output` | Characters of tool output kept before truncating     | 40,000  |

The system prompt the model sees is `SYSTEM_PROMPT` in the same file.

## Troubleshooting

| Message                                              | Cause and fix                                                                  |
| ---------------------------------------------------- | ------------------------------------------------------------------------------ |
| `No API key found.`                                  | Set `ANTHROPIC_API_KEY` or create an `API.KEY` file (see [Install](#install)). |
| `'x' is not a directory.`                            | The workspace argument must be an existing directory.                          |
| `Authentication failed. Check your API key.`         | The key was found but rejected. Check for whitespace or an expired key.        |
| `Rate limited. Wait a moment and try again.`         | You hit the API rate limit. The turn was rolled back; just resend.             |
| `API error <code>: ...`                              | Any other API error. The turn was rolled back.                                 |
| `Network error.`                                     | Could not reach the API. Check your connection or proxy.                       |
| `[the model declined this request]`                  | The model refused. Rephrase or narrow the request.                             |
| `[reply was cut off before the tool call completed]` | The response hit `max_tokens` mid tool call. Ask for a smaller step or raise `max_tokens`. |
| `[stopped after 50 tool rounds; ...]`                | The model used every allowed round. Say `continue`, or raise `max_rounds`.     |
| `path '...' is outside the workspace`                | The model tried to leave the workspace. Point the app at a wider directory if that was intended. |
| Every command asks, even `ls`                        | `--always-ask` is set, or `risk_model.json` is missing. The startup line says which mode is active. |
| No colours                                           | Output is not a terminal, or `NO_COLOR` is set.                                |

Run with `--verbose` to see each request, stop reason and tool call on stderr.

## Project layout

```
main.py                 convenience launcher for a checkout
pyproject.toml          package metadata, dependencies, and ruff / mypy / pytest / coverage config
uv.lock                 locked dependency versions
CHANGELOG.md            release notes
tiny_agent/
  __init__.py           version string and public exports
  __main__.py           `python -m tiny_agent`
  config.py             Settings dataclass, CLI arguments, API-key lookup, prices, system prompt
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
tests/                  unit tests for every module; no network or API key needed
.github/workflows/      CI on every push and PR; release on version tags
```

## How a turn flows

1. `ChatApp` reads a line. Slash commands are handled locally; anything else
   goes to `Agent.ask`.
2. The agent appends your message to the `Conversation` and streams a request
   to the API with the system prompt, tool schemas, adaptive thinking and the
   chosen effort level. Text is printed as it arrives.
3. If the reply contains tool calls, `ToolRegistry` validates each input
   against its schema, runs it, and the results are appended as a user
   message. For `run_command`, the `Approver` consults the risk gate and, if
   needed, you. Steps 2 and 3 repeat until the model replies in plain text or
   the round cap is reached.
4. Token usage from every request is added to `UsageTracker`.
5. If anything fails, or you press `Ctrl-C`, `ChatApp` rolls the conversation
   back to the checkpoint taken before your message, so the history sent to
   the API is always well-formed.

Two details worth knowing if you change the agent loop: tool inputs use eager
input streaming, so a malformed tool JSON is retried up to two times; and a
`pause_turn` stop reason simply re-sends the conversation.

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

1. Subclass `WorkspaceTool` in `tiny_agent/tools.py`, or `Tool` if you need
   other dependencies.
2. Declare `name`, `description`, `parameters` and `required`.
3. Implement `run` with keyword parameters matching the schema. Raise
   `WorkspaceError` for expected failures; the message is returned to the
   model.
4. Add an instance to the list in `build_app` in `tiny_agent/cli.py`.

Schema validation, error reporting and output truncation are handled by
`ToolRegistry`. Parameter types can be `string`, `boolean`, `integer` or
`number`.

```python
class WordCountTool(WorkspaceTool):
    name = "word_count"
    description = "Count the words in a text file."
    parameters = {"path": {"type": "string", "description": "File path relative to the workspace."}}
    required = ("path",)

    def run(self, path: str) -> str:
        return str(len(self.workspace.read(path).split()))
```
