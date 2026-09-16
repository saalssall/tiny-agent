# Changelog

All notable changes to this project are documented here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/).

## [Unreleased]

### Changed
- README expanded: install without uv, full flag table, chat walkthrough, tool table,
  safety model, configuration reference and troubleshooting.

## [0.3.0] - 2026-09-16

### Added
- Installable package with a `tiny-agent` console command and `python -m tiny_agent`.
- `--version` and `--verbose` flags; the latter logs each request, stop reason and tool call.
- A cap on tool rounds per turn (`max_rounds`, default 50) so a confused model cannot loop indefinitely.
- Secret-like environment variables (`*KEY*`, `*TOKEN*`, `*SECRET*`, ...) are withheld from shell commands.
- Strict mypy, coverage threshold, pre-commit hooks, Dependabot, and a `uv.lock` for reproducible installs.

### Changed
- Dependencies are declared in `pyproject.toml` only; `requirements*.txt` removed.
- The release workflow builds a wheel and source distribution and attaches them to the GitHub Release.

## [0.2.0] - 2026-09-16

### Added
- Command risk classifier: hard rules plus a trained model decide whether a shell command
  can run without asking. Ships as JSON weights scored in pure Python.
- `--always-ask` flag to bypass the classifier.
- `ml/` pipeline: dataset generator, grouped cross-validation, export verification.

## [0.1.0] - 2026-09-16

### Added
- Initial release: streaming Claude agent with list/read/write/edit/shell tools confined to a
  workspace, y/N approval for shell commands, cost tracking, and unit tests.
