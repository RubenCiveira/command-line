# Contributing to Asistente

Thank you for considering a contribution! This document explains how to get
started, the conventions we follow and what to expect during the review process.

## Getting started

```bash
git clone https://github.com/RubenCiveira/asistente.git
cd asistente
make build          # creates .venv and installs all deps (including dev)
source .venv/bin/activate
```

## Running tests

```bash
make test           # pytest with verbose output
```

All new code should include tests. Place them under `test/` and mirror the
source layout (e.g. tests for `src/app/ui/textual/form.py` go in
`test/test_textual_form.py`).

## Code style

We use **ruff** for linting and formatting:

```bash
make lint           # check for issues
make format         # auto-fix formatting
```

Please run both before opening a pull request.

## Branching and commits

- Create a feature branch from `main`: `git checkout -b feat/my-feature`.
- Use conventional commit messages (`feat:`, `fix:`, `docs:`, `test:`, `refactor:`).
- Keep commits small and focused.

## Pull requests

1. Make sure all tests pass (`make test`).
2. Make sure the linter is happy (`make lint`).
3. Open a pull request against `main`.
4. Describe **what** changed and **why** in the PR body.
5. Link any related issues.

A maintainer will review your PR. Feel free to ping if you don't hear back
within a few days.

## Reporting issues

Open an issue on GitHub describing:

- What you expected to happen.
- What actually happened.
- Steps to reproduce (ideally a minimal example).
- Python version and OS.

## Adding a new agent (legacy — `src/old/`)

> The agent system lives in `src/old/` and is not part of the active codebase.

1. Create a new file under `src/old/agents/`.
2. Subclass `BaseAgent` and implement at least the `plan()` method.
3. Set the class-level `agent_type` attribute.
4. The runtime auto-discovers agents — no manual registration is needed.
5. Add prompts under `src/old/agents/i18n/<agent_name>.prompts.<lang>.json`.

## Adding a new tool (legacy — `src/old/`)

1. Create a function with signature `(params: dict, ctx: ToolContext) -> Any`.
2. Register it via `runtime.register_tool("name", fn)`.

## License

By contributing you agree that your contributions will be licensed under the
[Apache License 2.0](LICENSE).
