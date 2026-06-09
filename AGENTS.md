# AGENTS.md

## Repository Purpose

This repository is the starter code for CS336 Assignment 1: Basics. It is an
implementation project for core language-modeling systems:

- tokenization
- model layers
- transformer blocks
- optimization
- data loading
- serialization
- training utilities

The tests in `tests/` define the expected behavior. `tests/adapters.py` connects
submitted implementations to the test suite.

## Assignment Goals

- Build a working byte-pair encoding tokenizer and understand how tokenization affects language-model training.
- Implement neural network primitives used in transformer language models, including embeddings, normalization, attention, feed-forward layers, and full transformer blocks.
- Understand training infrastructure such as batching, checkpointing, optimizer behavior, and reproducible testing.
- Practice reading specifications, matching reference behavior, and validating implementations with focused unit tests.

## Sources of Truth and Important References

Consult these files when they are relevant to the change:

- `README.md`: concise project overview, setup, data requirements, common commands, documentation map, project status, and license summary.
- `requirements_for_code_produced.md`: comprehensive coding restrictions for submitted assignment code, including allowed and forbidden library usage.
- `repository_structure.md`: maintained repository map covering implementation code, tests, scripts, data artifacts, and documentation.
- Folder-level `README.md` files: local purpose, important files, setup assumptions, commands, inputs/outputs, implementation status, and known gaps for important folders such as `cs336_basics/`, `tests/`, `data/`, `bpe_samples/`, and `crates/`.
- `cs336_basics/README.md`: package-level documentation for submitted Python implementation modules, current implementation status, development rules, and related commands.
- `cs336_basics/PUBLIC_API_INVENTORY.md`: inventory of public-facing reusable Python functions and classes in `cs336_basics/`; consult it before adding reusable implementation code.
- `tests/README.md`: test-suite overview, adapter status, fixture/snapshot guidance, and focused test commands.
- `data/README.md`: local data download expectations, ignored generated artifacts, tokenizer artifact conventions, and token-ID array notes.
- `tests/adapters.py`: the bridge between the tests and implementations in `cs336_basics/`; adapter functions should stay thin.
- `tests/`: unit tests that define the required behavior.
- `pyproject.toml`: Python version range, dependencies, pytest settings, and ruff settings.
- `uv.lock`: resolved dependency graph for reproducible environments.
- `SETUP.md`: broader setup guidance and assignment workflow notes.
- `BPE_TOKENIZER.md`: notes for the optional enhanced BPE trainer and retained tokenizer experiment artifacts.
- `CHANGELOG.md`: upstream assignment/code changes.
- `uv-docs/`: offline `uv` reference documentation.

## Submitted Implementation Code

- Put submitted assignment implementation code under `cs336_basics/`.
- Connect implementations to the tests through `tests/adapters.py`.
- Keep adapter functions thin: they should import and call code from `cs336_basics/` rather than contain substantial implementation logic.
- Future submitted implementations for model layers, optimization, data loading, serialization, and training utilities should also live under `cs336_basics/`, split into modules that match the assignment component being implemented.
- Before creating a new reusable class or function under `cs336_basics/`, check `cs336_basics/PUBLIC_API_INVENTORY.md` and compose existing primitives where practical rather than reproducing similar code or structures.
- Notebook-only work is not the source of submitted implementation behavior unless the same logic is also placed under `cs336_basics/` and connected through `tests/adapters.py`.

## Coding Restrictions

- All code generated that directly answers assignment questions must conform to `requirements_for_code_produced.md`.
- Code used purely for error checking, unit tests, or logging is exempt from those submitted-code restrictions.
- Required neural-network, optimization, and RLHF/PPO components must be implemented from scratch as described in `requirements_for_code_produced.md`.
- Do not use PyTorch or another machine-learning library to provide a component that the assignment asks you to implement.
- When in doubt about whether a helper is allowed in submitted implementation code, implement the operation yourself from permitted elementary operations.

## Documentation Maintenance

- Whenever new submitted code is written, moved, or reorganized outside Jupyter notebooks, update `repository_structure.md` and the relevant folder-level `README.md` files in the same change.
- Whenever a reusable public-facing Python function or class is added, renamed, removed, or materially repurposed under `cs336_basics/`, update `cs336_basics/PUBLIC_API_INVENTORY.md` in the same change.
- Whenever repository changes alter setup, data requirements, common commands, scripts, generated artifacts, project status, source layout, test coverage, or contributor workflows, update the affected documentation in the same change.
- When a new Python implementation module is added to `cs336_basics/`, update `cs336_basics/README.md`; when adapter coverage changes, update `tests/README.md`; when data scripts or artifact formats change, update `data/README.md` and any relevant BPE documentation.
- Updates that only affect Jupyter notebooks do not require `repository_structure.md` updates unless notebook changes also alter submitted code, tracked artifacts, scripts, data expectations, or documented workflows.
- Prefer folder-level `README.md` files for documentation tied to a specific directory. Add or keep root-level Markdown files only when the repository root is clearly the best location for broader reference material.

## Testing and Validation

- Use the tests in `tests/` as the behavioral specification.
- Do not change test assertions to make tests pass.
- Run focused tests while implementing when possible, then broader tests when the change warrants it.
- Common test commands:

```sh
uv run pytest
uv run pytest tests/test_tokenizer.py
```

## Project Workflow

This project uses `uv` for Python environment and dependency management.

Common commands:

```sh
uv sync
uv run pytest
uv run pytest tests/test_tokenizer.py
```

The required Python range and dependencies are declared in `pyproject.toml`.
The resolved environment is tracked in `uv.lock`.

Prefer the checked-in dependencies for normal assignment work. Use dependency-changing commands such as `uv add` or `uv lock --upgrade` only when deliberately changing project dependencies and after considering the assignment's library-use restrictions.

## Local uv Documentation

Offline `uv` documentation is available in `uv-docs/`:

- `uv-docs/README.md`: overview, quick command reference, WSL setup notes, virtual environment usage, dependency installation, script execution, and test-running examples.
- `uv-docs/uv-projects.md`: project creation, project layout, `pyproject.toml`, `.venv`, lockfiles, workspaces, builds, and project configuration.
- `uv-docs/uv-dependencies.md`: adding/removing dependencies, dependency sources, optional and development dependencies, locking, syncing, resolution, command execution, and lockfile export.
- `uv-docs/uv-github-actions.md`: installing uv in GitHub Actions, setting up Python, matrix testing, syncing and running tests, caching, private repositories, and PyPI publishing.

These docs are a local reference snapshot. For current `uv` behavior, compare
against the official documentation at <https://docs.astral.sh/uv/>.
