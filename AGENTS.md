# AGENTS.md

## Repository Purpose

This repository is the starter code for CS336 Assignment 1: Basics. It is an educational implementation project for core language-modeling systems: tokenization, model layers, transformer blocks, optimization, data loading, serialization, and training utilities.

The code is intentionally incomplete. Tests in `tests/` define the expected behavior, and `tests/adapters.py` connects student implementations to the public test suite.

## Goals and Educational Outcomes

Work in this repository should help students:

- Build a working byte-pair encoding tokenizer and understand how tokenization affects language-model training.
- Implement neural network primitives used in transformer language models, including embeddings, normalization, attention, feed-forward layers, and full transformer blocks.
- Understand training infrastructure such as batching, checkpointing, optimizer behavior, and reproducible testing.
- Practice reading specifications, matching reference behavior, and validating implementations with focused unit tests.

Prefer small, test-driven changes. Preserve the assignment structure and avoid replacing the intended implementations with large external abstractions that bypass the learning goals.

## Project Workflow

This project uses `uv` for Python environment and dependency management. Common commands:

```sh
uv sync
uv run pytest
uv run pytest tests/test_tokenizer.py
```

The required Python range and dependencies are declared in `pyproject.toml`; the resolved environment is tracked in `uv.lock`.

## Local uv Documentation

Offline uv documentation is available in `uv-docs/`:

- `uv-docs/README.md`: overview, quick command reference, WSL setup notes, virtual environment usage, dependency installation, script execution, and test-running examples.
- `uv-docs/uv-projects.md`: project creation, project layout, `pyproject.toml`, `.venv`, lockfiles, workspaces, builds, and project configuration.
- `uv-docs/uv-dependencies.md`: adding/removing dependencies, dependency sources, optional and development dependencies, locking, syncing, resolution, command execution, and lockfile export.
- `uv-docs/uv-github-actions.md`: installing uv in GitHub Actions, setting up Python, matrix testing, syncing and running tests, caching, private repositories, and PyPI publishing.

These docs are a local reference snapshot. For current uv behavior, compare against the official documentation at https://docs.astral.sh/uv/.
