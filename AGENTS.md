# AGENTS.md

## Repository Purpose

This repository is the starter code for CS336 Assignment 1: Basics. It is an implementation project for core language-modeling systems: tokenization, model layers, transformer blocks, optimization, data loading, serialization, and training utilities.

Tests in `tests/` define the expected behavior, and `tests/adapters.py` connects implementations to the test suite.

## Goals

- Build a working byte-pair encoding tokenizer and understand how tokenization affects language-model training.
- Implement neural network primitives used in transformer language models, including embeddings, normalization, attention, feed-forward layers, and full transformer blocks.
- Understand training infrastructure such as batching, checkpointing, optimizer behavior, and reproducible testing.
- Practice reading specifications, matching reference behavior, and validating implementations with focused unit tests.

## Coding Restrictions

Comprehensive coding restrictions are documented in `requirements_for_code_produced.md`. All code generated that directly answers assignment questions must conform to those requirements, except for code used purely for error checking, unit tests, or logging.

## Repository Structure Documentation

The root file `repository_structure.md` documents the repository layout, including that submitted implementation code should be housed in `cs336_basics/` and connected to tests through `tests/adapters.py`.

Whenever new submitted code is written, moved, or reorganized outside Jupyter notebooks, update `repository_structure.md` in the same change. Updates that only affect Jupyter notebooks are excluded from this requirement.

## Implementation Package Documentation

The folder-level file `cs336_basics/README.md` documents the Python applications in `cs336_basics/`, including each application's description, purpose, and methodology.

Whenever a new Python application is added to `cs336_basics/`, update `cs336_basics/README.md` in the same change so the package-level documentation remains current.

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
