# Repository Structure

This repository contains the starter code and tests for CS336 Assignment 1: Basics. The tests define the required behavior, while submitted implementation code should live in the package directory.

## Implementation Code

`cs336_basics/` is the home for submitted assignment code.

- `cs336_basics/__init__.py` defines package metadata behavior.
- `cs336_basics/tokenizer.py` contains the BPE tokenizer training implementation added for the tokenizer training task.
- Future submitted implementations for model layers, optimization, data loading, serialization, and training utilities should also be placed under `cs336_basics/`, split into modules that match the assignment component being implemented.

`tests/adapters.py` is the bridge between the test suite and submitted code. Adapter functions should stay thin: they should import and call implementations from `cs336_basics/` rather than housing substantial implementation logic themselves.

## Tests and Fixtures

`tests/` contains the unit tests used to validate assignment behavior.

- `tests/test_train_bpe.py` checks BPE tokenizer training correctness, special-token handling, and speed.
- `tests/test_tokenizer.py` checks tokenizer encoding and decoding behavior.
- `tests/test_model.py`, `tests/test_nn_utils.py`, `tests/test_optimizer.py`, `tests/test_data.py`, and `tests/test_serialization.py` cover the remaining assignment systems.
- `tests/fixtures/` contains reference vocabularies, merge files, sample corpora, model weights, and other test data.
- `tests/_snapshots/` contains saved numerical snapshots used by tests.

## Notebooks

`BPE_tokenizer.ipynb` is an exploratory notebook for the tokenizer assignment. It may include explanation, experiments, and notebook-local versions of code for study. Notebook updates are not the source of submitted implementation behavior unless the same logic is also placed under `cs336_basics/` and connected through `tests/adapters.py`.

## Supporting Documentation

- `README.md` describes setup and basic test execution.
- `SETUP.md` gives broader setup guidance and assignment workflow notes.
- `requirements_for_code_produced.md` lists coding restrictions that submitted code must follow.
- `AGENTS.md` gives standing instructions for coding agents working in this repository.
- `repository_structure.md` is this file and should be kept current as implementation modules are added or reorganized.
- `CHANGELOG.md` records upstream assignment changes.
- `uv-docs/` contains offline reference documentation for `uv`.

## Tooling and Project Files

- `pyproject.toml` declares the package, Python version range, dependencies, pytest settings, and ruff settings.
- `uv.lock` records the resolved dependency graph for reproducible environments.
- `make_submission.sh` and `delete_zone_identifiers.sh` are helper scripts.

## Pretokenization Example

`cs336_basics/pretokenization_example.py` is reference starter code for splitting a corpus into chunks on a special token boundary. It demonstrates how to find chunk boundaries aligned to a byte-string special token such as `<|endoftext|>`, then read each chunk independently for pre-token counting. Its purpose is instructional: it shows how pre-tokenization can be parallelized safely without allowing BPE merges across document boundaries. It is not itself the submitted BPE trainer.

## Maintenance Rule

Whenever new submitted code is written, moved, or reorganized outside Jupyter notebooks, update this file in the same change so the repository map remains accurate. Jupyter notebook-only updates are excluded from this requirement.
