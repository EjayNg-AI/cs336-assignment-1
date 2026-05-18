# Repository Structure

This repository contains the starter code and tests for CS336 Assignment 1: Basics. The tests define the required behavior, while submitted implementation code should live in the package directory.

## Implementation Code

`cs336_basics/` is the home for submitted assignment code.

- `cs336_basics/__init__.py` defines package metadata behavior.
- `cs336_basics/train_bpe.py` contains the BPE tokenizer training implementation added for the tokenizer training task.
- `cs336_basics/tokenizer.py` contains the BPE tokenizer encoder/decoder implementation, including
  GPT-2-style pre-tokenization, merge application, special-token handling, streaming encoding, and UTF-8
  replacement decoding.
- `cs336_basics/train_bpe_enhanced.py` contains an additive large-corpus BPE trainer variant with parallel
  pre-token counting, integer-token merge state, heap rebuild maintenance, artifact writing, and training
  metadata emission. It leaves the original trainer module unchanged and is not wired into the default test
  adapter unless explicitly imported.
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

`BPE_tokenizer.ipynb` is an exploratory notebook for the tokenizer assignment. It may include explanation, experiments, and notebook-local versions of code for study. Notebook updates are not the source of submitted implementation behavior unless the same logic is also placed under `cs336_basics/` and connected through `tests/adapters.py`. The notebook currently includes answers for BPE tokenizer sampling experiments, local compression-ratio measurements, a throughput estimate for tokenizing an 825 GB corpus, and a resource note explaining why full-corpus token-ID serialization is large.

## Supporting Documentation

- `README.md` describes setup and basic test execution.
- `BPE_TOKENIZER.md` documents enhanced BPE trainer usage and retained tokenizer experiment artifacts.
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
- `run_tinystories_bpe_enhanced.sh` runs the enhanced BPE trainer on the full TinyStories training corpus with
  a 10,000-token vocabulary target, the `<|endoftext|>` special token, and artifact output under
  `data/tinystories_bpe_10000/` by default.
- `run_openwebtext_bpe_enhanced.sh` runs the enhanced BPE trainer on the full OpenWebText training corpus with
  a 32,000-token vocabulary target, the `<|endoftext|>` special token, and artifact output under
  `data/openwebtext_bpe_32000/` by default.
- `download_data.sh` downloads the TinyStories and OpenWebText sample files listed in `README.md`, then unpacks the gzipped OpenWebText files into `data/`.

## Generated BPE Artifacts

When `cs336_basics.train_bpe_enhanced.train_bpe` is imported directly and run,
it writes `vocab.pkl`, `merges.pkl`, `vocab.json`, `merges.txt`, and
`metadata.json` to its configured `output_dir`. If no output directory is
provided, it creates a directory beside the input corpus named
`<input_stem>_bpe_<vocab_size>/`. `vocab.json`, `merges.txt`, and
`metadata.json` are intended for human inspection; the pickle files preserve
the exact Python return objects. The metadata file records the requested and
final vocabulary sizes, merge count, phase durations, merge-loop subphase
durations, and run stats.

The repository-level scripts write the following enhanced BPE training outputs
by default:

- `data/tinystories_bpe_10000/` contains the full TinyStories 10,000-token BPE
  vocabulary, merges, and metadata from `run_tinystories_bpe_enhanced.sh`.
- `data/openwebtext_bpe_32000/` contains the full OpenWebText 32,000-token BPE
  vocabulary, merges, and metadata from `run_openwebtext_bpe_enhanced.sh`.

Example full TinyStories command:

```sh
uv run python -u - <<'PY'
from cs336_basics.train_bpe_enhanced import train_bpe

train_bpe(
    input_path="data/TinyStoriesV2-GPT4-train.txt",
    vocab_size=10_000,
    special_tokens=["<|endoftext|>"],
    num_workers=8,
    chunk_bytes=64 * 1024 * 1024,
    heap_rebuild_factor=3.0,
    output_dir="data/tinystories_bpe_10000",
)
PY
```

## BPE Sample Experiment Artifacts

`bpe_samples/` contains the retained outputs from the notebook's BPE tokenizer
experiments:

- `bpe_samples/tinystories/` contains 10 deterministic TinyStories document
  samples and a `manifest.json` describing their source document indices and
  byte counts.
- `bpe_samples/openwebtext/` contains 10 deterministic OpenWebText document
  samples and a `manifest.json` describing their source document indices and
  byte counts.
- `bpe_samples/ids/` contains JSON token-ID outputs for TinyStories samples
  encoded with the TinyStories tokenizer, OpenWebText samples encoded with the
  OpenWebText tokenizer, and OpenWebText samples encoded with the TinyStories
  tokenizer.
- `bpe_samples/experiment_1_2_summary.json` records the aggregate byte counts,
  token counts, compression ratios, and throughput estimates used in
  `BPE_tokenizer.ipynb`.

Full-dataset Experiment 3 serialization artifacts are intentionally not kept:
the partial `uint16`, JSON, and pickle outputs were deleted after the experiment
was aborted because the expected full-corpus artifacts are multi-GB to tens of
GB in size.

## Pretokenization Example

`cs336_basics/pretokenization_example.py` is reference starter code for splitting a corpus into chunks on a special token boundary. It demonstrates how to find chunk boundaries aligned to a byte-string special token such as `<|endoftext|>`, then read each chunk independently for pre-token counting. Its purpose is instructional: it shows how pre-tokenization can be parallelized safely without allowing BPE merges across document boundaries. It is not itself the submitted BPE trainer.

## Maintenance Rule

Whenever new submitted code is written, moved, or reorganized outside Jupyter notebooks, update this file in the same change so the repository map remains accurate. Jupyter notebook-only updates are excluded from this requirement.
