# Repository Structure

This repository contains the starter code and tests for CS336 Assignment 1: Basics. The tests define the required behavior, while submitted implementation code should live in the package directory.

## Implementation Code

`cs336_basics/` is the home for submitted assignment code.

- `cs336_basics/__init__.py` defines package metadata behavior.
- `cs336_basics/train_bpe.py` contains the BPE tokenizer training implementation added for the tokenizer training task.
- `cs336_basics/tokenizer.py` contains the BPE tokenizer encoder/decoder implementation, including
  GPT-2-style pre-tokenization, integer-ID merge application, special-token handling, context-preserving
  streaming encoding, and UTF-8 replacement decoding.
- `cs336_basics/nn_linear_embedding_rope_rmsnorm.py` contains the custom neural-network `Linear`,
  `Embedding`, `RotaryPositionalEmbedding`, and `RMSNorm` modules for the transformer architecture
  task, using explicit `torch.nn.Parameter` tensors, assignment-specified initialization, matrix
  multiplication, direct embedding-table indexing, fixed RoPE sine/cosine buffers, and manual
  root-mean-square normalization.
- `cs336_basics/nn_feedforward.py` contains the manual SiLU activation and SwiGLU feed-forward
  network for transformer blocks. It composes the custom `Linear` module, uses an explicit stable
  sigmoid from elementary tensor operations, supports an explicit `d_ff`, and otherwise computes
  the SwiGLU hidden width as `8/3 * d_model` rounded to the nearest multiple of 64.
- `cs336_basics/nn_attention.py` contains the manual softmax helper, batched scaled dot-product
  attention, and causal multi-head self-attention module for the transformer architecture task. It
  composes the custom `Linear` and `RotaryPositionalEmbedding` modules, applies causal masking with
  `True` meaning "allowed to attend", splits projections into heads without prebuilt attention
  helpers, and optionally applies RoPE to queries and keys only.
- `cs336_basics/train_bpe_enhanced.py` contains an additive large-corpus BPE trainer variant with parallel
  pre-token counting, integer-token merge state, heap rebuild maintenance, artifact writing, and training
  metadata emission. It leaves the original trainer module unchanged and is not wired into the default test
  adapter unless explicitly imported.
- `crates/cs336_bpe_rs/` contains an additive Rust CLI/library implementation of the enhanced byte-level BPE
  trainer and tokenizer. It is designed as a correctness-equivalent sibling of `train_bpe_enhanced.py` and
  `tokenizer.py`, writes language-neutral `vocab.json`, `merges.txt`, and `metadata.json` training artifacts,
  can serialize encoded token IDs as NumPy `uint16` arrays, and is validated through parity tests rather than
  being wired into the default Python assignment adapters.
- Future submitted implementations for model layers, optimization, data loading, serialization, and training utilities should also be placed under `cs336_basics/`, split into modules that match the assignment component being implemented.

`tests/adapters.py` is the bridge between the test suite and submitted code. Adapter functions should stay thin: they should import and call implementations from `cs336_basics/` rather than housing substantial implementation logic themselves.

## Tests and Fixtures

`tests/` contains the unit tests used to validate assignment behavior.

- `tests/test_train_bpe.py` checks BPE tokenizer training correctness, special-token handling, and speed.
- `tests/test_tokenizer.py` checks tokenizer encoding and decoding behavior.
- `tests/test_rust_bpe_parity.py` checks that the Rust enhanced BPE trainer and encoder match the existing
  Python enhanced trainer/tokenizer on edge corpora and streaming encode scenarios. These tests are skipped
  when Cargo is unavailable.
- `tests/test_model.py`, `tests/test_nn_utils.py`, `tests/test_optimizer.py`, `tests/test_data.py`, and `tests/test_serialization.py` cover the remaining assignment systems.
- `tests/fixtures/` contains reference vocabularies, merge files, sample corpora, model weights, and other test data.
- `tests/_snapshots/` contains saved numerical snapshots used by tests.

## Notebooks

`BPE_tokenizer.ipynb` is an exploratory notebook for the tokenizer assignment. It may include explanation, experiments, and notebook-local versions of code for study. Notebook updates are not the source of submitted implementation behavior unless the same logic is also placed under `cs336_basics/` and connected through `tests/adapters.py`. The notebook currently includes answers for BPE tokenizer sampling experiments, local compression-ratio measurements, a throughput estimate for tokenizing an 825 GB corpus, and a resource note explaining why full-corpus token-ID serialization is large.

`transformer_llm_architecture.ipynb` is an exploratory and explanatory notebook for transformer language-model architecture components. It includes the assignment text for parameter initialization, linear layers, embedding layers, RoPE, RMS normalization, the pre-norm transformer block feed-forward network, and multi-head self-attention. Its code cells mirror the submitted `Linear`, `Embedding`, `RotaryPositionalEmbedding`, and `RMSNorm` implementations from `cs336_basics/nn_linear_embedding_rope_rmsnorm.py`, the submitted SiLU/SwiGLU implementation from `cs336_basics/nn_feedforward.py`, and the submitted softmax/attention implementation from `cs336_basics/nn_attention.py`, with explanatory cells for the layer parameter data structures and tensor operations.

## Supporting Documentation

- `README.md` is the concise project entry point with status, setup, data, common commands, and the documentation map.
- `SETUP.md` gives broader setup guidance, test commands, data-download steps, optional Rust commands, and troubleshooting notes.
- `BPE_TOKENIZER.md` documents enhanced BPE trainer usage and retained tokenizer experiment artifacts.
- `RUST_BPE_IMPLEMENTATION.md` explains the additive Rust enhanced BPE trainer/encoder implementation,
  including its module layout, training pipeline, encoder behavior, parity contract, optimization findings,
  recommended future run workflow, and validation commands.
- `crates/README.md` documents the Rust workspace role and common Cargo commands.
- `crates/cs336_bpe_rs/README.md` documents the Rust BPE trainer/encoder CLI, generated artifact formats, and
  validation commands.
- `crates/cs336_bpe_rs/src/README.md`, `src/bin/README.md`, `src/trainer/README.md`, and `src/encoder/README.md`
  document the Rust source layout, CLI entrypoints, training pipeline, and encoder modules.
- `cs336_basics/README.md` documents the submitted Python package, implemented modules, current gaps, and development rules.
- `tests/README.md` documents the pytest suite, adapter role, fixture folders, commands, and current adapter gaps.
- `tests/fixtures/README.md` documents small tracked test corpora, tokenizer references, and transformer fixtures.
- `tests/fixtures/ts_tests/README.md` documents the transformer model fixture files.
- `tests/_snapshots/README.md` documents numerical snapshot fixtures.
- `bpe_samples/README.md` documents retained tokenizer notebook sample artifacts.
- `bpe_samples/tinystories/README.md`, `bpe_samples/openwebtext/README.md`, and `bpe_samples/ids/README.md`
  document sample manifests and encoded-ID artifacts.
- `data/README.md` documents the local ignored data directory, data download script, generated tokenizer artifacts,
  and token-ID array conventions.
- `requirements_for_code_produced.md` lists coding restrictions that submitted code must follow.
- `AGENTS.md` gives standing instructions for coding agents working in this repository.
- `repository_structure.md` is this file and should be kept current as implementation modules are added or reorganized.
- `CHANGELOG.md` records upstream assignment changes.
- `uv-docs/` contains offline reference documentation for `uv`.

## Tooling and Project Files

- `pyproject.toml` declares the package, Python version range, dependencies, pytest settings, and ruff settings.
- `uv.lock` records the resolved dependency graph for reproducible environments.
- `Cargo.toml` declares the root Cargo workspace for Rust support.
- `Cargo.lock` records the resolved Rust dependency graph for the `cs336_bpe_rs` crate.
- `crates/cs336_bpe_rs/Cargo.toml` declares the Rust BPE crate, library, and `cs336-bpe-train` /
  `cs336-bpe-encode` binaries.
- `.gitignore` ignores local data, caches, virtual environments, generated submissions, and training outputs while
  allowing `data/README.md` to remain tracked.
- `make_submission.sh` and `delete_zone_identifiers.sh` are helper scripts.
- `run_tinystories_bpe_enhanced.sh` runs the enhanced BPE trainer on the full TinyStories training corpus with
  a 10,000-token vocabulary target, the `<|endoftext|>` special token, and artifact output under
  `data/tinystories_bpe_10000/` by default.
- `run_openwebtext_bpe_enhanced.sh` runs the enhanced BPE trainer on the full OpenWebText training corpus with
  a 32,000-token vocabulary target, the `<|endoftext|>` special token, and artifact output under
  `data/openwebtext_bpe_32000/` by default.
- `run_bpe_experiment_3_tokenization.sh` streams the full TinyStories and OpenWebText train/validation corpora
  through `cs336_basics.tokenizer.Tokenizer` and writes flat NumPy `uint16` token-ID arrays plus metadata under
  `data/bpe_tokenized_corpora/`.
- `run_bpe_experiment_3_tokenization_rs.sh` builds the Rust encoder, streams the same standard corpus splits
  through `cs336-bpe-encode`, and writes compatible NumPy `uint16` token-ID arrays plus metadata under
  `data/bpe_tokenized_corpora_rs/` by default.
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

## BPE Tokenizer Experiment Artifacts

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

Experiment 3 full-corpus serialization is produced by
`run_bpe_experiment_3_tokenization.sh` rather than by executing notebook cells.
The script uses `Tokenizer.from_files(...)` from `cs336_basics/tokenizer.py`,
streams each input file with `encode_iterable`, and stores all generated
outputs under the Git-ignored `data/` directory. The default output layout is:

- `data/bpe_tokenized_corpora/tinystories/train.npy` and `train.json`
  contain the TinyStories training split token IDs and metadata.
- `data/bpe_tokenized_corpora/tinystories/valid.npy` and `valid.json`
  contain the TinyStories validation split token IDs and metadata.
- `data/bpe_tokenized_corpora/openwebtext/train.npy` and `train.json`
  contain the OpenWebText training split token IDs and metadata.
- `data/bpe_tokenized_corpora/openwebtext/valid.npy` and `valid.json`
  contain the OpenWebText validation split token IDs and metadata.
- `data/bpe_tokenized_corpora/manifest.json` collects the split metadata and
  memory-mapped loading examples.

The `.npy` files are flat one-dimensional `uint16` arrays designed for later
language-model training with `np.load(..., mmap_mode="r")`. The sidecar JSON
files record source paths, tokenizer artifact paths, token counts, compression
ratios, throughput measurements, and a SHA-256 hash of the little endian
`uint16` token stream.

The Rust encoder wrapper `run_bpe_experiment_3_tokenization_rs.sh` writes the
same array and metadata format from `vocab.json` / `merges.txt` artifacts, using
`data/bpe_tokenized_corpora_rs/` as its default output directory.

## Pretokenization Example

`cs336_basics/pretokenization_example.py` is reference starter code for splitting a corpus into chunks on a special token boundary. It demonstrates how to find chunk boundaries aligned to a byte-string special token such as `<|endoftext|>`, then read each chunk independently for pre-token counting. Its purpose is instructional: it shows how pre-tokenization can be parallelized safely without allowing BPE merges across document boundaries. It is not itself the submitted BPE trainer.

## Maintenance Rule

Whenever new submitted code is written, moved, or reorganized outside Jupyter notebooks, update this file in the same change so the repository map remains accurate. Jupyter notebook-only updates are excluded from this requirement.
