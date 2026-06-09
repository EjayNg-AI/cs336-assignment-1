# CS336 Assignment 1: Basics

This repository is a working copy of the CS336 Assignment 1 starter project. It
contains tests and partial implementations for the basic systems behind a small
language model:

- byte-pair encoding tokenizer training and encoding
- transformer layer primitives such as embeddings, RoPE, RMSNorm, SwiGLU, and attention
- expected future work for transformer blocks, language-model wrappers, data loading,
  optimization, checkpointing, and training utilities
- optional large-corpus BPE experiments in Python and Rust

The test suite in [`tests/`](tests/) is the behavioral specification. Submitted
Python implementation code lives in [`cs336_basics/`](cs336_basics/) and is
connected to the tests through [`tests/adapters.py`](tests/adapters.py).
For a systematic inventory of reusable public Python functions and classes in
the package, see
[`cs336_basics/PUBLIC_API_INVENTORY.md`](cs336_basics/PUBLIC_API_INVENTORY.md).

## Project Status

This is a partially complete project. Tokenizer components and several model
layer primitives are implemented. Some adapter functions still raise
`NotImplementedError`, including transformer-block/language-model composition,
batch sampling, cross-entropy, AdamW, learning-rate scheduling, gradient
clipping, and checkpoint serialization.

TODO: The assignment handout PDF referenced by the upstream starter README is
not present in this checkout. Add or link the authoritative handout if this
repository is used outside the course context.

## Quick Setup

Install `uv` using the official instructions:

- <https://docs.astral.sh/uv/getting-started/installation/>
- Local reference snapshot: [`uv-docs/README.md`](uv-docs/README.md)

Create the environment and install locked dependencies:

```sh
uv sync
```

Run the full test suite:

```sh
uv run pytest
```

Run focused tests while developing:

```sh
uv run pytest tests/test_tokenizer.py
uv run pytest tests/test_model.py
uv run pytest tests/test_train_bpe.py
```

Optional Rust BPE validation requires Cargo:

```sh
cargo test -p cs336_bpe_rs
uv run pytest tests/test_rust_bpe_parity.py
```

## Data Requirements

The full training corpora are not tracked in Git. The repository includes
[`download_data.sh`](download_data.sh), which downloads TinyStories and the
CS336 OpenWebText sample into `data/` and unpacks the gzipped OpenWebText files.

```sh
bash download_data.sh
```

Expected downloaded files:

- `data/TinyStoriesV2-GPT4-train.txt`
- `data/TinyStoriesV2-GPT4-valid.txt`
- `data/owt_train.txt`
- `data/owt_valid.txt`

The downloads are large; do not commit files under `data/` except
[`data/README.md`](data/README.md). See [`data/README.md`](data/README.md) for
derived tokenizer artifacts and token-ID array conventions.

## Common Workflows

Train enhanced BPE tokenizers after the data files are present:

```sh
bash run_tinystories_bpe_enhanced.sh
bash run_openwebtext_bpe_enhanced.sh
```

Serialize full-corpus token IDs after the tokenizer artifacts exist:

```sh
bash run_bpe_experiment_3_tokenization.sh
bash run_bpe_experiment_3_tokenization_rs.sh  # Rust encoder variant
```

Build the assignment submission archive:

```sh
bash make_submission.sh
```

## Documentation Map

- [`SETUP.md`](SETUP.md): detailed setup, testing, data, and troubleshooting notes
- [`repository_structure.md`](repository_structure.md): maintained map of code,
  tests, scripts, data artifacts, and documentation
- [`requirements_for_code_produced.md`](requirements_for_code_produced.md):
  restrictions for submitted assignment code
- [`cs336_basics/README.md`](cs336_basics/README.md): Python implementation package
- [`cs336_basics/PUBLIC_API_INVENTORY.md`](cs336_basics/PUBLIC_API_INVENTORY.md):
  reusable public Python functions and classes in `cs336_basics/`
- [`tests/README.md`](tests/README.md): test suite, fixtures, snapshots, and adapter status
- [`bpe_samples/README.md`](bpe_samples/README.md): retained BPE notebook samples
- [`data/README.md`](data/README.md): ignored local corpora and generated artifacts
- [`BPE_TOKENIZER.md`](BPE_TOKENIZER.md): enhanced Python BPE experiments and commands
- [`crates/README.md`](crates/README.md): Rust workspace overview
- [`RUST_BPE_IMPLEMENTATION.md`](RUST_BPE_IMPLEMENTATION.md): Rust BPE design notes
- [`uv-docs/README.md`](uv-docs/README.md): offline `uv` reference snapshot

## License

This repository includes an MIT-style license in [`LICENSE`](LICENSE), with
copyright assigned to Stanford University.
