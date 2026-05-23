# tests

This folder contains the pytest suite that defines the required behavior for
CS336 Assignment 1. The tests call functions in [`adapters.py`](adapters.py),
which should import implementation code from [`../cs336_basics/`](../cs336_basics/).

## Important Files

- [`adapters.py`](adapters.py): assignment-facing bridge from tests to submitted
  implementation code.
- [`common.py`](common.py): shared test helpers.
- [`conftest.py`](conftest.py): pytest fixtures, snapshot helpers, and shared
  model fixture setup.
- `test_train_bpe.py` and `test_tokenizer.py`: tokenizer training and tokenizer
  behavior.
- `test_model.py` and `test_nn_utils.py`: neural-network primitives and utility
  math.
- `test_data.py`, `test_optimizer.py`, and `test_serialization.py`: data loading,
  optimizer/scheduler behavior, and checkpoints.
- `test_rust_bpe_parity.py`: optional parity checks for the Rust BPE crate. These
  tests are skipped when Cargo is unavailable.
- [`fixtures/`](fixtures/): small corpora, reference tokenizer files, and model
  fixtures.
- [`_snapshots/`](_snapshots/): numerical snapshot files used by snapshot tests.

## Current Adapter Status

Implemented adapter paths cover tokenizer training, tokenizer construction,
linear layers, embeddings, RoPE, RMSNorm, SiLU/SwiGLU, softmax, scaled
dot-product attention, and causal multi-head self-attention.

Adapter paths that still raise `NotImplementedError` cover transformer-block and
language-model composition, batch sampling, cross-entropy, gradient clipping,
AdamW, cosine learning-rate scheduling, and checkpoint save/load utilities.

## Common Commands

```sh
uv run pytest
uv run pytest tests/test_tokenizer.py
uv run pytest tests/test_model.py
uv run pytest tests/test_train_bpe.py
uv run pytest tests/test_rust_bpe_parity.py
```

Some tokenizer tests compare against `tiktoken.get_encoding("gpt2")`; first-time
runs on a new machine may need network access for tiktoken's cached GPT-2
assets.

Do not change test assertions to make implementation tests pass. Update
[`../cs336_basics/`](../cs336_basics/) and the thin adapter functions instead.
