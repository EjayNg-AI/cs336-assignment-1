# cs336_basics

[`cs336_basics`](../README.md) is the submitted Python implementation package for
CS336 Assignment 1. Tests call this package through
[`tests/adapters.py`](../tests/adapters.py); keep adapters thin and put
substantive implementation logic here.

## Folder Contents

- [`train_bpe.py`](train_bpe.py): assignment-facing byte-level BPE trainer.
- [`tokenizer.py`](tokenizer.py): BPE tokenizer encoder/decoder with special-token
  handling, GPT-style pre-tokenization, streaming encoding, and UTF-8 replacement
  decoding.
- [`train_bpe_enhanced.py`](train_bpe_enhanced.py): optional large-corpus BPE
  trainer with multiprocessing pre-token counting and artifact writing.
- [`pretokenization_example.py`](pretokenization_example.py): instructional helper
  for chunking a corpus on special-token boundaries.
- [`nn_linear_embedding_rope_rmsnorm.py`](nn_linear_embedding_rope_rmsnorm.py):
  custom `Linear`, `Embedding`, `RotaryPositionalEmbedding`, and `RMSNorm`.
- [`nn_feedforward.py`](nn_feedforward.py): manual stable sigmoid, SiLU, and
  SwiGLU feed-forward network.
- [`nn_attention.py`](nn_attention.py): manual softmax, scaled dot-product
  attention, and causal multi-head self-attention.
- [`__init__.py`](__init__.py): package metadata lookup.

## Implementation Status

Implemented and connected through adapters:

- BPE training and tokenizer construction
- linear and embedding modules
- RoPE and RMSNorm
- SiLU and SwiGLU
- softmax, scaled dot-product attention, and causal multi-head self-attention

Still missing in the assignment-facing adapter path:

- transformer block and transformer language model
- language-model batch sampling
- cross-entropy loss
- gradient clipping
- AdamW optimizer and cosine learning-rate schedule
- checkpoint save/load utilities

## Development Rules

- Follow [`../requirements_for_code_produced.md`](../requirements_for_code_produced.md)
  for submitted implementation code.
- Do not use prebuilt PyTorch modules or functions for components the assignment
  asks this package to implement manually.
- Add new submitted implementation modules here, then expose them through thin
  functions in [`../tests/adapters.py`](../tests/adapters.py).
- When submitted code is added, moved, or reorganized, update
  [`../repository_structure.md`](../repository_structure.md).

## Common Commands

```sh
uv run pytest tests/test_train_bpe.py
uv run pytest tests/test_tokenizer.py
uv run pytest tests/test_model.py
```

The enhanced BPE trainer expects full corpus files under `data/`; see
[`../data/README.md`](../data/README.md) and
[`../BPE_TOKENIZER.md`](../BPE_TOKENIZER.md).
