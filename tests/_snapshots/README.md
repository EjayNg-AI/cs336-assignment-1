# tests/_snapshots

This folder contains saved numerical reference outputs used by the pytest
snapshot tests in [`../`](../).

## Contents

- `.npz` files for model layers, attention, transformer blocks, the language
  model, and AdamW.
- `test_train_bpe_special_tokens.pkl` for tokenizer training behavior with
  special tokens.

## Maintenance Notes

Treat these files as test fixtures, not generated scratch output. Update them
only when the assignment reference behavior intentionally changes, and keep the
related test change in the same review.
