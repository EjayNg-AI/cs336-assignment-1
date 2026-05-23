# tests/fixtures

This folder stores small fixture data consumed by tests in [`../`](../).

## Contents

- `address.txt`, `german.txt`, `tinystories_sample.txt`,
  `tinystories_sample_5M.txt`, and `corpus.en`: text corpora for tokenizer and
  BPE training tests.
- `gpt2_vocab.json` and `gpt2_merges.txt`: GPT-2 tokenizer reference artifacts.
- `train-bpe-reference-vocab.json` and `train-bpe-reference-merges.txt`:
  reference outputs for BPE training tests.
- `special_token_trailing_newlines.txt` and
  `special_token_double_newlines_non_whitespace.txt`: edge cases for special
  token handling.
- `ts_tests/`: transformer model fixture files used by model tests.

## Maintenance Notes

Fixture files are intentionally small enough to keep in Git. Large local corpora
belong in [`../../data/`](../../data/) and should remain ignored.

When changing fixtures, update the corresponding tests and document why the
reference behavior changed.
