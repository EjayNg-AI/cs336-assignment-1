# crates/cs336_bpe_rs/src/trainer

This folder implements the Rust byte-level BPE training pipeline.

## Important Files

- [`mod.rs`](mod.rs): trainer entrypoint and pipeline orchestration.
- [`state.rs`](state.rs): mutable merge-loop state.
- [`counts.rs`](counts.rs): pre-token and pair counting.
- [`heap.rs`](heap.rs): deterministic candidate-pair heap.
- [`merge.rs`](merge.rs): word-rewrite and pair-update logic.
- [`artifacts.rs`](artifacts.rs): `vocab.json`, `merges.txt`, and
  `metadata.json` writing.

## Inputs and Outputs

Inputs are UTF-8 corpus files and trainer settings from the CLI or library API.
Outputs are language-neutral tokenizer artifacts; Python pickle files are not
written by the Rust trainer.

The trainer is optimized for large-corpus enhanced BPE runs while preserving
the Python enhanced trainer's merge semantics. Heap entries share token byte
storage, and the merge loop updates adjacent-pair counts by direct window scans
instead of allocating per-word frequency maps.

Run validation from the repository root:

```sh
cargo test -p cs336_bpe_rs
uv run pytest tests/test_rust_bpe_parity.py
```
