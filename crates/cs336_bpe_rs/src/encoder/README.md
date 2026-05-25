# crates/cs336_bpe_rs/src/encoder

This folder implements Rust tokenizer loading and encoding for artifacts written
by the Rust BPE trainer.

## Important Files

- [`mod.rs`](mod.rs): encoder module exports.
- [`tokenizer.rs`](tokenizer.rs): tokenizer type and whole-string encoding.
- [`streaming.rs`](streaming.rs): chunked file encoding path.
- [`vocab.rs`](vocab.rs): vocabulary artifact loading.
- [`merges.rs`](merges.rs): merge artifact loading and rank lookup.

## Inputs and Outputs

The encoder reads `vocab.json` and `merges.txt`, applies configured special
tokens, and emits token IDs through the `cs336-bpe-encode` binary. Small runs
can write JSON token-ID arrays. Full-corpus runs should write flat NumPy
`uint16` `.npy` arrays with sidecar metadata so downstream training can load
them with `np.load(..., mmap_mode="r")`.

The `.npy` path streams UTF-8 chunks through the tokenizer and buffers emitted
little-endian token bytes before writing and hashing them. This keeps token IDs
and SHA-256 metadata identical while reducing per-token I/O overhead.
