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
tokens, and emits JSON token-ID arrays through the `cs336-bpe-encode` binary.
It does not emit NumPy `.npy` arrays; Python scripts handle those full-corpus
training artifacts.
