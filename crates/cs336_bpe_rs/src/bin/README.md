# crates/cs336_bpe_rs/src/bin

This folder contains the Rust command-line entrypoints for the optional BPE
implementation.

## Binaries

- [`train_bpe.rs`](train_bpe.rs): parses trainer arguments and calls the Rust BPE
  training library.
- [`encode_bpe.rs`](encode_bpe.rs): loads tokenizer artifacts and writes token IDs
  for an input text file.

## Common Commands

From the repository root:

```sh
cargo run -p cs336_bpe_rs --bin cs336-bpe-train -- --help
cargo run -p cs336_bpe_rs --bin cs336-bpe-encode -- --help
```

Full examples are in [`../../README.md`](../../README.md).
