# crates/cs336_bpe_rs/src

This folder contains the Rust library and binaries for the optional
`cs336_bpe_rs` BPE implementation. See [`../README.md`](../README.md) for CLI
usage and [`../../../RUST_BPE_IMPLEMENTATION.md`](../../../RUST_BPE_IMPLEMENTATION.md)
for design notes.

## Layout

- [`lib.rs`](lib.rs): public module declarations and re-exports.
- [`main.rs`](main.rs): placeholder binary entrypoint retained with the crate.
- [`bin/`](bin/): command-line binaries for training and encoding.
- [`trainer/`](trainer/): BPE training pipeline.
- [`encoder/`](encoder/): artifact loading and tokenization.
- [`pretokenizer.rs`](pretokenizer.rs): GPT-style pre-tokenization.
- [`chunking.rs`](chunking.rs): corpus chunk-boundary support.
- [`bytes_repr.rs`](bytes_repr.rs): byte-token display and parsing helpers.
- [`config.rs`](config.rs): runtime configuration helpers.
- [`errors.rs`](errors.rs): crate-level error types.

## Status

The Rust implementation is additive. It is tested through Cargo tests and
Python parity tests, but Python code in [`../../../cs336_basics/`](../../../cs336_basics/)
remains the submitted assignment path.
