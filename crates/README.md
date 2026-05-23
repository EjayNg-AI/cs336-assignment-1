# crates

This folder contains optional Rust support code for the repository.

## Contents

- [`cs336_bpe_rs/`](cs336_bpe_rs/): Rust byte-level BPE trainer and encoder.

The root [`../Cargo.toml`](../Cargo.toml) declares a Cargo workspace containing
this crate. Rust code is additive: it supports faster BPE experiments and parity
checks, but it is not wired into [`../tests/adapters.py`](../tests/adapters.py)
as the submitted Python assignment path.

## Common Commands

```sh
cargo test -p cs336_bpe_rs
cargo build --release -p cs336_bpe_rs
uv run pytest tests/test_rust_bpe_parity.py
```

See [`cs336_bpe_rs/README.md`](cs336_bpe_rs/README.md) and
[`../RUST_BPE_IMPLEMENTATION.md`](../RUST_BPE_IMPLEMENTATION.md) for CLI usage,
artifact formats, and design notes.
