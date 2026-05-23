# cs336_bpe_rs

Rust implementation of the repository's enhanced byte-level BPE trainer and tokenizer.

This crate is an additive sibling of the Python implementation in `cs336_basics/`.
It is intended to match the Python enhanced trainer/tokenizer semantics before
optimizing for speed or memory usage.

## Binaries

Train a tokenizer:

```sh
cargo run -p cs336_bpe_rs --bin cs336-bpe-train -- \
  --input data/TinyStoriesV2-GPT4-train.txt \
  --vocab-size 10000 \
  --special-token '<|endoftext|>' \
  --num-workers 8 \
  --chunk-bytes 67108864 \
  --heap-rebuild-factor 3.0 \
  --output-dir data/tinystories_bpe_10000_rs
```

Encode a file:

```sh
cargo run -p cs336_bpe_rs --bin cs336-bpe-encode -- \
  --vocab data/tinystories_bpe_10000_rs/vocab.json \
  --merges data/tinystories_bpe_10000_rs/merges.txt \
  --special-token '<|endoftext|>' \
  --input data/TinyStoriesV2-GPT4-valid.txt \
  --output-ids-json data/tinystories_valid_ids_rs.json
```

The encoder also accepts `--stream-chunk-bytes <n>` to exercise the streaming
encoding path.

## Artifacts

The trainer writes language-neutral artifacts:

- `vocab.json`
- `merges.txt`
- `metadata.json`

It intentionally does not write Python pickle files. Python remains responsible
for pickle artifacts when those are needed.

## Validation

Run Rust unit tests:

```sh
cargo test -p cs336_bpe_rs
```

Run Python parity tests:

```sh
uv run pytest tests/test_rust_bpe_parity.py
```
