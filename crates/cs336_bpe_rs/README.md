# cs336_bpe_rs

Rust implementation of the repository's enhanced byte-level BPE trainer and
tokenizer.

This crate is an additive sibling of the Python implementation in
[`../../cs336_basics/`](../../cs336_basics/). It is intended to match the Python
enhanced trainer/tokenizer semantics while providing faster large-corpus Rust
training and encoding paths. It is not the submitted Python assignment path wired through
[`../../tests/adapters.py`](../../tests/adapters.py).

## Layout

- [`src/lib.rs`](src/lib.rs): public library surface.
- [`src/bin/train_bpe.rs`](src/bin/train_bpe.rs): `cs336-bpe-train` CLI.
- [`src/bin/encode_bpe.rs`](src/bin/encode_bpe.rs): `cs336-bpe-encode` CLI.
- [`src/trainer/`](src/trainer/): BPE training state, counts, heap, merge, and
  artifact writing.
- [`src/encoder/`](src/encoder/): tokenizer loading, merge application, and
  streaming encoding.
- [`src/pretokenizer.rs`](src/pretokenizer.rs): GPT-style pre-tokenization.
- [`Cargo.toml`](Cargo.toml): crate metadata, library, binaries, and dependencies.

## Binaries

For large corpora, build release binaries and use `target/release/...` directly:

```sh
cargo build --release -p cs336_bpe_rs --bins
```

Train a tokenizer:

```sh
target/release/cs336-bpe-train \
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
target/release/cs336-bpe-encode \
  --vocab data/tinystories_bpe_10000_rs/vocab.json \
  --merges data/tinystories_bpe_10000_rs/merges.txt \
  --special-token '<|endoftext|>' \
  --input data/TinyStoriesV2-GPT4-valid.txt \
  --output-ids-json data/tinystories_valid_ids_rs.json
```

The encoder also accepts `--stream-chunk-bytes <n>` to exercise the streaming
encoding path. For full-corpus tokenization, it can write memory-mappable
NumPy `uint16` arrays and sidecar metadata:

```sh
target/release/cs336-bpe-encode \
  --vocab data/tinystories_bpe_10000/vocab.json \
  --merges data/tinystories_bpe_10000/merges.txt \
  --special-token '<|endoftext|>' \
  --input data/TinyStoriesV2-GPT4-valid.txt \
  --output-ids-npy data/bpe_tokenized_corpora_rs/tinystories/valid.npy \
  --metadata-json data/bpe_tokenized_corpora_rs/tinystories/valid.json \
  --manifest-json data/bpe_tokenized_corpora_rs/manifest.json \
  --split-name tinystories_valid \
  --corpus tinystories \
  --split valid
```

The current optimized encoder batches token-byte writes and SHA-256 updates
during `.npy` serialization. The current optimized trainer reduces allocation
overhead in large-vocabulary merge loops while preserving deterministic merge
selection and artifact parity. Use these release binaries for future full
TinyStories/OpenWebText training and encoding runs.

## Artifacts

The trainer writes language-neutral artifacts:

- `vocab.json`
- `merges.txt`
- `metadata.json`

It intentionally does not write Python pickle files. Python remains responsible
for pickle artifacts when those are needed. The encoder's `.npy` mode writes
flat little-endian `uint16` token-ID arrays compatible with
`np.load(..., mmap_mode="r")`.

For standard full-corpus tokenization, prefer the repository wrapper with a
fresh output directory:

```sh
EXPERIMENT3_OUTPUT_DIR=data/bpe_tokenized_corpora_rs_new \
TINYSTORIES_TOKENIZER_DIR=data/tinystories_bpe_10000 \
OWT_TOKENIZER_DIR=data/owt_bpe_32000 \
SPLITS="tinystories_train tinystories_valid owt_train owt_valid" \
bash run_bpe_experiment_3_tokenization_rs.sh
```

Use `FORCE=1` only when intentionally replacing outputs in the selected
`EXPERIMENT3_OUTPUT_DIR`.

## Validation

Run Rust unit tests:

```sh
cargo test -p cs336_bpe_rs
```

Run Python parity tests:

```sh
uv run pytest tests/test_rust_bpe_parity.py
```

More implementation detail is in
[`../../RUST_BPE_IMPLEMENTATION.md`](../../RUST_BPE_IMPLEMENTATION.md).
