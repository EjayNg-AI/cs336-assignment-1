# BPE Tokenizer Notes

## Run the enhanced BPE trainer

The enhanced BPE trainer is an optional large-corpus variant of the tokenizer
trainer. It is not wired into `tests/adapters.py`; import it directly when you
want multiprocessing pre-token counting and saved training artifacts.

After downloading the full data files, the repository-level shell wrappers can
run the full TinyStories and OpenWebText BPE jobs:

```sh
bash run_tinystories_bpe_enhanced.sh
bash run_openwebtext_bpe_enhanced.sh
```

The wrappers add the `<|endoftext|>` special token and default to:

- TinyStories: `data/TinyStoriesV2-GPT4-train.txt`, vocabulary target 10,000,
  output `data/tinystories_bpe_10000/`.
- OpenWebText: `data/owt_train.txt`, vocabulary target 32,000, output
  `data/openwebtext_bpe_32000/`.

Each run writes `vocab.pkl`, `merges.pkl`, `vocab.json`, `merges.txt`, and
`metadata.json` into its output directory. If `output_dir` is omitted when
calling the trainer directly, artifacts are written beside the input corpus in
`<input_stem>_bpe_<vocab_size>/`.

Local full-corpus runs with `num_workers=8`, 64 MiB chunks, and
`heap_rebuild_factor=3.0` produced:

| Corpus | Vocab target | Total time | Longest token | Slowest phase |
| --- | ---: | ---: | --- | --- |
| TinyStories | 10,000 | 0 min 52.50 sec | 15-byte ties: `" accomplishment"`, `" disappointment"`, `" responsibility"` | `pretoken_counting`: 0 min 50.18 sec |
| OpenWebText | 32,000 | 17 min 48.67 sec | 64-byte ties: 64 hyphens and repeated mojibake bytes | `merge_loop`: 13 min 3.97 sec |

These BPE trainer runs are CPU/Python multiprocessing jobs; they do not use the
GPU.

## Run the optimized Rust BPE trainer and encoder

The repository also includes an additive Rust implementation of the enhanced
byte-level BPE trainer and encoder under `crates/cs336_bpe_rs/`. It is a
correctness-equivalent sibling of the Python enhanced trainer/tokenizer, not a
replacement for `tests/adapters.py` or the submitted Python assignment path.

Use the current optimized Rust release binaries for future large BPE training
and encoding runs. The optimized trainer keeps the same vocabulary/merge
semantics while reducing merge-loop allocation overhead on large-vocabulary
OpenWebText-style jobs. The optimized encoder batches token-byte writes and
SHA-256 updates, which substantially improves full-corpus `.npy` serialization
without changing token IDs or metadata contracts.

Build release binaries before timing or running large corpora:

```sh
cargo build --release -p cs336_bpe_rs --bins
```

Example full TinyStories training command:

```sh
target/release/cs336-bpe-train \
  --input data/TinyStoriesV2-GPT4-train.txt \
  --vocab-size 10000 \
  --special-token '<|endoftext|>' \
  --num-workers 8 \
  --chunk-bytes 67108864 \
  --heap-rebuild-factor 3.0 \
  --output-dir data/rust/tinystories_bpe_10000
```

Example full OpenWebText training command:

```sh
target/release/cs336-bpe-train \
  --input data/owt_train.txt \
  --vocab-size 32000 \
  --special-token '<|endoftext|>' \
  --num-workers 8 \
  --chunk-bytes 67108864 \
  --heap-rebuild-factor 3.0 \
  --output-dir data/rust/owt_bpe_32000
```

The Rust trainer writes language-neutral artifacts only:

- `vocab.json`
- `merges.txt`
- `metadata.json`

It intentionally does not write Python `vocab.pkl` / `merges.pkl` files. The
Rust encoder can consume the generated JSON and text artifacts and can write
either JSON token IDs for small parity checks or NumPy `.npy` token-ID arrays
for full-corpus serialization:

```sh
target/release/cs336-bpe-encode \
  --vocab data/rust/tinystories_bpe_10000/vocab.json \
  --merges data/rust/tinystories_bpe_10000/merges.txt \
  --special-token '<|endoftext|>' \
  --input data/TinyStoriesV2-GPT4-valid.txt \
  --output-ids-json data/rust/tinystories_bpe_10000/valid_ids.json
```

```sh
target/release/cs336-bpe-encode \
  --vocab data/rust/tinystories_bpe_10000/vocab.json \
  --merges data/rust/tinystories_bpe_10000/merges.txt \
  --special-token '<|endoftext|>' \
  --input data/TinyStoriesV2-GPT4-valid.txt \
  --output-ids-npy data/bpe_tokenized_corpora_rs/tinystories/valid.npy \
  --metadata-json data/bpe_tokenized_corpora_rs/tinystories/valid.json \
  --manifest-json data/bpe_tokenized_corpora_rs/manifest.json \
  --split-name tinystories_valid \
  --corpus tinystories \
  --split valid
```

For standard full-corpus token-ID serialization, prefer the Rust wrapper with a
fresh output directory so previous token arrays are not overwritten:

```sh
EXPERIMENT3_OUTPUT_DIR=data/bpe_tokenized_corpora_rs_new \
TINYSTORIES_TOKENIZER_DIR=data/rust/tinystories_bpe_10000 \
OWT_TOKENIZER_DIR=data/rust/owt_bpe_32000 \
SPLITS="tinystories_train tinystories_valid owt_train owt_valid" \
bash run_bpe_experiment_3_tokenization_rs.sh
```

Use `FORCE=1` only when intentionally replacing outputs in the selected
`EXPERIMENT3_OUTPUT_DIR`.

A completed full TinyStories Rust run is stored under
`data/rust/tinystories_bpe_10000/`, including `run_timing.txt`. Compared with
the current Python enhanced metadata in `data/tinystories_bpe_10000/`, the
matching Rust run produced:

| Implementation | Total time |
| --- | ---: |
| Python enhanced trainer | 85.57 sec |
| Rust trainer metadata | 30.07 sec |
| Rust `/usr/bin/time` wall clock | 30.10 sec |

The Rust run was about 2.84x faster overall for this TinyStories configuration.
The full run matched the Python trainer statistics: 10,000 final vocabulary
items, 9,743 merges, 59,933 unique pre-tokens, and 536,592,168 total
pre-tokens. `merges.txt` was byte-for-byte identical to the Python output, and
`vocab.json` was equal after parsing as JSON. Raw `vocab.json` bytes differ
because Python and Rust serialize JSON strings differently.

See `RUST_BPE_IMPLEMENTATION.md` for implementation details, parity notes, and
the full phase-level timing table.

Recent optimization findings are also documented there. On one warmup plus one
timed run for full TinyStories training and two delimiter-aligned OpenWebText
training subsamples of similar size, the optimized Rust paths improved the
geometric-mean wall-clock time by about `1.49x`. The encoder saw the largest
gain: full TinyStories `.npy` serialization improved from `240.73s` to
`105.72s`, and OpenWebText sample serialization improved by about `1.83x` to
`1.89x`. Trainer output parity and encoder token-stream SHA-256 checks matched
for all benchmarked tasks.

## BPE tokenizer experiment artifacts

`BPE_tokenizer.ipynb` contains the tokenizer experiment writeups. The sampled
documents and encoded sample IDs are stored under `bpe_samples/`:

- `bpe_samples/tinystories/` contains the deterministic TinyStories document
  samples and `manifest.json`.
- `bpe_samples/openwebtext/` contains the deterministic OpenWebText document
  samples and `manifest.json`.
- `bpe_samples/ids/` contains JSON-serialized token IDs for the sample
  tokenizations.
- `bpe_samples/experiment_1_2_summary.json` records the measured compression
  ratios and throughput used in the notebook answers.

Measured sample compression ratios were 4.137 bytes/token for the TinyStories
tokenizer on TinyStories samples, 3.774 bytes/token for the OpenWebText
tokenizer on OpenWebText samples, and 2.816 bytes/token for the TinyStories
tokenizer on OpenWebText samples.

## Run Experiment 3 full-corpus tokenization

Experiment 3 token-ID serialization is run from the shell, not from the
notebook:

```sh
bash run_bpe_experiment_3_tokenization.sh
```

The script imports `cs336_basics.tokenizer.Tokenizer`, loads the trained
`vocab.pkl` and `merges.pkl` files, streams each corpus split through
`Tokenizer.encode_iterable`, and writes NumPy `uint16` arrays under
`data/bpe_tokenized_corpora/`.

Default inputs and tokenizer artifact directories:

- TinyStories train: `data/TinyStoriesV2-GPT4-train.txt` with
  `data/tinystories_bpe_10000/`.
- TinyStories validation: `data/TinyStoriesV2-GPT4-valid.txt` with
  `data/tinystories_bpe_10000/`.
- OpenWebText train: `data/owt_train.txt` with `data/owt_bpe_32000/`.
- OpenWebText validation: `data/owt_valid.txt` with `data/owt_bpe_32000/`.

If the OpenWebText tokenizer was produced by the current training wrapper, the
script also falls back from `data/owt_bpe_32000/` to
`data/openwebtext_bpe_32000/` when the shorter directory name is absent.

Default outputs:

- `data/bpe_tokenized_corpora/tinystories/train.npy`
- `data/bpe_tokenized_corpora/tinystories/train.json`
- `data/bpe_tokenized_corpora/tinystories/valid.npy`
- `data/bpe_tokenized_corpora/tinystories/valid.json`
- `data/bpe_tokenized_corpora/openwebtext/train.npy`
- `data/bpe_tokenized_corpora/openwebtext/train.json`
- `data/bpe_tokenized_corpora/openwebtext/valid.npy`
- `data/bpe_tokenized_corpora/openwebtext/valid.json`
- `data/bpe_tokenized_corpora/manifest.json`

Each `.npy` file is a flat one-dimensional `uint16` token-ID array suitable for
memory-mapped training reads, for example:

```py
import numpy as np

ids = np.load("data/bpe_tokenized_corpora/openwebtext/train.npy", mmap_mode="r")
```

Each sidecar JSON records the source corpus, tokenizer artifact paths, shape,
token count, compression ratio, throughput, and a SHA-256 hash of the little
endian `uint16` token stream. The top-level `manifest.json` collects the split
metadata for later retrieval.

A completed local run produced these token arrays:

| Split | Tokens | Bytes/token | `.npy` size |
| --- | ---: | ---: | ---: |
| TinyStories train | 541,229,347 | 4.116 | 1,082,458,822 bytes |
| TinyStories validation | 5,465,883 | 4.117 | 10,931,894 bytes |
| OpenWebText train | 2,727,120,452 | 4.371 | 5,454,241,032 bytes |
| OpenWebText validation | 66,401,098 | 4.367 | 132,802,324 bytes |

Useful environment overrides:

```sh
SPLITS="tinystories_valid owt_valid" bash run_bpe_experiment_3_tokenization.sh
FORCE=1 SPLITS="owt_train" bash run_bpe_experiment_3_tokenization.sh
OWT_TOKENIZER_DIR=data/openwebtext_bpe_32000 bash run_bpe_experiment_3_tokenization.sh
```

Supported split names are `tinystories_train`, `tinystories_valid`,
`owt_train`, and `owt_valid`. Existing complete outputs are skipped unless
`FORCE=1` is set. The script also defaults `UV_CACHE_DIR` to
`data/.uv-cache`, so generated arrays, metadata, temporary files, and cache
writes from the run live under `data/`, which is ignored by Git.

The equivalent Rust encoder wrapper is:

```sh
bash run_bpe_experiment_3_tokenization_rs.sh
```

It consumes `vocab.json` and `merges.txt`, writes the same flat little-endian
`uint16` `.npy` array format plus sidecar JSON metadata, and defaults to
`data/bpe_tokenized_corpora_rs/` so it does not overwrite Python-generated
arrays. It accepts the same `SPLITS`, `SPECIAL_TOKEN`,
`TINYSTORIES_TOKENIZER_DIR`, `OWT_TOKENIZER_DIR`, and `FORCE=1` overrides, plus
`STREAM_CHUNK_BYTES` for the Rust file reader. The byte-level `.npy`
construction and temporary-file write sequence are documented in
[`RUST_BPE_IMPLEMENTATION.md`](RUST_BPE_IMPLEMENTATION.md#numpy-npy-serialization).
