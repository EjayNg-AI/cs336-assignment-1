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
