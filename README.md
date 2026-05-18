# CS336 Spring 2025 Assignment 1: Basics

For a full description of the assignment, see the assignment handout at
[cs336_assignment1_basics.pdf](./cs336_assignment1_basics.pdf)

## Setup

### Environment
We manage our environments with `uv` to ensure reproducibility, portability, and ease of use.
Install `uv` [here](https://github.com/astral-sh/uv#installation) (recommended), or run `pip install uv`/`brew install uv`.
We recommend reading a bit about managing projects in `uv` [here](https://docs.astral.sh/uv/guides/projects/#managing-dependencies) (you will not regret it!).

You can now run any code in the repo using
```sh
uv run <python_file_path>
```
and the environment will be automatically solved and activated when necessary.

### Run unit tests


```sh
uv run pytest
```

Initially, all tests should fail with `NotImplementedError`s.
To connect your implementation to the tests, complete the
functions in [./tests/adapters.py](./tests/adapters.py).

### Download data
Download the TinyStories data and a subsample of OpenWebText

``` sh
mkdir -p data
cd data

wget https://huggingface.co/datasets/roneneldan/TinyStories/resolve/main/TinyStoriesV2-GPT4-train.txt
wget https://huggingface.co/datasets/roneneldan/TinyStories/resolve/main/TinyStoriesV2-GPT4-valid.txt

wget https://huggingface.co/datasets/stanford-cs336/owt-sample/resolve/main/owt_train.txt.gz
gunzip owt_train.txt.gz
wget https://huggingface.co/datasets/stanford-cs336/owt-sample/resolve/main/owt_valid.txt.gz
gunzip owt_valid.txt.gz

cd ..
```

### Run the enhanced BPE trainer

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

### BPE tokenizer experiment artifacts

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
tokenizer on OpenWebText samples. Experiment 3 full-dataset serialization was
aborted because the expected full-corpus `uint16`, JSON, and pickle outputs are
resource intensive; no full-dataset Experiment 3 artifacts are retained under
`bpe_samples/`.
