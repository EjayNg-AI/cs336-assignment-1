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

After downloading the full TinyStories data, run:

```sh
uv run python -u - <<'PY'
from cs336_basics.train_bpe_enhanced import train_bpe

train_bpe(
    input_path="data/TinyStoriesV2-GPT4-train.txt",
    vocab_size=10_000,
    special_tokens=["<|endoftext|>"],
    num_workers=8,
    chunk_bytes=64 * 1024 * 1024,
    heap_rebuild_factor=3.0,
    output_dir="data/tinystories_bpe_10000",
)
PY
```

This writes `vocab.pkl`, `merges.pkl`, `vocab.json`, and `merges.txt` into
`data/tinystories_bpe_10000/`. If `output_dir` is omitted, artifacts are written
beside the input corpus in `<input_stem>_bpe_<vocab_size>/`.
