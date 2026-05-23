# CS336 Assignment 1: Setup Instructions

## Prerequisites

- **Python 3.12 or 3.13** (required by `pyproject.toml`)
- **Git** (to clone the repository)
- **curl** (for installing `uv`)
- **wget** and **gunzip** (only needed if you download the full training datasets)
- **Rust/Cargo** (optional, only needed for the additive Rust BPE trainer and
  encoder under `crates/cs336_bpe_rs/`)

---

## Step 1: Install uv (Python Package Manager)

This project uses **uv** instead of pip/conda for environment management.

### On Linux/macOS/WSL:

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

### Verify installation:

```bash
uv --version
command -v uv
```

For WSL-specific PATH setup, virtual environment usage, dependency installation patterns (`uv sync`, `uv pip`, `requirements.txt`), running scripts, and test commands, use the consolidated guide in [`uv-docs/README.md`](./uv-docs/README.md).

---

## Step 2: Clone the Repository (if not already done)

```bash
git clone <repository-url>
cd <repo-directory>
```

---

## Step 3: Create Virtual Environment and Install Dependencies

With **uv**, this happens automatically when you run any command. Simply run:

```bash
uv sync
```

This will:
1. Create a virtual environment (in `.venv/`)
2. Install a compatible Python version if needed
3. Install all dependencies from `uv.lock`

The environment includes:
- **PyTorch** (`torch~=2.11.0`)
- **NumPy**, **einops**, **einx** (tensor operations)
- **jaxtyping** (type hints)
- **pytest** (testing)
- **regex** (advanced regex)
- **tiktoken**, **tqdm**, **wandb** (ML utilities)

Rust dependencies are managed separately by Cargo. They are not installed by
`uv sync`; Cargo resolves them from `Cargo.lock` when you run a Rust build or
test command.

---

## Step 4: Verify Setup by Running Tests

```bash
uv run pytest
```

In a fresh starter checkout, the tests are expected to fail because `tests/adapters.py`
is still a set of stubs that raise `NotImplementedError`. This is the expected
starting point for the assignment.

Some tokenizer comparison tests call `tiktoken.get_encoding("gpt2")`. On a machine
where `tiktoken` has not cached the GPT-2 assets yet, those tests may need network
access the first time they run.

To run a specific test file:

```bash
uv run pytest tests/test_model.py      # Model component tests
uv run pytest tests/test_tokenizer.py  # Tokenizer tests
uv run pytest tests/test_optimizer.py  # AdamW optimizer tests
```

If Cargo is installed, the optional Rust BPE parity tests can be run with:

```bash
uv run pytest tests/test_rust_bpe_parity.py
```

The Rust crate's native test suite can be run with:

```bash
cargo test -p cs336_bpe_rs
```

---

## Step 5: Download Training Data

Create a data directory and download the datasets:

```bash
mkdir -p data
cd data

# TinyStories dataset
wget https://huggingface.co/datasets/roneneldan/TinyStories/resolve/main/TinyStoriesV2-GPT4-train.txt
wget https://huggingface.co/datasets/roneneldan/TinyStories/resolve/main/TinyStoriesV2-GPT4-valid.txt

# OpenWebText sample
wget https://huggingface.co/datasets/stanford-cs336/owt-sample/resolve/main/owt_train.txt.gz
gunzip owt_train.txt.gz
wget https://huggingface.co/datasets/stanford-cs336/owt-sample/resolve/main/owt_valid.txt.gz
gunzip owt_valid.txt.gz

cd ..
```

These downloads are large. As of the current Hugging Face listings, the TinyStories
train file is about 2.23 GB, and the OpenWebText training sample is about 4.59 GB
compressed. The repository ignores `data/`, so downloaded datasets should not be
committed.

---

## Step 6: Start Implementing

Before implementing, read:

- `cs336_assignment1_basics.pdf` for the assignment specification
- `requirements_for_code_produced.md` for allowed and forbidden library usage

Your implementation normally goes in **two places**:

### 1. `cs336_basics/` folder
Create your implementation modules here (e.g., `tokenizer.py`, `model.py`, `optimizer.py`).

### 2. `tests/adapters.py`
This file connects your implementation to the test suite. Fill in the adapter
functions that currently raise `NotImplementedError`. Do not change the test
assertions to make them pass.

Key functions to implement in `adapters.py`:

| Category | Functions |
|----------|-----------|
| **Tokenizer** | `run_train_bpe`, `get_tokenizer` |
| **Model** | `run_linear`, `run_embedding`, `run_swiglu`, `run_silu`, `run_scaled_dot_product_attention`, `run_multihead_self_attention`, `run_multihead_self_attention_with_rope`, `run_rope`, `run_rmsnorm`, `run_transformer_block`, `run_transformer_lm` |
| **Training** | `run_softmax`, `run_cross_entropy`, `get_adamw_cls`, `run_gradient_clipping`, `run_get_lr_cosine_schedule` |
| **Data** | `run_get_batch` |
| **Checkpoints** | `run_save_checkpoint`, `run_load_checkpoint` |

---

## Common uv Commands

```bash
uv run <script.py>         # Run a Python script
uv run pytest              # Run tests
uv add <package>           # Add a new dependency
uv sync                    # Sync environment with lockfile
uv lock --upgrade          # Intentionally upgrade all packages and rewrite uv.lock
```

For this assignment, prefer the checked-in dependencies. Only use `uv add` or
`uv lock --upgrade` if you deliberately want to change project dependencies and
understand the assignment's library-use restrictions.

## Optional Rust BPE Commands

The Rust BPE trainer and encoder are additive tools. They are useful for
large-corpus BPE experiments and parity checks, but they are not wired into
`tests/adapters.py`.

Build optimized Rust binaries:

```bash
cargo build --release -p cs336_bpe_rs
```

Train the TinyStories BPE tokenizer into a Rust-specific data subdirectory:

```bash
target/release/cs336-bpe-train \
  --input data/TinyStoriesV2-GPT4-train.txt \
  --vocab-size 10000 \
  --special-token '<|endoftext|>' \
  --num-workers 8 \
  --chunk-bytes 67108864 \
  --heap-rebuild-factor 3.0 \
  --output-dir data/rust/tinystories_bpe_10000
```

Encode a file with the Rust tokenizer:

```bash
target/release/cs336-bpe-encode \
  --vocab data/rust/tinystories_bpe_10000/vocab.json \
  --merges data/rust/tinystories_bpe_10000/merges.txt \
  --special-token '<|endoftext|>' \
  --input data/TinyStoriesV2-GPT4-valid.txt \
  --output-ids-json data/rust/tinystories_bpe_10000/valid_ids.json
```

More detail is documented in `RUST_BPE_IMPLEMENTATION.md` and
`crates/cs336_bpe_rs/README.md`.

---

## Recommended Workflow

1. Read the assignment PDF and `requirements_for_code_produced.md`
2. Start with the tokenizer (BPE training)
3. Then implement model components (attention, transformer blocks)
4. Then training infrastructure (loss, optimizer, data loading)
5. Run tests frequently: `uv run pytest tests/test_<component>.py`
6. Treat the files in `tests/_snapshots/` as test fixtures used by pytest

---

## Troubleshooting

### uv not found after installation
Use the WSL troubleshooting steps in [`uv-docs/README.md`](./uv-docs/README.md), including PATH fixes and install-location checks.

### Dependency issues
```bash
uv sync --reinstall  # Force reinstall packages from the checked-in lockfile
```

Avoid `uv lock --upgrade` for normal setup troubleshooting, because it rewrites
`uv.lock` and can move you away from the assignment's tested dependency set.

### uv cache permission issues
If `uv` cannot write to its default cache directory, point the cache at a writable
location:

```bash
UV_CACHE_DIR=.uv-cache uv sync
UV_CACHE_DIR=.uv-cache uv run pytest
```

### For more uv help
See the offline documentation in `uv-docs/`:
- `uv-docs/uv-projects.md` - Project configuration
- `uv-docs/uv-dependencies.md` - Dependency management
- `uv-docs/uv-github-actions.md` - CI/CD setup
