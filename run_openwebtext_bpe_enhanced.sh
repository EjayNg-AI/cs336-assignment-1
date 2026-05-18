#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

export INPUT_PATH="${INPUT_PATH:-data/owt_train.txt}"
export OUTPUT_DIR="${OUTPUT_DIR:-data/openwebtext_bpe_32000}"
export VOCAB_SIZE="${VOCAB_SIZE:-32000}"
export SPECIAL_TOKEN="${SPECIAL_TOKEN:-<|endoftext|>}"
export BPE_NUM_WORKERS="${BPE_NUM_WORKERS:-8}"
export BPE_CHUNK_BYTES="${BPE_CHUNK_BYTES:-67108864}"
export BPE_HEAP_REBUILD_FACTOR="${BPE_HEAP_REBUILD_FACTOR:-3.0}"

uv run python -u - <<'PY'
import os

from cs336_basics.train_bpe_enhanced import train_bpe


train_bpe(
    input_path=os.environ["INPUT_PATH"],
    vocab_size=int(os.environ["VOCAB_SIZE"]),
    special_tokens=[os.environ["SPECIAL_TOKEN"]],
    num_workers=int(os.environ["BPE_NUM_WORKERS"]),
    chunk_bytes=int(os.environ["BPE_CHUNK_BYTES"]),
    heap_rebuild_factor=float(os.environ["BPE_HEAP_REBUILD_FACTOR"]),
    output_dir=os.environ["OUTPUT_DIR"],
)
PY
