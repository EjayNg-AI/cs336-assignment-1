#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

export EXPERIMENT3_OUTPUT_DIR="${EXPERIMENT3_OUTPUT_DIR:-data/bpe_tokenized_corpora}"
export TINYSTORIES_TOKENIZER_DIR="${TINYSTORIES_TOKENIZER_DIR:-data/tinystories_bpe_10000}"
export OWT_TOKENIZER_DIR="${OWT_TOKENIZER_DIR:-data/owt_bpe_32000}"
export SPECIAL_TOKEN="${SPECIAL_TOKEN:-<|endoftext|>}"
export SPLITS="${SPLITS:-tinystories_train tinystories_valid owt_train owt_valid}"
export TEXT_CHUNK_CHARS="${TEXT_CHUNK_CHARS:-1048576}"
export TOKEN_BUFFER_SIZE="${TOKEN_BUFFER_SIZE:-1048576}"
export COPY_BUFFER_TOKENS="${COPY_BUFFER_TOKENS:-16777216}"
export TOKEN_PROGRESS_INTERVAL="${TOKEN_PROGRESS_INTERVAL:-50000000}"
export FORCE="${FORCE:-0}"
export UV_CACHE_DIR="${UV_CACHE_DIR:-data/.uv-cache}"

if [[ "$OWT_TOKENIZER_DIR" == "data/owt_bpe_32000" \
  && ! -f "$OWT_TOKENIZER_DIR/vocab.pkl" \
  && -f "data/openwebtext_bpe_32000/vocab.pkl" ]]; then
  export OWT_TOKENIZER_DIR="data/openwebtext_bpe_32000"
fi

uv run python -u - <<'PY'
from __future__ import annotations

import datetime as dt
import hashlib
import json
import os
import time
from pathlib import Path
from typing import Any

import numpy as np

from cs336_basics.tokenizer import Tokenizer


UINT16_DTYPE = np.dtype("<u2")
UINT16_MAX = np.iinfo(np.uint16).max


def env_int(name: str) -> int:
    value = int(os.environ[name])
    if value < 1:
        raise ValueError(f"{name} must be at least 1")
    return value


def env_flag(name: str) -> bool:
    return os.environ.get(name, "0") in {"1", "true", "TRUE", "yes", "YES"}


def utc_now() -> str:
    return dt.datetime.now(dt.UTC).isoformat().replace("+00:00", "Z")


output_dir = Path(os.environ["EXPERIMENT3_OUTPUT_DIR"])
special_token = os.environ["SPECIAL_TOKEN"]
text_chunk_chars = env_int("TEXT_CHUNK_CHARS")
token_buffer_size = env_int("TOKEN_BUFFER_SIZE")
copy_buffer_tokens = env_int("COPY_BUFFER_TOKENS")
token_progress_interval = env_int("TOKEN_PROGRESS_INTERVAL")
force = env_flag("FORCE")

split_configs: dict[str, dict[str, Any]] = {
    "tinystories_train": {
        "corpus": "tinystories",
        "split": "train",
        "input_path": Path("data/TinyStoriesV2-GPT4-train.txt"),
        "tokenizer_dir": Path(os.environ["TINYSTORIES_TOKENIZER_DIR"]),
    },
    "tinystories_valid": {
        "corpus": "tinystories",
        "split": "valid",
        "input_path": Path("data/TinyStoriesV2-GPT4-valid.txt"),
        "tokenizer_dir": Path(os.environ["TINYSTORIES_TOKENIZER_DIR"]),
    },
    "owt_train": {
        "corpus": "openwebtext",
        "split": "train",
        "input_path": Path("data/owt_train.txt"),
        "tokenizer_dir": Path(os.environ["OWT_TOKENIZER_DIR"]),
    },
    "owt_valid": {
        "corpus": "openwebtext",
        "split": "valid",
        "input_path": Path("data/owt_valid.txt"),
        "tokenizer_dir": Path(os.environ["OWT_TOKENIZER_DIR"]),
    },
}

selected_split_names = os.environ["SPLITS"].split()
unknown_split_names = [name for name in selected_split_names if name not in split_configs]
if unknown_split_names:
    known = ", ".join(sorted(split_configs))
    unknown = ", ".join(unknown_split_names)
    raise ValueError(f"Unknown split(s): {unknown}. Known splits: {known}")

output_dir.mkdir(parents=True, exist_ok=True)
tokenizer_cache: dict[Path, Tokenizer] = {}


def load_tokenizer(tokenizer_dir: Path) -> Tokenizer:
    tokenizer_dir = tokenizer_dir.resolve()
    cached = tokenizer_cache.get(tokenizer_dir)
    if cached is not None:
        return cached

    vocab_path = tokenizer_dir / "vocab.pkl"
    merges_path = tokenizer_dir / "merges.pkl"
    if not vocab_path.is_file():
        raise FileNotFoundError(f"Missing tokenizer vocabulary: {vocab_path}")
    if not merges_path.is_file():
        raise FileNotFoundError(f"Missing tokenizer merges: {merges_path}")

    print(f"Loading tokenizer from {tokenizer_dir}", flush=True)
    tokenizer = Tokenizer.from_files(str(vocab_path), str(merges_path), special_tokens=[special_token])
    max_token_id = max(tokenizer.vocab, default=-1)
    if max_token_id > UINT16_MAX:
        raise ValueError(f"Tokenizer max token ID {max_token_id} exceeds uint16 max {UINT16_MAX}")

    tokenizer_cache[tokenizer_dir] = tokenizer
    return tokenizer


def iter_text_chunks(path: Path):
    with path.open("r", encoding="utf-8", newline="") as f:
        while True:
            chunk = f.read(text_chunk_chars)
            if not chunk:
                break
            yield chunk


def flush_token_buffer(raw_file, token_buffer: list[int], token_stream_sha256) -> tuple[int, int, int]:
    if not token_buffer:
        return 0, UINT16_MAX, 0

    token_array = np.asarray(token_buffer, dtype=UINT16_DTYPE)
    token_stream_sha256.update(token_array.tobytes(order="C"))
    token_array.tofile(raw_file)
    flushed_count = int(token_array.size)
    min_token_id = int(token_array.min(initial=UINT16_MAX))
    max_token_id = int(token_array.max(initial=0))
    token_buffer.clear()
    return flushed_count, min_token_id, max_token_id


def write_manifest() -> None:
    metadata_files = sorted(path for path in output_dir.glob("*/*.json") if path.name != "manifest.json")
    split_metadata = []
    for metadata_path in metadata_files:
        with metadata_path.open(encoding="utf-8") as f:
            split_metadata.append(json.load(f))

    manifest = {
        "format": "cs336_basics.bpe_experiment_3_manifest.v1",
        "updated_utc": utc_now(),
        "output_dir": str(output_dir),
        "dtype": "uint16",
        "load_example": "np.load('data/bpe_tokenized_corpora/tinystories/train.npy', mmap_mode='r')",
        "splits": split_metadata,
    }
    manifest_tmp = output_dir / "manifest.json.tmp"
    manifest_path = output_dir / "manifest.json"
    with manifest_tmp.open("w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)
        f.write("\n")
    os.replace(manifest_tmp, manifest_path)


def copy_raw_tokens_to_npy(raw_path: Path, npy_tmp_path: Path, token_count: int) -> None:
    token_ids = np.lib.format.open_memmap(
        npy_tmp_path,
        mode="w+",
        dtype=UINT16_DTYPE,
        shape=(token_count,),
    )
    offset = 0
    with raw_path.open("rb") as raw_file:
        while offset < token_count:
            count = min(copy_buffer_tokens, token_count - offset)
            token_chunk = np.fromfile(raw_file, dtype=UINT16_DTYPE, count=count)
            if token_chunk.size == 0:
                raise RuntimeError(f"Unexpected end of temporary token stream: {raw_path}")
            token_ids[offset : offset + token_chunk.size] = token_chunk
            offset += int(token_chunk.size)
    token_ids.flush()
    del token_ids


def tokenize_split(split_name: str, config: dict[str, Any]) -> None:
    input_path = config["input_path"]
    tokenizer_dir = config["tokenizer_dir"]
    corpus = config["corpus"]
    split = config["split"]
    split_output_dir = output_dir / corpus
    split_output_dir.mkdir(parents=True, exist_ok=True)

    final_npy_path = split_output_dir / f"{split}.npy"
    metadata_path = split_output_dir / f"{split}.json"
    raw_tmp_path = split_output_dir / f"{split}.uint16.tmp"
    npy_tmp_path = split_output_dir / f"{split}.npy.tmp"
    metadata_tmp_path = split_output_dir / f"{split}.json.tmp"

    if final_npy_path.exists() and metadata_path.exists() and not force:
        print(f"Skipping {split_name}; output already exists at {final_npy_path}", flush=True)
        return

    for temporary_path in (raw_tmp_path, npy_tmp_path, metadata_tmp_path):
        if temporary_path.exists():
            temporary_path.unlink()
    if force:
        for existing_path in (final_npy_path, metadata_path):
            if existing_path.exists():
                existing_path.unlink()

    if not input_path.is_file():
        raise FileNotFoundError(f"Missing input corpus: {input_path}")

    tokenizer = load_tokenizer(tokenizer_dir)
    vocab_path = tokenizer_dir / "vocab.pkl"
    merges_path = tokenizer_dir / "merges.pkl"
    input_bytes = input_path.stat().st_size
    start_time = time.perf_counter()
    token_stream_sha256 = hashlib.sha256()
    token_buffer: list[int] = []
    token_count = 0
    min_observed_token_id = UINT16_MAX
    max_observed_token_id = 0
    next_token_report = token_progress_interval

    print(f"Tokenizing {split_name}: {input_path} -> {final_npy_path}", flush=True)
    with raw_tmp_path.open("wb") as raw_file:
        for token_id in tokenizer.encode_iterable(iter_text_chunks(input_path)):
            token_buffer.append(token_id)
            if len(token_buffer) >= token_buffer_size:
                flushed_count, min_token_id, max_token_id = flush_token_buffer(
                    raw_file,
                    token_buffer,
                    token_stream_sha256,
                )
                token_count += flushed_count
                min_observed_token_id = min(min_observed_token_id, min_token_id)
                max_observed_token_id = max(max_observed_token_id, max_token_id)
                if token_count >= next_token_report:
                    elapsed = time.perf_counter() - start_time
                    print(
                        f"{split_name}: {token_count:,} tokens written to temporary stream "
                        f"after {elapsed:.1f} sec",
                        flush=True,
                    )
                    while next_token_report <= token_count:
                        next_token_report += token_progress_interval

        flushed_count, min_token_id, max_token_id = flush_token_buffer(raw_file, token_buffer, token_stream_sha256)
        token_count += flushed_count
        min_observed_token_id = min(min_observed_token_id, min_token_id)
        max_observed_token_id = max(max_observed_token_id, max_token_id)

    if token_count == 0:
        min_observed_token_id = 0

    copy_raw_tokens_to_npy(raw_tmp_path, npy_tmp_path, token_count)
    raw_tmp_path.unlink()

    elapsed_seconds = time.perf_counter() - start_time
    metadata = {
        "format": "cs336_basics.bpe_tokenized_corpus.v1",
        "status": "complete",
        "created_utc": utc_now(),
        "split_name": split_name,
        "corpus": corpus,
        "split": split,
        "input_path": str(input_path),
        "input_bytes": input_bytes,
        "tokenizer_vocab_path": str(vocab_path),
        "tokenizer_merges_path": str(merges_path),
        "special_tokens": [special_token],
        "output_path": str(final_npy_path),
        "dtype": "uint16",
        "numpy_dtype_descr": UINT16_DTYPE.str,
        "shape": [token_count],
        "token_count": token_count,
        "min_token_id": min_observed_token_id,
        "max_token_id": max_observed_token_id,
        "token_stream_sha256_uint16_le": token_stream_sha256.hexdigest(),
        "bytes_per_token": input_bytes / token_count if token_count else None,
        "elapsed_seconds": elapsed_seconds,
        "tokens_per_second": token_count / elapsed_seconds if elapsed_seconds else None,
        "input_bytes_per_second": input_bytes / elapsed_seconds if elapsed_seconds else None,
        "text_chunk_chars": text_chunk_chars,
        "token_buffer_size": token_buffer_size,
        "copy_buffer_tokens": copy_buffer_tokens,
        "load_example": f"np.load('{final_npy_path}', mmap_mode='r')",
    }

    with metadata_tmp_path.open("w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)
        f.write("\n")
    os.replace(npy_tmp_path, final_npy_path)
    os.replace(metadata_tmp_path, metadata_path)
    write_manifest()
    print(
        f"Completed {split_name}: {token_count:,} tokens, "
        f"{metadata['bytes_per_token']:.3f} bytes/token, {elapsed_seconds:.1f} sec",
        flush=True,
    )


for selected_split_name in selected_split_names:
    tokenize_split(selected_split_name, split_configs[selected_split_name])

write_manifest()
print(f"Experiment 3 tokenization manifest: {output_dir / 'manifest.json'}", flush=True)
PY
