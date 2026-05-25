#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

export EXPERIMENT3_OUTPUT_DIR="${EXPERIMENT3_OUTPUT_DIR:-data/bpe_tokenized_corpora_rs}"
export TINYSTORIES_TOKENIZER_DIR="${TINYSTORIES_TOKENIZER_DIR:-data/tinystories_bpe_10000}"
export OWT_TOKENIZER_DIR="${OWT_TOKENIZER_DIR:-data/owt_bpe_32000}"
export SPECIAL_TOKEN="${SPECIAL_TOKEN:-<|endoftext|>}"
export SPLITS="${SPLITS:-tinystories_train tinystories_valid owt_train owt_valid}"
export STREAM_CHUNK_BYTES="${STREAM_CHUNK_BYTES:-1048576}"
export TOKEN_PROGRESS_INTERVAL="${TOKEN_PROGRESS_INTERVAL:-50000000}"
export FORCE="${FORCE:-0}"
export RUST_BPE_BIN="${RUST_BPE_BIN:-target/release/cs336-bpe-encode}"

if [[ "$OWT_TOKENIZER_DIR" == "data/owt_bpe_32000" \
  && ! -f "$OWT_TOKENIZER_DIR/vocab.json" \
  && -f "data/openwebtext_bpe_32000/vocab.json" ]]; then
  export OWT_TOKENIZER_DIR="data/openwebtext_bpe_32000"
fi

cargo build --release -p cs336_bpe_rs --bin cs336-bpe-encode

run_split() {
  local split_name="$1"
  local corpus="$2"
  local split="$3"
  local input_path="$4"
  local tokenizer_dir="$5"

  local vocab_path="$tokenizer_dir/vocab.json"
  local merges_path="$tokenizer_dir/merges.txt"
  local split_output_dir="$EXPERIMENT3_OUTPUT_DIR/$corpus"
  local npy_path="$split_output_dir/$split.npy"
  local metadata_path="$split_output_dir/$split.json"
  local manifest_path="$EXPERIMENT3_OUTPUT_DIR/manifest.json"

  if [[ -f "$npy_path" && -f "$metadata_path" && "$FORCE" != "1" ]]; then
    printf 'Skipping %s; output already exists at %s\n' "$split_name" "$npy_path"
    return
  fi
  if [[ ! -f "$input_path" ]]; then
    printf 'Missing input corpus: %s\n' "$input_path" >&2
    return 1
  fi
  if [[ ! -f "$vocab_path" ]]; then
    printf 'Missing tokenizer vocabulary: %s\n' "$vocab_path" >&2
    return 1
  fi
  if [[ ! -f "$merges_path" ]]; then
    printf 'Missing tokenizer merges: %s\n' "$merges_path" >&2
    return 1
  fi

  mkdir -p "$split_output_dir"
  local args=(
    "$RUST_BPE_BIN"
    --vocab "$vocab_path"
    --merges "$merges_path"
    --special-token "$SPECIAL_TOKEN"
    --input "$input_path"
    --output-ids-npy "$npy_path"
    --metadata-json "$metadata_path"
    --manifest-json "$manifest_path"
    --split-name "$split_name"
    --corpus "$corpus"
    --split "$split"
    --stream-chunk-bytes "$STREAM_CHUNK_BYTES"
    --token-progress-interval "$TOKEN_PROGRESS_INTERVAL"
  )
  if [[ "$FORCE" == "1" ]]; then
    args+=(--force)
  fi

  printf 'Tokenizing %s with Rust: %s -> %s\n' "$split_name" "$input_path" "$npy_path"
  "${args[@]}"
}

for split_name in $SPLITS; do
  case "$split_name" in
    tinystories_train)
      run_split "$split_name" "tinystories" "train" \
        "data/TinyStoriesV2-GPT4-train.txt" "$TINYSTORIES_TOKENIZER_DIR"
      ;;
    tinystories_valid)
      run_split "$split_name" "tinystories" "valid" \
        "data/TinyStoriesV2-GPT4-valid.txt" "$TINYSTORIES_TOKENIZER_DIR"
      ;;
    owt_train)
      run_split "$split_name" "openwebtext" "train" \
        "data/owt_train.txt" "$OWT_TOKENIZER_DIR"
      ;;
    owt_valid)
      run_split "$split_name" "openwebtext" "valid" \
        "data/owt_valid.txt" "$OWT_TOKENIZER_DIR"
      ;;
    *)
      printf 'Unknown split: %s\n' "$split_name" >&2
      printf 'Known splits: tinystories_train tinystories_valid owt_train owt_valid\n' >&2
      exit 1
      ;;
  esac
done

printf 'Rust Experiment 3 tokenization manifest: %s\n' "$EXPERIMENT3_OUTPUT_DIR/manifest.json"
