#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
data_dir="$repo_root/data"

tiny_train_url="https://huggingface.co/datasets/roneneldan/TinyStories/resolve/main/TinyStoriesV2-GPT4-train.txt"
tiny_valid_url="https://huggingface.co/datasets/roneneldan/TinyStories/resolve/main/TinyStoriesV2-GPT4-valid.txt"
owt_train_url="https://huggingface.co/datasets/stanford-cs336/owt-sample/resolve/main/owt_train.txt.gz"
owt_valid_url="https://huggingface.co/datasets/stanford-cs336/owt-sample/resolve/main/owt_valid.txt.gz"

download_file() {
    local url="$1"
    local output_path="$2"
    local tmp_path="${output_path}.tmp"

    if [[ -s "$output_path" ]]; then
        echo "Found $(basename "$output_path"); skipping download."
        return
    fi

    rm -f "$tmp_path"
    if command -v wget >/dev/null 2>&1; then
        wget -O "$tmp_path" "$url"
    elif command -v curl >/dev/null 2>&1; then
        curl -fL "$url" -o "$tmp_path"
    else
        echo "Error: install wget or curl before running this script." >&2
        exit 1
    fi
    mv "$tmp_path" "$output_path"
}

download_and_unpack_gzip() {
    local url="$1"
    local gzip_path="$2"
    local output_path="${gzip_path%.gz}"

    if [[ -s "$output_path" ]]; then
        echo "Found $(basename "$output_path"); skipping download and unpack."
        return
    fi

    download_file "$url" "$gzip_path"

    if ! command -v gunzip >/dev/null 2>&1; then
        echo "Error: install gunzip before unpacking $(basename "$gzip_path")." >&2
        exit 1
    fi
    gunzip -f "$gzip_path"
}

mkdir -p "$data_dir"

download_file "$tiny_train_url" "$data_dir/TinyStoriesV2-GPT4-train.txt"
download_file "$tiny_valid_url" "$data_dir/TinyStoriesV2-GPT4-valid.txt"
download_and_unpack_gzip "$owt_train_url" "$data_dir/owt_train.txt.gz"
download_and_unpack_gzip "$owt_valid_url" "$data_dir/owt_valid.txt.gz"

echo "Data files are ready in $data_dir."
