from __future__ import annotations

import json
import shutil
import subprocess
from pathlib import Path

import pytest

from cs336_basics.tokenizer import Tokenizer
from cs336_basics.train_bpe_enhanced import train_bpe


REPO_ROOT = Path(__file__).resolve().parents[1]

pytestmark = pytest.mark.skipif(
    shutil.which("cargo") is None,
    reason="Rust parity tests require cargo",
)


def run_rust_train(corpus: Path, out_dir: Path, vocab_size: int, special_tokens: list[str]) -> None:
    cmd = [
        "cargo",
        "run",
        "-q",
        "-p",
        "cs336_bpe_rs",
        "--bin",
        "cs336-bpe-train",
        "--",
        "--input",
        str(corpus),
        "--vocab-size",
        str(vocab_size),
        "--num-workers",
        "1",
        "--output-dir",
        str(out_dir),
    ]
    for special_token in special_tokens:
        cmd += ["--special-token", special_token]
    subprocess.run(cmd, cwd=REPO_ROOT, check=True)


def run_rust_encode(
    vocab_path: Path,
    merges_path: Path,
    corpus: Path,
    ids_path: Path,
    special_tokens: list[str],
    stream_chunk_bytes: int | None = None,
) -> None:
    cmd = [
        "cargo",
        "run",
        "-q",
        "-p",
        "cs336_bpe_rs",
        "--bin",
        "cs336-bpe-encode",
        "--",
        "--vocab",
        str(vocab_path),
        "--merges",
        str(merges_path),
        "--input",
        str(corpus),
        "--output-ids-json",
        str(ids_path),
    ]
    for special_token in special_tokens:
        cmd += ["--special-token", special_token]
    if stream_chunk_bytes is not None:
        cmd += ["--stream-chunk-bytes", str(stream_chunk_bytes)]
    subprocess.run(cmd, cwd=REPO_ROOT, check=True)


def write_edge_corpus(path: Path) -> None:
    path.write_text(
        "hello world\n"
        "hello  world\n"
        "don't we'll they're\n"
        "$a^2 + b^2 = c^2$\n"
        "中文测试 русский текст عربى\n"
        "emoji: 😀😃😄\n"
        "<|endoftext|>after special\n"
        "trailing whitespace    \n",
        encoding="utf-8",
    )


def test_rust_trainer_matches_python_enhanced_on_edge_corpus(tmp_path: Path) -> None:
    corpus = tmp_path / "edge.txt"
    write_edge_corpus(corpus)
    py_out = tmp_path / "py"
    rs_out = tmp_path / "rs"

    train_bpe(
        input_path=corpus,
        vocab_size=400,
        special_tokens=["<|endoftext|>"],
        num_workers=1,
        output_dir=py_out,
    )
    run_rust_train(corpus, rs_out, 400, ["<|endoftext|>"])

    assert (py_out / "merges.txt").read_text(encoding="utf-8") == (rs_out / "merges.txt").read_text(encoding="utf-8")
    assert json.loads((py_out / "vocab.json").read_text(encoding="utf-8")) == json.loads(
        (rs_out / "vocab.json").read_text(encoding="utf-8")
    )


def test_rust_encoder_matches_python_tokenizer_on_rust_artifacts(tmp_path: Path) -> None:
    corpus = tmp_path / "edge.txt"
    corpus.write_text(
        "Hello, how <|endoftext|><|endoftext|> are you?\nHéllò hôw are ü? 🙃\n",
        encoding="utf-8",
    )
    out_dir = tmp_path / "rs"
    run_rust_train(corpus, out_dir, 400, ["<|endoftext|>"])

    py_tokenizer = Tokenizer.from_files(
        str(out_dir / "vocab.json"),
        str(out_dir / "merges.txt"),
        special_tokens=["<|endoftext|>"],
    )
    py_ids = py_tokenizer.encode(corpus.read_text(encoding="utf-8"))

    ids_path = tmp_path / "ids.json"
    run_rust_encode(
        out_dir / "vocab.json",
        out_dir / "merges.txt",
        corpus,
        ids_path,
        ["<|endoftext|>"],
    )
    assert json.loads(ids_path.read_text(encoding="utf-8")) == py_ids


def test_rust_streaming_encoder_matches_whole_file_encoding(tmp_path: Path) -> None:
    corpus = tmp_path / "edge.txt"
    corpus.write_text(
        "abc <|endoftext|><|endoftext|> Héllò 🙃\ntrailing whitespace    \n",
        encoding="utf-8",
    )
    out_dir = tmp_path / "rs"
    run_rust_train(corpus, out_dir, 320, ["<|endoftext|>"])

    whole_ids_path = tmp_path / "whole.json"
    run_rust_encode(
        out_dir / "vocab.json",
        out_dir / "merges.txt",
        corpus,
        whole_ids_path,
        ["<|endoftext|>"],
    )
    whole_ids = json.loads(whole_ids_path.read_text(encoding="utf-8"))

    for chunk_bytes in (1, 2, 7, 4096):
        stream_ids_path = tmp_path / f"stream-{chunk_bytes}.json"
        run_rust_encode(
            out_dir / "vocab.json",
            out_dir / "merges.txt",
            corpus,
            stream_ids_path,
            ["<|endoftext|>"],
            stream_chunk_bytes=chunk_bytes,
        )
        assert json.loads(stream_ids_path.read_text(encoding="utf-8")) == whole_ids
