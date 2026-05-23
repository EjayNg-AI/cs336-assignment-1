# Workflow for a parallel Rust implementation of the enhanced BPE tokenizer

## 1. Design objective

The Rust implementation should be a **parallel, correctness-equivalent sibling** of the existing Python enhanced BPE trainer and tokenizer, not a rewrite that silently changes semantics.

The current Python enhanced trainer already has the essential structure we want to preserve: GPT-style pre-tokenization, special-token-aware corpus chunking, parallel pre-token counting, integer-token internal merge state, lazy heap maintenance, artifact writing, and metadata emission. The current Python tokenizer already supports loading pickle/JSON/GPT-2-style files, special-token handling, merge-rank encoding, streaming `encode_iterable`, UTF-8 replacement decoding, and a pretoken cache. 

The Rust implementation should therefore target:

=> same corpus + same vocab_size + same special_tokens + same pretokenizer
=> same pretoken counts
=> same merge order
=> same vocab.json and merges.txt
=> same encoded token IDs
=> same decoded text

The priority order should be:

1. semantic equivalence
2. deterministic artifacts
3. test coverage
4. streaming behavior
5. performance
6. memory optimization

Do **not** optimize first. For this project, an incorrect fast tokenizer is worse than a slower but parity-tested Rust implementation.

## 2. Repository layout

Add Rust at the repository root as a Cargo workspace while leaving the existing Python project intact:

```text
cs336-assignment-1/
├── pyproject.toml
├── uv.lock
├── Cargo.toml
├── crates/
│   └── cs336_bpe_rs/
│       ├── Cargo.toml
│       ├── README.md
│       └── src/
│           ├── lib.rs
│           ├── main.rs
│           ├── config.rs
│           ├── errors.rs
│           ├── bytes_repr.rs
│           ├── pretokenizer.rs
│           ├── chunking.rs
│           ├── trainer/
│           │   ├── mod.rs
│           │   ├── state.rs
│           │   ├── counts.rs
│           │   ├── heap.rs
│           │   ├── merge.rs
│           │   └── artifacts.rs
│           ├── encoder/
│           │   ├── mod.rs
│           │   ├── vocab.rs
│           │   ├── merges.rs
│           │   ├── tokenizer.rs
│           │   └── streaming.rs
│           └── bin/
│               ├── train_bpe.rs
│               └── encode_bpe.rs
├── cs336_basics/
│   ├── train_bpe.py
│   ├── train_bpe_enhanced.py
│   └── tokenizer.py
├── tests/
│   ├── test_train_bpe.py
│   ├── test_tokenizer.py
│   └── test_rust_bpe_parity.py
└── repository_structure.md
```

Use this root `Cargo.toml`:

```toml
[workspace]
resolver = "2"
members = [
    "crates/cs336_bpe_rs",
]
```

The Rust crate should initially be a standalone CLI plus library. Do **not** start with PyO3/maturin. A CLI is easier to parity-test from Python using `subprocess`, and it avoids coupling Rust correctness to Python packaging.

Suggested `crates/cs336_bpe_rs/Cargo.toml`:

```toml
[package]
name = "cs336_bpe_rs"
version = "0.1.0"
edition = "2021"
description = "Rust implementation of the CS336 enhanced byte-level BPE trainer and tokenizer"

[dependencies]
anyhow = "1"
clap = { version = "4", features = ["derive"] }
fancy-regex = "0.16"
rayon = "1"
serde = { version = "1", features = ["derive"] }
serde_json = "1"
thiserror = "2"

[dev-dependencies]
tempfile = "3"
```

Use `fancy-regex` first because the Python pre-tokenizer uses a negative lookahead in `\s+(?!\S)`. Rust’s normal `regex` crate is not a drop-in replacement for this pattern. Later, once parity is established, replace this with a custom scanner if profiling says regex is the bottleneck.

## 3. Command-line interface

Expose two binaries:

```text
cs336-bpe-train
cs336-bpe-encode
```

The trainer command should mirror the Python enhanced trainer:

```bash
cargo run -p cs336_bpe_rs --bin cs336-bpe-train -- \
  --input data/TinyStoriesV2-GPT4-train.txt \
  --vocab-size 10000 \
  --special-token '<|endoftext|>' \
  --num-workers 8 \
  --chunk-bytes 67108864 \
  --heap-rebuild-factor 3.0 \
  --output-dir data/tinystories_bpe_10000_rs
```

The encoder command should support both whole-file and streaming output:

```bash
cargo run -p cs336_bpe_rs --bin cs336-bpe-encode -- \
  --vocab data/tinystories_bpe_10000_rs/vocab.json \
  --merges data/tinystories_bpe_10000_rs/merges.txt \
  --special-token '<|endoftext|>' \
  --input data/TinyStoriesV2-GPT4-valid.txt \
  --output-ids-json data/tinystories_valid_ids_rs.json
```

Later, add `.npy` output. For the first implementation, JSON IDs are enough for parity tests.

## 4. Rust trainer architecture

### 4.1 Core types

Use compact integer IDs internally:

```rust
pub type TokenId = u32;
pub type WordId = usize;
pub type Count = u64;
pub type TokenPair = (TokenId, TokenId);
pub type BytePair = (Vec<u8>, Vec<u8>);
```

Core state:

```rust
pub struct TrainerState {
    pub id_to_bytes: Vec<Vec<u8>>,
    pub words: Vec<Vec<TokenId>>,
    pub word_counts: Vec<Count>,
    pub pair_counts: HashMap<TokenPair, Count>,
    pub pair_to_word_ids: HashMap<TokenPair, HashSet<WordId>>,
    pub heap: BinaryHeap<HeapEntry>,
    pub merges: Vec<BytePair>,
}
```

Start with standard `HashMap`/`HashSet`. Only move to `hashbrown`, `rustc_hash`, `FxHashMap`, packed postings, or arena-encoded words after the parity suite passes.

### 4.2 Vocabulary initialization

Replicate Python exactly:

```text
id 0..255 => one-byte tokens b"\x00" ... b"\xff"
then append each special token as UTF-8 bytes if absent
```

The current Python enhanced trainer creates byte tokens for all 256 byte values, appends special tokens if not already present, and uses token IDs as integer vocabulary keys. 

### 4.3 Pre-tokenization

Replicate this pattern exactly:

```text
'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+
```

The Python implementation removes/splits around special tokens before applying the normal pre-token regex.  Rust should do the same:

```text
for each normal text segment between special-token matches:
    run GPT-style pre-tokenizer
    increment Counter<Vec<u8>>
```

Important edge cases:

```text
empty input
ASCII only
UTF-8 multilingual text
emoji
NUL byte after UTF-8 decoding if present
trailing whitespace
multiple newlines
special token adjacent to normal text
overlapping special tokens
special token repeated with no separator
```

The tokenizer tests already cover special-token preservation, overlapping special tokens, matching GPT-2/tiktoken behavior, streaming encoding, and memory-sensitive streaming behavior.  

### 4.4 Parallel chunking

Mirror the Python enhanced trainer’s safe chunking rule:

```text
If special_tokens is empty:
    do not chunk for training parity unless a custom safe tokenizer-boundary chunker is implemented.

If special_tokens is nonempty:
    choose the first special token as the chunk boundary delimiter.
    divide file into approximate byte chunks.
    move each internal boundary forward until the delimiter is found.
    deduplicate sorted boundaries.
    decode each byte range as UTF-8.
    count pre-tokens independently.
```

The existing docs say `pretokenization_example.py` exists specifically to demonstrate safe splitting at a special-token boundary so BPE merges do not cross document boundaries. 

For the initial Rust version, use `rayon`:

```rust
let partial_counts: Vec<HashMap<Vec<u8>, Count>> = ranges
    .par_iter()
    .map(|range| count_pretokens_for_range(input_path, range, special_tokens))
    .collect();

let mut global_counts = HashMap::new();
for partial in partial_counts {
    reduce_counts(&mut global_counts, partial);
}
```

Parallelism is appropriate for:

```text
pre-token counting
initial pair-state construction
possibly file encoding by independent document chunks
```

It is **not** appropriate for the main merge loop at first. The merge sequence is inherently sequential because merge `k + 1` depends on the tokenization state produced by merge `k`.

### 4.5 Word materialization

Convert each unique pre-token byte string into byte-token IDs:

```text
pretoken b"abc" -> [97, 98, 99]
```

Preserve the pre-token count separately:

```rust
words: Vec<Vec<TokenId>>
word_counts: Vec<Count>
```

The Python enhanced trainer does exactly this, with `words.append(list(pretoken))` and `word_counts.append(count)`. 

### 4.6 Initial pair state

For each unique word:

```text
word = [t0, t1, t2, ...]
pairs = (t0,t1), (t1,t2), ...
weighted count = pair frequency inside word * word_count
```

Build:

```text
pair_counts: pair -> weighted frequency
pair_to_word_ids: pair -> set of word IDs containing that pair
```

The Python enhanced trainer already parallelizes this by chunking word records and reducing local maps.  Do the same with `rayon`, but keep reduction deterministic by summing integer counts only. Map iteration order must never affect merge order.

### 4.7 Heap and tie-breaking

This is the most important correctness point.

The Python trainer chooses:

```text
highest frequency wins
if frequency ties, lexicographically larger underlying byte pair wins
stale heap entries are lazily discarded
```

This behavior comes from the heap key using negative count plus a reversed byte-pair comparator. 

Rust’s `BinaryHeap` is a max-heap, so implement:

```rust
#[derive(Clone, Eq, PartialEq)]
pub struct HeapEntry {
    pub count: Count,
    pub left_bytes: Vec<u8>,
    pub right_bytes: Vec<u8>,
    pub pair: TokenPair,
}

impl Ord for HeapEntry {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        self.count
            .cmp(&other.count)
            .then_with(|| self.left_bytes.cmp(&other.left_bytes))
            .then_with(|| self.right_bytes.cmp(&other.right_bytes))
    }
}

impl PartialOrd for HeapEntry {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        Some(self.cmp(other))
    }
}
```

Do not tie-break by token ID. That will diverge from Python.

`pop_best_pair` must discard stale entries:

```rust
while let Some(entry) = heap.pop() {
    if pair_counts.get(&entry.pair) == Some(&entry.count) {
        return Some(entry.pair);
    }
}
None
```

### 4.8 Merge loop

For each selected pair:

```text
1. create merged token bytes = vocab[left] + vocab[right]
2. assign new token ID = id_to_bytes.len()
3. append byte pair to merges
4. collect affected word IDs from pair_to_word_ids[best_pair]
5. for each affected word:
   a. compute old pair frequencies
   b. subtract old weighted pair frequencies
   c. remove word ID from old pair postings
   d. rewrite the word by replacing non-overlapping best_pair occurrences
   e. compute new pair frequencies
   f. add new weighted pair frequencies
   g. add word ID to new pair postings
6. push every changed pair back into the heap
7. rebuild heap if lazy-invalidated entries exceed threshold
```

Preserve non-overlapping left-to-right merge semantics. For example:

```text
[a, a, a], merge (a, a) -> [aa, a]
not [a, aa]
not [aa, aa]
```

This is exactly how `_merge_word` behaves in the Python trainer. 

### 4.9 Artifact writing

The Rust trainer should write language-neutral artifacts first:

```text
vocab.json
merges.txt
metadata.json
```

Do not attempt to write Python pickle in the first implementation. The repository docs already treat `vocab.json`, `merges.txt`, and `metadata.json` as human-inspectable artifacts, while pickle preserves Python objects. 

Match the existing enhanced format as closely as possible:

```json
{
  "format": "cs336_basics.enhanced_bpe.v1",
  "tokens": [
    {
      "id": 0,
      "byte_values": [0],
      "hex": "00",
      "repr": "b'\\x00'",
      "utf8": "\u0000"
    }
  ]
}
```

For `merges.txt`, match the existing tab-separated style:

```text
# cs336_basics enhanced BPE merges v1
# rank	left_repr	right_repr	merged_repr
0	b'a'	b'b'	b'ab'
```

For `metadata.json`, include both Rust-specific and Python-compatible fields:

```json
{
  "format": "cs336_basics.enhanced_bpe.rust.metadata.v1",
  "compatibility_target": "cs336_basics.train_bpe_enhanced",
  "input_path": "...",
  "output_dir": "...",
  "requested_vocab_size": 10000,
  "vocab_size": 10000,
  "merge_count": 9743,
  "special_tokens": ["<|endoftext|>"],
  "num_workers": 8,
  "chunk_bytes": 67108864,
  "heap_rebuild_factor": 3.0,
  "input_file_bytes": 123456,
  "unique_pretoken_count": 59933,
  "total_pretoken_count": 536592168,
  "initial_pair_count": 12345,
  "final_pair_count": 6789,
  "initial_heap_size": 12345,
  "final_heap_size": 6789,
  "heap_rebuild_count": 3,
  "phase_durations_seconds": {
    "vocab_setup": 0.0,
    "pretoken_counting": 0.0,
    "word_materialization": 0.0,
    "initial_pair_state": 0.0,
    "initial_heap_build": 0.0,
    "merge_loop": 0.0,
    "artifact_writing": 0.0,
    "total_training": 0.0
  }
}
```

## 5. Rust encoder architecture

The Rust encoder should be a runtime equivalent of `cs336_basics/tokenizer.py`.

### 5.1 Supported input artifact formats

Initial support should include:

```text
vocab.json produced by Python enhanced trainer
vocab.json produced by Rust trainer
merges.txt produced by Python enhanced trainer
merges.txt produced by Rust trainer
GPT-2 vocab.json fixture
GPT-2 merges.txt fixture
```

Defer pickle support unless needed. Python can continue loading pickle; Rust should prefer language-neutral files.

### 5.2 Tokenizer state

```rust
pub struct Tokenizer {
    vocab: Vec<Vec<u8>>,
    token_to_id: HashMap<Vec<u8>, TokenId>,
    special_tokens: Vec<String>,
    special_token_ids: HashMap<String, TokenId>,
    merge_ranks_by_id: HashMap<TokenPair, usize>,
    merge_output_by_pair_id: HashMap<TokenPair, TokenId>,
    byte_token_ids: [TokenId; 256],
    encode_cache: LruOrSimpleCache<Vec<u8>, Vec<TokenId>>,
    max_special_token_length: usize,
}
```

The Python tokenizer constructs reverse lookup tables for byte tokens, merge ranks, and merge outputs; encoding repeatedly applies the lowest-ranked adjacent merge until no configured merge applies. 

### 5.3 Special-token handling

Special tokens must use longest-match precedence, matching Python:

```text
special_tokens sorted by descending length
normal spans are pre-tokenized
special-token spans emit the special token ID directly
```

This matters for overlapping tokens such as:

```text
<|endoftext|>
<|endoftext|><|endoftext|>
```

The tests explicitly check overlapping special-token behavior. 

### 5.4 Encoding a pre-token

Algorithm:

```text
input pretoken bytes
tokens = byte_token_ids[each byte]
while tokens has at least 2:
    scan adjacent pairs
    find pair with smallest merge rank
    if no ranked pair exists: stop
    rewrite all non-overlapping occurrences of that pair into merged token ID
return tokens
```

This exactly mirrors the Python `_encode_pretoken` loop. 

### 5.5 Streaming encoding

Implement a Rust equivalent of `encode_iterable`.

Initial CLI streaming can be line/chunk based:

```text
read input in chunks
append to string buffer
identify complete token segments
flush only complete safe prefix
retain suffix that might belong to:
    last regex token
    partial special-token prefix
encode flushed prefix
at EOF, encode remaining buffer
```

The Python implementation keeps a rolling buffer and avoids splitting either regex tokens or special-token prefixes.  The repository tests already check that `encode_iterable` matches tiktoken on TinyStories and stays memory-conscious. 

For parity, add a Rust CLI test that encodes the same file with:

```text
whole input at once
streaming chunks of size 1
streaming chunks of size 2
streaming chunks of size 7
streaming chunks of size 4096
```

All should produce identical token IDs.

## 6. Python–Rust parity tests

Add `tests/test_rust_bpe_parity.py`. These tests should be skipped if Cargo is unavailable, so normal Python assignment tests are not broken on machines without Rust.

Example structure:

```python
from __future__ import annotations

import json
import shutil
import subprocess
from pathlib import Path

import pytest

from cs336_basics.train_bpe_enhanced import train_bpe
from cs336_basics.tokenizer import Tokenizer


pytestmark = pytest.mark.skipif(
    shutil.which("cargo") is None,
    reason="Rust parity tests require cargo",
)


def run_rust_train(corpus: Path, out_dir: Path, vocab_size: int, special_tokens: list[str]) -> None:
    cmd = [
        "cargo", "run", "-q", "-p", "cs336_bpe_rs", "--bin", "cs336-bpe-train", "--",
        "--input", str(corpus),
        "--vocab-size", str(vocab_size),
        "--output-dir", str(out_dir),
    ]
    for tok in special_tokens:
        cmd += ["--special-token", tok]
    subprocess.run(cmd, check=True)


def test_rust_trainer_matches_python_enhanced_on_edge_corpus(tmp_path: Path):
    corpus = tmp_path / "edge.txt"
    corpus.write_text(
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

    assert (py_out / "merges.txt").read_text(encoding="utf-8") == (
        rs_out / "merges.txt"
    ).read_text(encoding="utf-8")

    assert json.loads((py_out / "vocab.json").read_text(encoding="utf-8")) == json.loads(
        (rs_out / "vocab.json").read_text(encoding="utf-8")
    )


def test_rust_encoder_matches_python_tokenizer_on_rust_artifacts(tmp_path: Path):
    corpus = tmp_path / "edge.txt"
    corpus.write_text(
        "Hello, how <|endoftext|><|endoftext|> are you?\n"
        "Héllò hôw are ü? 🙃\n",
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
    subprocess.run(
        [
            "cargo", "run", "-q", "-p", "cs336_bpe_rs", "--bin", "cs336-bpe-encode", "--",
            "--vocab", str(out_dir / "vocab.json"),
            "--merges", str(out_dir / "merges.txt"),
            "--special-token", "<|endoftext|>",
            "--input", str(corpus),
            "--output-ids-json", str(ids_path),
        ],
        check=True,
    )

    rs_ids = json.loads(ids_path.read_text(encoding="utf-8"))
    assert rs_ids == py_ids
```

Also add Rust-native unit tests for:

```text
pretokenizer
special-token splitting
chunk boundary finding
word pair frequency counting
merge_word non-overlap behavior
heap tie-breaking
stale heap entry discard
vocab.json loading
merges.txt loading
encode_pretoken
decode replacement behavior
streaming chunk equivalence
```

## 7. Validation commands

Codex should use these commands incrementally:

```bash
cargo fmt --all
cargo test -p cs336_bpe_rs
uv run pytest tests/test_train_bpe.py tests/test_tokenizer.py
uv run pytest tests/test_rust_bpe_parity.py
```

Then broader validation:

```bash
uv run pytest
cargo test --workspace
```

Do not require full TinyStories/OpenWebText runs for ordinary PR validation. The repository’s BPE notes show full-corpus runs can be substantial; OpenWebText training was documented as taking about 17 minutes locally, with merge-loop work dominating.  Keep full-corpus validation as an optional manual benchmark.

## 8. Documentation updates required

Because this change adds new code outside notebooks, Codex must update:

```text
repository_structure.md
crates/cs336_bpe_rs/README.md
```

If any new Python wrapper under `cs336_basics/` is added, update:

```text
cs336_basics/README.md
```

Do not modify the root `README.md`; the repository agent instructions explicitly forbid it. 