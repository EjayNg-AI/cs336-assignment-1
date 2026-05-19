# cs336_basics

This folder is the submitted implementation package for CS336 Assignment 1:
Basics. Code here is imported by `tests/adapters.py`, which presents the
assignment-facing API expected by the test suite. The package currently focuses
on tokenizer training support and starter pretokenization guidance.

## Python Applications

### `__init__.py`

**Description:** Package initialization module for `cs336_basics`.

**Purpose:** Exposes package metadata without requiring the source tree to be
installed before it can be imported. This lets local development and test runs
import `cs336_basics` directly from the repository while still making
`__version__` available when the package has installed distribution metadata.

**Methodology:** The module uses `importlib.metadata.version("cs336_basics")` to
look up the installed package version. If the package metadata is unavailable,
it catches `importlib.metadata.PackageNotFoundError` and leaves the package
importable without defining `__version__`. It performs no assignment logic and
has no dependencies on the tokenizer implementation.

### `pretokenization_example.py`

**Description:** Instructional pretokenization helper showing how to split a
large byte stream into independently processable chunks at special-token
boundaries.

**Purpose:** Demonstrates a safe way to prepare corpus chunks for parallel
pre-token counting. In BPE tokenizer training, document-boundary special tokens
such as `<|endoftext|>` should prevent merges from crossing document boundaries.
This helper shows how a large file can be divided near evenly sized byte offsets
while moving each internal boundary forward until a special-token delimiter is
found.

**Methodology:** The `find_chunk_boundaries` function accepts an open binary
file, a desired number of chunks, and a byte-string special token. It measures
the file size, creates uniformly spaced initial byte offsets, and then scans
forward from each internal offset in 4096-byte mini-chunks until it finds the
special token or reaches end-of-file. The returned boundary list is sorted and
deduplicated because multiple guessed boundaries can collapse to the same token
position. The usage block at the bottom illustrates reading each resulting
`start`/`end` byte range, decoding it as UTF-8 with ignored errors, and running
pre-token counting independently per chunk. That usage block is an adaptation
template rather than submitted tokenizer-training logic.

### `train_bpe.py`

**Description:** Byte-pair encoding tokenizer training implementation.

**Purpose:** Provides the `train_bpe` function used by `tests/adapters.py` to
train a BPE vocabulary and ordered merge list from a text corpus. The function
returns the vocabulary as `dict[int, bytes]` and merges as `list[tuple[bytes,
bytes]]`, matching the assignment tests for tokenizer training.

**Methodology:** The module starts from the 256 single-byte tokens and appends
requested special tokens as UTF-8 byte sequences if they are not already present
in the vocabulary. It reads the input corpus as UTF-8 text, removes special
tokens from the normal pre-token stream, and applies the GPT-2-style regex
pattern stored in `PAT` to count repeated pre-tokens. Each pre-token is then
represented as a list of byte tokens so BPE merges can be performed at the byte
sequence level.

The trainer maintains weighted pair frequencies across all pre-token instances,
using the pre-token count as the weight for each word representation. It also
keeps a reverse index from each byte pair to the word IDs containing that pair,
which limits recomputation after each merge to only the affected word
representations. Candidate pairs are stored in a heap keyed by negative
frequency and a reverse lexicographic tie-break wrapper so the highest-frequency
pair is selected first and assignment-compatible tie behavior is preserved.
Because heap entries can become stale after later merges, `pop_best_pair`
validates each popped candidate against the current frequency table before
accepting it.

For each merge, the implementation creates a new byte token by concatenating the
selected pair, appends the pair to the merge list, adds the merged token to the
vocabulary, subtracts the old affected pair counts, rewrites only affected word
representations with `_merge_word`, adds the new pair counts, and pushes changed
pairs back onto the heap. Training stops when the requested vocabulary size is
reached or no mergeable pairs remain.

### `tokenizer.py`

**Description:** Byte-pair encoding tokenizer encoder/decoder implementation.

**Purpose:** Provides the `Tokenizer` class used by `tests/adapters.py` to
encode Unicode text into BPE token IDs and decode token IDs back into text from
a supplied `dict[int, bytes]` vocabulary and ordered merge list.

**Methodology:** The tokenizer copies the provided vocabulary, appends missing
special tokens as UTF-8 byte sequences, and builds reverse lookup tables for
byte tokens and merge ranks. Encoding first separates configured special tokens
with longest-match precedence, then applies the GPT-2-style regex
pre-tokenizer to ordinary text. Each pre-token starts as single-byte tokens and
is repeatedly merged by the lowest-ranked adjacent BPE pair until no configured
merge applies. `encode_iterable` keeps a small rolling buffer so chunks can be
encoded lazily without splitting possible regex tokens or special-token
prefixes. Decoding concatenates token bytes and decodes UTF-8 with replacement
for malformed byte sequences.

### `nn_linear_embedding_rmsnorm.py`

**Description:** Custom neural-network modules for the transformer architecture
assignment.

**Purpose:** Provides the `Linear`, `Embedding`, and `RMSNorm` classes used by
`tests/adapters.py` for the first model-layer tasks. The module avoids
`torch.nn.Linear`, `torch.nn.Embedding`, built-in normalization modules, and
`torch.nn.functional` helpers while preserving PyTorch module and parameter
registration behavior.

**Methodology:** `Linear` stores a single weight parameter `W` with shape
`(out_features, in_features)`, initializes it with the assignment's truncated
normal distribution, and applies the transformation with `x @ W.T` so all
leading input dimensions are preserved. `Embedding` stores a parameter named
`weight` with shape `(num_embeddings, embedding_dim)`, initializes it from the
assignment's embedding distribution, and returns rows selected by integer token
IDs through direct tensor indexing. `RMSNorm` stores a learnable scale parameter
named `weight`, initializes it to ones, upcasts input activations to
`torch.float32` for the mean-square normalization calculation, rescales by the
root mean square over the final dimension, and returns the result in the
original input dtype.

### `nn_feedforward.py`

**Description:** Manual feed-forward components for transformer blocks.

**Purpose:** Provides the SiLU activation helper and `SwiGLU` module used by
`tests/adapters.py` for the position-wise feed-forward network task. The module
keeps the implementation in submitted package code instead of in the adapter or
notebook-only cells.

**Methodology:** `stable_sigmoid` computes the logistic sigmoid from elementary
tensor operations using `exp(-abs(x))`, avoiding the unstable branch of the
naive formula for large positive or negative inputs. `silu` multiplies the input
by this sigmoid. `SwiGLU` composes three custom bias-free `Linear` projections:
`w1` and `w3` project from `d_model` to `d_ff`, their outputs are combined as
`silu(w1(x)) * w3(x)`, and `w2` projects the gated hidden activations back to
`d_model`. The constructor accepts an explicit `d_ff`; when omitted, it computes
`8/3 * d_model` rounded to the nearest multiple of 64.

### `train_bpe_enhanced.py`

**Description:** Additive large-corpus variant of the byte-pair encoding
tokenizer trainer.

**Purpose:** Provides an enhanced `train_bpe` implementation that can be
imported directly for larger corpora without replacing the original
assignment-facing `train_bpe.py` module. It keeps the same return contract,
special-token handling, and deterministic merge ordering while adding optional
parallelism, lower-overhead internal token state, and artifact writing.

**Methodology:** The enhanced trainer can split a corpus into byte ranges whose
boundaries align to the first configured special token, then use
`multiprocessing` workers to count pre-tokens independently before reducing the
worker-local counters in the parent process. If safe chunking is unavailable or
the file is small, it falls back to single-process pre-tokenization for
correctness.

During the merge loop, pre-token representations use integer token IDs rather
than Python `bytes` objects. The vocabulary still maps token IDs to bytes, so
merge outputs are converted back to `tuple[bytes, bytes]` and lexicographic
tie-breaking is computed from the underlying bytes. The implementation keeps
the original incremental `pair_counts` and `pair_to_word_ids` strategy, and it
periodically rebuilds the candidate heap when lazy-invalidated entries grow too
large.

After training, the enhanced trainer writes five artifacts to disk while still
returning `(vocab, merges)` to the caller. The binary `vocab.pkl` and
`merges.pkl` files preserve the exact Python objects. The human-inspectable
`vocab.json` file lists each token ID with its byte values, hex string, Python
`repr`, and UTF-8 text when valid. The human-inspectable `merges.txt` file lists
merge rank, left token, right token, and merged token as tab-separated byte
representations. The human-inspectable `metadata.json` file records the
requested and final vocabulary sizes, merge count, phase durations, merge-loop
subphase durations, and run stats. If `output_dir` is omitted, the default
output directory is created beside the input corpus as
`<input_stem>_bpe_<vocab_size>/`.

Example command for the full TinyStories training corpus:

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

This writes `vocab.pkl`, `merges.pkl`, `vocab.json`, `merges.txt`, and
`metadata.json` under `data/tinystories_bpe_10000/`.

#### New local findings from full-corpus enhanced BPE runs

The following are new findings based on running the enhanced BPE trainer locally
on a laptop with the following provided specs: Nvidia RTX 4060 GPU, 32 MB RAM,
and Intel Core i7 CPU. The runs used `num_workers=8`,
`chunk_bytes=64 * 1024 * 1024`, `heap_rebuild_factor=3.0`, and the
`<|endoftext|>` special token. These BPE training runs did not use the GPU: the
trainer is a CPU/Python multiprocessing pipeline and does not dispatch work to
CUDA, so GPU power is not relevant for these measurements.

| Corpus | Input | Vocab target | Final vocab | Merges | Unique pre-tokens | Total pre-tokens | Total time | Slowest phase |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| TinyStories | `data/TinyStoriesV2-GPT4-train.txt` | 10,000 | 10,000 | 9,743 | 59,933 | 536,592,168 | 0 min 52.50 sec | `pretoken_counting`: 0 min 50.18 sec |
| OpenWebText | `data/owt_train.txt` | 32,000 | 32,000 | 31,743 | 6,601,892 | 2,471,753,092 | 17 min 48.67 sec | `merge_loop`: 13 min 3.97 sec |

For TinyStories, the longest vocabulary tokens were tied at 15 bytes:
`" accomplishment"`, `" disappointment"`, and `" responsibility"`. Within its
merge loop, the slowest subphase was `word_rewrite_and_pair_update`, which took
0 min 1.34 sec.

For OpenWebText, the longest vocabulary tokens were tied at 64 bytes:
`b'----------------------------------------------------------------'` and a repeated mojibake byte pattern
`b'\xc3\x83\xc3\x82'` repeated 16 times. Within its merge loop, the slowest
subphase was `word_rewrite_and_pair_update`, which took 6 min 43.39 sec.
Pretoken counting took 4 min 21.42 sec.

The TinyStories and OpenWebText tokenizers differ mainly because the corpora and
vocabulary targets are very different. TinyStories produced a narrower
story-domain vocabulary with child-story words such as `granddaughter`,
`butterflies`, `strawberries`, and the longest tokens above. OpenWebText
produced a broader and noisier web-text vocabulary with long punctuation runs,
encoding artifacts, and general web vocabulary such as `telecommunications`,
`cryptocurrencies`, and `unconstitutional`. OpenWebText also had far more unique
pre-tokens, shifting the bottleneck from pre-token counting to merge-loop pair
updates and heap maintenance.
