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
