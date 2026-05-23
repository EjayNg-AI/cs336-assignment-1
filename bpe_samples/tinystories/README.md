# bpe_samples/tinystories

This folder contains 10 deterministic TinyStories document samples used by the
BPE tokenizer notebook.

## Important Files

- [`manifest.json`](manifest.json): source path, sampling seed, total documents
  seen, source document indices, sample paths, and byte counts.
- `tinystories_sample_*_doc_*.txt`: sampled TinyStories documents.

The manifest records `data/TinyStoriesV2-GPT4-train.txt` as the source corpus.
That full corpus is not tracked; see [`../../data/README.md`](../../data/README.md)
for data setup.
