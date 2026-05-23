# bpe_samples/openwebtext

This folder contains 10 deterministic OpenWebText sample documents used by the
BPE tokenizer notebook.

## Important Files

- [`manifest.json`](manifest.json): source path, sampling seed, total documents
  seen, source document indices, sample paths, and byte counts.
- `openwebtext_sample_*_doc_*.txt`: sampled OpenWebText documents.

The manifest records `data/owt_train.txt` as the source corpus. That full corpus
is not tracked; see [`../../data/README.md`](../../data/README.md) for data
setup.
