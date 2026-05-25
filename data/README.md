# data

This folder is for local corpora and generated experiment artifacts. Most files
under `data/` are intentionally ignored by Git because they can be large.

## Downloaded Corpora

Use the repository script to download the required text corpora:

```sh
bash download_data.sh
```

The script writes:

- `TinyStoriesV2-GPT4-train.txt`
- `TinyStoriesV2-GPT4-valid.txt`
- `owt_train.txt`
- `owt_valid.txt`

It uses `wget` when available, otherwise `curl`, and requires `gunzip` for the
OpenWebText `.gz` files.

## Generated Artifacts

Enhanced BPE training scripts write tokenizer artifacts here by default:

- `tinystories_bpe_10000/`
- `openwebtext_bpe_32000/`

Each Python enhanced BPE output directory contains:

- `vocab.pkl` and `merges.pkl`: Python objects consumed by
  `Tokenizer.from_files(...)`
- `vocab.json`, `merges.txt`, and `metadata.json`: inspectable artifacts and run
  metadata

Full-corpus tokenization writes arrays and metadata under:

- `bpe_tokenized_corpora/`
- `bpe_tokenized_corpora_rs/` when using the Rust encoder wrapper

Those `.npy` arrays are intended for memory-mapped reads:

```py
import numpy as np

ids = np.load("data/bpe_tokenized_corpora/tinystories/train.npy", mmap_mode="r")
```

## Related Docs

- [`../README.md`](../README.md): project overview and quick setup
- [`../BPE_TOKENIZER.md`](../BPE_TOKENIZER.md): BPE experiment commands and
  artifact formats
- [`../SETUP.md`](../SETUP.md): detailed setup notes

Do not commit downloaded corpora, tokenized arrays, or generated tokenizer
artifacts unless the repository policy changes.
