# bpe_samples

This folder contains retained, deterministic sample artifacts from
[`../BPE_tokenizer.ipynb`](../BPE_tokenizer.ipynb). They support the tokenizer
experiment writeups without requiring a future reader to rerun full-corpus
sampling.

## Contents

- [`tinystories/`](tinystories/): 10 TinyStories text samples and a manifest.
- [`openwebtext/`](openwebtext/): 10 OpenWebText sample documents and a manifest.
- [`ids/`](ids/): JSON token-ID outputs for tokenizer comparison experiments.
- [`experiment_1_2_summary.json`](experiment_1_2_summary.json): aggregate byte
  counts, token counts, compression ratios, and throughput measurements used by
  the notebook.

## Data Dependencies

The manifests reference source files under [`../data/`](../data/), but the sample
texts and encoded IDs in this folder are tracked. Full source corpora are not
tracked and can be downloaded with:

```sh
bash download_data.sh
```

For experiment commands and interpretation notes, see
[`../BPE_TOKENIZER.md`](../BPE_TOKENIZER.md).
