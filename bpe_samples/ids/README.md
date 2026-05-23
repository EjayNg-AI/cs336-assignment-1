# bpe_samples/ids

This folder stores JSON-serialized token-ID sequences generated for the BPE
tokenizer notebook experiments.

## Naming Convention

Files follow this pattern:

```text
<sample_corpus>_<tokenizer_corpus>_sample_<index>_ids.json
```

Examples:

- `tinystories_tinystories_sample_00_ids.json`: TinyStories tokenizer on a
  TinyStories sample.
- `openwebtext_openwebtext_sample_00_ids.json`: OpenWebText tokenizer on an
  OpenWebText sample.
- `openwebtext_tinystories_sample_00_ids.json`: OpenWebText sample encoded with
  the TinyStories tokenizer.

See [`../README.md`](../README.md) and
[`../../BPE_TOKENIZER.md`](../../BPE_TOKENIZER.md) for the surrounding experiment
context.
