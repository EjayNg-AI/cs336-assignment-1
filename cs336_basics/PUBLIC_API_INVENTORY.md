# Public API Inventory

This file inventories reusable public Python functions and classes implemented
under `cs336_basics/`. It covers top-level functions and classes whose names do
not start with `_`, plus explicit public aliases. For classes, it lists the
public construction and call methods that are intended to be used from tests,
scripts, or future implementation modules.

Private helpers, nested functions, type aliases, constants, and exploratory
notebook-only code are intentionally excluded.

## Maintenance Rules

- Update this inventory whenever a reusable public function or class is added,
  renamed, removed, or materially repurposed under `cs336_basics/`.
- Before adding a new reusable function or class, check this file and prefer
  composing existing primitives when they already cover the behavior.
- Keep `tests/adapters.py` thin: adapters should import these implementations
  rather than duplicate their logic.

## Tokenization And BPE

### `cs336_basics.train_bpe.train_bpe`

```python
train_bpe(
    input_path: str | os.PathLike,
    vocab_size: int,
    special_tokens: list[str],
) -> tuple[dict[int, bytes], list[tuple[bytes, bytes]]]
```

Assignment-facing byte-level BPE trainer. It reads the input corpus, reserves
byte tokens and requested special tokens, counts GPT-style pre-tokens, performs
deterministic pair merges, and returns a vocabulary mapping token IDs to bytes
plus the ordered merge list.

Reuse this for correctness-oriented tokenizer training and adapter-backed tests.
Do not reimplement BPE merge selection or special-token handling in adapters.

### `cs336_basics.train_bpe_enhanced.train_bpe`

```python
train_bpe(
    input_path: str | os.PathLike,
    vocab_size: int,
    special_tokens: list[str],
    *,
    num_workers: int | None = None,
    chunk_bytes: int | None = None,
    heap_rebuild_factor: float = 3.0,
    output_dir: str | os.PathLike | None = None,
) -> tuple[dict[int, bytes], list[tuple[bytes, bytes]]]
```

Optional large-corpus BPE trainer variant. It adds parallel pre-token counting,
integer-token merge state, heap rebuild maintenance, artifact writing, and
training metadata emission while returning the same vocabulary and merge shapes
as the assignment-facing trainer.

Reuse this for full-corpus experiments and scripts that need serialized
tokenizer artifacts. It is additive and is not the default assignment adapter
path unless explicitly imported.

### `cs336_basics.train_bpe_enhanced.train_bpe_enhanced`

```python
train_bpe_enhanced = train_bpe
```

Public alias for the enhanced BPE trainer. Reuse it when a call site benefits
from naming the enhanced trainer distinctly from the assignment-facing trainer.

### `cs336_basics.tokenizer.Tokenizer`

```python
Tokenizer(
    vocab: dict[int, bytes],
    merges: list[tuple[bytes, bytes]],
    special_tokens: list[str] | None = None,
)
```

BPE tokenizer encoder and decoder. It supports GPT-style pre-tokenization,
integer-ID merge application, special-token handling, cached pre-token encoding,
context-preserving streaming encoding, and UTF-8 replacement decoding.

Public methods:

- `Tokenizer.from_files(vocab_filepath, merges_filepath, special_tokens=None) -> Tokenizer`
- `encode(text: str) -> list[int]`
- `encode_iterable(iterable: Iterable[str]) -> Iterator[int]`
- `decode(ids: list[int]) -> str`

Reuse this for tokenizer tests, corpus token-ID serialization, and scripts that
load either pickle artifacts, enhanced JSON artifacts, or GPT-2-style
vocabulary/merge files. Do not duplicate its special-token splitting or
streaming-buffer logic.

### `cs336_basics.pretokenization_example.find_chunk_boundaries`

```python
find_chunk_boundaries(
    file: BinaryIO,
    desired_num_chunks: int,
    split_special_token: bytes,
) -> list[int]
```

Instructional helper for splitting a binary corpus into chunk boundaries aligned
to a special-token byte string. It demonstrates the safe boundary-finding
strategy used for independent pre-token counting.

Reuse it as reference starter code for corpus chunking. For production enhanced
BPE training, prefer the enhanced trainer's internal chunking path.

## Neural Network Primitives

### `cs336_basics.nn_linear_embedding_rope_rmsnorm.Linear`

```python
Linear(
    in_features: int,
    out_features: int,
    device: torch.device | None = None,
    dtype: torch.dtype | None = None,
)
```

Custom affine-free linear projection module with weight parameter `W` shaped
`(out_features, in_features)` and `forward(x)` implemented as `x @ W.T`.

Reuse this for assignment components that need learned projections, including
attention projections, feed-forward projections, and future language-model
heads. Do not replace it with `torch.nn.Linear` in submitted implementation
code.

### `cs336_basics.nn_linear_embedding_rope_rmsnorm.Embedding`

```python
Embedding(
    num_embeddings: int,
    embedding_dim: int,
    device: torch.device | None = None,
    dtype: torch.dtype | None = None,
)
```

Custom embedding table module with weight parameter `weight` shaped
`(num_embeddings, embedding_dim)` and `forward(token_ids)` implemented by direct
table indexing.

Reuse this for token embeddings and any future assignment path that needs
embedding lookup behavior. Do not replace it with `torch.nn.Embedding` in
submitted implementation code.

### `cs336_basics.nn_linear_embedding_rope_rmsnorm.RotaryPositionalEmbedding`

```python
RotaryPositionalEmbedding(
    theta: float,
    d_k: int,
    max_seq_len: int,
    device: torch.device | None = None,
)
```

RoPE module that precomputes non-persistent sine and cosine buffers and applies
pairwise rotations in `forward(x, token_positions)`.

Reuse this for query/key rotation in attention modules. It expects an even
`d_k` and supports broadcast-compatible token position tensors.

### `cs336_basics.nn_linear_embedding_rope_rmsnorm.RMSNorm`

```python
RMSNorm(
    d_model: int,
    eps: float = 1e-5,
    device: torch.device | None = None,
    dtype: torch.dtype | None = None,
)
```

Custom RMS normalization module with learned scale parameter `weight` and
`forward(x)` implemented from elementary tensor operations in float32 for the
normalization calculation.

Reuse this for transformer blocks and final normalization. Do not replace it
with a prebuilt normalization module in submitted implementation code.

### `cs336_basics.nn_feedforward.swiglu_d_ff`

```python
swiglu_d_ff(d_model: int) -> int
```

Computes the default SwiGLU hidden width as `8/3 * d_model`, rounded down to the
nearest multiple of 64 after adding 32, with a minimum of 64.

Reuse this when future modules need the package's default feed-forward hidden
dimension.

### `cs336_basics.nn_feedforward.stable_sigmoid`

```python
stable_sigmoid(x: torch.Tensor) -> torch.Tensor
```

Numerically stable sigmoid implemented from elementary tensor operations.

Reuse this inside custom activation functions when sigmoid behavior is needed
without relying on a prebuilt activation helper.

### `cs336_basics.nn_feedforward.silu`

```python
silu(x: torch.Tensor) -> torch.Tensor
```

SiLU activation implemented as `x * stable_sigmoid(x)`.

Reuse this anywhere submitted code needs SiLU behavior, including SwiGLU and
future feed-forward blocks.

### `cs336_basics.nn_feedforward.SwiGLU`

```python
SwiGLU(
    d_model: int,
    d_ff: int | None = None,
    device: torch.device | None = None,
    dtype: torch.dtype | None = None,
)
```

Position-wise SwiGLU feed-forward module. It composes three custom `Linear`
modules and applies `w2(silu(w1(x)) * w3(x))`.

Reuse this for transformer feed-forward sublayers. Do not duplicate its
projection layout or SiLU gating logic in future transformer components.

### `cs336_basics.nn_attention.softmax`

```python
softmax(x: torch.Tensor, dim: int) -> torch.Tensor
```

Manual numerically stable softmax that subtracts the maximum along `dim` before
exponentiation and normalization.

Reuse this for attention and future loss or probability utilities when a manual
softmax is required.

### `cs336_basics.nn_attention.scaled_dot_product_attention`

```python
scaled_dot_product_attention(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    mask: torch.Tensor | None = None,
) -> torch.Tensor
```

Manual scaled dot-product attention. It computes `q @ k.transpose(-2, -1) /
sqrt(d_k)`, applies an optional boolean mask where `True` means attention is
allowed, then returns the weighted value tensor.

Reuse this inside attention modules rather than reimplementing score scaling,
mask application, or attention-weight normalization.

### `cs336_basics.nn_attention.CausalMultiHeadSelfAttention`

```python
CausalMultiHeadSelfAttention(
    d_model: int,
    num_heads: int,
    max_seq_len: int | None = None,
    theta: float | None = None,
    use_rope: bool = False,
    device: torch.device | None = None,
    dtype: torch.dtype | None = None,
)
```

Causal multi-head self-attention module. It composes custom `Linear` projections,
splits projected tensors into heads, optionally applies `RotaryPositionalEmbedding`
to queries and keys, applies a lower-triangular causal mask, and projects the
concatenated head output back to `d_model`.

Reuse this for transformer blocks and language-model components. When
`use_rope=True`, provide `max_seq_len` and `theta`; call `forward(x,
token_positions=None)` with explicit token positions when processing nonzero
or externally supplied positions.

## Dependency Map

- `SwiGLU` depends on `Linear`, `silu`, and `swiglu_d_ff`.
- `CausalMultiHeadSelfAttention` depends on `Linear`,
  `RotaryPositionalEmbedding`, and `scaled_dot_product_attention`.
- `scaled_dot_product_attention` depends on the package's manual `softmax`.
- `Tokenizer.from_files` loads artifacts compatible with `train_bpe`,
  `train_bpe_enhanced`, and GPT-2-style vocab/merge files.
