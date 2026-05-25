# Rust BPE Implementation

This document explains the Rust byte-level BPE trainer and encoder added under
`crates/cs336_bpe_rs/`. The Rust implementation is an additive sibling of the
existing Python enhanced trainer and tokenizer. It is not wired into
`tests/adapters.py` and does not replace the submitted Python assignment path.

The implementation goal is parity with:

- `cs336_basics/train_bpe_enhanced.py`
- `cs336_basics/tokenizer.py`

The current Rust implementation writes language-neutral tokenizer artifacts and
supports CLI training and encoding. It does not write Python pickle files. The
encoder can serialize token IDs as JSON for small parity checks or as NumPy
`.npy` arrays for full-corpus token-ID datasets.

## Layout

The root `Cargo.toml` declares a Cargo workspace:

```toml
[workspace]
resolver = "2"
members = [
    "crates/cs336_bpe_rs",
]
```

The crate provides a library plus two binaries:

```text
crates/cs336_bpe_rs/
|-- Cargo.toml
|-- README.md
`-- src/
    |-- lib.rs
    |-- pretokenizer.rs
    |-- chunking.rs
    |-- bytes_repr.rs
    |-- npy.rs
    |-- sha256.rs
    |-- trainer/
    |   |-- mod.rs
    |   |-- state.rs
    |   |-- counts.rs
    |   |-- heap.rs
    |   |-- merge.rs
    |   `-- artifacts.rs
    |-- encoder/
    |   |-- mod.rs
    |   |-- vocab.rs
    |   |-- merges.rs
    |   |-- tokenizer.rs
    |   `-- streaming.rs
    `-- bin/
        |-- train_bpe.rs
        `-- encode_bpe.rs
```

The public library surface is intentionally small:

```rust
pub mod bytes_repr;
pub mod chunking;
pub mod config;
pub mod encoder;
pub mod errors;
pub mod npy;
pub mod pretokenizer;
pub mod sha256;
pub mod trainer;

pub use encoder::Tokenizer;
pub use trainer::{train_bpe, TrainConfig, TrainOutput};
```

## Command Line Interfaces

The trainer binary is `cs336-bpe-train`. Its CLI maps to `TrainConfig`:

```rust
#[derive(Debug, Parser)]
#[command(name = "cs336-bpe-train")]
struct Args {
    #[arg(long)]
    input: PathBuf,

    #[arg(long)]
    vocab_size: usize,

    #[arg(long = "special-token")]
    special_tokens: Vec<String>,

    #[arg(long)]
    num_workers: Option<usize>,

    #[arg(long)]
    chunk_bytes: Option<usize>,

    #[arg(long, default_value_t = 3.0)]
    heap_rebuild_factor: f64,

    #[arg(long)]
    output_dir: Option<PathBuf>,
}
```

Example:

```sh
cargo run -p cs336_bpe_rs --bin cs336-bpe-train -- \
  --input data/TinyStoriesV2-GPT4-train.txt \
  --vocab-size 10000 \
  --special-token '<|endoftext|>' \
  --num-workers 8 \
  --chunk-bytes 67108864 \
  --heap-rebuild-factor 3.0 \
  --output-dir data/tinystories_bpe_10000_rs
```

The encoder binary is `cs336-bpe-encode`:

```rust
#[derive(Debug, Parser)]
#[command(name = "cs336-bpe-encode")]
struct Args {
    #[arg(long)]
    vocab: PathBuf,

    #[arg(long)]
    merges: PathBuf,

    #[arg(long = "special-token")]
    special_tokens: Vec<String>,

    #[arg(long)]
    input: PathBuf,

    #[arg(long)]
    output_ids_json: Option<PathBuf>,

    #[arg(long)]
    output_ids_npy: Option<PathBuf>,

    #[arg(long)]
    metadata_json: Option<PathBuf>,

    #[arg(long)]
    manifest_json: Option<PathBuf>,

    #[arg(long)]
    split_name: Option<String>,

    #[arg(long)]
    corpus: Option<String>,

    #[arg(long)]
    split: Option<String>,

    #[arg(long, default_value_t = false)]
    force: bool,

    #[arg(long)]
    stream_chunk_bytes: Option<usize>,

    #[arg(long)]
    token_progress_interval: Option<u64>,
}
```

Example:

```sh
cargo run -p cs336_bpe_rs --bin cs336-bpe-encode -- \
  --vocab data/tinystories_bpe_10000_rs/vocab.json \
  --merges data/tinystories_bpe_10000_rs/merges.txt \
  --special-token '<|endoftext|>' \
  --input data/TinyStoriesV2-GPT4-valid.txt \
  --output-ids-json data/tinystories_valid_ids_rs.json
```

The optional `--stream-chunk-bytes` flag exercises the streaming encoder path.
For full-corpus tokenization, use `--output-ids-npy` with optional
`--metadata-json` and `--manifest-json` to write a flat little-endian `uint16`
NumPy array compatible with `np.load(..., mmap_mode="r")`.

The repository wrapper for the standard Experiment 3 splits is:

```sh
bash run_bpe_experiment_3_tokenization_rs.sh
```

## NumPy `.npy` Serialization

Rust `.npy` output is implemented for the Experiment 3 token-ID datasets, where
the desired artifact is a flat memory-mappable array of tokenizer IDs. The
writer intentionally supports only the format this repository needs:

- NumPy format version: `1.0`
- dtype: little-endian unsigned 16-bit integers, written as `'<u2'`
- shape: one-dimensional `(token_count,)`
- order: C order, recorded as `fortran_order: False`

The output is equivalent to a Python array created with:

```py
np.asarray(token_ids, dtype=np.dtype("<u2"))
```

and saved in a way that can later be loaded with:

```py
ids = np.load("data/bpe_tokenized_corpora_rs/tinystories/valid.npy", mmap_mode="r")
```

### On-Disk File Layout

The `.npy` file is written as:

1. Magic bytes: `\x93NUMPY`
2. Version bytes: `\x01\x00`
3. Two-byte little-endian header length
4. ASCII Python-literal header dictionary
5. Raw little-endian `uint16` token bytes

For a split with `N` tokens, the header dictionary has this logical content:

```text
{'descr': '<u2', 'fortran_order': False, 'shape': (N,), }
```

The header is padded with spaces and terminated by a newline so the full header
prefix is 16-byte aligned, following NumPy v1 format expectations. The payload
then contains exactly `2 * N` bytes. Token ID `513`, for example, is written as
the two bytes `0x01 0x02`.

### Write Sequence

The Rust encoder writes `.npy` output in two phases so the final file has a
correct shape without storing the full token stream in memory:

1. Load `vocab.json` and `merges.txt` into the Rust `Tokenizer`.
2. Read the input corpus as UTF-8 chunks, preserving character boundaries.
3. Stream chunks through `Tokenizer::encode_iterable_result_to_sink`.
4. For each emitted token ID:
   - validate it is `<= 65535`;
   - write it to a temporary raw `.uint16.tmp` stream as little-endian bytes;
   - update token count, min/max token ID, throughput counters, and SHA-256.
5. After encoding completes, verify the raw stream is exactly `2 * token_count`
   bytes.
6. Create a temporary `.npy.tmp` file, write the NumPy header using the final
   token count, then copy the raw token payload after the header.
7. Remove the raw temporary stream and atomically rename `.npy.tmp` into the
   requested final `.npy` path.

This is why the implementation does not need to know `token_count` before
encoding starts, and also does not need a large in-memory `Vec<TokenId>` for
full corpora.

### Sidecar Metadata and Manifest

When `--metadata-json` is provided, the encoder writes a sidecar JSON file next
to the array. It records:

- source input path and byte size;
- tokenizer artifact paths;
- special tokens;
- output path, dtype, NumPy dtype descriptor, and shape;
- token count plus observed min/max token IDs;
- SHA-256 of the little-endian `uint16` token stream;
- bytes/token and throughput measurements;
- a `np.load(..., mmap_mode="r")` loading example.

The SHA-256 is computed over the payload token bytes only, not over the `.npy`
header. This matches the Python wrapper's `token_stream_sha256_uint16_le`
contract and makes hashes independent of header formatting.

When `--manifest-json` is provided, the encoder scans split metadata JSON files
under the output directory and writes a top-level manifest using the same
`cs336_basics.bpe_experiment_3_manifest.v1` format as the Python wrapper.

### Failure and Replacement Behavior

The final `.npy` file is not replaced until the temporary raw stream has been
fully encoded and the temporary `.npy` file has been constructed. Existing final
outputs are rejected unless `--force` is supplied. Temporary files use sibling
paths such as `valid.uint16.tmp` and `valid.npy.tmp`, so interrupted runs leave
recoverable scratch files without silently corrupting a completed final array.

## Trainer Pipeline

The trainer follows the Python enhanced trainer's phases:

1. Initialize byte vocabulary and special tokens.
2. Count pretokens.
3. Materialize unique pretokens as byte-token words.
4. Build initial pair counts and pair postings.
5. Build a deterministic lazy heap.
6. Run the sequential merge loop.
7. Write `vocab.json`, `merges.txt`, and `metadata.json`.

The main entrypoint is:

```rust
pub fn train_bpe(config: TrainConfig) -> Result<TrainOutput> {
    let resolved_num_workers = resolve_num_workers(config.num_workers)?;
    let input_file_bytes = fs::metadata(&config.input_path)?.len();

    let mut id_to_bytes: Vec<Vec<u8>> =
        (0u16..=255).map(|value| vec![value as u8]).collect();
    let mut vocab_values: HashSet<Vec<u8>> = id_to_bytes.iter().cloned().collect();
    for special_token in &config.special_tokens {
        let special_bytes = special_token.as_bytes().to_vec();
        if vocab_values.insert(special_bytes.clone()) {
            id_to_bytes.push(special_bytes);
        }
    }

    let pretoken_counts = pretoken_counts_from_path(
        &config.input_path,
        &config.special_tokens,
        resolved_num_workers,
        config.chunk_bytes,
        input_file_bytes,
        &pool,
    )?;

    // Build state, run merges, then write artifacts.
}
```

The first 256 token IDs are always the raw byte tokens. Special tokens are
appended only when they are not already present.

## Pretokenization

The Rust pretokenizer uses the same GPT-style pattern as the Python tokenizer
and enhanced trainer:

```rust
pub const PAT: &str =
    r#"'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"#;

static PRETOKEN_RE: LazyLock<Regex> =
    LazyLock::new(|| Regex::new(PAT).expect("GPT-style pretokenizer pattern compiles"));
```

The implementation uses `fancy-regex` because Rust's standard `regex` crate
does not support the negative lookahead in `\s+(?!\S)`.

Special tokens are removed from normal text before pretoken counting, matching
the Python implementation:

```rust
pub fn pretoken_byte_counts(
    text: &str,
    special_tokens: &[String],
) -> Result<HashMap<Vec<u8>, Count>> {
    let sorted_specials = sorted_special_tokens(special_tokens);
    let mut counts = HashMap::new();

    if sorted_specials.is_empty() {
        count_segment(text, &mut counts)?;
        return Ok(counts);
    }

    let mut start = 0;
    while let Some((special_start, special_end, _)) =
        find_next_special(text, start, &sorted_specials)
    {
        if special_start > start {
            count_segment(&text[start..special_start], &mut counts)?;
        }
        start = special_end;
    }
    if start < text.len() {
        count_segment(&text[start..], &mut counts)?;
    }

    Ok(counts)
}
```

Special tokens are sorted by descending length before matching. This preserves
longest-match behavior for overlapping tokens such as:

```text
<|endoftext|>
<|endoftext|><|endoftext|>
```

## Safe Chunking

The Rust trainer mirrors the Python enhanced trainer's safe chunking rule:

- If there are no special tokens, training uses one range for semantic parity.
- If special tokens exist, the first special token is used as the chunk
  delimiter.
- Internal chunk boundaries are moved forward to the delimiter.

The boundary finder scans forward from approximate offsets:

```rust
pub fn find_chunk_boundaries(
    file: &mut File,
    desired_num_chunks: usize,
    split_special_token: &[u8],
) -> Result<Vec<u64>> {
    file.seek(SeekFrom::End(0))?;
    let file_size = file.stream_position()?;
    file.seek(SeekFrom::Start(0))?;

    let desired_num_chunks = desired_num_chunks.max(1).min(file_size as usize);
    let chunk_size = (file_size / desired_num_chunks as u64).max(1);
    let mut boundaries = Vec::with_capacity(desired_num_chunks + 1);
    for i in 0..=desired_num_chunks {
        boundaries.push((i as u64 * chunk_size).min(file_size));
    }
    *boundaries.last_mut().unwrap() = file_size;

    for boundary in boundaries.iter_mut().take(desired_num_chunks).skip(1) {
        let mut position = *boundary;
        file.seek(SeekFrom::Start(position))?;
        loop {
            let bytes_read = file.read(&mut buffer)?;
            if bytes_read == 0 {
                *boundary = file_size;
                break;
            }
            if let Some(found_at) = find_subslice(&buffer[..bytes_read], split_special_token) {
                *boundary = position + found_at as u64;
                break;
            }
            position += bytes_read as u64;
        }
    }

    boundaries.sort_unstable();
    boundaries.dedup();
    Ok(boundaries)
}
```

Pretoken counting across safe ranges is parallelized with Rayon. The merge loop
is deliberately not parallelized because each merge depends on the state after
the previous merge.

## Trainer State

The trainer uses integer token IDs internally:

```rust
pub type TokenId = u32;
pub type WordId = usize;
pub type Count = u64;
pub type TokenPair = (TokenId, TokenId);
pub type BytePair = (Vec<u8>, Vec<u8>);
```

The full mutable state is:

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

Each unique pretoken becomes one `word`, represented as its raw byte token IDs.
The count for that pretoken is stored separately:

```rust
for (pretoken, count) in pretoken_counts {
    words.push(pretoken.into_iter().map(TokenId::from).collect());
    word_counts.push(count);
}
```

Pair counts are weighted by the number of times each pretoken appears:

```rust
pub fn word_pair_frequencies(word: &[TokenId]) -> HashMap<TokenPair, Count> {
    let mut frequencies = HashMap::new();
    for window in word.windows(2) {
        let pair = (window[0], window[1]);
        *frequencies.entry(pair).or_insert(0) += 1;
    }
    frequencies
}
```

## Heap Ordering

The heap is one of the most important parity points. The Python enhanced trainer
selects:

1. Highest pair count.
2. On count ties, lexicographically larger underlying byte pair.

The Rust heap entry stores both token IDs and byte values:

```rust
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct HeapEntry {
    pub count: Count,
    pub left_bytes: Vec<u8>,
    pub right_bytes: Vec<u8>,
    pub pair: TokenPair,
}
```

The ordering intentionally compares byte values, not token IDs:

```rust
impl Ord for HeapEntry {
    fn cmp(&self, other: &Self) -> Ordering {
        self.count
            .cmp(&other.count)
            .then_with(|| self.left_bytes.cmp(&other.left_bytes))
            .then_with(|| self.right_bytes.cmp(&other.right_bytes))
            .then_with(|| self.pair.cmp(&other.pair))
    }
}
```

The heap uses lazy invalidation. Old heap entries are not removed immediately
when counts change. Instead, popping discards entries whose count no longer
matches `pair_counts`:

```rust
pub fn pop_best_pair(
    heap: &mut BinaryHeap<HeapEntry>,
    pair_counts: &HashMap<TokenPair, Count>,
) -> Option<TokenPair> {
    while let Some(entry) = heap.pop() {
        if pair_counts.get(&entry.pair) == Some(&entry.count) {
            return Some(entry.pair);
        }
    }
    None
}
```

## Merge Loop

The merge loop is sequential. For each best pair, the trainer:

1. Creates a new token by concatenating the left and right token bytes.
2. Assigns the next token ID.
3. Appends the byte pair to the merge sequence.
4. Finds affected words via `pair_to_word_ids`.
5. Removes old weighted pair counts for each affected word.
6. Rewrites the word with a non-overlapping left-to-right merge.
7. Adds new weighted pair counts.
8. Pushes changed pairs back onto the heap.

The non-overlapping word merge is:

```rust
pub fn merge_word(word: &[TokenId], pair: TokenPair, merged_token_id: TokenId) -> Vec<TokenId> {
    let mut merged_word = Vec::with_capacity(word.len());
    let mut i = 0;
    while i < word.len() {
        if i + 1 < word.len() && word[i] == pair.0 && word[i + 1] == pair.1 {
            merged_word.push(merged_token_id);
            i += 2;
        } else {
            merged_word.push(word[i]);
            i += 1;
        }
    }
    merged_word
}
```

For example:

```text
[a, a, a] with merge (a, a) becomes [aa, a]
```

It does not become `[a, aa]` or `[aa, aa]`.

The merge update returns the set of changed pairs so they can be pushed back
onto the lazy heap:

```rust
fn apply_merge(state: &mut TrainerState, best_pair: TokenPair) -> HashSet<TokenPair> {
    let mut changed_pairs = HashSet::new();
    let mut merged_token = state.id_to_bytes[best_pair.0 as usize].clone();
    merged_token.extend(&state.id_to_bytes[best_pair.1 as usize]);
    let merged_token_id = state.id_to_bytes.len() as TokenId;

    state.merges.push((
        state.id_to_bytes[best_pair.0 as usize].clone(),
        state.id_to_bytes[best_pair.1 as usize].clone(),
    ));
    state.id_to_bytes.push(merged_token);

    // For each affected word:
    // - subtract old pair frequencies
    // - rewrite the word
    // - add new pair frequencies
    // - record changed pairs

    changed_pairs.insert(best_pair);
    changed_pairs
}
```

The actual implementation keeps the bookkeeping explicit so counts and postings
stay in sync.

## Artifact Writing

The Rust trainer writes:

```text
vocab.json
merges.txt
metadata.json
```

It does not write `vocab.pkl` or `merges.pkl`.

The vocabulary JSON uses the same object shape as the Python enhanced trainer:

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

The Rust code builds each entry from raw bytes:

```rust
VocabJsonEntry {
    id,
    byte_values: token.clone(),
    hex: token
        .iter()
        .map(|byte| format!("{byte:02x}"))
        .collect::<Vec<_>>()
        .join(""),
    repr: python_bytes_repr(token),
    utf8: String::from_utf8(token.clone()).ok(),
}
```

The merge text is byte-for-byte compatible with the Python enhanced merge text
format:

```text
# cs336_basics enhanced BPE merges v1
# rank	left_repr	right_repr	merged_repr
0	b'a'	b'b'	b'ab'
```

The byte-literal formatter is implemented in `bytes_repr.rs`:

```rust
pub fn python_bytes_repr(bytes: &[u8]) -> String {
    let quote = if bytes.contains(&b'\'') && !bytes.contains(&b'"') {
        b'"'
    } else {
        b'\''
    };

    let mut out = String::new();
    out.push('b');
    out.push(quote as char);
    for &byte in bytes {
        match byte {
            b'\\' => out.push_str("\\\\"),
            b'\n' => out.push_str("\\n"),
            b'\t' => out.push_str("\\t"),
            b'\r' => out.push_str("\\r"),
            b if b == quote => {
                out.push('\\');
                out.push(quote as char);
            }
            0x20..=0x7e => out.push(byte as char),
            _ => out.push_str(&format!("\\x{byte:02x}")),
        }
    }
    out.push(quote as char);
    out
}
```

## Encoder Loading

The encoder can load:

- Enhanced `vocab.json` files produced by Python.
- Enhanced `vocab.json` files produced by Rust.
- GPT-2-style vocabulary JSON fixtures.
- Enhanced `merges.txt` files.
- GPT-2-style merges fixtures.

Enhanced vocabulary loading reads the `tokens` list:

```rust
if let Some(tokens) = data.get("tokens").and_then(Value::as_array) {
    return load_enhanced_vocab(tokens);
}
```

GPT-2 vocabulary loading uses the same bytes-to-unicode mapping used by the
Python tests:

```rust
pub fn gpt2_byte_decoder() -> HashMap<char, u8> {
    let mut byte_values: Vec<u16> = (b'!' as u16..=b'~' as u16).collect();
    byte_values.extend(0x00a1..=0x00ac);
    byte_values.extend(0x00ae..=0x00ff);
    let mut code_points = byte_values.clone();
    let mut next_shifted = 0u16;
    for byte_value in 0u16..=255 {
        if !byte_values.contains(&byte_value) {
            byte_values.push(byte_value);
            code_points.push(256 + next_shifted);
            next_shifted += 1;
        }
    }
    // Return unicode character -> original byte.
}
```

Enhanced merge loading parses Python-style bytes literals from tab-separated
merge files:

```rust
let tab_parts: Vec<_> = line.split('\t').collect();
if tab_parts.len() >= 3
    && tab_parts[0]
        .chars()
        .all(|character| character.is_ascii_digit())
{
    merges.push((
        parse_python_bytes_literal(tab_parts[1])?,
        parse_python_bytes_literal(tab_parts[2])?,
    ));
    continue;
}
```

## Encoder State

`Tokenizer` stores byte vocabulary, special token IDs, merge ranks, merge
outputs, and a simple pretoken cache:

```rust
pub struct Tokenizer {
    vocab: Vec<Vec<u8>>,
    token_to_id: HashMap<Vec<u8>, TokenId>,
    special_tokens: Vec<String>,
    sorted_special_tokens: Vec<String>,
    special_token_ids: HashMap<String, TokenId>,
    merge_ranks_by_id: HashMap<TokenPair, usize>,
    merge_output_by_pair_id: HashMap<TokenPair, TokenId>,
    byte_token_ids: [TokenId; 256],
    encode_cache: HashMap<Vec<u8>, Vec<TokenId>>,
    max_special_token_length: usize,
}
```

During construction, missing special tokens are appended to the vocabulary, just
as in the Python tokenizer:

```rust
for special_token in &special_tokens {
    let special_bytes = special_token.as_bytes().to_vec();
    let token_id = match token_to_id.get(&special_bytes).copied() {
        Some(token_id) => token_id,
        None => {
            let token_id = vocab.len() as TokenId;
            vocab.push(special_bytes.clone());
            token_to_id.insert(special_bytes, token_id);
            token_id
        }
    };
    special_token_ids.insert(special_token.clone(), token_id);
}
```

## Encoding Algorithm

Text encoding splits around special tokens first. Normal text spans are
pretokenized and encoded with the merge table:

```rust
fn encode_text(&mut self, text: &str) -> Result<Vec<TokenId>> {
    let mut ids = Vec::new();
    if self.sorted_special_tokens.is_empty() {
        ids.extend(self.encode_normal_text(text)?);
        return Ok(ids);
    }

    let mut start = 0;
    while let Some((special_start, special_end, special_token)) =
        find_next_special(text, start, &self.sorted_special_tokens)
    {
        if special_start > start {
            ids.extend(self.encode_normal_text(&text[start..special_start])?);
        }
        ids.push(*self.special_token_ids.get(&special_token).unwrap());
        start = special_end;
    }
    if start < text.len() {
        ids.extend(self.encode_normal_text(&text[start..])?);
    }
    Ok(ids)
}
```

Pretoken encoding mirrors `cs336_basics/tokenizer.py`:

1. Start with byte token IDs.
2. Scan adjacent pairs.
3. Find the pair with the smallest merge rank.
4. Rewrite all non-overlapping occurrences of that pair.
5. Repeat until no ranked pair remains.

```rust
pub fn encode_pretoken(&mut self, pretoken: &[u8]) -> Result<Vec<TokenId>> {
    let mut tokens: Vec<TokenId> = pretoken
        .iter()
        .map(|byte| self.byte_token_ids[*byte as usize])
        .collect();

    while tokens.len() > 1 {
        let mut best_pair: Option<TokenPair> = None;
        let mut best_rank: Option<usize> = None;
        for window in tokens.windows(2) {
            let pair = (window[0], window[1]);
            if let Some(&rank) = self.merge_ranks_by_id.get(&pair) {
                if best_rank.is_none_or(|current| rank < current) {
                    best_pair = Some(pair);
                    best_rank = Some(rank);
                }
            }
        }

        let Some(best_pair) = best_pair else {
            break;
        };
        let merged_token = self.merge_output_by_pair_id[&best_pair];

        let mut merged_tokens = Vec::with_capacity(tokens.len());
        let mut i = 0;
        while i < tokens.len() {
            if i + 1 < tokens.len() && tokens[i] == best_pair.0 && tokens[i + 1] == best_pair.1 {
                merged_tokens.push(merged_token);
                i += 2;
            } else {
                merged_tokens.push(tokens[i]);
                i += 1;
            }
        }
        tokens = merged_tokens;
    }

    Ok(tokens)
}
```

Decoding concatenates token bytes and performs UTF-8 replacement decoding:

```rust
pub fn decode(&self, ids: &[TokenId]) -> Result<String> {
    let mut bytes = Vec::new();
    for &token_id in ids {
        let token = self.vocab.get(token_id as usize).unwrap();
        bytes.extend(token);
    }
    Ok(String::from_utf8_lossy(&bytes).into_owned())
}
```

## Streaming Encoding

Streaming encoding keeps a rolling text buffer. It only flushes a prefix when
the prefix ends on a safe token boundary:

- It does not split a regex pretoken.
- It keeps enough suffix text to detect a special-token prefix.

The flush logic mirrors Python's `encode_iterable` behavior:

```rust
pub fn encode_iterable<I>(&mut self, iterable: I) -> Result<Vec<TokenId>>
where
    I: IntoIterator<Item = String>,
{
    let mut output = Vec::new();
    let mut buffer = String::new();
    for chunk in iterable {
        if chunk.is_empty() {
            continue;
        }
        buffer.push_str(&chunk);
        let segments = self.token_segments(&buffer)?;
        let flush_index = self.stream_flush_index_from_segments(&buffer, &segments);
        if flush_index > 0 {
            output.extend(self.encode_prefix_with_context(&buffer, flush_index, &segments)?);
            buffer = buffer[flush_index..].to_string();
        }
    }
    if !buffer.is_empty() {
        output.extend(self.encode_text(&buffer)?);
    }
    Ok(output)
}
```

The CLI exposes this path with:

```sh
--stream-chunk-bytes 7
```

## Parity and Test Coverage

Rust unit tests cover the critical local behavior:

- Python bytes literal formatting and parsing.
- GPT-style pretokenization.
- Special-token splitting.
- Chunk boundary discovery.
- Word pair frequency counting.
- Non-overlapping merge behavior.
- Heap tie-breaking and stale entry discard.
- Enhanced vocab and merge loading.
- Pretoken encoding.
- UTF-8 replacement decoding.
- Streaming equivalence.

Python parity tests live in `tests/test_rust_bpe_parity.py`. They are skipped if
Cargo is unavailable:

```python
pytestmark = pytest.mark.skipif(
    shutil.which("cargo") is None,
    reason="Rust parity tests require cargo",
)
```

The trainer parity test compares Rust artifacts with Python enhanced artifacts:

```python
assert (py_out / "merges.txt").read_text(encoding="utf-8") == (
    rs_out / "merges.txt"
).read_text(encoding="utf-8")
assert json.loads((py_out / "vocab.json").read_text(encoding="utf-8")) == json.loads(
    (rs_out / "vocab.json").read_text(encoding="utf-8")
)
```

The encoder parity test checks that Rust token IDs match the Python tokenizer on
Rust-generated artifacts:

```python
py_tokenizer = Tokenizer.from_files(
    str(out_dir / "vocab.json"),
    str(out_dir / "merges.txt"),
    special_tokens=["<|endoftext|>"],
)
py_ids = py_tokenizer.encode(corpus.read_text(encoding="utf-8"))

run_rust_encode(
    out_dir / "vocab.json",
    out_dir / "merges.txt",
    corpus,
    ids_path,
    ["<|endoftext|>"],
)
assert json.loads(ids_path.read_text(encoding="utf-8")) == py_ids
```

Streaming parity checks chunk sizes `1`, `2`, `7`, and `4096` against whole-file
Rust encoding. The `.npy` parity test loads Rust output with NumPy and checks
that the `uint16` token array and SHA-256 sidecar metadata match the Python
tokenizer IDs.

## Current Parity Contract

The current implementation is intended to provide:

- Identical merge sequence for validated cases.
- Equivalent vocabulary token IDs and byte values.
- Matching encoded token IDs.
- Matching decoded text with UTF-8 replacement behavior.

Known intentional differences:

- Rust does not write `vocab.pkl` or `merges.pkl`.
- Rust training does not write `.npy` token arrays; Rust encoding can serialize
  `.npy` arrays with sidecar metadata.
- `metadata.json` has Rust-specific fields and format naming.
- Raw `vocab.json` bytes may differ because Python and Rust serialize JSON
  differently, but the parsed JSON object is expected to match.

## Full TinyStories Run

The Rust trainer has been run on the full TinyStories training corpus:

```text
data/TinyStoriesV2-GPT4-train.txt
```

The run used the same configuration as the existing Python enhanced trainer
artifact in `data/tinystories_bpe_10000/`:

```sh
target/release/cs336-bpe-train \
  --input data/TinyStoriesV2-GPT4-train.txt \
  --vocab-size 10000 \
  --special-token '<|endoftext|>' \
  --num-workers 8 \
  --chunk-bytes 67108864 \
  --heap-rebuild-factor 3.0 \
  --output-dir data/rust/tinystories_bpe_10000
```

The release binary was built before timing:

```sh
cargo build --release -p cs336_bpe_rs
```

Generated Rust artifacts are stored in:

```text
data/rust/tinystories_bpe_10000/
|-- merges.txt
|-- metadata.json
|-- run_timing.txt
`-- vocab.json
```

Timing summary:

| Implementation | Total time |
| --- | ---: |
| Python enhanced trainer | `85.57s` |
| Rust trainer metadata | `30.07s` |
| Rust `/usr/bin/time` wall clock | `30.10s` |

Using the Python metadata total and the Rust wall-clock time, the Rust trainer
was about `2.84x` faster overall and saved about `55.5s`.

Phase-level timing comparison:

| Phase | Python | Rust | Speedup |
| --- | ---: | ---: | ---: |
| Pretoken counting | `82.05s` | `29.44s` | `2.79x` |
| Word materialization | `0.013s` | `0.0067s` | `1.88x` |
| Initial pair state | `0.160s` | `0.030s` | `5.30x` |
| Initial heap build | `0.00090s` | `0.00027s` | `3.34x` |
| Merge loop | `3.24s` | `0.59s` | `5.52x` |
| Artifact writing | `0.095s` | `0.010s` | `9.27x` |

The `/usr/bin/time -v` run reported:

```text
Elapsed (wall clock) time: 0:30.10
User time: 211.56s
System time: 2.93s
CPU utilization: 712%
Maximum resident set size: 573388 KB
```

The Rust and Python full TinyStories runs produced matching training statistics:

| Field | Value |
| --- | ---: |
| Input bytes | `2227753162` |
| Requested vocab size | `10000` |
| Final vocab size | `10000` |
| Merge count | `9743` |
| Unique pretokens | `59933` |
| Total pretokens | `536592168` |
| Initial pair count | `2108` |
| Final pair count | `47278` |
| Heap rebuild count | `8` |

Artifact parity for the full TinyStories run:

- `merges.txt` is byte-for-byte identical to the Python enhanced trainer output.
- `vocab.json` is equal after parsing as JSON.
- Raw `vocab.json` bytes differ because Python and Rust serialize JSON strings
  differently.
- Rust training does not produce `vocab.pkl` or `merges.pkl`; the encoder can
  produce `.npy` token-ID arrays from `vocab.json` and `merges.txt`.

## Validation Commands

Run Rust formatting and tests:

```sh
cargo fmt --all --check
cargo test -p cs336_bpe_rs
cargo test --workspace
```

Run Python parity tests:

```sh
uv run pytest tests/test_rust_bpe_parity.py
```

Run the existing tokenizer and BPE trainer tests:

```sh
uv run pytest tests/test_train_bpe.py tests/test_tokenizer.py
```
