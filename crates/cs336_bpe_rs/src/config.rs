pub const DEFAULT_CHUNK_BYTES: usize = 64 * 1024 * 1024;
pub const MIN_PARALLEL_BYTES: u64 = 16 * 1024 * 1024;
pub const MIN_PARALLEL_WORDS: usize = 20_000;

pub const VOCAB_JSON_FILENAME: &str = "vocab.json";
pub const MERGES_TEXT_FILENAME: &str = "merges.txt";
pub const METADATA_FILENAME: &str = "metadata.json";
