use std::fs::{self, File};
use std::io::{self, BufWriter, Read, Write};
use std::path::{Path, PathBuf};
use std::time::{Instant, SystemTime, UNIX_EPOCH};

use anyhow::{bail, Context, Result};
use clap::Parser;
use cs336_bpe_rs::npy::copy_raw_uint16_to_npy;
use cs336_bpe_rs::sha256::{hex_digest, Sha256};
use cs336_bpe_rs::Tokenizer;
use serde_json::{json, Value};

const DEFAULT_STREAM_CHUNK_BYTES: usize = 1_048_576;
const TOKEN_BYTE_BUFFER_BYTES: usize = 1_048_576;
const UINT16_MAX: u32 = u16::MAX as u32;

#[derive(Debug, Parser)]
#[command(name = "cs336-bpe-encode")]
#[command(about = "Encode text with a byte-level BPE tokenizer.")]
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

fn main() -> Result<()> {
    let args = Args::parse();
    if args.output_ids_json.is_none() && args.output_ids_npy.is_none() {
        bail!("provide --output-ids-json, --output-ids-npy, or both");
    }

    if let Some(output_npy_path) = args.output_ids_npy.as_ref() {
        encode_to_npy(&args, output_npy_path)?;
    } else {
        let output_json_path = args
            .output_ids_json
            .as_ref()
            .context("--output-ids-json is required when --output-ids-npy is not set")?;
        encode_to_json(&args, output_json_path)?;
    }
    Ok(())
}

fn encode_to_json(args: &Args, output_json_path: &Path) -> Result<()> {
    let mut tokenizer =
        Tokenizer::from_files(&args.vocab, &args.merges, args.special_tokens.clone())?;
    let text = fs::read_to_string(&args.input)?;
    let ids = if let Some(chunk_bytes) = args.stream_chunk_bytes {
        let chunks = split_string_on_char_boundaries(&text, chunk_bytes);
        tokenizer.encode_iterable(chunks)?
    } else {
        tokenizer.encode(&text)?
    };
    let mut json = serde_json::to_string(&ids)?;
    json.push('\n');
    write_text_atomically(output_json_path, &json, args.force)?;
    Ok(())
}

fn encode_to_npy(args: &Args, output_npy_path: &Path) -> Result<()> {
    if output_npy_path.exists() && !args.force {
        bail!(
            "output already exists at {}; pass --force to replace it",
            output_npy_path.display()
        );
    }
    if let Some(metadata_path) = args.metadata_json.as_ref() {
        if metadata_path.exists() && !args.force {
            bail!(
                "metadata already exists at {}; pass --force to replace it",
                metadata_path.display()
            );
        }
    }
    if let Some(output_json_path) = args.output_ids_json.as_ref() {
        if output_json_path.exists() && !args.force {
            bail!(
                "JSON output already exists at {}; pass --force to replace it",
                output_json_path.display()
            );
        }
    }
    if let Some(parent) = output_npy_path.parent() {
        fs::create_dir_all(parent)?;
    }

    let raw_tmp_path = output_npy_path.with_extension("uint16.tmp");
    let npy_tmp_path = output_npy_path.with_extension("npy.tmp");
    remove_if_exists(&raw_tmp_path)?;
    remove_if_exists(&npy_tmp_path)?;

    let input_bytes = args
        .input
        .metadata()
        .with_context(|| format!("failed to stat input {}", args.input.display()))?
        .len();
    let start_time = Instant::now();
    let mut tokenizer =
        Tokenizer::from_files(&args.vocab, &args.merges, args.special_tokens.clone())?;
    let stream_chunk_bytes = args
        .stream_chunk_bytes
        .unwrap_or(DEFAULT_STREAM_CHUNK_BYTES);
    let chunks = Utf8FileChunks::new(&args.input, stream_chunk_bytes)?;
    let raw_file = File::create(&raw_tmp_path).with_context(|| {
        format!(
            "failed to create temporary token stream {}",
            raw_tmp_path.display()
        )
    })?;
    let mut raw_file = BufWriter::with_capacity(TOKEN_BYTE_BUFFER_BYTES, raw_file);
    let mut token_byte_buffer = Vec::with_capacity(TOKEN_BYTE_BUFFER_BYTES);
    let mut hasher = Sha256::new();
    let mut token_count = 0u64;
    let mut min_token_id = UINT16_MAX;
    let mut max_token_id = 0u32;
    let progress_interval = args.token_progress_interval.unwrap_or(50_000_000);
    let mut next_progress = progress_interval;

    tokenizer.encode_iterable_result_to_sink(chunks, |token_id| {
        if token_id > UINT16_MAX {
            bail!("token id {token_id} exceeds uint16 max {UINT16_MAX}");
        }
        let bytes = (token_id as u16).to_le_bytes();
        token_byte_buffer.extend_from_slice(&bytes);
        if token_byte_buffer.len() >= TOKEN_BYTE_BUFFER_BYTES {
            raw_file.write_all(&token_byte_buffer)?;
            hasher.update(&token_byte_buffer);
            token_byte_buffer.clear();
        }
        token_count += 1;
        min_token_id = min_token_id.min(token_id);
        max_token_id = max_token_id.max(token_id);
        if progress_interval > 0 && token_count >= next_progress {
            eprintln!(
                "{}: {token_count} tokens written to temporary stream after {:.1} sec",
                args.split_name.as_deref().unwrap_or("encode"),
                start_time.elapsed().as_secs_f64()
            );
            while next_progress <= token_count {
                next_progress = next_progress.saturating_add(progress_interval);
                if next_progress == u64::MAX {
                    break;
                }
            }
        }
        Ok(())
    })?;
    if !token_byte_buffer.is_empty() {
        raw_file.write_all(&token_byte_buffer)?;
        hasher.update(&token_byte_buffer);
    }
    raw_file.flush()?;
    drop(raw_file);

    if token_count == 0 {
        min_token_id = 0;
    }

    copy_raw_uint16_to_npy(&raw_tmp_path, &npy_tmp_path, token_count)?;
    remove_if_exists(&raw_tmp_path)?;
    if fs::rename(&npy_tmp_path, output_npy_path).is_err() {
        remove_if_exists(output_npy_path)?;
        fs::rename(&npy_tmp_path, output_npy_path)?;
    }

    let elapsed_seconds = start_time.elapsed().as_secs_f64();
    let token_stream_sha256 = hex_digest(hasher.finalize());
    let metadata = json!({
        "format": "cs336_basics.bpe_tokenized_corpus.v1",
        "status": "complete",
        "created_utc": utc_now_iso8601(),
        "split_name": args.split_name.as_deref().unwrap_or("encode"),
        "corpus": args.corpus.as_deref(),
        "split": args.split.as_deref(),
        "input_path": args.input.to_string_lossy().to_string(),
        "input_bytes": input_bytes,
        "tokenizer_vocab_path": args.vocab.to_string_lossy().to_string(),
        "tokenizer_merges_path": args.merges.to_string_lossy().to_string(),
        "special_tokens": &args.special_tokens,
        "output_path": output_npy_path.to_string_lossy().to_string(),
        "dtype": "uint16",
        "numpy_dtype_descr": "<u2",
        "shape": [token_count],
        "token_count": token_count,
        "min_token_id": min_token_id,
        "max_token_id": max_token_id,
        "token_stream_sha256_uint16_le": token_stream_sha256,
        "bytes_per_token": if token_count == 0 { Value::Null } else { json!(input_bytes as f64 / token_count as f64) },
        "elapsed_seconds": elapsed_seconds,
        "tokens_per_second": if elapsed_seconds == 0.0 { Value::Null } else { json!(token_count as f64 / elapsed_seconds) },
        "input_bytes_per_second": if elapsed_seconds == 0.0 { Value::Null } else { json!(input_bytes as f64 / elapsed_seconds) },
        "stream_chunk_bytes": stream_chunk_bytes,
        "load_example": format!("np.load('{}', mmap_mode='r')", output_npy_path.display()),
    });

    if let Some(metadata_path) = args.metadata_json.as_ref() {
        write_json_atomically(metadata_path, &metadata, args.force)?;
    }
    if let Some(output_json_path) = args.output_ids_json.as_ref() {
        let ids = read_npy_payload_as_json_array(output_npy_path, token_count)?;
        write_text_atomically(output_json_path, &ids, args.force)?;
    }
    if let Some(manifest_path) = args.manifest_json.as_ref() {
        write_manifest(manifest_path)?;
    }

    eprintln!(
        "Completed {}: {token_count} tokens, {:.3} bytes/token, {:.1} sec",
        args.split_name.as_deref().unwrap_or("encode"),
        if token_count == 0 {
            0.0
        } else {
            input_bytes as f64 / token_count as f64
        },
        elapsed_seconds
    );
    Ok(())
}

fn split_string_on_char_boundaries(text: &str, chunk_bytes: usize) -> Vec<String> {
    if chunk_bytes == 0 {
        return vec![text.to_string()];
    }
    let mut chunks = Vec::new();
    let mut start = 0;
    while start < text.len() {
        let mut end = (start + chunk_bytes).min(text.len());
        while end > start && !text.is_char_boundary(end) {
            end -= 1;
        }
        if end == start {
            end = text[start..]
                .char_indices()
                .nth(1)
                .map(|(relative, _)| start + relative)
                .unwrap_or(text.len());
        }
        chunks.push(text[start..end].to_string());
        start = end;
    }
    chunks
}

struct Utf8FileChunks {
    file: File,
    pending: Vec<u8>,
    chunk_bytes: usize,
    done: bool,
}

impl Utf8FileChunks {
    fn new(path: &Path, chunk_bytes: usize) -> Result<Self> {
        Ok(Self {
            file: File::open(path)
                .with_context(|| format!("failed to open input corpus {}", path.display()))?,
            pending: Vec::new(),
            chunk_bytes: chunk_bytes.max(1),
            done: false,
        })
    }
}

impl Iterator for Utf8FileChunks {
    type Item = io::Result<String>;

    fn next(&mut self) -> Option<Self::Item> {
        let mut read_buffer = vec![0u8; self.chunk_bytes];
        while !self.done && self.pending.len() < self.chunk_bytes {
            match self.file.read(&mut read_buffer) {
                Ok(0) => self.done = true,
                Ok(n) => self.pending.extend_from_slice(&read_buffer[..n]),
                Err(error) => return Some(Err(error)),
            }
        }

        if self.pending.is_empty() {
            return None;
        }

        let limit = self.pending.len().min(self.chunk_bytes);
        let mut end = valid_utf8_prefix_len(&self.pending, limit);
        if end == 0 && !self.done && self.pending.len() < 4 {
            return self.next();
        }
        if end == 0 {
            end = valid_utf8_prefix_len(&self.pending, self.pending.len());
        }
        if end == 0 {
            return Some(Err(io::Error::new(
                io::ErrorKind::InvalidData,
                "input is not valid UTF-8",
            )));
        }

        let rest = self.pending.split_off(end);
        let chunk_bytes = std::mem::replace(&mut self.pending, rest);
        Some(
            String::from_utf8(chunk_bytes)
                .map_err(|error| io::Error::new(io::ErrorKind::InvalidData, error)),
        )
    }
}

fn valid_utf8_prefix_len(bytes: &[u8], limit: usize) -> usize {
    for end in (1..=limit).rev() {
        if std::str::from_utf8(&bytes[..end]).is_ok() {
            return end;
        }
    }
    0
}

fn remove_if_exists(path: &Path) -> Result<()> {
    match fs::remove_file(path) {
        Ok(()) => Ok(()),
        Err(error) if error.kind() == io::ErrorKind::NotFound => Ok(()),
        Err(error) => Err(error).with_context(|| format!("failed to remove {}", path.display())),
    }
}

fn write_json_atomically(path: &Path, value: &Value, force: bool) -> Result<()> {
    let mut text = serde_json::to_string_pretty(value)?;
    text.push('\n');
    write_text_atomically(path, &text, force)
}

fn write_text_atomically(path: &Path, text: &str, force: bool) -> Result<()> {
    if path.exists() && !force {
        bail!(
            "output already exists at {}; pass --force to replace it",
            path.display()
        );
    }
    if let Some(parent) = path.parent() {
        fs::create_dir_all(parent)?;
    }
    let tmp_path = path.with_extension(format!(
        "{}.tmp",
        path.extension()
            .and_then(|extension| extension.to_str())
            .unwrap_or("out")
    ));
    fs::write(&tmp_path, text)?;
    if fs::rename(&tmp_path, path).is_err() {
        remove_if_exists(path)?;
        fs::rename(&tmp_path, path)?;
    }
    Ok(())
}

fn read_npy_payload_as_json_array(path: &Path, token_count: u64) -> Result<String> {
    let mut file = File::open(path)?;
    let mut header_prefix = [0u8; 10];
    file.read_exact(&mut header_prefix)?;
    let header_len = u16::from_le_bytes([header_prefix[8], header_prefix[9]]) as usize;
    let mut header = vec![0u8; header_len];
    file.read_exact(&mut header)?;
    let mut ids = Vec::with_capacity(token_count as usize);
    let mut bytes = [0u8; 2];
    for _ in 0..token_count {
        file.read_exact(&mut bytes)?;
        ids.push(u16::from_le_bytes(bytes));
    }
    let mut json = serde_json::to_string(&ids)?;
    json.push('\n');
    Ok(json)
}

fn write_manifest(path: &Path) -> Result<()> {
    let output_dir = path.parent().unwrap_or_else(|| Path::new("."));
    let mut splits = Vec::new();
    for entry in fs::read_dir(output_dir)? {
        let entry = entry?;
        if !entry.file_type()?.is_dir() {
            continue;
        }
        for nested in fs::read_dir(entry.path())? {
            let nested = nested?;
            let nested_path = nested.path();
            if nested_path.extension().and_then(|value| value.to_str()) != Some("json") {
                continue;
            }
            let value: Value = serde_json::from_str(&fs::read_to_string(&nested_path)?)?;
            if value.get("format").and_then(Value::as_str)
                == Some("cs336_basics.bpe_tokenized_corpus.v1")
            {
                splits.push(value);
            }
        }
    }
    splits.sort_by(|left, right| {
        left.get("split_name")
            .and_then(Value::as_str)
            .cmp(&right.get("split_name").and_then(Value::as_str))
    });
    let manifest = json!({
        "format": "cs336_basics.bpe_experiment_3_manifest.v1",
        "updated_utc": utc_now_iso8601(),
        "output_dir": output_dir.to_string_lossy().to_string(),
        "dtype": "uint16",
        "load_example": "np.load('data/bpe_tokenized_corpora_rs/tinystories/train.npy', mmap_mode='r')",
        "splits": splits,
    });
    write_json_atomically(path, &manifest, true)
}

fn utc_now_iso8601() -> String {
    let now = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_secs() as i64;
    unix_seconds_to_utc_iso8601(now)
}

fn unix_seconds_to_utc_iso8601(seconds: i64) -> String {
    let days = seconds.div_euclid(86_400);
    let seconds_of_day = seconds.rem_euclid(86_400);
    let (year, month, day) = civil_from_days(days);
    let hour = seconds_of_day / 3_600;
    let minute = (seconds_of_day % 3_600) / 60;
    let second = seconds_of_day % 60;
    format!("{year:04}-{month:02}-{day:02}T{hour:02}:{minute:02}:{second:02}Z")
}

fn civil_from_days(days_since_epoch: i64) -> (i64, i64, i64) {
    let z = days_since_epoch + 719_468;
    let era = if z >= 0 { z } else { z - 146_096 }.div_euclid(146_097);
    let doe = z - era * 146_097;
    let yoe = (doe - doe / 1_460 + doe / 36_524 - doe / 146_096).div_euclid(365);
    let mut year = yoe + era * 400;
    let doy = doe - (365 * yoe + yoe / 4 - yoe / 100);
    let mp = (5 * doy + 2).div_euclid(153);
    let day = doy - (153 * mp + 2).div_euclid(5) + 1;
    let month = mp + if mp < 10 { 3 } else { -9 };
    if month <= 2 {
        year += 1;
    }
    (year, month, day)
}

#[cfg(test)]
mod tests {
    use super::unix_seconds_to_utc_iso8601;

    #[test]
    fn formats_unix_epoch_as_utc() {
        assert_eq!(unix_seconds_to_utc_iso8601(0), "1970-01-01T00:00:00Z");
    }
}
