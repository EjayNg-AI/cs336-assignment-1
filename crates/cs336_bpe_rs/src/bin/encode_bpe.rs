use std::fs;
use std::path::PathBuf;

use anyhow::Result;
use clap::Parser;
use cs336_bpe_rs::Tokenizer;

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
    output_ids_json: PathBuf,

    #[arg(long)]
    stream_chunk_bytes: Option<usize>,
}

fn main() -> Result<()> {
    let args = Args::parse();
    let mut tokenizer = Tokenizer::from_files(args.vocab, args.merges, args.special_tokens)?;
    let text = fs::read_to_string(args.input)?;
    let ids = if let Some(chunk_bytes) = args.stream_chunk_bytes {
        let chunks = split_string_on_char_boundaries(&text, chunk_bytes);
        tokenizer.encode_iterable(chunks)?
    } else {
        tokenizer.encode(&text)?
    };
    let mut json = serde_json::to_string(&ids)?;
    json.push('\n');
    fs::write(args.output_ids_json, json)?;
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
