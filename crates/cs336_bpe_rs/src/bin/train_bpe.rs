use std::path::PathBuf;

use anyhow::Result;
use clap::Parser;
use cs336_bpe_rs::{train_bpe, TrainConfig};

#[derive(Debug, Parser)]
#[command(name = "cs336-bpe-train")]
#[command(about = "Train a byte-level BPE tokenizer with CS336 enhanced-trainer semantics.")]
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

fn main() -> Result<()> {
    let args = Args::parse();
    let output = train_bpe(TrainConfig {
        input_path: args.input,
        vocab_size: args.vocab_size,
        special_tokens: args.special_tokens,
        num_workers: args.num_workers,
        chunk_bytes: args.chunk_bytes,
        heap_rebuild_factor: args.heap_rebuild_factor,
        output_dir: args.output_dir,
    })?;
    println!("{}", output.output_dir.display());
    Ok(())
}
