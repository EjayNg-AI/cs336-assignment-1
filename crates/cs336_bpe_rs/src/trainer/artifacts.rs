use std::fs;
use std::path::{Path, PathBuf};

use anyhow::Result;
use serde::Serialize;

use crate::bytes_repr::python_bytes_repr;
use crate::config::{MERGES_TEXT_FILENAME, VOCAB_JSON_FILENAME};

use super::state::BytePair;

#[derive(Serialize)]
struct VocabJson<'a> {
    format: &'a str,
    tokens: Vec<VocabJsonEntry>,
}

#[derive(Serialize)]
struct VocabJsonEntry {
    id: usize,
    byte_values: Vec<u8>,
    hex: String,
    repr: String,
    utf8: Option<String>,
}

pub fn default_output_dir(input_path: &Path, vocab_size: usize) -> PathBuf {
    let stem = input_path
        .file_stem()
        .and_then(|value| value.to_str())
        .unwrap_or("corpus");
    input_path.with_file_name(format!("{stem}_bpe_{vocab_size}"))
}

pub fn write_training_artifacts(
    vocab: &[Vec<u8>],
    merges: &[BytePair],
    output_dir: Option<&Path>,
    input_path: &Path,
    vocab_size: usize,
) -> Result<PathBuf> {
    let resolved_output_dir = output_dir
        .map(Path::to_path_buf)
        .unwrap_or_else(|| default_output_dir(input_path, vocab_size));
    fs::create_dir_all(&resolved_output_dir)?;
    write_vocab_json(vocab, &resolved_output_dir.join(VOCAB_JSON_FILENAME))?;
    write_merges_text(merges, &resolved_output_dir.join(MERGES_TEXT_FILENAME))?;
    Ok(resolved_output_dir)
}

pub fn write_vocab_json(vocab: &[Vec<u8>], output_path: &Path) -> Result<()> {
    let payload = VocabJson {
        format: "cs336_basics.enhanced_bpe.v1",
        tokens: vocab
            .iter()
            .enumerate()
            .map(|(id, token)| VocabJsonEntry {
                id,
                byte_values: token.clone(),
                hex: token
                    .iter()
                    .map(|byte| format!("{byte:02x}"))
                    .collect::<Vec<_>>()
                    .join(""),
                repr: python_bytes_repr(token),
                utf8: String::from_utf8(token.clone()).ok(),
            })
            .collect(),
    };
    let mut json = serde_json::to_string_pretty(&payload)?;
    json.push('\n');
    fs::write(output_path, json)?;
    Ok(())
}

pub fn write_merges_text(merges: &[BytePair], output_path: &Path) -> Result<()> {
    let mut out = String::new();
    out.push_str("# cs336_basics enhanced BPE merges v1\n");
    out.push_str("# rank\tleft_repr\tright_repr\tmerged_repr\n");
    for (rank, (left, right)) in merges.iter().enumerate() {
        let mut merged = left.clone();
        merged.extend(right);
        out.push_str(&format!(
            "{rank}\t{}\t{}\t{}\n",
            python_bytes_repr(left),
            python_bytes_repr(right),
            python_bytes_repr(&merged)
        ));
    }
    fs::write(output_path, out)?;
    Ok(())
}
