use std::fs;
use std::path::Path;

use anyhow::{bail, Result};

use crate::bytes_repr::parse_python_bytes_literal;
use crate::encoder::vocab::gpt2_byte_decoder;
use crate::errors::BpeError;
use crate::trainer::state::BytePair;

pub fn load_merges(path: &Path) -> Result<Vec<BytePair>> {
    if path.extension().and_then(|value| value.to_str()) == Some("pkl") {
        bail!(BpeError::UnsupportedMergesFormat(
            "pickle merges are intentionally unsupported by the Rust loader".to_string()
        ));
    }

    let byte_decoder = gpt2_byte_decoder();
    let mut merges = Vec::new();
    for line in fs::read_to_string(path)?.lines() {
        let line = line.trim_end_matches('\n');
        if line.is_empty() || line.starts_with('#') {
            continue;
        }

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

        let space_parts: Vec<_> = line.split(' ').collect();
        if space_parts.len() == 2 {
            merges.push((
                decode_gpt2_token(space_parts[0], &byte_decoder)?,
                decode_gpt2_token(space_parts[1], &byte_decoder)?,
            ));
        }
    }
    Ok(merges)
}

fn decode_gpt2_token(
    token: &str,
    byte_decoder: &std::collections::HashMap<char, u8>,
) -> Result<Vec<u8>> {
    token
        .chars()
        .map(|character| {
            byte_decoder
                .get(&character)
                .copied()
                .ok_or_else(|| anyhow::anyhow!("unknown GPT-2 byte character {character:?}"))
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use std::fs;

    use tempfile::tempdir;

    use super::load_merges;

    #[test]
    fn loads_enhanced_merges_text() {
        let dir = tempdir().unwrap();
        let path = dir.path().join("merges.txt");
        fs::write(
            &path,
            "# cs336_basics enhanced BPE merges v1\n# rank\tleft_repr\tright_repr\tmerged_repr\n0\tb'a'\tb'b'\tb'ab'\n",
        )
        .unwrap();
        assert_eq!(
            load_merges(&path).unwrap(),
            vec![(b"a".to_vec(), b"b".to_vec())]
        );
    }
}
