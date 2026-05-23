use std::collections::HashMap;
use std::fs;
use std::path::Path;

use anyhow::{bail, Context, Result};
use serde_json::Value;

use crate::errors::BpeError;

pub fn load_vocab(path: &Path) -> Result<Vec<Vec<u8>>> {
    if path.extension().and_then(|value| value.to_str()) == Some("pkl") {
        bail!(BpeError::UnsupportedVocabFormat(
            "pickle vocabularies are intentionally unsupported by the Rust loader".to_string()
        ));
    }

    let data: Value = serde_json::from_str(&fs::read_to_string(path)?)?;
    if let Some(tokens) = data.get("tokens").and_then(Value::as_array) {
        return load_enhanced_vocab(tokens);
    }

    let object = data
        .as_object()
        .ok_or_else(|| BpeError::UnsupportedVocabFormat(path.display().to_string()))?;

    if object.values().all(Value::is_number) {
        return load_gpt2_vocab(object);
    }

    if object.values().all(Value::is_array) {
        return load_id_to_byte_array_vocab(object);
    }

    Err(BpeError::UnsupportedVocabFormat(path.display().to_string()).into())
}

fn load_enhanced_vocab(tokens: &[Value]) -> Result<Vec<Vec<u8>>> {
    let max_id = tokens
        .iter()
        .filter_map(|token| token.get("id").and_then(Value::as_u64))
        .max()
        .unwrap_or(0) as usize;
    let mut vocab = vec![Vec::new(); max_id + 1];
    for token in tokens {
        let id = token
            .get("id")
            .and_then(Value::as_u64)
            .context("enhanced vocab token missing numeric id")? as usize;
        let byte_values = token
            .get("byte_values")
            .and_then(Value::as_array)
            .context("enhanced vocab token missing byte_values")?;
        vocab[id] = byte_values
            .iter()
            .map(|value| {
                let byte = value.as_u64().context("byte value must be an integer")?;
                if byte > 255 {
                    bail!("byte value {byte} is outside 0..=255");
                }
                Ok(byte as u8)
            })
            .collect::<Result<Vec<_>>>()?;
    }
    Ok(vocab)
}

fn load_gpt2_vocab(object: &serde_json::Map<String, Value>) -> Result<Vec<Vec<u8>>> {
    let byte_decoder = gpt2_byte_decoder();
    let max_id = object.values().filter_map(Value::as_u64).max().unwrap_or(0) as usize;
    let mut vocab = vec![Vec::new(); max_id + 1];
    for (token, id_value) in object {
        let id = id_value.as_u64().unwrap() as usize;
        let mut bytes = Vec::new();
        for character in token.chars() {
            let byte = byte_decoder
                .get(&character)
                .copied()
                .with_context(|| format!("unknown GPT-2 byte character {character:?}"))?;
            bytes.push(byte);
        }
        vocab[id] = bytes;
    }
    Ok(vocab)
}

fn load_id_to_byte_array_vocab(object: &serde_json::Map<String, Value>) -> Result<Vec<Vec<u8>>> {
    let max_id = object
        .keys()
        .filter_map(|key| key.parse::<usize>().ok())
        .max()
        .unwrap_or(0);
    let mut vocab = vec![Vec::new(); max_id + 1];
    for (id_text, bytes_value) in object {
        let id = id_text.parse::<usize>()?;
        let byte_values = bytes_value
            .as_array()
            .ok_or_else(|| BpeError::UnsupportedVocabFormat("expected byte arrays".to_string()))?;
        vocab[id] = byte_values
            .iter()
            .map(|value| {
                let byte = value.as_u64().context("byte value must be an integer")?;
                if byte > 255 {
                    bail!("byte value {byte} is outside 0..=255");
                }
                Ok(byte as u8)
            })
            .collect::<Result<Vec<_>>>()?;
    }
    Ok(vocab)
}

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

    byte_values
        .into_iter()
        .zip(code_points)
        .map(|(byte_value, code_point)| {
            (
                char::from_u32(code_point as u32).expect("valid GPT-2 byte unicode code point"),
                byte_value as u8,
            )
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use std::fs;

    use tempfile::tempdir;

    use super::load_vocab;

    #[test]
    fn loads_enhanced_vocab_json() {
        let dir = tempdir().unwrap();
        let path = dir.path().join("vocab.json");
        fs::write(
            &path,
            r#"{"format":"cs336_basics.enhanced_bpe.v1","tokens":[{"id":0,"byte_values":[97],"hex":"61","repr":"b'a'","utf8":"a"}]}"#,
        )
        .unwrap();
        assert_eq!(load_vocab(&path).unwrap(), vec![b"a".to_vec()]);
    }
}
