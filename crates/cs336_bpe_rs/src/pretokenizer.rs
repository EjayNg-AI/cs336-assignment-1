use std::collections::HashMap;
use std::sync::LazyLock;

use anyhow::Result;
use fancy_regex::Regex;

use crate::trainer::state::Count;

pub const PAT: &str = r#"'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"#;

static PRETOKEN_RE: LazyLock<Regex> =
    LazyLock::new(|| Regex::new(PAT).expect("GPT-style pretokenizer pattern compiles"));

pub fn sorted_special_tokens(special_tokens: &[String]) -> Vec<String> {
    let mut sorted = special_tokens.to_vec();
    sorted.sort_by(|left, right| right.len().cmp(&left.len()));
    sorted
}

pub fn find_next_special(
    text: &str,
    start: usize,
    sorted_special_tokens: &[String],
) -> Option<(usize, usize, String)> {
    let mut best: Option<(usize, usize, String)> = None;
    for token in sorted_special_tokens {
        if token.is_empty() {
            continue;
        }
        if let Some(relative_start) = text[start..].find(token) {
            let absolute_start = start + relative_start;
            let absolute_end = absolute_start + token.len();
            match &best {
                None => best = Some((absolute_start, absolute_end, token.clone())),
                Some((best_start, _, _)) if absolute_start < *best_start => {
                    best = Some((absolute_start, absolute_end, token.clone()));
                }
                _ => {}
            }
        }
    }
    best
}

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

pub fn pretoken_byte_strings(text: &str) -> Result<Vec<Vec<u8>>> {
    let mut pretokens = Vec::new();
    for match_result in PRETOKEN_RE.find_iter(text) {
        let matched = match_result?;
        pretokens.push(matched.as_str().as_bytes().to_vec());
    }
    Ok(pretokens)
}

pub fn pretoken_spans(text: &str) -> Result<Vec<(usize, usize)>> {
    let mut spans = Vec::new();
    for match_result in PRETOKEN_RE.find_iter(text) {
        let matched = match_result?;
        spans.push((matched.start(), matched.end()));
    }
    Ok(spans)
}

fn count_segment(segment: &str, counts: &mut HashMap<Vec<u8>, Count>) -> Result<()> {
    for match_result in PRETOKEN_RE.find_iter(segment) {
        let matched = match_result?;
        *counts
            .entry(matched.as_str().as_bytes().to_vec())
            .or_insert(0) += 1;
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::{pretoken_byte_counts, pretoken_byte_strings};

    #[test]
    fn pretokenizes_gpt_style_segments() {
        let tokens = pretoken_byte_strings("Hello, how are you?\n\n").unwrap();
        assert_eq!(
            tokens,
            vec![
                b"Hello".to_vec(),
                b",".to_vec(),
                b" how".to_vec(),
                b" are".to_vec(),
                b" you".to_vec(),
                b"?".to_vec(),
                b"\n\n".to_vec(),
            ]
        );
    }

    #[test]
    fn splits_around_special_tokens_before_counting() {
        let counts = pretoken_byte_counts(
            "hello<|endoftext|>world<|endoftext|><|endoftext|>",
            &["<|endoftext|>".to_string()],
        )
        .unwrap();
        assert_eq!(counts.get(&b"hello".to_vec()), Some(&1));
        assert_eq!(counts.get(&b"world".to_vec()), Some(&1));
        assert!(!counts.contains_key(&b"<|endoftext|>".to_vec()));
    }
}
