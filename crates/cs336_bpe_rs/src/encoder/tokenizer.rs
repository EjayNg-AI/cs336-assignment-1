use std::collections::HashMap;
use std::path::Path;

use anyhow::{anyhow, bail, Context, Error, Result};

use crate::encoder::merges::load_merges;
use crate::encoder::streaming::TokenSegment;
use crate::encoder::vocab::load_vocab;
use crate::pretokenizer::{
    find_next_special, pretoken_byte_strings, pretoken_spans, sorted_special_tokens,
};
use crate::trainer::state::{BytePair, TokenId, TokenPair};

const MAX_CACHE_SIZE: usize = 50_000;

#[derive(Debug, Clone)]
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

impl Tokenizer {
    pub fn from_files(
        vocab_filepath: impl AsRef<Path>,
        merges_filepath: impl AsRef<Path>,
        special_tokens: Vec<String>,
    ) -> Result<Self> {
        let vocab = load_vocab(vocab_filepath.as_ref())?;
        let merges = load_merges(merges_filepath.as_ref())?;
        Self::new(vocab, merges, special_tokens)
    }

    pub fn new(
        mut vocab: Vec<Vec<u8>>,
        merges: Vec<BytePair>,
        special_tokens: Vec<String>,
    ) -> Result<Self> {
        let mut token_to_id = HashMap::new();
        for (token_id, token) in vocab.iter().enumerate() {
            token_to_id.insert(token.clone(), token_id as TokenId);
        }

        let mut special_token_ids = HashMap::new();
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

        let mut byte_token_ids = [0; 256];
        for byte in 0u16..=255 {
            let token = vec![byte as u8];
            byte_token_ids[byte as usize] = *token_to_id
                .get(&token)
                .with_context(|| format!("vocabulary missing byte token {byte}"))?;
        }

        let mut merge_ranks_by_id = HashMap::new();
        let mut merge_output_by_pair_id = HashMap::new();
        for (rank, (left, right)) in merges.iter().enumerate() {
            let left_id = *token_to_id
                .get(left)
                .with_context(|| format!("merge left token missing from vocab: {left:?}"))?;
            let right_id = *token_to_id
                .get(right)
                .with_context(|| format!("merge right token missing from vocab: {right:?}"))?;
            let mut merged = left.clone();
            merged.extend(right);
            let merged_id = *token_to_id
                .get(&merged)
                .with_context(|| format!("merge output token missing from vocab: {merged:?}"))?;
            let pair = (left_id, right_id);
            merge_ranks_by_id.insert(pair, rank);
            merge_output_by_pair_id.insert(pair, merged_id);
        }

        let sorted_special_tokens = sorted_special_tokens(&special_tokens);
        let max_special_token_length = special_tokens
            .iter()
            .map(|token| token.len())
            .max()
            .unwrap_or(0);

        Ok(Self {
            vocab,
            token_to_id,
            special_tokens,
            sorted_special_tokens,
            special_token_ids,
            merge_ranks_by_id,
            merge_output_by_pair_id,
            byte_token_ids,
            encode_cache: HashMap::new(),
            max_special_token_length,
        })
    }

    pub fn encode(&mut self, text: &str) -> Result<Vec<TokenId>> {
        self.encode_text(text)
    }

    pub fn encode_iterable<I>(&mut self, iterable: I) -> Result<Vec<TokenId>>
    where
        I: IntoIterator<Item = String>,
    {
        let iterable = iterable.into_iter().map(Ok::<String, Error>);
        let mut output = Vec::new();
        self.encode_iterable_result_to_sink(iterable, |token_id| {
            output.push(token_id);
            Ok(())
        })?;
        Ok(output)
    }

    pub fn encode_iterable_result_to_sink<I, E, F>(
        &mut self,
        iterable: I,
        mut sink: F,
    ) -> Result<()>
    where
        I: IntoIterator<Item = std::result::Result<String, E>>,
        E: std::fmt::Display,
        F: FnMut(TokenId) -> Result<()>,
    {
        let mut buffer = String::new();
        for chunk in iterable {
            let chunk = chunk.map_err(|error| anyhow!("{error}"))?;
            if chunk.is_empty() {
                continue;
            }
            buffer.push_str(&chunk);
            let segments = self.token_segments(&buffer)?;
            let flush_index = self.stream_flush_index_from_segments(&buffer, &segments);
            if flush_index > 0 {
                for token_id in self.encode_prefix_with_context(&buffer, flush_index, &segments)? {
                    sink(token_id)?;
                }
                buffer = buffer[flush_index..].to_string();
            }
        }
        if !buffer.is_empty() {
            for token_id in self.encode_text(&buffer)? {
                sink(token_id)?;
            }
        }
        Ok(())
    }

    pub fn decode(&self, ids: &[TokenId]) -> Result<String> {
        let mut bytes = Vec::new();
        for &token_id in ids {
            let token = self
                .vocab
                .get(token_id as usize)
                .with_context(|| format!("unknown token id {token_id}"))?;
            bytes.extend(token);
        }
        Ok(String::from_utf8_lossy(&bytes).into_owned())
    }

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
            ids.push(
                *self
                    .special_token_ids
                    .get(&special_token)
                    .with_context(|| format!("missing special token id for {special_token}"))?,
            );
            start = special_end;
        }
        if start < text.len() {
            ids.extend(self.encode_normal_text(&text[start..])?);
        }
        Ok(ids)
    }

    fn encode_normal_text(&mut self, text: &str) -> Result<Vec<TokenId>> {
        let mut ids = Vec::new();
        for pretoken in pretoken_byte_strings(text)? {
            ids.extend(self.encode_pretoken(&pretoken)?);
        }
        Ok(ids)
    }

    pub fn encode_pretoken(&mut self, pretoken: &[u8]) -> Result<Vec<TokenId>> {
        if let Some(cached) = self.encode_cache.get(pretoken) {
            return Ok(cached.clone());
        }

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
            let merged_token = *self
                .merge_output_by_pair_id
                .get(&best_pair)
                .context("merge rank exists without merge output")?;
            let mut merged_tokens = Vec::with_capacity(tokens.len());
            let mut i = 0;
            while i < tokens.len() {
                if i + 1 < tokens.len() && tokens[i] == best_pair.0 && tokens[i + 1] == best_pair.1
                {
                    merged_tokens.push(merged_token);
                    i += 2;
                } else {
                    merged_tokens.push(tokens[i]);
                    i += 1;
                }
            }
            tokens = merged_tokens;
        }

        if self.encode_cache.len() >= MAX_CACHE_SIZE {
            self.encode_cache.clear();
        }
        self.encode_cache.insert(pretoken.to_vec(), tokens.clone());
        Ok(tokens)
    }

    fn encode_prefix_with_context(
        &mut self,
        text: &str,
        end_index: usize,
        segments: &[TokenSegment],
    ) -> Result<Vec<TokenId>> {
        let mut ids = Vec::new();
        for (start, end, special_token) in segments {
            if *end <= end_index {
                if let Some(special_token) = special_token {
                    ids.push(*self.special_token_ids.get(special_token).with_context(|| {
                        format!("missing special token id for {special_token}")
                    })?);
                } else {
                    ids.extend(self.encode_pretoken(text[*start..*end].as_bytes())?);
                }
            } else if *start < end_index {
                bail!("stream flush boundary split a token");
            } else {
                break;
            }
        }
        Ok(ids)
    }

    fn stream_flush_index_from_segments(&self, text: &str, segments: &[TokenSegment]) -> usize {
        let Some((last_start, _, _)) = segments.last() else {
            return 0;
        };

        let mut keep_start = *last_start;
        if self.max_special_token_length > 1 {
            let special_keep_start = text
                .len()
                .saturating_sub(self.max_special_token_length)
                .saturating_add(1);
            keep_start = keep_start.min(previous_char_boundary(text, special_keep_start));
        }

        for (start, end, _) in segments {
            if *start < keep_start && keep_start < *end {
                return *start;
            }
        }
        keep_start
    }

    fn token_segments(&self, text: &str) -> Result<Vec<TokenSegment>> {
        let mut segments = Vec::new();
        if self.sorted_special_tokens.is_empty() {
            for (start, end) in pretoken_spans(text)? {
                segments.push((start, end, None));
            }
            return Ok(segments);
        }

        let mut start = 0;
        while let Some((special_start, special_end, special_token)) =
            find_next_special(text, start, &self.sorted_special_tokens)
        {
            if special_start > start {
                for (relative_start, relative_end) in pretoken_spans(&text[start..special_start])? {
                    segments.push((start + relative_start, start + relative_end, None));
                }
            }
            segments.push((special_start, special_end, Some(special_token)));
            start = special_end;
        }
        if start < text.len() {
            for (relative_start, relative_end) in pretoken_spans(&text[start..])? {
                segments.push((start + relative_start, start + relative_end, None));
            }
        }
        Ok(segments)
    }

    pub fn vocab(&self) -> &[Vec<u8>] {
        &self.vocab
    }

    pub fn special_tokens(&self) -> &[String] {
        &self.special_tokens
    }

    pub fn token_to_id_len(&self) -> usize {
        self.token_to_id.len()
    }
}

fn previous_char_boundary(text: &str, mut index: usize) -> usize {
    index = index.min(text.len());
    while index > 0 && !text.is_char_boundary(index) {
        index -= 1;
    }
    index
}

#[cfg(test)]
mod tests {
    use super::Tokenizer;

    fn tiny_tokenizer() -> Tokenizer {
        let mut vocab: Vec<Vec<u8>> = (0u16..=255).map(|value| vec![value as u8]).collect();
        vocab.push(b"ab".to_vec());
        vocab.push(b"abc".to_vec());
        Tokenizer::new(
            vocab,
            vec![
                (b"a".to_vec(), b"b".to_vec()),
                (b"ab".to_vec(), b"c".to_vec()),
            ],
            vec!["<|endoftext|>".to_string()],
        )
        .unwrap()
    }

    #[test]
    fn encodes_pretoken_by_lowest_rank_merge() {
        let mut tokenizer = tiny_tokenizer();
        assert_eq!(tokenizer.encode_pretoken(b"abc").unwrap(), vec![257]);
    }

    #[test]
    fn preserves_special_tokens_with_longest_match() {
        let mut vocab: Vec<Vec<u8>> = (0u16..=255).map(|value| vec![value as u8]).collect();
        vocab.push(b"<|endoftext|>".to_vec());
        vocab.push(b"<|endoftext|><|endoftext|>".to_vec());
        let mut tokenizer = Tokenizer::new(
            vocab,
            Vec::new(),
            vec![
                "<|endoftext|>".to_string(),
                "<|endoftext|><|endoftext|>".to_string(),
            ],
        )
        .unwrap();
        let ids = tokenizer
            .encode("a<|endoftext|><|endoftext|>b<|endoftext|>")
            .unwrap();
        assert!(ids.contains(&257));
        assert!(ids.contains(&256));
    }

    #[test]
    fn decodes_with_utf8_replacement() {
        let tokenizer = tiny_tokenizer();
        assert_eq!(tokenizer.decode(&[255]).unwrap(), "\u{fffd}");
    }

    #[test]
    fn streaming_matches_whole_input() {
        let mut tokenizer = tiny_tokenizer();
        let text = "abc abc<|endoftext|>abc";
        let whole = tokenizer.encode(text).unwrap();
        for chunk_size in [1, 2, 7] {
            let chunks = text
                .as_bytes()
                .chunks(chunk_size)
                .map(|chunk| String::from_utf8(chunk.to_vec()).unwrap())
                .collect::<Vec<_>>();
            let mut streaming_tokenizer = tiny_tokenizer();
            assert_eq!(streaming_tokenizer.encode_iterable(chunks).unwrap(), whole);
        }
    }
}
