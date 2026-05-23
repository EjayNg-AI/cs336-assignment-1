use std::collections::{HashMap, HashSet};

use rayon::prelude::*;

use crate::config::MIN_PARALLEL_WORDS;

use super::state::{Count, TokenId, TokenPair, WordId};

pub fn word_pair_frequencies(word: &[TokenId]) -> HashMap<TokenPair, Count> {
    let mut frequencies = HashMap::new();
    for window in word.windows(2) {
        let pair = (window[0], window[1]);
        *frequencies.entry(pair).or_insert(0) += 1;
    }
    frequencies
}

pub fn build_initial_pair_state(
    words: &[Vec<TokenId>],
    word_counts: &[Count],
    num_workers: usize,
) -> (
    HashMap<TokenPair, Count>,
    HashMap<TokenPair, HashSet<WordId>>,
) {
    if num_workers == 1 || words.len() < MIN_PARALLEL_WORDS {
        return initial_pair_state_worker(words, word_counts, 0);
    }

    let worker_count = num_workers.min(words.len()).max(1);
    let chunk_size = words.len().div_ceil(worker_count * 4).max(1);
    let partials: Vec<_> = words
        .par_chunks(chunk_size)
        .zip(word_counts.par_chunks(chunk_size))
        .enumerate()
        .map(|(chunk_index, (word_chunk, count_chunk))| {
            initial_pair_state_worker(word_chunk, count_chunk, chunk_index * chunk_size)
        })
        .collect();

    let mut pair_counts = HashMap::new();
    let mut pair_to_word_ids: HashMap<TokenPair, HashSet<WordId>> = HashMap::new();
    for (local_counts, local_postings) in partials {
        for (pair, count) in local_counts {
            *pair_counts.entry(pair).or_insert(0) += count;
        }
        for (pair, word_ids) in local_postings {
            pair_to_word_ids.entry(pair).or_default().extend(word_ids);
        }
    }
    (pair_counts, pair_to_word_ids)
}

fn initial_pair_state_worker(
    words: &[Vec<TokenId>],
    word_counts: &[Count],
    base_word_id: WordId,
) -> (
    HashMap<TokenPair, Count>,
    HashMap<TokenPair, HashSet<WordId>>,
) {
    let mut pair_counts = HashMap::new();
    let mut pair_to_word_ids: HashMap<TokenPair, HashSet<WordId>> = HashMap::new();
    for (offset, word) in words.iter().enumerate() {
        let word_id = base_word_id + offset;
        let word_count = word_counts[offset];
        for (pair, frequency) in word_pair_frequencies(word) {
            *pair_counts.entry(pair).or_insert(0) += frequency * word_count;
            pair_to_word_ids.entry(pair).or_default().insert(word_id);
        }
    }
    (pair_counts, pair_to_word_ids)
}

#[cfg(test)]
mod tests {
    use super::word_pair_frequencies;

    #[test]
    fn counts_word_pair_frequencies() {
        let frequencies = word_pair_frequencies(&[1, 2, 1, 2, 2]);
        assert_eq!(frequencies[&(1, 2)], 2);
        assert_eq!(frequencies[&(2, 1)], 1);
        assert_eq!(frequencies[&(2, 2)], 1);
    }
}
