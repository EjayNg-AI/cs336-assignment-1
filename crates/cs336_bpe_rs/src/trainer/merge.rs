use super::state::{TokenId, TokenPair};

pub fn merge_word(word: &[TokenId], pair: TokenPair, merged_token_id: TokenId) -> Vec<TokenId> {
    let mut merged_word = Vec::with_capacity(word.len());
    let mut i = 0;
    while i < word.len() {
        if i + 1 < word.len() && word[i] == pair.0 && word[i + 1] == pair.1 {
            merged_word.push(merged_token_id);
            i += 2;
        } else {
            merged_word.push(word[i]);
            i += 1;
        }
    }
    merged_word
}

#[cfg(test)]
mod tests {
    use super::merge_word;

    #[test]
    fn merges_non_overlapping_left_to_right() {
        assert_eq!(merge_word(&[1, 1, 1], (1, 1), 2), vec![2, 1]);
    }
}
