use std::collections::{BinaryHeap, HashMap, HashSet};
use std::sync::Arc;

pub type TokenId = u32;
pub type WordId = usize;
pub type Count = u64;
pub type TokenPair = (TokenId, TokenId);
pub type BytePair = (Vec<u8>, Vec<u8>);
pub type TokenBytes = Arc<[u8]>;

use super::heap::HeapEntry;

#[derive(Debug)]
pub struct TrainerState {
    pub id_to_bytes: Vec<TokenBytes>,
    pub words: Vec<Vec<TokenId>>,
    pub word_counts: Vec<Count>,
    pub pair_counts: HashMap<TokenPair, Count>,
    pub pair_to_word_ids: HashMap<TokenPair, HashSet<WordId>>,
    pub heap: BinaryHeap<HeapEntry>,
    pub merges: Vec<BytePair>,
}
