use std::cmp::Ordering;
use std::collections::{BinaryHeap, HashMap};

use super::state::{Count, TokenBytes, TokenPair};

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct HeapEntry {
    pub count: Count,
    pub left_bytes: TokenBytes,
    pub right_bytes: TokenBytes,
    pub pair: TokenPair,
}

impl Ord for HeapEntry {
    fn cmp(&self, other: &Self) -> Ordering {
        self.count
            .cmp(&other.count)
            .then_with(|| self.left_bytes.cmp(&other.left_bytes))
            .then_with(|| self.right_bytes.cmp(&other.right_bytes))
            .then_with(|| self.pair.cmp(&other.pair))
    }
}

impl PartialOrd for HeapEntry {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

pub fn push_pair(
    heap: &mut BinaryHeap<HeapEntry>,
    pair_counts: &HashMap<TokenPair, Count>,
    id_to_bytes: &[TokenBytes],
    pair: TokenPair,
) {
    let count = pair_counts.get(&pair).copied().unwrap_or(0);
    if count == 0 {
        return;
    }
    heap.push(HeapEntry {
        count,
        left_bytes: id_to_bytes[pair.0 as usize].clone(),
        right_bytes: id_to_bytes[pair.1 as usize].clone(),
        pair,
    });
}

pub fn rebuild_heap(
    pair_counts: &HashMap<TokenPair, Count>,
    id_to_bytes: &[TokenBytes],
) -> BinaryHeap<HeapEntry> {
    let mut heap = BinaryHeap::new();
    for &pair in pair_counts.keys() {
        push_pair(&mut heap, pair_counts, id_to_bytes, pair);
    }
    heap
}

pub fn pop_best_pair(
    heap: &mut BinaryHeap<HeapEntry>,
    pair_counts: &HashMap<TokenPair, Count>,
) -> Option<TokenPair> {
    while let Some(entry) = heap.pop() {
        if pair_counts.get(&entry.pair) == Some(&entry.count) {
            return Some(entry.pair);
        }
    }
    None
}

#[cfg(test)]
mod tests {
    use std::collections::HashMap;
    use std::sync::Arc;

    use super::{pop_best_pair, push_pair, rebuild_heap};

    fn token_bytes(values: &[&[u8]]) -> Vec<Arc<[u8]>> {
        values
            .iter()
            .map(|value| Arc::<[u8]>::from(*value))
            .collect()
    }

    #[test]
    fn tie_breaks_by_larger_underlying_bytes() {
        let id_to_bytes = token_bytes(&[b"a", b"b", b"c"]);
        let pair_counts = HashMap::from([((0, 1), 3), ((0, 2), 3)]);
        let mut heap = rebuild_heap(&pair_counts, &id_to_bytes);
        assert_eq!(pop_best_pair(&mut heap, &pair_counts), Some((0, 2)));
    }

    #[test]
    fn discards_stale_heap_entries() {
        let id_to_bytes = token_bytes(&[b"a", b"b", b"c"]);
        let mut old_counts = HashMap::from([((0, 1), 10), ((1, 2), 2)]);
        let mut heap = rebuild_heap(&old_counts, &id_to_bytes);
        old_counts.insert((0, 1), 1);
        push_pair(&mut heap, &old_counts, &id_to_bytes, (0, 1));
        assert_eq!(pop_best_pair(&mut heap, &old_counts), Some((1, 2)));
    }
}
