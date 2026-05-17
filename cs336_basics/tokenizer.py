from __future__ import annotations

import heapq
import os
from collections import Counter, defaultdict
from dataclasses import dataclass

import regex as re


PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
PRETOKEN_RE = re.compile(PAT)

Pair = tuple[bytes, bytes]
BYTE_TOKENS = tuple(bytes([i]) for i in range(256))


@dataclass(frozen=True)
class _ReversePair:
    pair: Pair

    def __lt__(self, other: _ReversePair) -> bool:
        return self.pair > other.pair


def _pretoken_counts(text: str, special_tokens: list[str]) -> Counter[bytes]:
    counts: Counter[bytes] = Counter()
    if special_tokens:
        escaped_specials = [re.escape(token) for token in sorted(special_tokens, key=len, reverse=True)]
        chunks = re.split("|".join(escaped_specials), text)
    else:
        chunks = [text]

    for chunk in chunks:
        for match in PRETOKEN_RE.finditer(chunk):
            counts[match.group(0).encode("utf-8")] += 1
    return counts


def _word_pair_frequencies(word: list[bytes]) -> dict[Pair, int]:
    frequencies: dict[Pair, int] = {}
    for i in range(len(word) - 1):
        pair = (word[i], word[i + 1])
        frequencies[pair] = frequencies.get(pair, 0) + 1
    return frequencies


def _merge_word(word: list[bytes], pair: Pair, merged_token: bytes) -> list[bytes]:
    merged_word: list[bytes] = []
    i = 0
    while i < len(word):
        if i + 1 < len(word) and word[i] == pair[0] and word[i + 1] == pair[1]:
            merged_word.append(merged_token)
            i += 2
        else:
            merged_word.append(word[i])
            i += 1
    return merged_word


def train_bpe(
    input_path: str | os.PathLike,
    vocab_size: int,
    special_tokens: list[str],
) -> tuple[dict[int, bytes], list[Pair]]:
    with open(input_path, encoding="utf-8") as f:
        text = f.read()

    vocab: dict[int, bytes] = {i: BYTE_TOKENS[i] for i in range(256)}
    vocab_values = set(vocab.values())
    for special_token in special_tokens:
        special_bytes = special_token.encode("utf-8")
        if special_bytes not in vocab_values:
            vocab[len(vocab)] = special_bytes
            vocab_values.add(special_bytes)

    pretoken_counts = _pretoken_counts(text, special_tokens)
    word_counts: list[int] = []
    words: list[list[bytes]] = []
    for pretoken, count in pretoken_counts.items():
        word_counts.append(count)
        words.append([BYTE_TOKENS[byte] for byte in pretoken])

    pair_counts: dict[Pair, int] = defaultdict(int)
    pair_to_word_ids: dict[Pair, set[int]] = defaultdict(set)
    for word_id, word in enumerate(words):
        for pair, frequency in _word_pair_frequencies(word).items():
            pair_counts[pair] += frequency * word_counts[word_id]
            pair_to_word_ids[pair].add(word_id)

    heap: list[tuple[int, _ReversePair, Pair]] = []

    def push_pair(pair: Pair) -> None:
        count = pair_counts.get(pair, 0)
        if count > 0:
            heapq.heappush(heap, (-count, _ReversePair(pair), pair))

    for pair in pair_counts:
        push_pair(pair)

    def pop_best_pair() -> Pair | None:
        while heap:
            neg_count, _, pair = heapq.heappop(heap)
            count = pair_counts.get(pair, 0)
            if count > 0 and count == -neg_count:
                return pair
        return None

    merges: list[Pair] = []
    while len(vocab) < vocab_size:
        best_pair = pop_best_pair()
        if best_pair is None:
            break

        merged_token = best_pair[0] + best_pair[1]
        merges.append(best_pair)
        vocab[len(vocab)] = merged_token

        affected_word_ids = list(pair_to_word_ids.get(best_pair, ()))
        changed_pairs: set[Pair] = set()
        for word_id in affected_word_ids:
            word_count = word_counts[word_id]
            old_word = words[word_id]
            old_pairs = _word_pair_frequencies(old_word)

            for pair, frequency in old_pairs.items():
                next_count = pair_counts[pair] - frequency * word_count
                if next_count > 0:
                    pair_counts[pair] = next_count
                else:
                    del pair_counts[pair]

                word_ids = pair_to_word_ids.get(pair)
                if word_ids is not None:
                    word_ids.discard(word_id)
                    if not word_ids:
                        del pair_to_word_ids[pair]
                changed_pairs.add(pair)

            new_word = _merge_word(old_word, best_pair, merged_token)
            words[word_id] = new_word
            new_pairs = _word_pair_frequencies(new_word)

            for pair, frequency in new_pairs.items():
                pair_counts[pair] += frequency * word_count
                pair_to_word_ids[pair].add(word_id)
                changed_pairs.add(pair)

        for pair in changed_pairs:
            push_pair(pair)

    return vocab, merges
