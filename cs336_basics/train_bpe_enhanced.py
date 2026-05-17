from __future__ import annotations

import heapq
import math
import multiprocessing as mp
import os
from collections import Counter, defaultdict
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from typing import BinaryIO

import regex as re


PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
PRETOKEN_RE = re.compile(PAT)

Pair = tuple[bytes, bytes]
TokenPair = tuple[int, int]
BYTE_TOKENS = tuple(bytes([i]) for i in range(256))

_DEFAULT_CHUNK_BYTES = 64 * 1024 * 1024
_MIN_PARALLEL_BYTES = 16 * 1024 * 1024
_MIN_PARALLEL_WORDS = 20_000


@dataclass(frozen=True)
class _ReverseBytesPair:
    pair: Pair

    def __lt__(self, other: _ReverseBytesPair) -> bool:
        return self.pair > other.pair


def _multiprocessing_context() -> mp.context.BaseContext:
    try:
        return mp.get_context("fork")
    except ValueError:
        return mp.get_context()


def _resolve_num_workers(num_workers: int | None) -> int:
    if num_workers is None:
        return max(1, min(os.cpu_count() or 1, 8))
    if num_workers < 1:
        raise ValueError("num_workers must be at least 1")
    return num_workers


def _pretoken_counts_from_text(text: str, special_tokens: list[str]) -> Counter[bytes]:
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


def _pretoken_counts_for_range(args: tuple[str, int, int, list[str]]) -> Counter[bytes]:
    input_path, start, end, special_tokens = args
    with open(input_path, "rb") as f:
        f.seek(start)
        text = f.read(end - start).decode("utf-8")
    return _pretoken_counts_from_text(text, special_tokens)


def _find_chunk_boundaries(
    file: BinaryIO,
    desired_num_chunks: int,
    split_special_token: bytes,
) -> list[int]:
    assert isinstance(split_special_token, bytes), "Must represent special token as a bytestring"

    file.seek(0, os.SEEK_END)
    file_size = file.tell()
    file.seek(0)
    if file_size == 0:
        return [0]

    desired_num_chunks = max(1, min(desired_num_chunks, file_size))
    chunk_size = max(1, file_size // desired_num_chunks)
    chunk_boundaries = [min(i * chunk_size, file_size) for i in range(desired_num_chunks + 1)]
    chunk_boundaries[-1] = file_size

    mini_chunk_size = 4096
    for boundary_index in range(1, len(chunk_boundaries) - 1):
        initial_position = chunk_boundaries[boundary_index]
        file.seek(initial_position)
        while True:
            mini_chunk = file.read(mini_chunk_size)
            if mini_chunk == b"":
                chunk_boundaries[boundary_index] = file_size
                break

            found_at = mini_chunk.find(split_special_token)
            if found_at != -1:
                chunk_boundaries[boundary_index] = initial_position + found_at
                break
            initial_position += mini_chunk_size

    return sorted(set(chunk_boundaries))


def _chunk_ranges(
    input_path: str,
    num_workers: int,
    chunk_bytes: int | None,
    special_tokens: list[str],
) -> list[tuple[int, int]]:
    file_size = os.path.getsize(input_path)
    if file_size == 0:
        return [(0, 0)]
    if not special_tokens:
        return [(0, file_size)]

    target_chunk_bytes = chunk_bytes or _DEFAULT_CHUNK_BYTES
    if target_chunk_bytes < 1:
        raise ValueError("chunk_bytes must be at least 1")

    desired_chunks = max(num_workers, math.ceil(file_size / target_chunk_bytes))
    with open(input_path, "rb") as f:
        boundaries = _find_chunk_boundaries(f, desired_chunks, special_tokens[0].encode("utf-8"))

    ranges = [(start, end) for start, end in zip(boundaries[:-1], boundaries[1:]) if end > start]
    return ranges or [(0, file_size)]


def _pretoken_counts(
    input_path: str | os.PathLike,
    special_tokens: list[str],
    num_workers: int,
    chunk_bytes: int | None,
) -> Counter[bytes]:
    path = os.fspath(input_path)
    file_size = os.path.getsize(path)
    if num_workers == 1 or file_size < _MIN_PARALLEL_BYTES or not special_tokens:
        with open(path, encoding="utf-8") as f:
            return _pretoken_counts_from_text(f.read(), special_tokens)

    ranges = _chunk_ranges(path, num_workers, chunk_bytes, special_tokens)
    if len(ranges) == 1:
        with open(path, encoding="utf-8") as f:
            return _pretoken_counts_from_text(f.read(), special_tokens)

    counts: Counter[bytes] = Counter()
    jobs = [(path, start, end, special_tokens) for start, end in ranges]
    context = _multiprocessing_context()
    worker_count = min(num_workers, len(jobs))
    with context.Pool(processes=worker_count) as pool:
        for partial_counts in pool.imap_unordered(_pretoken_counts_for_range, jobs, chunksize=1):
            counts.update(partial_counts)
    return counts


def _word_pair_frequencies(word: Sequence[int]) -> dict[TokenPair, int]:
    frequencies: dict[TokenPair, int] = {}
    for i in range(len(word) - 1):
        pair = (word[i], word[i + 1])
        frequencies[pair] = frequencies.get(pair, 0) + 1
    return frequencies


def _merge_word(word: Sequence[int], pair: TokenPair, merged_token_id: int) -> list[int]:
    merged_word: list[int] = []
    i = 0
    while i < len(word):
        if i + 1 < len(word) and word[i] == pair[0] and word[i + 1] == pair[1]:
            merged_word.append(merged_token_id)
            i += 2
        else:
            merged_word.append(word[i])
            i += 1
    return merged_word


def _initial_pair_state_worker(
    records: list[tuple[int, list[int], int]],
) -> tuple[dict[TokenPair, int], dict[TokenPair, set[int]]]:
    pair_counts: dict[TokenPair, int] = {}
    pair_to_word_ids: dict[TokenPair, set[int]] = defaultdict(set)
    for word_id, word, word_count in records:
        for pair, frequency in _word_pair_frequencies(word).items():
            pair_counts[pair] = pair_counts.get(pair, 0) + frequency * word_count
            pair_to_word_ids[pair].add(word_id)
    return pair_counts, dict(pair_to_word_ids)


def _word_jobs(
    words: list[list[int]],
    word_counts: list[int],
    chunk_size: int,
) -> Iterable[list[tuple[int, list[int], int]]]:
    records: list[tuple[int, list[int], int]] = []
    for word_id, word in enumerate(words):
        records.append((word_id, word, word_counts[word_id]))
        if len(records) == chunk_size:
            yield records
            records = []
    if records:
        yield records


def _build_initial_pair_state(
    words: list[list[int]],
    word_counts: list[int],
    num_workers: int,
) -> tuple[dict[TokenPair, int], dict[TokenPair, set[int]]]:
    if num_workers == 1 or len(words) < _MIN_PARALLEL_WORDS:
        return _initial_pair_state_worker([(word_id, word, word_counts[word_id]) for word_id, word in enumerate(words)])

    pair_counts: dict[TokenPair, int] = defaultdict(int)
    pair_to_word_ids: dict[TokenPair, set[int]] = defaultdict(set)
    worker_count = min(num_workers, len(words))
    chunk_size = max(1, math.ceil(len(words) / (worker_count * 4)))
    context = _multiprocessing_context()

    with context.Pool(processes=worker_count) as pool:
        for local_pair_counts, local_pair_to_word_ids in pool.imap_unordered(
            _initial_pair_state_worker,
            _word_jobs(words, word_counts, chunk_size),
            chunksize=1,
        ):
            for pair, count in local_pair_counts.items():
                pair_counts[pair] += count
            for pair, word_ids in local_pair_to_word_ids.items():
                pair_to_word_ids[pair].update(word_ids)

    return dict(pair_counts), dict(pair_to_word_ids)


def train_bpe(
    input_path: str | os.PathLike,
    vocab_size: int,
    special_tokens: list[str],
    *,
    num_workers: int | None = None,
    chunk_bytes: int | None = None,
    heap_rebuild_factor: float = 3.0,
) -> tuple[dict[int, bytes], list[Pair]]:
    resolved_num_workers = _resolve_num_workers(num_workers)

    id_to_bytes: dict[int, bytes] = {i: BYTE_TOKENS[i] for i in range(256)}
    vocab_values = set(id_to_bytes.values())
    for special_token in special_tokens:
        special_bytes = special_token.encode("utf-8")
        if special_bytes not in vocab_values:
            id_to_bytes[len(id_to_bytes)] = special_bytes
            vocab_values.add(special_bytes)

    pretoken_counts = _pretoken_counts(input_path, special_tokens, resolved_num_workers, chunk_bytes)
    word_counts: list[int] = []
    words: list[list[int]] = []
    for pretoken, count in pretoken_counts.items():
        word_counts.append(count)
        words.append(list(pretoken))

    pair_counts, pair_to_word_ids = _build_initial_pair_state(words, word_counts, resolved_num_workers)
    heap: list[tuple[int, _ReverseBytesPair, TokenPair]] = []

    def push_pair(pair: TokenPair) -> None:
        count = pair_counts.get(pair, 0)
        if count > 0:
            pair_bytes = (id_to_bytes[pair[0]], id_to_bytes[pair[1]])
            heapq.heappush(heap, (-count, _ReverseBytesPair(pair_bytes), pair))

    def rebuild_heap() -> None:
        heap.clear()
        for pair in pair_counts:
            push_pair(pair)

    rebuild_heap()

    def pop_best_pair() -> TokenPair | None:
        while heap:
            neg_count, _, pair = heapq.heappop(heap)
            count = pair_counts.get(pair, 0)
            if count > 0 and count == -neg_count:
                return pair
        return None

    def maybe_rebuild_heap() -> None:
        if heap_rebuild_factor <= 0 or not pair_counts:
            return
        if len(heap) > heap_rebuild_factor * len(pair_counts):
            rebuild_heap()

    merges: list[Pair] = []
    while len(id_to_bytes) < vocab_size:
        best_pair = pop_best_pair()
        if best_pair is None:
            break

        merged_token = id_to_bytes[best_pair[0]] + id_to_bytes[best_pair[1]]
        merged_token_id = len(id_to_bytes)
        merges.append((id_to_bytes[best_pair[0]], id_to_bytes[best_pair[1]]))
        id_to_bytes[merged_token_id] = merged_token

        affected_word_ids = list(pair_to_word_ids.get(best_pair, ()))
        changed_pairs: set[TokenPair] = set()
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

            new_word = _merge_word(old_word, best_pair, merged_token_id)
            words[word_id] = new_word
            new_pairs = _word_pair_frequencies(new_word)

            for pair, frequency in new_pairs.items():
                pair_counts[pair] = pair_counts.get(pair, 0) + frequency * word_count
                pair_to_word_ids.setdefault(pair, set()).add(word_id)
                changed_pairs.add(pair)

        for pair in changed_pairs:
            push_pair(pair)
        maybe_rebuild_heap()

    return id_to_bytes, merges


train_bpe_enhanced = train_bpe

__all__ = ["Pair", "PAT", "PRETOKEN_RE", "TokenPair", "train_bpe", "train_bpe_enhanced"]
