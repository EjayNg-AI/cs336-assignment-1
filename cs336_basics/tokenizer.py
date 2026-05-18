from __future__ import annotations

import ast
import json
import pickle
from collections.abc import Iterable, Iterator
from pathlib import Path

import regex as re


PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
PRETOKEN_RE = re.compile(PAT)
BYTE_TOKENS = tuple(bytes([i]) for i in range(256))

Pair = tuple[bytes, bytes]
TokenSegment = tuple[int, int, str | None]


class Tokenizer:
    @staticmethod
    def _gpt2_byte_decoder() -> dict[str, int]:
        byte_values = list(range(ord("!"), ord("~") + 1))
        byte_values += list(range(ord("¡"), ord("¬") + 1))
        byte_values += list(range(ord("®"), ord("ÿ") + 1))
        code_points = byte_values[:]
        next_shifted = 0
        for byte_value in range(256):
            if byte_value not in byte_values:
                byte_values.append(byte_value)
                code_points.append(256 + next_shifted)
                next_shifted += 1
        return {chr(code_point): byte_value for byte_value, code_point in zip(byte_values, code_points)}

    @staticmethod
    def _decode_gpt2_token(token: str, byte_decoder: dict[str, int]) -> bytes:
        return bytes(byte_decoder[character] for character in token)

    @classmethod
    def _load_vocab(cls, vocab_filepath: str) -> dict[int, bytes]:
        path = Path(vocab_filepath)
        if path.suffix == ".pkl":
            with path.open("rb") as f:
                return pickle.load(f)

        with path.open(encoding="utf-8") as f:
            data = json.load(f)

        if isinstance(data, dict) and isinstance(data.get("tokens"), list):
            vocab: dict[int, bytes] = {}
            for token in data["tokens"]:
                vocab[int(token["id"])] = bytes(token["byte_values"])
            return vocab

        if isinstance(data, dict) and all(isinstance(value, int) for value in data.values()):
            byte_decoder = cls._gpt2_byte_decoder()
            return {token_id: cls._decode_gpt2_token(token, byte_decoder) for token, token_id in data.items()}

        if isinstance(data, dict):
            return {int(token_id): bytes(token_bytes) for token_id, token_bytes in data.items()}

        raise ValueError(f"Unsupported vocabulary file format: {vocab_filepath}")

    @classmethod
    def _load_merges(cls, merges_filepath: str) -> list[Pair]:
        path = Path(merges_filepath)
        if path.suffix == ".pkl":
            with path.open("rb") as f:
                return pickle.load(f)

        byte_decoder = cls._gpt2_byte_decoder()
        merges: list[Pair] = []
        with path.open(encoding="utf-8") as f:
            for line in f:
                line = line.rstrip("\n")
                if not line or line.startswith("#"):
                    continue

                parts = line.split("\t")
                if len(parts) >= 3 and parts[0].isdigit():
                    merges.append((ast.literal_eval(parts[1]), ast.literal_eval(parts[2])))
                    continue

                parts = line.split(" ")
                if len(parts) == 2:
                    merges.append(
                        (
                            cls._decode_gpt2_token(parts[0], byte_decoder),
                            cls._decode_gpt2_token(parts[1], byte_decoder),
                        )
                    )

        return merges

    def __init__(
        self,
        vocab: dict[int, bytes],
        merges: list[Pair],
        special_tokens: list[str] | None = None,
    ) -> None:
        self.vocab = dict(vocab)
        self.token_to_id = {token: token_id for token_id, token in self.vocab.items()}

        self.special_tokens = list(special_tokens) if special_tokens is not None else []
        self.special_token_ids: dict[str, int] = {}
        next_token_id = max(self.vocab, default=-1) + 1
        for special_token in self.special_tokens:
            special_bytes = special_token.encode("utf-8")
            token_id = self.token_to_id.get(special_bytes)
            if token_id is None:
                token_id = next_token_id
                next_token_id += 1
                self.vocab[token_id] = special_bytes
                self.token_to_id[special_bytes] = token_id
            self.special_token_ids[special_token] = token_id

        self.merge_ranks = {pair: rank for rank, pair in enumerate(merges)}
        self.byte_token_ids = tuple(self.token_to_id[token] for token in BYTE_TOKENS)
        self.merge_ranks_by_id: dict[tuple[int, int], int] = {}
        self.merge_output_by_pair_id: dict[tuple[int, int], int] = {}
        for pair, rank in self.merge_ranks.items():
            left, right = pair
            pair_ids = (self.token_to_id[left], self.token_to_id[right])
            self.merge_ranks_by_id[pair_ids] = rank
            self.merge_output_by_pair_id[pair_ids] = self.token_to_id[left + right]
        self._encode_cache: dict[bytes, tuple[int, ...]] = {}
        self._max_cache_size = 50_000

        self._max_special_token_length = max((len(token) for token in self.special_tokens), default=0)
        if self.special_tokens:
            special_pattern = "|".join(re.escape(token) for token in sorted(self.special_tokens, key=len, reverse=True))
            self._special_re: re.Pattern[str] | None = re.compile(special_pattern)
        else:
            self._special_re = None

    @classmethod
    def from_files(
        cls,
        vocab_filepath: str,
        merges_filepath: str,
        special_tokens: list[str] | None = None,
    ) -> Tokenizer:
        vocab = cls._load_vocab(vocab_filepath)
        merges = cls._load_merges(merges_filepath)
        return cls(vocab, merges, special_tokens)

    def encode(self, text: str) -> list[int]:
        return list(self._encode_text(text))

    def encode_iterable(self, iterable: Iterable[str]) -> Iterator[int]:
        buffer = ""
        for chunk in iterable:
            if not chunk:
                continue
            buffer += chunk
            segments = list(self._token_segments(buffer))
            flush_index = self._stream_flush_index_from_segments(buffer, segments)
            if flush_index > 0:
                yield from self._encode_prefix_with_context(buffer, flush_index, segments)
                buffer = buffer[flush_index:]
        if buffer:
            yield from self._encode_text(buffer)

    def decode(self, ids: list[int]) -> str:
        return b"".join(self.vocab[token_id] for token_id in ids).decode("utf-8", errors="replace")

    def _encode_text(self, text: str) -> Iterator[int]:
        if self._special_re is None:
            yield from self._encode_normal_text(text)
            return

        start = 0
        for match in self._special_re.finditer(text):
            if match.start() > start:
                yield from self._encode_normal_text(text[start : match.start()])
            yield self.special_token_ids[match.group(0)]
            start = match.end()
        if start < len(text):
            yield from self._encode_normal_text(text[start:])

    def _encode_normal_text(self, text: str) -> Iterator[int]:
        for match in PRETOKEN_RE.finditer(text):
            yield from self._encode_pretoken(match.group(0).encode("utf-8"))

    def _encode_prefix_with_context(
        self,
        text: str,
        end_index: int,
        segments: Iterable[TokenSegment] | None = None,
    ) -> Iterator[int]:
        if segments is None:
            segments = self._token_segments(text)

        for start, end, special_token in segments:
            if end <= end_index:
                if special_token is None:
                    yield from self._encode_pretoken(text[start:end].encode("utf-8"))
                else:
                    yield self.special_token_ids[special_token]
            elif start < end_index:
                raise ValueError("stream flush boundary split a token")
            else:
                break

    def _encode_pretoken(self, pretoken: bytes) -> tuple[int, ...]:
        cached = self._encode_cache.get(pretoken)
        if cached is not None:
            return cached

        tokens = [self.byte_token_ids[byte] for byte in pretoken]
        while len(tokens) > 1:
            best_pair: tuple[int, int] | None = None
            best_rank: int | None = None
            for i in range(len(tokens) - 1):
                pair = (tokens[i], tokens[i + 1])
                rank = self.merge_ranks_by_id.get(pair)
                if rank is not None and (best_rank is None or rank < best_rank):
                    best_pair = pair
                    best_rank = rank

            if best_pair is None:
                break

            merged_token = self.merge_output_by_pair_id[best_pair]
            merged_tokens: list[int] = []
            i = 0
            while i < len(tokens):
                if i + 1 < len(tokens) and tokens[i] == best_pair[0] and tokens[i + 1] == best_pair[1]:
                    merged_tokens.append(merged_token)
                    i += 2
                else:
                    merged_tokens.append(tokens[i])
                    i += 1
            tokens = merged_tokens

        token_ids = tuple(tokens)
        if len(self._encode_cache) >= self._max_cache_size:
            self._encode_cache.clear()
        self._encode_cache[pretoken] = token_ids
        return token_ids

    def _stream_flush_index(self, text: str) -> int:
        return self._stream_flush_index_from_segments(text, list(self._token_segments(text)))

    def _stream_flush_index_from_segments(self, text: str, segments: list[TokenSegment]) -> int:
        if not segments:
            return 0

        keep_start = segments[-1][0]
        if self._max_special_token_length > 1:
            keep_start = min(keep_start, max(0, len(text) - self._max_special_token_length + 1))

        for start, end, _ in segments:
            if start < keep_start < end:
                return start
        return keep_start

    def _token_spans(self, text: str) -> Iterator[tuple[int, int]]:
        for start, end, _ in self._token_segments(text):
            yield start, end

    def _token_segments(self, text: str) -> Iterator[TokenSegment]:
        if self._special_re is None:
            for match in PRETOKEN_RE.finditer(text):
                yield match.start(), match.end(), None
            return

        start = 0
        for special_match in self._special_re.finditer(text):
            if special_match.start() > start:
                for match in PRETOKEN_RE.finditer(text, start, special_match.start()):
                    yield match.start(), match.end(), None
            yield special_match.start(), special_match.end(), special_match.group(0)
            start = special_match.end()
        if start < len(text):
            for match in PRETOKEN_RE.finditer(text, start):
                yield match.start(), match.end(), None
