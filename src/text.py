"""Tokenization and vocabulary handling for COCO captions."""

from __future__ import annotations

import json
import re
from collections import Counter
from pathlib import Path
from typing import Iterable, Sequence

PAD_TOKEN = "<pad>"
BOS_TOKEN = "<bos>"
EOS_TOKEN = "<eos>"
UNK_TOKEN = "<unk>"
SPECIAL_TOKENS = [PAD_TOKEN, BOS_TOKEN, EOS_TOKEN, UNK_TOKEN]
PAD_ID, BOS_ID, EOS_ID, UNK_ID = 0, 1, 2, 3

_TOKEN_RE = re.compile(r"[a-z0-9]+(?:'[a-z]+)?")


def tokenize(text: str) -> list[str]:
    """Lowercase and split a caption into word/number tokens.

    Punctuation is discarded. Possessives are normalized (``dog's`` -> ``dog``)
    and common contractions are kept as a single token (``don't``) so the
    vocabulary stays compact and stable.
    """
    text = text.lower().replace("\u2019", "'")
    tokens: list[str] = []
    for match in _TOKEN_RE.finditer(text):
        token = match.group(0)
        if token.endswith("'s"):
            token = token[:-2]
        if token:
            tokens.append(token)
    return tokens


def detokenize(tokens: Sequence[str]) -> str:
    """Join tokens into a readable sentence (light punctuation cleanup)."""
    text = " ".join(tokens).strip()
    text = re.sub(r"\s+([,.;:!?])", r"\1", text)
    if text:
        text = text[0].upper() + text[1:]
    return text


class Vocab:
    """Integer vocabulary with reserved special tokens."""

    def __init__(
        self,
        itos: Sequence[str],
        min_freq: int = 1,
        max_len: int = 22,
    ) -> None:
        if list(itos[: len(SPECIAL_TOKENS)]) != SPECIAL_TOKENS:
            raise ValueError("Vocabulary must start with the reserved special tokens")
        self.itos = list(itos)
        self.stoi = {token: idx for idx, token in enumerate(self.itos)}
        self.min_freq = int(min_freq)
        self.max_len = int(max_len)

    def __len__(self) -> int:
        return len(self.itos)

    @property
    def pad_id(self) -> int:
        return PAD_ID

    @property
    def bos_id(self) -> int:
        return BOS_ID

    @property
    def eos_id(self) -> int:
        return EOS_ID

    @property
    def unk_id(self) -> int:
        return UNK_ID

    def encode(self, text: str, max_len: int | None = None) -> list[int]:
        """Encode text as ``<bos> ... <eos>``, truncated to ``max_len``."""
        limit = int(max_len if max_len is not None else self.max_len)
        if limit < 2:
            raise ValueError("max_len must be at least 2 (BOS + EOS)")
        ids = [self.stoi.get(token, UNK_ID) for token in tokenize(text)]
        ids = ids[: limit - 2]
        return [BOS_ID, *ids, EOS_ID]

    def decode(self, ids: Iterable[int], skip_special: bool = True) -> str:
        """Decode ids, optionally omitting special tokens."""
        tokens: list[str] = []
        specials = set(SPECIAL_TOKENS)
        for idx in ids:
            idx = int(idx)
            if idx < 0 or idx >= len(self.itos):
                continue
            token = self.itos[idx]
            if skip_special and token in specials:
                continue
            tokens.append(token)
        return " ".join(tokens)

    def to_dict(self) -> dict:
        return {
            "itos": self.itos,
            "min_freq": self.min_freq,
            "max_len": self.max_len,
        }

    @classmethod
    def from_dict(cls, payload: dict) -> "Vocab":
        return cls(payload["itos"], payload.get("min_freq", 1), payload.get("max_len", 22))

    def save(self, path: str | Path) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8") as handle:
            json.dump(self.to_dict(), handle, indent=2)
            handle.write("\n")
        return path

    @classmethod
    def load(cls, path: str | Path) -> "Vocab":
        with Path(path).open("r", encoding="utf-8") as handle:
            return cls.from_dict(json.load(handle))

    @classmethod
    def build(
        cls,
        captions: Iterable[str],
        min_freq: int = 3,
        max_len: int = 22,
        max_size: int | None = None,
    ) -> "Vocab":
        """Build a vocabulary from captions, keeping frequent words."""
        counter: Counter[str] = Counter()
        for caption in captions:
            counter.update(tokenize(caption))
        words = [(word, count) for word, count in counter.items() if count >= min_freq]
        words.sort(key=lambda item: (-item[1], item[0]))
        if max_size is not None:
            words = words[:max_size]
        itos = [*SPECIAL_TOKENS, *(word for word, _ in words)]
        return cls(itos, min_freq=min_freq, max_len=max_len)

    def coverage(self, captions: Iterable[str]) -> dict:
        """Report token coverage and unknown-token rate for diagnostics."""
        total = unknown = 0
        for caption in captions:
            for token in tokenize(caption):
                total += 1
                unknown += int(token not in self.stoi)
        return {
            "tokens": total,
            "unknown": unknown,
            "unknown_rate": (unknown / total) if total else 0.0,
            "size": len(self),
        }


def caption_length_stats(captions: Iterable[str]) -> dict:
    lengths = [len(tokenize(caption)) for caption in captions]
    if not lengths:
        return {"count": 0}
    array = sorted(lengths)
    return {
        "count": len(lengths),
        "mean": sum(lengths) / len(lengths),
        "min": array[0],
        "p50": array[len(array) // 2],
        "p95": array[min(len(array) - 1, int(0.95 * len(array)))],
        "max": array[-1],
    }
