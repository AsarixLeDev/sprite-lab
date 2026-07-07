"""Tiny deterministic tokenizer for baseline sprite training."""

from __future__ import annotations

import json
import re
from collections import Counter
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

SPECIAL_TOKENS: tuple[str, ...] = ("<pad>", "<unk>", "<bos>", "<eos>")
PAD_TOKEN = "<pad>"
UNK_TOKEN = "<unk>"
BOS_TOKEN = "<bos>"
EOS_TOKEN = "<eos>"
TOKEN_RE = re.compile(r"[a-z0-9]+")


def tokenize_text(text: str) -> list[str]:
    """Split prose and snake_case into lowercase alphanumeric tokens."""

    normalized = str(text).replace("_", " ").replace("-", " ").lower()
    return TOKEN_RE.findall(normalized)


def semantic_strings_from_record(record: Mapping[str, Any]) -> list[str]:
    """Flatten grounded semantic manifest fields into tokenizer text chunks."""

    chunks: list[str] = []
    for key in (
        "category",
        "base_object",
        "object_name",
        "colors",
        "materials",
        "effects",
        "function",
        "style",
        "caption_type",
        "caption_source",
    ):
        _extend_labeled_value(chunks, key, record.get(key))

    conditioning = record.get("conditioning") if isinstance(record.get("conditioning"), Mapping) else {}
    semantic = conditioning.get("semantic_v3") if isinstance(conditioning, Mapping) else {}
    if isinstance(semantic, Mapping):
        for key in ("base_object", "open_name"):
            value = semantic.get(key)
            if value:
                chunks.extend([key, str(value)])
        attributes = semantic.get("attributes") if isinstance(semantic.get("attributes"), Mapping) else {}
        for group in ("colors", "materials", "shapes", "effects", "state", "function"):
            chunks.append(group)
            chunks.extend(str(value) for value in attributes.get(group) or ())

    for section_name in ("kept_attributes", "dropped_attributes"):
        section = conditioning.get(section_name) if isinstance(conditioning, Mapping) else {}
        if isinstance(section, Mapping):
            for group, values in sorted(section.items()):
                chunks.extend([section_name, str(group)])
                chunks.extend(str(value) for value in values or ())

    if isinstance(conditioning, Mapping):
        chunks.append("dropout_ops")
        chunks.extend(str(value) for value in conditioning.get("dropout_ops") or ())

    target_semantics = record.get("target_semantics") if isinstance(record.get("target_semantics"), Mapping) else {}
    if isinstance(target_semantics, Mapping):
        for key in ("base_object", "open_name", "object_name"):
            _extend_labeled_value(chunks, key, target_semantics.get(key))
        attributes = target_semantics.get("attributes") if isinstance(target_semantics.get("attributes"), Mapping) else {}
        for group in ("colors", "materials", "shapes", "effects", "state", "function", "style"):
            _extend_labeled_value(chunks, group, attributes.get(group))

    negative_tags = record.get("negative_tags")
    if isinstance(negative_tags, Sequence) and not isinstance(negative_tags, str):
        chunks.append("negative_tags")
        chunks.extend(str(value) for value in negative_tags)

    return chunks


def _extend_labeled_value(chunks: list[str], label: str, value: Any) -> None:
    if value is None or value == "":
        return
    chunks.append(str(label))
    if isinstance(value, Mapping):
        for key, nested in sorted(value.items()):
            _extend_labeled_value(chunks, str(key), nested)
    elif isinstance(value, Sequence) and not isinstance(value, str):
        chunks.extend(str(item) for item in value if item is not None and item != "")
    else:
        chunks.append(str(value))


def record_texts(record: Mapping[str, Any]) -> list[str]:
    """Return all text fields used to build the baseline vocabulary."""

    caption = str(record.get("caption", ""))
    return [caption, *semantic_strings_from_record(record)]


@dataclass
class SpriteTextTokenizer:
    """Small deterministic vocabulary tokenizer with fixed-length encoding."""

    token_to_id: dict[str, int]
    max_length: int = 32

    @classmethod
    def build(
        cls,
        texts: Iterable[str],
        *,
        max_length: int = 32,
        min_freq: int = 1,
        max_vocab_size: int | None = None,
    ) -> "SpriteTextTokenizer":
        counter: Counter[str] = Counter()
        for text in texts:
            counter.update(tokenize_text(text))

        token_to_id = {token: index for index, token in enumerate(SPECIAL_TOKENS)}
        ordered = sorted(
            (token, count) for token, count in counter.items() if count >= min_freq and token not in token_to_id
        )
        ordered.sort(key=lambda item: (-item[1], item[0]))
        if max_vocab_size is not None:
            limit = max(0, int(max_vocab_size) - len(token_to_id))
            ordered = ordered[:limit]
        for token, _count in ordered:
            token_to_id[token] = len(token_to_id)
        return cls(token_to_id=token_to_id, max_length=max_length)

    @classmethod
    def build_from_records(
        cls,
        records: Iterable[Mapping[str, Any]],
        *,
        max_length: int = 32,
        min_freq: int = 1,
        max_vocab_size: int | None = None,
    ) -> "SpriteTextTokenizer":
        texts: list[str] = []
        for record in records:
            texts.extend(record_texts(record))
        return cls.build(texts, max_length=max_length, min_freq=min_freq, max_vocab_size=max_vocab_size)

    @property
    def id_to_token(self) -> dict[int, str]:
        return {index: token for token, index in self.token_to_id.items()}

    @property
    def pad_id(self) -> int:
        return self.token_to_id[PAD_TOKEN]

    @property
    def unk_id(self) -> int:
        return self.token_to_id[UNK_TOKEN]

    @property
    def bos_id(self) -> int:
        return self.token_to_id[BOS_TOKEN]

    @property
    def eos_id(self) -> int:
        return self.token_to_id[EOS_TOKEN]

    def __len__(self) -> int:
        return len(self.token_to_id)

    def encode(
        self,
        text: str | Sequence[str],
        *,
        max_length: int | None = None,
        add_special_tokens: bool = True,
    ) -> list[int]:
        if isinstance(text, str):
            tokens = tokenize_text(text)
        else:
            tokens = [token for value in text for token in tokenize_text(str(value))]
        ids = [self.token_to_id.get(token, self.unk_id) for token in tokens]
        if add_special_tokens:
            ids = [self.bos_id, *ids, self.eos_id]
        length = int(max_length or self.max_length)
        if len(ids) > length:
            ids = ids[:length]
            if add_special_tokens and length > 0:
                ids[-1] = self.eos_id
        if len(ids) < length:
            ids.extend([self.pad_id] * (length - len(ids)))
        return ids

    def encode_record_semantics(self, record: Mapping[str, Any], *, max_length: int | None = None) -> list[int]:
        return self.encode(semantic_strings_from_record(record), max_length=max_length)

    def decode(self, ids: Iterable[int], *, skip_special_tokens: bool = True) -> str:
        id_to_token = self.id_to_token
        tokens: list[str] = []
        for value in ids:
            token = id_to_token.get(int(value), UNK_TOKEN)
            if token == EOS_TOKEN:
                if not skip_special_tokens:
                    tokens.append(token)
                break
            if skip_special_tokens and token in SPECIAL_TOKENS:
                continue
            tokens.append(token)
        return " ".join(tokens)

    def to_json_dict(self) -> dict[str, Any]:
        return {
            "token_to_id": dict(sorted(self.token_to_id.items(), key=lambda item: item[1])),
            "max_length": int(self.max_length),
            "special_tokens": list(SPECIAL_TOKENS),
        }

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.to_json_dict(), indent=2, sort_keys=True) + "\n", encoding="utf-8")

    @classmethod
    def load(cls, path: str | Path) -> "SpriteTextTokenizer":
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        token_to_id = {str(token): int(index) for token, index in data["token_to_id"].items()}
        for index, token in enumerate(SPECIAL_TOKENS):
            if token_to_id.get(token) != index:
                raise ValueError(f"vocabulary special token {token!r} must have id {index}")
        return cls(token_to_id=token_to_id, max_length=int(data.get("max_length", 32)))
