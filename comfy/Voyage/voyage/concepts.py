"""Concept store + novelty check (DESIGN §§21, 74; task group G).

V1 policy: immutable append-only JSONL history, deterministic token-set
similarity fallback. A sentence-transformer embedding backend can be
plugged in later behind the same interface without changing callers.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

from pydantic import BaseModel

_WORD = re.compile(r"[a-z0-9]+")


def tokenize(text: str) -> frozenset[str]:
    return frozenset(_WORD.findall(text.lower()))


def token_set_similarity(a: str, b: str) -> float:
    tokens_a = tokenize(a)
    tokens_b = tokenize(b)
    if not tokens_a and not tokens_b:
        return 1.0
    if not tokens_a or not tokens_b:
        return 0.0
    return len(tokens_a & tokens_b) / len(tokens_a | tokens_b)


class ConceptRecord(BaseModel):
    index: int
    text: str
    accepted: bool = True


class ConceptStore:
    """Append-only concept history. Rejections are recorded, never deleted."""

    def __init__(self, path: Path, similarity_threshold: float = 0.55) -> None:
        self._path = path
        self._threshold = similarity_threshold
        self._records: list[ConceptRecord] = []
        if path.exists():
            for line in path.read_text(encoding="utf-8").splitlines():
                if line.strip():
                    self._records.append(ConceptRecord.model_validate_json(line))

    def __len__(self) -> int:
        return len(self._records)

    def history_texts(self) -> list[str]:
        return [record.text for record in self._records if record.accepted]

    def check_novel(self, text: str) -> tuple[bool, float]:
        """Return (accepted, max_similarity_to_accepted_history)."""
        best = 0.0
        for known in self.history_texts():
            score = token_set_similarity(text, known)
            best = max(best, score)
        return (best < self._threshold, best)

    def append(self, text: str, accepted: bool) -> ConceptRecord:
        record = ConceptRecord(index=len(self._records), text=text, accepted=accepted)
        self._path.parent.mkdir(parents=True, exist_ok=True)
        with self._path.open("a", encoding="utf-8") as handle:
            handle.write(record.model_dump_json() + "\n")
        self._records.append(record)
        return record

    def propose(self, text: str) -> tuple[ConceptRecord, float]:
        accepted, score = self.check_novel(text)
        return self.append(text, accepted), score

    @staticmethod
    def load_jsonl(path: Path) -> list[dict[str, object]]:
        if not path.exists():
            return []
        return [
            json.loads(line)
            for line in path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
