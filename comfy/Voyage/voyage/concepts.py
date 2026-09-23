"""Concept store + novelty policy (DESIGN §§21, 74; task group G).

Persistent memory lives in the run directory:

- `concepts.jsonl` — append-only concept records (accepted and rejected);
- `concept_vectors.npy` — embedding rows aligned with `embedding_index`;
- `concept_index.json` — concept id → embedding row.

The novelty system is independent of the director: it embeds the
canonical concept text, compares cosine similarity against accepted
history, and rejects above the configured threshold. Embeddings come
from the director worker's `embed` op; the store itself only does
deterministic policy math. A token-set similarity fallback covers runs
where no embedding backend is available.
"""

from __future__ import annotations

import json
import math
import re
from datetime import datetime, timezone
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


def canonicalize(text: str) -> str:
    """Canonical concept text for embedding (§21.2 step 1)."""
    return " ".join(sorted(tokenize(text)))


def cosine_similarity(a: list[float], b: list[float]) -> float:
    """Cosine similarity over plain float lists (no tensor deps here)."""
    if len(a) != len(b) or not a:
        return 0.0
    dot = sum(x * y for x, y in zip(a, b, strict=True))
    norm_a = math.sqrt(sum(x * x for x in a))
    norm_b = math.sqrt(sum(y * y for y in b))
    if norm_a == 0.0 or norm_b == 0.0:
        return 0.0
    return dot / (norm_a * norm_b)


class ConceptRecord(BaseModel):
    id: str
    canonical_name: str
    summary: str = ""
    embedding_index: int = -1
    first_segment: int = 0
    created_at: str = ""
    accepted: bool = True


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")  # noqa: UP017 — worker image is py3.10, datetime.UTC needs 3.11+


class ConceptStore:
    """Append-only concept history. Rejections are recorded, never deleted."""

    def __init__(
        self,
        directory: Path,
        similarity_threshold: float = 0.85,
        legacy_path: Path | None = None,
    ) -> None:
        self._directory = directory
        self._threshold = similarity_threshold
        self._concepts_path = directory / "concepts.jsonl"
        self._vectors_path = directory / "concept_vectors.npy"
        self._index_path = directory / "concept_index.json"
        self._records: list[ConceptRecord] = []
        if legacy_path is not None and legacy_path.exists() and not self._concepts_path.exists():
            self._migrate_legacy(legacy_path)
        if self._concepts_path.exists():
            for line in self._concepts_path.read_text(encoding="utf-8").splitlines():
                if line.strip():
                    self._records.append(ConceptRecord.model_validate_json(line))

    def __len__(self) -> int:
        return len(self._records)

    def records(self) -> list[ConceptRecord]:
        """All records (accepted + rejected), oldest first."""
        return list(self._records)

    def history_texts(self) -> list[str]:
        return [record.canonical_name for record in self._records if record.accepted]

    def _migrate_legacy(self, legacy_path: Path) -> None:
        """Adopt Phase 0/2 token-set history into §21 records (no vectors)."""
        directory = self._directory
        directory.mkdir(parents=True, exist_ok=True)
        records: list[ConceptRecord] = []
        for line in legacy_path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            old = json.loads(line)
            records.append(
                ConceptRecord(
                    id=f"concept-{len(records):06d}",
                    canonical_name=canonicalize(str(old.get("text", ""))),
                    embedding_index=-1,
                    accepted=bool(old.get("accepted", True)),
                    created_at=_utc_now(),
                )
            )
        with self._concepts_path.open("w", encoding="utf-8") as handle:
            for record in records:
                handle.write(record.model_dump_json() + "\n")

    def _load_vectors(self) -> list[list[float]]:
        if not self._vectors_path.exists():
            return []
        import numpy as np

        matrix = np.load(str(self._vectors_path))
        return [[float(value) for value in row] for row in matrix.tolist()]

    def _append_vector(self, vector: list[float]) -> int:
        import numpy as np

        if self._vectors_path.exists():
            matrix = np.load(str(self._vectors_path))
            rows = matrix.tolist()
            rows.append([float(value) for value in vector])
            stacked = np.asarray(rows, dtype=np.float32)
        else:
            stacked = np.asarray([[float(value) for value in vector]], dtype=np.float32)
        self._vectors_path.parent.mkdir(parents=True, exist_ok=True)
        np.save(str(self._vectors_path), stacked)
        return int(stacked.shape[0]) - 1

    def check_novel(self, text: str, vector: list[float] | None = None) -> tuple[bool, float]:
        """Return (accepted, max_similarity_to_accepted_history).

        Cosine against stored vectors when an embedding is supplied;
        token-set fallback otherwise.
        """
        canonical = canonicalize(text)
        if vector is not None:
            stored = self._load_vectors()
            best = 0.0
            for record in self._records:
                if not record.accepted or record.embedding_index < 0:
                    continue
                if record.embedding_index < len(stored):
                    best = max(best, cosine_similarity(vector, stored[record.embedding_index]))
            return (best < self._threshold, best)
        best = 0.0
        for known in self.history_texts():
            best = max(best, token_set_similarity(canonical, canonicalize(known)))
        return (best < self._threshold, best)

    def append(
        self,
        text: str,
        accepted: bool,
        summary: str = "",
        vector: list[float] | None = None,
        segment: int = 0,
    ) -> ConceptRecord:
        embedding_index = self._append_vector(vector) if vector is not None else -1
        record = ConceptRecord(
            id=f"concept-{len(self._records):06d}",
            canonical_name=canonicalize(text),
            summary=summary,
            embedding_index=embedding_index,
            first_segment=segment,
            created_at=_utc_now(),
            accepted=accepted,
        )
        self._directory.mkdir(parents=True, exist_ok=True)
        with self._concepts_path.open("a", encoding="utf-8") as handle:
            handle.write(record.model_dump_json() + "\n")
        self._records.append(record)
        self._write_index()
        return record

    def _write_index(self) -> None:
        index = {record.id: record.embedding_index for record in self._records if record.accepted}
        self._index_path.write_text(json.dumps(index, indent=2), encoding="utf-8")

    def propose(
        self, text: str, summary: str = "", vector: list[float] | None = None, segment: int = 0
    ) -> tuple[ConceptRecord, float]:
        accepted, score = self.check_novel(text, vector)
        return self.append(text, accepted, summary, vector, segment), score

    @staticmethod
    def load_jsonl(path: Path) -> list[dict[str, object]]:
        if not path.exists():
            return []
        return [
            json.loads(line)
            for line in path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
