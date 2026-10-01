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
import os
import re
import warnings
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from numpy.typing import NDArray
from pydantic import BaseModel

from voyage.atomic import JsonValue, atomic_write_bytes, atomic_write_json
from voyage.atomic import fsync_dir as fsync_dir
from voyage.errors import StateError

LEGACY_MIGRATION_REMOVE_AFTER = "2026-12-31"
"""Time-box for the Phase 0/2 `legacy_path` migration (issue 046).

The migration has served: live callers (`supervisor`, `cli`) still thread
`legacy_path` through, but no new code may depend on it. After the date
above, delete the `legacy_path` parameter, `_migrate_legacy`, and this
constant — every use site warns until then.
"""

_WORD = re.compile(r"[^\W_]+")
# Word runs in any script (issue 117): `[a-z0-9]+` collapsed every
# non-Latin concept to the empty token set, so unrelated CJK concepts scored
# similarity 1.0 (false duplicates) and accented Latin shredded into
# fragments. `[^\W_]+` keeps Unicode letters + digits in any script
# (`.lower()` is already Unicode-aware); `_` stays a separator as before.
# repairs the token-set fallback.


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


def _max_cosine_to_accepted(
    vector: list[float], matrix: NDArray[np.float32], accepted_indices: list[int]
) -> float:
    """Max cosine of `vector` against accepted matrix rows (issue 027).

    Vectorized twin of the `cosine_similarity` loop it replaces: empty
    queries, dimension mismatches, and zero-norm rows all score 0.0,
    and the floor stays 0.0 (the loop never returned a negative best).
    """
    rows = [index for index in accepted_indices if 0 <= index < matrix.shape[0]]
    if not rows or not vector:
        return 0.0
    query = np.asarray([float(value) for value in vector], dtype=np.float64)
    if query.shape[0] != matrix.shape[1]:
        return 0.0
    query_norm = float(np.linalg.norm(query))
    if query_norm == 0.0:
        return 0.0
    candidates = matrix[rows].astype(np.float64)
    row_norms = np.linalg.norm(candidates, axis=1)
    dots = candidates @ query
    with np.errstate(divide="ignore", invalid="ignore"):
        similarities = np.divide(dots, row_norms * query_norm)
    similarities[row_norms == 0.0] = 0.0
    return float(max(0.0, np.max(similarities)))


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
        # Issue 027: in-memory vector matrix (None until first read).
        # The store is the single writer per run (DESIGN §72), so the
        # cache only goes stale on our own appends, which refresh it —
        # repeated check/append cycles in one commit stop re-reading
        # the full file from disk.
        self._matrix_cache: NDArray[np.float32] | None = None
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
        """Adopt Phase 0/2 token-set history into §21 records (no vectors).

        Time-boxed (see `LEGACY_MIGRATION_REMOVE_AFTER`): warns on every
        use so remaining callers show up in logs before removal.
        """
        warnings.warn(
            "ConceptStore legacy_path migration is deprecated and will be removed after "
            f"{LEGACY_MIGRATION_REMOVE_AFTER}; stop threading legacy_path through new code",
            DeprecationWarning,
            stacklevel=3,
        )
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
        atomic_write_bytes(
            self._concepts_path,
            "".join(record.model_dump_json() + "\n" for record in records).encode("utf-8"),
        )

    def _load_matrix(self) -> NDArray[np.float32]:
        """Full vector matrix as float32, cached in memory (issue 027).

        A missing matrix reads as a (0, 0) empty — callers treat "no
        rows" uniformly instead of branching on file existence.
        """
        if self._matrix_cache is None:
            if not self._vectors_path.exists():
                self._matrix_cache = np.zeros((0, 0), dtype=np.float32)
            else:
                self._matrix_cache = np.asarray(np.load(str(self._vectors_path)), dtype=np.float32)
        return self._matrix_cache

    def _load_vectors(self) -> list[list[float]]:
        matrix = self._load_matrix()
        return [[float(value) for value in row] for row in matrix.tolist()]

    def _dangling_vector_records(self, stored_count: int) -> list[str]:
        """Accepted record ids whose embedding row is missing (issue 059).

        A deleted/truncated `concept_vectors.npy` leaves records pointing
        past the end of the matrix; without this check every duplicate
        scores 0.0 and is silently accepted as novel.
        """
        return [
            record.id
            for record in self._records
            if record.accepted
            and record.embedding_index >= 0
            and record.embedding_index >= stored_count
        ]

    def _append_vector(self, vector: list[float]) -> int:
        """Append one row atomically: save temp + fsync + rename.

        A crash mid-save must never leave a torn .npy behind — the
        previous valid matrix survives. (Single writer, DESIGN §72.)

        Issue 027: single disk load (was two `np.load` calls: one for
        the dangling guard, one for the rows) plus a C-speed
        `concatenate` (was `tolist → append → asarray`: a Python-float
        roundtrip of every stored value per segment).
        """
        if not vector:
            raise ValueError("concept vector must be non-empty")
        # Issue 059 fail-loud: never stack a fresh matrix under records
        # that reference lost rows — the indices would silently dangle.
        existing_rows = int(self._load_matrix().shape[0]) if self._vectors_path.exists() else 0
        dangling = self._dangling_vector_records(existing_rows)
        if dangling:
            raise StateError(
                f"concept vectors missing rows for records {dangling} "
                f"({existing_rows} rows stored); restore concept_vectors.npy "
                "before appending"
            )
        row = np.asarray([[float(value) for value in vector]], dtype=np.float32)
        if self._vectors_path.exists():
            stacked = np.asarray(
                np.concatenate([self._load_matrix(), row], axis=0), dtype=np.float32
            )
        else:
            stacked = row
        self._vectors_path.parent.mkdir(parents=True, exist_ok=True)
        # np.save appends .npy when missing, so the temp name keeps the suffix.
        tmp_npy = self._vectors_path.parent / f"{self._vectors_path.name}.{os.getpid()}.tmp.npy"
        try:
            np.save(str(tmp_npy), stacked)
            with tmp_npy.open("rb+") as handle:
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(tmp_npy, self._vectors_path)
            fsync_dir(self._vectors_path.parent)
        except BaseException:
            try:
                tmp_npy.unlink()
            except OSError:
                pass
            raise
        self._matrix_cache = stacked
        return int(stacked.shape[0]) - 1

    def check_novel(self, text: str, vector: list[float] | None = None) -> tuple[bool, float]:
        """Return (accepted, max_similarity_to_accepted_history).

        Cosine against stored vectors when an embedding is supplied;
        token-set fallback otherwise.
        """
        canonical = canonicalize(text)
        if vector is not None:
            matrix = self._load_matrix()
            # Issue 059 fail-loud: missing rows must abort the novelty
            # decision, never score the duplicate 0.0 (accepted as novel).
            dangling = self._dangling_vector_records(int(matrix.shape[0]))
            if dangling:
                raise StateError(
                    f"concept vectors missing rows for records {dangling} "
                    f"({int(matrix.shape[0])} rows stored); restore concept_vectors.npy "
                    "before proposing"
                )
            best = _max_cosine_to_accepted(
                vector,
                matrix,
                [record.embedding_index for record in self._records if record.accepted],
            )
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
        """Append one concept record across the three-file group (101).

        Order is vector (.npy, fully atomic) → jsonl (file fsync +
        `fsync_dir`) → index (atomic). The group is not a single atomic
        rename, so crash windows exist — every window is fail-loud via
        `validate_concepts`, which now covers both directions (missing
        vector rows *and* missing index keys). A torn group never scores
        silently; it errors on the next validate/load.
        """
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
            handle.flush()
            os.fsync(handle.fileno())
        fsync_dir(self._concepts_path.parent)
        self._records.append(record)
        self._write_index()
        return record

    def _write_index(self) -> None:
        index: dict[str, JsonValue] = {
            record.id: record.embedding_index for record in self._records if record.accepted
        }
        atomic_write_json(self._index_path, index)

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


def validate_concepts(directory: Path) -> list[str]:
    """Read-only concept-state consistency check (issues 059/101, for validate).

    Covers what checksums/numbering never did: vectors row-count vs the
    highest accepted embedding_index, index keys ⊆ accepted record ids,
    index values matching the records, *and* accepted ids ⊆ index keys
    (101: a crash between the jsonl append and `_write_index` leaves a
    record without an index entry — the reverse of the 059 dangling-row
    direction, and equally fail-loud now). A directory with none of the
    three files is a fresh run — no errors. Token-only records
    (embedding_index == -1, e.g. legacy migrations) legitimately coexist
    with vector-backed ones, so mixed -1 is not an error.
    """
    errors: list[str] = []
    concepts_path = directory / "concepts.jsonl"
    vectors_path = directory / "concept_vectors.npy"
    index_path = directory / "concept_index.json"
    if not concepts_path.exists() and not vectors_path.exists() and not index_path.exists():
        return []
    try:
        records = [
            ConceptRecord.model_validate_json(line)
            for line in concepts_path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
    except (OSError, ValueError) as exc:
        return [f"novelty/concepts.jsonl unreadable: {exc}"]
    if vectors_path.exists():
        try:
            import numpy as np

            row_count = int(np.load(str(vectors_path)).shape[0])
        except (OSError, ValueError, IndexError) as exc:
            errors.append(f"novelty/concept_vectors.npy unreadable: {exc}")
            row_count = -1
    else:
        row_count = 0
    accepted = {record.id: record.embedding_index for record in records if record.accepted}
    if row_count >= 0:
        dangling = sorted(
            record_id for record_id, index in accepted.items() if index >= 0 and index >= row_count
        )
        if dangling:
            errors.append(
                f"novelty/concept_vectors.npy has {row_count} rows "
                f"but records reference missing rows: {dangling}"
            )
    if not concepts_path.exists() and vectors_path.exists():
        errors.append("novelty/concept_vectors.npy present but concepts.jsonl missing")
    if index_path.exists():
        try:
            index = json.loads(index_path.read_text(encoding="utf-8"))
        except (OSError, ValueError) as exc:
            errors.append(f"novelty/concept_index.json unreadable: {exc}")
            index = None
        if index is not None:
            if not isinstance(index, dict):
                errors.append("novelty/concept_index.json is not an object")
            else:
                for record_id, row in index.items():
                    if record_id not in accepted:
                        errors.append(
                            f"novelty/concept_index.json references "
                            f"unknown/rejected record {record_id}"
                        )
                    elif accepted[record_id] != row:
                        errors.append(
                            f"novelty/concept_index.json row mismatch for "
                            f"{record_id}: index={row}, record={accepted[record_id]}"
                        )
                for record_id in sorted(accepted):
                    if record_id not in index:
                        errors.append(
                            f"novelty/concept_index.json missing key "
                            f"for accepted record {record_id}"
                        )
    elif accepted:
        errors.append(
            f"novelty/concept_index.json missing but {len(accepted)} accepted "
            "records exist (crash between jsonl append and index write?)"
        )
    return errors
