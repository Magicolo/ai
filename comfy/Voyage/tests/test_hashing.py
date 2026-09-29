"""Shared hashing helpers (issue 021).

CPU-only, stdlib only: known vectors, chunked-vs-oneshot equivalence, and
delegation of the four non-supervisor call sites. `supervisor.py` keeps its
own copy — it belongs to another track (noted, not touched).
"""

from __future__ import annotations

import hashlib
from pathlib import Path

import pytest

from voyage import hashing, media, model_registry
from voyage.hashing import sha256_file, sha256_text
from voyage.workers import video_causvid, video_ltxv


def test_sha256_file_matches_hashlib(tmp_path: Path) -> None:
    target = tmp_path / "weights.bin"
    target.write_bytes(b"voyage-weights-bytes" * 4096)
    assert sha256_file(target) == hashlib.sha256(target.read_bytes()).hexdigest()


def test_sha256_file_empty(tmp_path: Path) -> None:
    target = tmp_path / "empty.bin"
    target.write_bytes(b"")
    assert sha256_file(target) == hashlib.sha256(b"").hexdigest()


def test_sha256_file_streams_in_chunks(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Multi-chunk reads hash identically (chunk size is an impl detail)."""
    monkeypatch.setattr(hashing, "CHUNK_SIZE_BYTES", 7)
    target = tmp_path / "chunked.bin"
    target.write_bytes(bytes(range(256)) * 64)
    assert sha256_file(target) == hashlib.sha256(target.read_bytes()).hexdigest()


def test_sha256_text_known_vector() -> None:
    assert sha256_text("abc") == "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad"


def test_registry_alias_delegates(tmp_path: Path) -> None:
    target = tmp_path / "model.pt"
    target.write_bytes(b"registry-bytes")
    assert model_registry._sha256(target) == sha256_file(target)


def test_media_alias_delegates(tmp_path: Path) -> None:
    target = tmp_path / "audio.wav"
    target.write_bytes(b"media-bytes")
    assert media._sha256_file(target) == sha256_file(target)


def test_ltxv_sha256_file_delegates(tmp_path: Path) -> None:
    target = tmp_path / "video_tail.mp4"
    target.write_bytes(b"tail-bytes")
    assert video_ltxv.sha256_file(target) == sha256_file(target)


def test_causvid_sha256_helpers_delegate(tmp_path: Path) -> None:
    target = tmp_path / "video_tail.mp4"
    target.write_bytes(b"causvid-tail-bytes")
    assert video_causvid.sha256_file(target) == sha256_file(target)
    assert video_causvid.sha256_text("causvid") == sha256_text("causvid")
