"""Issue 117: concept tokenizer must not collapse non-Latin scripts.

`tokenize` kept only `[a-z0-9]+` runs, so every CJK/Thai/Arabic concept
tokenized to the empty set, `canonicalize` stored `""`, and
`token_set_similarity` scored unrelated non-Latin concepts 1.0 (duplicate).
Pure logic — CPU-only, no supervisor needed.
"""

from __future__ import annotations

from voyage.concepts import canonicalize, token_set_similarity, tokenize


def test_cjk_tokenizes_to_nonempty() -> None:
    tokens = tokenize("霧の港 ガラスの砂漠 港")
    assert tokens != frozenset()


def test_distinct_cjk_concepts_are_not_duplicates() -> None:
    assert token_set_similarity("霧の港", "ガラスの砂漠") < 0.85


def test_cjk_canonicalize_is_nonempty() -> None:
    assert canonicalize("霧の港") != ""


def test_accented_latin_keeps_whole_words() -> None:
    tokens = tokenize("café naïve façade")
    assert "café" in tokens
    assert "naïve" in tokens
    assert "façade" in tokens


def test_ascii_behavior_unchanged() -> None:
    assert tokenize("Misty Glass Harbor 123") == frozenset({"misty", "glass", "harbor", "123"})
    assert token_set_similarity("", "") == 1.0
    assert token_set_similarity("", "neon reef") == 0.0
    assert token_set_similarity("Neon Reef", "reef neon") == 1.0
