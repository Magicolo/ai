# 117 — Concept tokenizer is ASCII-only: non-Latin concepts collapse to empty and false-match as duplicates

- **Severity:** Medium (novelty policy — silent false-reject of entire language families)
- **File:line:** `Voyage/voyage/concepts.py:43-47` (`_WORD = re.compile(r"[a-z0-9]+")`, `tokenize`), `:60-62` (`canonicalize`), `:50-57` (`token_set_similarity`)
- **Area:** workers-internals tail — `voyage/concepts.py` similarity/novelty math (below pass-1 coverage, which handled vectorization/dangling-rows per the 000_INDEX)

## Description

`tokenize` keeps only `[a-z0-9]+` runs after `.lower()`. Consequences, all verified live (evidence below):

1. Any concept written in a non-Latin script (CJK, Thai, Arabic, …) tokenizes to the **empty set**; `canonicalize` then returns `""`, so every such record is stored under the identical `canonical_name == ""`.
2. `token_set_similarity("", "")` returns `1.0` (both-empty branch, `:53-54`), so the **second and every later non-Latin concept scores max similarity against the first and is rejected as a duplicate** (`check_novel`, `:270-297`, token fallback path `:294-297`), regardless of meaning.
3. Accented Latin text is silently shredded: `"café naïve façade"` → `{'caf', 'fa', 'na', 've', 'ade'}` — word fragments that then participate in Jaccard scoring as if they were words, depressing match quality for French/German/Spanish concepts.

The embedding path (`check_novel` with a vector) is unaffected — MiniLM handles Unicode — but the token-set fallback (runs with no embedding backend, plus `canonicalize` used for every stored record name) is the broken one, and there is no test pinning any non-ASCII behavior.

## Rationale

The novelty gate is a *policy* control: a false accept lets a repeat through (cosmetic), but a false reject kills a director proposal the bounded accept loop (DESIGN §74) must then work around, burning Qwen decisions. Collapsing all CJK input to a single canonical form is not a gradual quality loss — it is a total novelty blackout for those scripts from the second concept on. The regex also contradicts `canonicalize`'s own contract ("Canonical concept text for embedding (§21.2 step 1)") since the canonical form can be empty for non-empty input.

## Evidence (verified live 2026-09-30, `voyage:latest` container, `PYTHONPATH=/app/Voyage`)

```
$ docker run --rm -v "$PWD:/app" -w /app -e PYTHONPATH=/app/Voyage voyage:latest python3 -c "
from voyage.concepts import tokenize, canonicalize, token_set_similarity
..."
ascii: ['123', 'desert', 'glass', 'harbor', 'misty']
cjk: []
accents: ['ade', 'caf', 'fa', 'na', 've']
canon cjk: ''
sim cjk-vs-cjk: 1.0
```

- `tokenize('霧の港 ガラスの砂漠 港')` → `[]`; `canonicalize(...)` → `''`.
- `token_set_similarity('霧の港', 'ガラスの砂漠')` → `1.0` (two unrelated concepts, max score).
- `tokenize('café naïve façade')` → fragments, not words.

## Repro

```bash
docker run --rm -v "$PWD:/app" -w /app -e PYTHONPATH=/app/Voyage voyage:latest python3 -c "
from voyage.concepts import ConceptStore, token_set_similarity
from pathlib import Path; import tempfile
store = ConceptStore(Path(tempfile.mkdtemp()))
print(store.propose('霧の港')[0].accepted)   # True (first one lands)
print(store.propose('ガラスの砂漠')[1])      # (record, 1.0) — rejected as duplicate
print(token_set_similarity('霧の港', 'ガラスの砂漠'))  # 1.0
"
```

## Fix candidates

1. Unicode-aware word pattern: `re.compile(r"[^\W_]+", re.UNICODE)` (letters + digits in any script) instead of `[a-z0-9]+`; keep `.lower()` (Unicode-aware in Python).
2. Belt-and-braces: make the both-empty branch return `0.0` unless both inputs are literally empty strings after stripping (an empty *token set from non-empty text* must never score 1.0).
3. Add property + example tests: `tokenize` never drops a non-empty alphabetic script to empty; `token_set_similarity` of two distinct CJK strings `< threshold`; accented Latin round-trips whole words.
4. Note in DESIGN §21 that the token fallback is script-sensitive and the embedding path is authoritative for non-Latin text.

## Refs

- `Voyage/voyage/concepts.py:43-62` (tokenizer + canonicalize + Jaccard); `:270-297` (`check_novel` fallback); DESIGN §§21, 74.
- Adjacent, not overlapping: 103 (`_embed_texts` garbage bypass — embedding side); 059 (dangling vector rows — storage side). Neither touches the tokenizer regex.

## Progress log

- 2026-09-30 (Group B): re-verified live first (`voyage:latest`, CPU-only):
  `tokenize('霧の港 ガラスの砂漠 港')` → `[]`, `canonicalize` → `''`,
  `token_set_similarity` → `1.0`, accents → fragments. Premise confirmed.
- TDD: wrote `tests/test_concepts_unicode_117.py` first — 4 failed / 1 passed
  (ascii pin green) before the fix.
- Fix: `_WORD` `[a-z0-9]+` → `[^\W_]+` (letters + digits, any script;
  `_` stays a separator as before) with a why-comment at
  `voyage/concepts.py:43-50`. Deliberately did NOT touch the both-empty
  branch: the reflexive property (`test_similarity_is_reflexive`, any drawn
  text scores 1.0 against itself) plus the `("", "") == 1.0` pin would break
  for tokenless inputs, and with the Unicode pattern the branch is
  unreachable for real concepts (any letter/digit in any script yields
  tokens) — the tokenizer was the whole defect.
- Gates (in-container): new tests + `test_similarity_properties` +
  `test_concept_integrity` + `test_concepts_pruning` +
  `test_issue_027_concepts_perf` = 34 passed; `ruff check` + `ruff format
  --check` + `mypy voyage/concepts.py` clean.

## Resolution

- Verdict: FIXED. Files changed: `voyage/concepts.py` (one-line pattern +
  comment), `tests/test_concepts_unicode_117.py` (new: CJK nonempty,
  distinct-CJK `< 0.85`, CJK canonicalize nonempty, accented whole words,
  ASCII pins).
- DESIGN proposal (quoted text only, for the DESIGN owner): in §21, after
  the token-fallback description, add: "The token-set fallback is
  script-aware (`[^\W_]+` word runs): non-Latin concepts tokenize to
  non-empty sets and score genuine Jaccard values. The embedding path
  remains authoritative for non-Latin text — the fallback only decides when
  no embedding backend is available."
- Residuals: none. Candidate 2 (both-empty → 0.0) explicitly rejected —
  breaks the reflexive property pin; candidate 4 partially adopted as the
  DESIGN note above.
