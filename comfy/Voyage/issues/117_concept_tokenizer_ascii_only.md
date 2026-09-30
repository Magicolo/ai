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
