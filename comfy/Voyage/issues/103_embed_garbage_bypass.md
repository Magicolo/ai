# 103 — `_embed_texts` sanitization sits outside the VoyageError guard: worker garbage bypasses the embed fallback, NaN poisons novelty

**Severity:** MEDIUM

**File:line (verified live 2026-09-30):**
- `voyage/supervisor.py:801-819` (`_embed_texts`: `try/except VoyageError` covers only the RPC at lines 807-810; the cleaning loop at lines 811-818 is outside)
- `voyage/concepts.py:270-297` (`check_novel`: vector path → `_max_cosine_to_accepted`)
- `voyage/concepts.py:77-101` (`_max_cosine_to_accepted`: `np.max` over similarities, `max(0.0, …)` floor at line 101)
- `voyage/concepts.py:65-74` (`cosine_similarity`: NaN in → NaN out, no guard)
- `voyage/errors.py:7-25` (taxonomy: `VoyageError` → `WorkerError` → `Recoverable`/`Fatal`; raw `ValueError`/`TypeError` are outside it)
- `voyage/supervisor.py:666-690` (belt-and-braces `except Exception`: non-`VoyageError` rests the run `FAILED` and re-raises `FatalWorkerError` — sweep said `:659-690`, drifted)

**Description:**
```python
# supervisor.py:801-819
def _embed_texts(self, texts: list[str]) -> list[list[float]] | None:
    try:
        result = self._director.call("embed", {"texts": texts}, timeout=EMBED_TIMEOUT_SECONDS)
    except VoyageError:
        return None
    vectors = result.get("vectors")
    if not isinstance(vectors, list):
        return None
    cleaned: list[list[float]] = []
    for row in vectors:
        if not isinstance(row, list):
            return None
        cleaned.append([float(value) for value in row])   # <-- outside the try
    return cleaned
```

The `try` ends at line 810. Everything after it — the shape checks (fine) and the `[float(value) for value in row]` coercion (not fine) — runs unguarded. A compromised or buggy director worker returning `{"vectors": [["abc"]]}` (or `[None]`, nested dicts, or any non-numeric junk — the RPC boundary validates the *envelope* via `decode_response`, not the *result payload*, which is `dict[str, Any]` by design: `rpc.py:270-272`) raises raw `ValueError`/`TypeError` from `float()`. Only `VoyageError` is caught, so the embed fallback (return `None` → token-set similarity in `check_novel`) is skipped and the raw exception propagates out of the commit into the belt-and-braces handler (lines 666-690): run rests `FAILED`, wrapped as `FatalWorkerError`. A degraded-embedding situation the code explicitly designed for ("a wedged director degrades to the token-set fallback instead of stalling the commit", lines 795-799) instead kills the run — and worse, it kills it as *fatal* (no restart), when the correct disposition for a bad embed response is *fallback*, several rungs below even *recoverable*.

The second half is quieter: `float("nan")` and `float("inf")` *succeed* in the cleaning loop, so NaN vectors pass into the novelty math. `cosine_similarity` (`concepts.py:65-74`) has no NaN guard (NaN in → NaN out); on the vectorized path, NaN similarities flow into `np.max` (line 101 area), and NaN never compares equal or greater sanely — the resulting novelty score is arbitrary (in practice it collapses toward the `0.0` floor, i.e. *always-novel*, silently defeating the dedup gate of DESIGN §21). A worker that emits one NaN embedding per response permanently disables novelty detection with no error anywhere.

**Rationale:**
Boundary validation is otherwise the tree's religion (issue 036's typed JSONL boundary, `checked_request`, per-op validators on every worker). `_embed_texts` is the one place where untrusted worker output is coerced with a bare `float()` outside any guard — a three-line scope error with run-fatal consequences. The NaN half matters because embeddings are the *authority* path for novelty (vectors preferred over token-set whenever present): poisoned authority silently outranks the healthy fallback.

**Live evidence (current tree, host stdlib — the exact coercion primitive):**
```
$ python3 -c "<[float(v) for v in row] over hostile rows>"
garbage-str -> RAW ValueError (escapes _embed_texts: only VoyageError is caught)
none        -> RAW TypeError  (escapes _embed_texts: only VoyageError is caught)
nan-str     -> cleaned: [nan]   <- passes cleaning, poisons downstream math
ok          -> cleaned: [0.5]
max(0.0, nan): 0.0 | nan==nan: False  <- NaN novelty scores never compare sanely
```
Code-shape proof (live-read, no interpretation needed): the `try:` at line 807 pairs with `except VoyageError:` at 809-810; lines 811-819 are at function-body indent, outside the `try` block. `float()` raises `ValueError` on bad strings and `TypeError` on `None`/dicts — neither subclasses `VoyageError` (`errors.py:7`). The commit-path consequence is mechanical: any non-`VoyageError` from `_propose_segment` reaches `supervisor.py:659` (`except Exception`) → `FAILED` + `FatalWorkerError`.

**Repro (CPU, no GPU):**
1. Point the director worker (or a test double honoring the JSONL envelope) at an `embed` op returning `{"vectors": [["not-a-float"]]}` with `ok: true` — envelope-valid, payload-hostile.
2. Commit a segment: observe raw `ValueError` out of `_embed_texts`, run resting `FAILED` (fatal, no retry), instead of the documented token-set fallback.
3. NaN variant: return `{"vectors": [[0.1, float("nan")]]}` across several commits with near-duplicate concepts — observe every proposal scoring ~0.0 similarity (always accepted as novel) while the token-set fallback would have rejected them. (NumPy leg: assert in-container that `np.max` propagates NaN through `_max_cosine_to_accepted`; NumPy is container-only so this half needs `docker run --rm -v "$PWD:/app" -w /app voyage:latest python3`.)

**Fix candidates:**
- Move the cleaning loop inside the `try` (return `None` on `ValueError`/`TypeError`), i.e. extend lines 807-810 to cover 811-819: any uncoercible payload degrades to the token-set fallback, matching the docstring.
- Add a per-component `math.isfinite` check in the cleaning loop (mirroring `beat._require_finite`, the in-tree precedent): non-finite component → `return None` (fallback), never a poisoned vector. Optionally log an `embed_garbage` metric event so a persistently hostile worker is visible instead of silently bypassed.
- Belt-and-braces alternative: catch `ValueError`/`TypeError` narrowly around the coercion only (not a broadened `except Exception`, which would mask programming errors elsewhere in the method).
- Tests: hostile-payload matrix (`["abc"]`, `[None]`, `[{}]`, `["nan"]`, `[1e999]` → inf, ragged rows, non-list `vectors`) asserting `None` (fallback) in every case; NaN-vector novelty test asserting fallback scores, not `0.0`-floor accepts.

**Refs:**
- In-tree: `voyage/supervisor.py:801-819` (the method), `voyage/rpc.py:270-272` ("`payload` stays `dict[str, Any]` … not JSON-shaped" — why the result payload is untrusted by construction), `voyage/errors.py:1-48` (class-based branching the raw exceptions defeat), `voyage/concepts.py:270-297` + `:77-101` (novelty math with no NaN guard), `voyage/audio/beat.py:24-33` (`_require_finite` precedent).
- Python semantics (verified live above): `float("abc")` → `ValueError`, `float(None)` → `TypeError`, `float("nan")` → `nan` (no error); NaN comparisons are always `False`, so `max()`/sorting-based novelty over NaN is arbitrary.

## Progress log

- 2026-09-30: re-verified live in-container before touching anything — holds as-read: `voyage/supervisor.py:1012-1030` (`try` ends at the `except VoyageError` for the RPC; shape checks + `[float(value) for value in row]` at `:1029` run unguarded at function-body indent); `concepts.py:65-74` (`cosine_similarity`, no NaN guard) and `:101` (`max(0.0, …)` floor) as cited; `errors.py:7` taxonomy confirmed (`ValueError`/`TypeError` outside `VoyageError`). Docstring still promises the token-set fallback.
- 2026-09-30 (TDD red): wrote `Voyage/tests/test_media_robustness_rank2.py` first; the five 103 tests failed as required (`ValueError`/`TypeError` escaping on garbage/`None`/dict; `[nan]`/`[inf]` passing through as poisoned vectors).
- 2026-09-30 (implement, `voyage/supervisor.py::_embed_texts` only — belt-and-braces alternative, narrow scope): coercion loop wrapped in `try/except (ValueError, TypeError) → None` (RPC `except VoyageError` untouched — no broadened `except Exception`); per-component `math.isfinite` check (mirrors `beat._require_finite`) → `None` on any non-finite value; happy path byte-identical. Note: the owned-files directive for this batch names `media.py`/`cli.py`, but this fix is unimplementable there — `_embed_texts` lives in `supervisor.py`, so the minimal two-hunk change (`import math` + method body) lands here with unique anchors; the concurrent issue-139 hunk (~line 751) is adjacent-but-untouched.
- 2026-09-30 (TDD green + gates): 18 passed in the new file; related suites 93 passed + 1 torch-gated skip; `ruff check` + `ruff format --check` + `mypy` (strict) clean on all touched files.

## Resolution

- Verdict: fixed in scope (method-level). Any uncoercible payload (`"abc"`, `None`, `{}`, ragged rows) and any non-finite payload (`"nan"`, `"inf"`, `1e999`) now degrades to the documented token-set fallback instead of resting the run `FAILED` (fatal) or silently defeating the §21 dedup gate.
- Files changed: `Voyage/voyage/supervisor.py` (`import math` + `_embed_texts`), `Voyage/tests/test_media_robustness_rank2.py` (new: 5 hostile-payload tests + happy-path pin).
- Test evidence: `test_embed_garbage_string/none_component/dict_component_degrades_to_fallback` (were raw raises, now `None`), `test_embed_nan/inf_degrades_to_fallback` (were `[nan]`/`[inf]`, now `None`), `test_embed_healthy_vectors_still_pass_through` (`[[0.5, -0.25]]` round-trips).
- DESIGN.md as-built proposal (not applied — DESIGN.md untouched per directive): in §21 (novelty/embeds), add "untrusted director `embed` vectors are coerced inside the `VoyageError` guard: uncoercible (`ValueError`/`TypeError`) or non-finite components degrade to the token-set fallback (optionally with an `embed_garbage` metric event for visibility)."
- Residuals: no `embed_garbage` metric event added (candidate mentions it as optional — visibility vs log-spam tradeoff belongs to the observability track); `cosine_similarity`/`_max_cosine_to_accepted` carry no NaN guard of their own (defense stays at the boundary — any future vector producer must go through the same finite check).
