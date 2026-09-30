# 127 — Director `handle_init` silently ignores unknown fields: a typo'd model id boots the wrong weights

- **Severity:** Low (operator footgun — misconfiguration succeeds with defaults instead of failing)
- **File:line:** `Voyage/voyage/workers/director.py:43-52` (`INIT_STR_KEYS`), `:457-470` (`handle_init`: validates *present known* keys, ignores everything else)
- **Area:** workers-internals tail — director worker init (below pass-1 coverage)

## Description

`handle_init` iterates `INIT_STR_KEYS` and records the ones present:

```python
for key in INIT_STR_KEYS:
    if key in payload and not isinstance(payload[key], str):
        raise TypeError(...)
_CONFIG.update({key: payload[key] for key in INIT_STR_KEYS if key in payload})
```

Any key *not* in the tuple is never looked at. So `init {"model_is": "/models/qwen-custom", "backemd": "qwen"}` returns `{"status": "READY", "backend": "qwen"}` and the worker proceeds with the **default** model id and backend — the operator's pinned weights silently unused. Concretely dangerous pairs: `model_id` vs `model`/`modelid`/`qwen_model_id`; `inspector_model_id` vs `inspector_model`; `models_dir` vs `model_dir`/`models_path` (the last one also defeats the /models snapshot resolution added 2026-09-29, since `_models_dir()` falls back to env/`/models`).

The strictness precedent exists in-tree: `checked_request` fails fast on missing/mistyped fields precisely so "a mistyped `init` fails as INVALID_PAYLOAD (fatal) instead of misdirecting every later `decide`" (the function's own docstring, `:458-464`). Unknown-key silence defeats that exact goal through the complementary hole — present-but-misspelled instead of absent.

## Rationale

Director weights decide every prompt on the voyage. A silent fallback to defaults means an operator who *thinks* they pinned a custom model (or an offline snapshot path) runs the default hub id instead — and `_resolve_model_source` may then *download* weights mid-commit on a box the operator believed was offline-capable. One set-membership check converts this to a loud config error at startup, the cheapest place in the system.

## Evidence (verified live 2026-09-30, tree read + logic trace)

- `director.py:466-469`: the loop is over `INIT_STR_KEYS` membership in `payload`, never over `payload` membership in `INIT_STR_KEYS` — unknown keys cannot raise by construction.
- `director.py:470`: returns READY unconditionally after the partial update.
- Contrast `handle_decide → checked_request(payload, decision_index=int, phase=str)` (`:417`), which *does* reject unknown shapes at the next op — init is the odd one out.

## Repro

```bash
# No GPU needed: import the module's init path with stubbed deps is heavy;
# the logic repro is the dict comprehension itself:
python3 -c "
INIT_STR_KEYS = ('backend','model_id','embedding_model_id','inspector_model_id','models_dir')
payload = {'model_is': '/models/custom', 'backemd': 'qwen'}
print({k: payload[k] for k in INIT_STR_KEYS if k in payload})  # {} — everything dropped, READY anyway
"
```

## Fix candidates

1. Reject unknown keys: `unknown = set(payload) - set(INIT_STR_KEYS)` → `TypeError` (INVALID_PAYLOAD, fatal) naming the key and the known set (typo-visible error message).
2. Same treatment for the ACE/SFX workers' `handle_init` (`audio_acestep.py:79-97`, `sfx_mmaudio.py:77-101` — identical record-if-known pattern) — one shared `strict_init_update` helper (084's `_validators.py` is the natural home).
3. Test: `init` with `{"model_is": ...}` → `TypeError`; `init` with each documented key → still accepted.

## Refs

 - `Voyage/voyage/workers/director.py:43-52,100-105,457-470`; `Voyage/voyage/workers/audio_acestep.py:79-97`; `Voyage/voyage/workers/sfx_mmaudio.py:77-101`.
 - Adjacent, not overlapping: 075-era reload-on-id-change (resident-stack correctness — this file is *which id gets recorded*); 084 (structural collapse — the shared-helper home, not this behavior).

## Progress log

- 2026-09-30 (Group E2): evaluated live first. Premise CONFIRMED as-read for all three workers: `director.py:591-595` iterates `INIT_STR_KEYS` membership in payload (never the reverse — unknown keys cannot raise by construction, READY returned unconditionally); `audio_acestep.py` and `sfx_mmaudio.py` had the identical record-if-known pattern. Per the task brief ("if the fix fits your worker files do it, else residual"): the two Group E2 workers are fixed here; `director.py` is out of scope (owned by another track) and logged as residual. TDD: `tests/test_e2_worker_init_strict_127.py` — 2 failed pre-fix, 5/5 green post-fix (incl. globals-restore fixture so sibling tests keep defaults).

## Resolution

- Verdict: FIXED in the two Group E2 workers; director leg RESIDUAL (below).
- Changes: `_INIT_STR_KEYS` tuples + unknown-key `TypeError` (naming the key and the known set) in `voyage/workers/sfx_mmaudio.py` (`models_dir/device/model_size`) and `voyage/workers/audio_acestep.py` (`models_dir/device`). ACE validation runs BEFORE `_redirect_upstream_writes`, so a typo fails with zero side effects (CWD untouched — pinned by test).
- Files changed: `voyage/workers/sfx_mmaudio.py`, `voyage/workers/audio_acestep.py` (+ new `tests/test_e2_worker_init_strict_127.py`). Existing `test_audio_workers`, `test_sfx_contract`, and the concurrent `test_audio_acestep_cwd` all green (the ACE reorder is side-effect compatible).
- Test evidence (in-container `voyage:latest`, CPU-only): new file 5 passed. Ruff + format + mypy strict clean.
- DESIGN proposal (quoted text only, for the DESIGN owner — worker discipline §12): "Every worker `init` rejects unknown fields as `TypeError` (INVALID_PAYLOAD, fatal) naming the key and the known set — a mistyped model id fails at startup instead of booting defaults."
 - Residuals (out of scope, precise): `voyage/workers/director.py:45-52` (`INIT_STR_KEYS`) + `:591-595` (`handle_init` loop) need the identical `unknown = set(payload) - set(INIT_STR_KEYS)` guard — the dangerous pairs from the filing (`model_id` vs `model`/`modelid`/`qwen_model_id`, `models_dir` vs `model_dir`/`models_path` defeating the 2026-09-29 /models snapshot resolution) are still silent there. Suggested home per fix candidate 2 (`_validators.py` shared `strict_init_update`) belongs to the 084 owner; the per-worker pattern landed here ports verbatim.

## Progress log (continued)

- 2026-09-30 (video-workers track): re-verified the director leg live first — premise STILL HOLDS: `voyage/workers/director.py:591-594` iterates `INIT_STR_KEYS` membership in payload and returns READY unconditionally (no unknown-key guard), while the `sfx_mmaudio.py:149-151` / `audio_acestep.py:174-176` guards from the Group E2 fix are intact (the mirror pattern ports verbatim). Disjointness check at runtime: `git diff HEAD -- Voyage/voyage/workers/director.py` is EMPTY (no overlapping hunks — the file is clean; concurrent dirt is in `voyage/supervisor.py`, `voyage/config.py`, `tests/test_stage_a_telemetry.py`, none of which this leg touches).

## Resolution (continued)

- Verdict: DEFERRED — file left untouched (no edit to `voyage/workers/director.py` in this track).
- Reason: this track's owned scope is `voyage/workers/video_{longlive,causvid,ltxv}.py` + new tests only, with `voyage/workers/director.py` explicitly listed as read-only (AVOID: concurrent-dirty, do not edit). The disjointness gate passes (empty diff), so the deferral is scope-based, not conflict-based — the owning track can apply the patch below with no merge risk.
- Files changed: none (this issue file only).
- Test evidence: live reads 2026-09-30 (cites above: `:591-594` loop shape, `:595` unconditional READY, empty-diff disjointness proof); no test added here — the pins from fix candidate 3 (`init` with `{"model_is": ...}` → `TypeError`; each documented key still accepted) belong to the director owner with the patch.
- Patch proposal (quoted text only, for the director owner — mirrors the landed `sfx_mmaudio`/`audio_acestep` guards verbatim): "In `handle_init`, after the per-key type loop and before `_CONFIG.update`, insert `unknown = sorted(set(payload) - set(INIT_STR_KEYS))` followed by `if unknown: raise TypeError(f\"init got unknown field(s) {unknown} (known: {sorted(INIT_STR_KEYS)})\")`."
- Residuals (director owner, precise): `voyage/workers/director.py:45-52` (`INIT_STR_KEYS`) + `:591-595` (`handle_init`) — apply the quoted patch + add the candidate-3 tests; shared-helper home (`_validators.py strict_init_update`, fix candidate 2) still belongs to the 084 owner.
