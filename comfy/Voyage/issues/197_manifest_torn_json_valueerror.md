# 197 — Manifest JSON reads are unguarded: a torn `manifest.json` crashes ensure/repair as `ValueError`, not `OSError`

- Severity: LOW
- Area: model registry / ensure — exception-type gap
- Files (as-read 2026-09-30):
  - `voyage/models_ensure.py:150-168` (`_repair_manifest`: `json.loads` inside `except OSError`)
  - `voyage/model_registry.py:393-404` (`_merge_manifest_record`: bare `json.loads`, no handler at all)

## Description

Both manifest read sites assume valid JSON. `_repair_manifest` reads inside a
`try` that catches only `OSError`:

```python
# models_ensure.py:150-168 (as-read)
with _MANIFEST_LOCK:
    for entry in entries:
        try:
            …
            if manifest_path.is_file():
                raw: Any = json.loads(manifest_path.read_text(encoding="utf-8"))
                …
        except OSError:
            continue
```

A torn `manifest.json` (killed parallel download — the exact race the repair
pass exists to heal, `models_ensure.py:16-21`; torn write from
`_merge_manifest_record`'s own non-atomic `write_text`, `model_registry.py:403`)
raises `json.JSONDecodeError`, a `ValueError` subclass — not `OSError` — so it
propagates out of the repair, out of `ensure_models`, and fails `generate`
*after* every weight verified and downloaded. `_merge_manifest_record` itself
(`:393-404`) does the same bare `json.loads`, so the download path crashes
identically on a torn manifest instead of starting from `{}`. The failure is
loud but misplaced: the operator gets a traceback about JSON in a step whose
job is healing manifest damage, and a retry re-downloads nothing (weights
verify) yet crashes in the same place until the file is hand-deleted.

## Rationale

A repair pass must be total over the damage it repairs — torn JSON is the most
likely manifest damage under the documented parallel-write race. The fix is a
two-line `except (OSError, ValueError)` (repair: treat torn as missing and
re-merge) plus a guard in `_merge_manifest_record` (torn → start from `{}` or
back up aside). The durability rule (§12: atomic writes + fsync for critical
state) would remove the tear class, but the read guard is needed regardless —
readers must never trust a file writers can tear.

## Live evidence (verified live 2026-09-30, host reads)

- `sed -n '150,168p' voyage/models_ensure.py` — `json.loads` at line 157,
  handler at 167-168 catches `OSError` only; `JSONDecodeError` subclasses
  `ValueError` (stdlib), disjoint from `OSError`.
- `sed -n '393,404p' voyage/model_registry.py` — `loaded = json.loads(…)` with
  no `try` at all; torn file crashes `download_model` before any fetch.
- `python3 -c "import json; json.loads('{torn')"` → `json.decoder.JSONDecodeError:
  Expecting property name…` (ValueError family — propagates through both sites).

## Repro

1. `echo '{torn' > <models>/manifest.json` (simulates a kill -9 mid-merge).
2. `ensure_models(config, …)` with all weights present → verifies pass, then
   `_repair_manifest` raises `JSONDecodeError` instead of returning 0.
3. Same file through `download_model(…)` → raises before fetching.

## Fix candidates

1. (Preferred) `_repair_manifest`: `except (OSError, ValueError): continue`
   (torn entry re-merges from stats, same as missing); `_merge_manifest_record`:
   torn → `record = {}` (or rename aside to `manifest.json.torn.<ts>` for
   forensics, then start empty).
2. Promote the write to atomic (`temp + fsync + os.replace`, the §12 rule —
   `voyage/atomic.py` already exists) so tears stop occurring; keep the read
   guard for crash-window files.
3. Tests: torn manifest + present weights → ensure returns 0 with a healed
   manifest; torn manifest + download → downloads and heals.

## Refs

- In-tree: `voyage/models_ensure.py:16-21,142-168,188-233`;
  `voyage/model_registry.py:393-404,1153-1157`; `voyage/atomic.py` (the write
  discipline to adopt); `DESIGN §85` (manifest merge contract).
- Not-a-duplicate: 077 is the *silent* hole — `except OSError: continue`
  blessing hash-less state and reporting "models ready" (a soundness gap where
  repair succeeds too quietly). This file is the *loud* hole — the same handler
  missing `ValueError`, so repair crashes on the torn input it was built to
  heal (a completeness gap where repair fails too noisily). Fixing 077's
  fail-loud rule does not add the missing exception class; fixing this class
  does not add 077's outcome verification. Land both.

## Progress log — 2026-09-30 (this pass, models_ensure owner)

- Re-verified live: the `models_ensure.py` half NO LONGER HOLDS —
  already fixed (077-era rework). `_read_manifest_keys`
  (`models_ensure.py:154-172`) catches `(OSError, ValueError)` and
  returns `None` for torn; `_repair_manifest` (`:175-228`) treats
  `None` as unrepairable and returns `still_missing` (clean fail-loud
  via "manifest record missing after repair", never a traceback);
  `test_containers_rank2.py:181-188`
  (`test_repair_treats_torn_manifest_as_unrepairable`) already pins it.
  No `json.loads` inside a bare-`OSError` handler remains in this
  module (single `json.loads` at `:167`, guarded). Per contract, no
  fix invented for an already-fixed premise.
- The `_merge_manifest_record` half (`model_registry.py:465-476`) still
  HOLDS — bare `json.loads` with no handler, torn crashes
  `download_model` before any fetch — but `model_registry.py` is
  outside this group's file scope (AVOID list), so NOT touched here.
- Characterization tests added (prove the owned half stays loud-clean):
  `tests/test_issue197_torn_manifest.py` 2/2 green pre- and
  post-pass (torn to `None`, torn to `still_missing` — no raise).
- Files changed: `Voyage/tests/test_issue197_torn_manifest.py` (new,
  2 tests); `Voyage/voyage/models_ensure.py` untouched.

## Resolution — 2026-09-30 (this pass)

- Verdict: ALREADY-FIXED (owned half) + DEFERRED (registry half).
  Owned `models_ensure.py` maps torn to the loud clean error the issue
  asks for (`still_missing` to `ensure_models` return 1 with console
  errors, contrasted with 077's silent-repair half which stays resolved
  and pinned by `test_containers_rank2`).
- Test evidence: new 2 characterization tests green (would have failed
  red on the as-filed code, pass on live code — premise drift proven);
  related `test_containers_rank2` repair tests green.
- Residual with handoff: `Voyage/voyage/model_registry.py:465-476`
  `_merge_manifest_record` needs `except (OSError, ValueError)` to
  `record = {}` (or torn-aside + start empty) + the atomic-write
  promotion (`voyage/atomic.py`) per fix candidates 1-2 — owner:
  registry track. Repro: `echo '{torn' > <models>/manifest.json`
  then `download_model(...)` raises `JSONDecodeError` before fetching.
