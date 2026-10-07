# 202 — `_merge_manifest_record` writes `models/manifest.json` non-atomically (HIGH)

## Technical description
Every run manifest/state goes through `atomic_write_json` (temp `*.partial`
+ flush + fsync + `os.replace` + fsync dir). The models-tree manifest merge
does a direct in-place write:

```python
def _merge_manifest_record(                          # voyage/model_registry.py:554-566
    manifest_path.write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")
```

vs `atomic_write_json` (`voyage/atomic.py:97-107`), which stages bytes
atomically. SIGKILL mid-write leaves torn JSON; the next
`verify_model`/`_manifest_hash_mismatches` reads it as `[]`/mismatch
(fail-open to presence-only, see #230) or crashes outright.

## Rationale
Durability rule (§12): all critical writes via temp + fsync + replace. The
models manifest is the tamper-evidence baseline — corrupting it silently
downgrades every later verify.

## Live evidence
Code-shape finding (read both functions fully):

```
def atomic_write_json(destination: Path, payload: Any) -> None:   # voyage/atomic.py:97-107
    atomic_write_bytes(destination, (json.dumps(payload, indent=2) + "\n").encode("utf-8"))
```

## Repro
Kill between `read_text` and `write_text` (or fill disk) → truncated
`models/manifest.json`; `verify_model` then returns `missing`/hash-mismatch
or `json.loads` raises raw (no taxonomy wrapper here, unlike
`persistence.read_manifest`).

## Source refs
- `Voyage/voyage/model_registry.py:554-566`
- `Voyage/voyage/atomic.py:97-107`
- `Voyage/voyage/persistence.py:52-53`

## Online sources
- Atomic-write durability pattern (temp + fsync + rename).
- Torn JSON documents need atomic replace + version field —
  https://jsonic.io/guides/json-migrations

## Fix candidates
1. Route through `atomic_write_json`; add `fsync_dir`.
2. Wrap read/parse errors as `ValueError`/`StateError` per taxonomy.
3. Test with a fault-injected partial write.

## Log
- Track B sweep, 2026-10-07. Read-only; nothing fixed.

## Evaluation (2026-10-07)
Re-verified live against the current tree before fixing — all
load-bearing claims hold, nothing stale:
- Direct `manifest_path.write_text(...)` confirmed at
  `voyage/model_registry.py:565` (only non-atomic manifest write in
  the module; `json` import retained — still used by the merge read
  and by `verify_checkpoint_against_manifest:530`).
- `atomic_write_json` stages temp `*.partial` + flush + fsync +
  `os.replace` + fsync dir, confirmed at `voyage/atomic.py:97-107`.
- Raw read/parse confirmed: `_merge_manifest_record` called
  `json.loads(manifest_path.read_text(...))` with no taxonomy
  wrapper, unlike `persistence.read_manifest` (`voyage/persistence.py:91-97`,
  `(OSError, ValueError)` → `StateError`). Adjacent same-shape raw
  read at `verify_checkpoint_against_manifest:530` noted — left
  untouched (outside this issue's file scope).
- No existing test referenced `_merge_manifest_record` (grep clean).

## Resolution (2026-10-07)
- `_merge_manifest_record` stages through `atomic_write_json`
  (`voyage/model_registry.py:554+`); unreadable manifests
  (torn bytes/hand-edit corruption via `(OSError, ValueError)`,
  non-object JSON) raise `StateError` per the error taxonomy.
  Top-level imports gained `atomic_write_json` + `StateError` only —
  no other function in `model_registry.py` touched.
- Mode composition: the shared atomic path now preserves the
  destination mode (issue 222), so merged manifests inherit it.
- New `Voyage/tests/test_models_manifest_atomic_202.py` (6 tests):
  merge round-trip + no `.partial` litter, routing through
  `atomic_write_json` (monkeypatched recording wrapper), fresh
  manifest lands 0644, torn/non-object manifests raise `StateError`
  with the previous file untouched, fault-injected crash between
  temp-stage and replace keeps the previous manifest byte-identical.
- Verify (in-container): scoped pytest 49 passed (12 new + 37
  `test_unit` incl. the existing atomic round-trip pins), `ruff
  check` + `ruff format --check` clean on all touched files, `mypy`
  strict clean on `voyage/atomic.py` + `voyage/model_registry.py`.
- Uncommitted; left for orchestrator review.
