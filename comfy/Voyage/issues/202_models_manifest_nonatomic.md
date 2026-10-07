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
