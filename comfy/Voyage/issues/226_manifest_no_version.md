# 226 — Run manifest has no schema/format version; `paths.SCHEMA_VERSION = 1` is dead (MEDIUM)

## Technical description
`build_manifest` dumps `config.model_dump()` +
`segments/final_video/skip_bad/no_sfx/final_coverage` with no
`schema_version`/`format` key. `read_effective_config` detects exactly two
legacy shapes (nested `effective_config`, pre-multiplier augment keys) with
bespoke `StateError`s, then `ProjectConfig.model_validate`.
`paths.SCHEMA_VERSION = 1` is defined but imported nowhere (only
`METRICS_SCHEMA_VERSION` in `logrotate.py` is live). Verified live
2026-10-07: `grep -rn SCHEMA_VERSION voyage/ tests/ scripts/` (excluding
METRICS + pycache) hits only the definition at `paths.py:20`.

## Rationale
Without a version field every breaking change needs another content-sniff
carve-out; downgrades misread silently (extra keys ignored by pydantic
default). Chrome/MDN, winget, Slack, and JSON-migration guides all mandate a
required `manifest_version`/`schemaVersion` for exactly this reason.

## Live evidence
```
SCHEMA_VERSION refs (excluding METRICS): voyage/paths.py:20 only
manifest = config.model_dump(mode="json")   # voyage/persistence.py:34
manifest["segments"]=...; manifest["final_video"]=...  # :45-48, no version key
```

## Repro
Hand-add `"schema_version": 999` to a manifest → `read_effective_config`
succeeds (extra ignored); remove `interp_backend` from augment → silently
backfilled to `"film"` (`persistence.py:134-140`) with no version to
distinguish intentional-v1 from corrupt-v2.

## Source refs
- `Voyage/voyage/paths.py:20`
- `Voyage/voyage/persistence.py:18-49,103-146`

## Online sources
- https://developer.chrome.com/docs/extensions/reference/manifest/manifest-version
- https://developer.mozilla.org/en-US/docs/Mozilla/Add-ons/WebExtensions/manifest.json/manifest_version
- https://jsonic.io/guides/json-migrations ("numeric schemaVersion field
  embedded in each document")

## Fix candidates
1. Emit `manifest["schema_version"]=SCHEMA_VERSION`;
   `read_manifest`/`read_effective_config` fail loud on unknown-future (and
   document v1→v2 migration functions); wire `SCHEMA_VERSION` into
   `build_manifest` + tests.
2. Or delete the constant if versioning is truly unwanted (don't leave a
   dead promise).

## Log
- Track B sweep, 2026-10-07. Dead-constant verified live by orchestrator.
  Read-only; nothing fixed.

## Progress (2026-10-07, group O)

- `voyage/persistence.py`: `build_manifest` emits
  `manifest["schema_version"] = paths.SCHEMA_VERSION`; `read_manifest`
  gates it — missing key reads as legacy v1 (the only format ever
  shipped), non-integer fails loud, integer newer than supported fails
  loud with the upgrade hint. `read_effective_config` inherits the gate
  (it reads through `read_manifest`); `record_final_coverage` preserves
  the key (dict mutation, no rebuild).
- `voyage/paths.py`: `SCHEMA_VERSION` gains the ownership comment
  (emit in `build_manifest`, gate in `read_manifest`, bump with a
  migration note on breaking changes).
- New `tests/test_issue_226_manifest_version.py` (6 tests: stamp equals
  the constant, stamped round-trip, future version fails loud on both
  readers, missing key reads as legacy, non-integer fails loud,
  freshness-stamp rewrite preserves the key).
- Verified in-container: `ruff check` clean, `ruff format --check`
  clean, `mypy` strict clean on `persistence.py` + `paths.py`, scoped
  pytest 6/6 green; manifest neighbors (`test_manifest_no_stale_fields`,
  `test_final_coverage`, `test_configure`, `test_generation_stack`,
  `test_definition_tiers`) green.

## Resolution (2026-10-07)

FIXED (candidate 1). The dead constant is now live: every newly
configured run is stamped, downgrades/future files fail loud instead of
misreading, and legacy key-less manifests keep loading. Left open:
nothing — v1→v2 migration functions are deferred to the change that
first needs them.

## Evaluation (2026-10-07, group O)

Re-verified live; all claims hold:
- `paths.SCHEMA_VERSION = 1` (`voyage/paths.py:20`) is still imported
  nowhere (grep excluding `METRICS_SCHEMA_VERSION` hits only the
  definition).
- `build_manifest` (`voyage/persistence.py:18-49`) still emits no version
  key; `read_effective_config` (`:103-146`) still sniffs exactly two
  legacy shapes with bespoke `StateError`s.
- Extra-key tolerance confirmed: manifests already carry `segments`,
  `final_video`, `skip_bad`, `no_sfx`, `final_coverage` alongside the
  config root and validate cleanly, so a new top-level `schema_version`
  key is inert for `ProjectConfig.model_validate`.
- `tests/test_manifest_no_stale_fields.py` (4 tests) asserts only
  absences inside `director`/`video`/`audio` sections — a new top-level
  key cannot break it. Fix direction: candidate 1 (emit + fail loud on
  unknown-future, missing key = legacy accept so every existing
  hand-made test manifest keeps reading).
