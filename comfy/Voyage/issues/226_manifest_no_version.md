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
