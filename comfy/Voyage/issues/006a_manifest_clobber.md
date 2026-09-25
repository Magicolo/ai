# 006a — `download_longlive2_bf16` overwrites `manifest.json` instead of merging (record loss)

- Status: open
- Severity: high (silent data loss — sibling backend records deleted)
- Area: model registry — `voyage/model_registry.py:227,260-261`
- Rank rationale: filed in pass 2 with HIGH severity; prefixed `006a` so it sorts
  with the criticals. Running it after any other download deletes that download's
  record.

## Technical description

```python
manifest_path = models_dir / "manifest.json"
manifest_path.write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")
```

where `record` contains only the `"video"` key. Every other downloader goes
through `_merge_manifest_record` (`:285`), which preserves sibling keys — so
`models download longlive2-bf16` run *after* any other download silently deletes
the `director`/`audio`/`ltxv`/`causvid`/`inspector` records.

## Why this is an issue

This is silent record loss on a routine path: a user who provisions the
director LLM and then downloads the video weights ends up with a manifest
that claims the director was never downloaded — so later `verify` runs,
fresh-container rebuilds, and audits all operate on a lie, potentially
re-downloading gigabytes or shipping an image believed to contain models it
cannot prove. One downloader in six bypasses the merge helper the other five
use, so the fix is literally one line plus a regression test. HIGH severity
is warranted: data loss with no error, on the happy path, in ordering-
dependent fashion.

## Evidence

(code experiment, host, verified by orchestrator 2026-09-25; re-verified
repair pass — output below)

```
$ rg -n "manifest_path.write_text|_merge_manifest_record\(" Voyage/voyage/model_registry.py
261:    manifest_path.write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")
285:def _merge_manifest_record(models_dir: Path, key: str, value: dict[str, Any]) -> dict[str, Any]:
294:    manifest_path.write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")
325:    record = _merge_manifest_record(
397:    record = _merge_manifest_record(
460:    record = _merge_manifest_record(
537:    record = _merge_manifest_record(
603:    record = _merge_manifest_record(
```

Raw `write_text` only at :261 (inside `download_longlive2_bf16`); all five others
merge. `rg -n "download_longlive2_bf16" tests/*.py` → zero hits: no test pins
merge behavior for this target.

## Reproduction

With a tmp models dir: write sentinel `manifest.json` with `{"director": {...}}`,
call `download_longlive2_bf16` (or just inspect that `record` at :245-259 has only
`"video"` and is written verbatim) → sentinel gone.

## Source references

- `voyage/model_registry.py:227,245-261,285-294,325,397,460,537,603`.

## Resolution candidates

Route through `_merge_manifest_record(models_dir, "video", record["video"])` like
the other five; add a test that pre-seeds `manifest.json` with a foreign key and
asserts it survives.

## Investigation / progress / resolution log

- 2026-09-25: found by pass-2 tests/docs sweep; `rg` re-verified live.
- 2026-09-25 (repair): re-verified live — raw `write_text` still only at
  `:261` (inside `download_longlive2_bf16`, record built `:245-259` with only
  the `"video"` key); `_merge_manifest_record` at `:285-294` still used by
  all five others (`:325,397,460,537,603`). Fixed Evidence header to the
  exact `## Evidence` (was `## Evidence (…)`). Added
  `## Why this is an issue`. No staleness.
- Open: one-line fix + regression test.
