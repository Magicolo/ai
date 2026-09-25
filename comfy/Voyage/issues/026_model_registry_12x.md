# 026 — `model_registry.py`: 6 download + 6 verify functions are one table-driven function copied 12×

- Status: resolved (fixed 2026-09-25)
- Severity: medium-high (~400 lines; adding a backend = 13th copy)
- Area: structure — `voyage/model_registry.py` (662 lines)
- Rank rationale: textbook table-driven refactor; each pair differs only in
  repo/revision/glob/MIN_BYTES.

## Technical description

`rg -n "def download_|def verify_" Voyage/voyage/model_registry.py` → 12 hits
(`:227,265,298,346,377,412,430,478,509,554,574,624`). Every pair follows
download(snapshot→glob→bytes→merge-manifest) / verify(glob→size-sanity→
missing-list) with only constants differing:

```python
def download_longlive2_bf16(models_dir) / download_director_models / ...
    snapshot_download(repo_id=..., revision=..., local_dir=..., allow_patterns=...)
    shards = sorted(...glob("model-*-of-*.safetensors"))
    weights_bytes = sum(p.stat().st_size for p in shards)
    record = _merge_manifest_record(models_dir, key, {...})
def verify_longlive2_bf16 / ... -> tuple[bool, str]:
    missing: list[str] = []
    ...glob...; ...stat().st_size < MIN_BYTES → missing.append(...)
    if missing: return False, f"missing {len(missing)} files: {missing[:5]}"
    return True, f"... OK (... GiB)"
```

## Why this is an issue

Twelve near-identical functions mean the thirteenth backend costs a thirteenth copy instead of one table row, and each copy is a fresh chance for download/verify constants to drift apart — a shard glob that downloads files the verifier never checks, or a size floor updated in one twin but not the other. Reviewers must diff 40-line bodies to find the single differing constant, which is exactly the kind of review that waves real bugs through. The blast radius is every future model pin plus the integrity of the models already on disk.

## Evidence

`diff <(sed -n '345,375p' model_registry.py) <(sed -n '411,427p' model_registry.py)`
— director vs inspector verify differ only in constants (sweep output).

Verified live 2026-09-25 (line numbers shifted +1 since the sweep):

```
$ rg -n "def download_|def verify_" voyage/model_registry.py
227:def download_longlive2_bf16  265:def verify_longlive2_bf16
298:def download_director_models  346:def verify_director_models
377:def download_inspector_models  412:def verify_inspector_models
430:def download_audio_models  478:def verify_audio_models
509:def download_ltxv_models  554:def verify_ltxv_models
574:def download_causvid_models  624:def verify_causvid_models
```

## Reproduction

The `rg` above.

## Source references

- `voyage/model_registry.py` lines above.

## Resolution candidates

`MODEL_SPECS: dict[name, ModelSpec(repo, revision, subdir, shard_glob, min_bytes,
license…)]` + generic `download_model(spec)` / `verify_model(spec)`; per-backend
functions become one-line wrappers (or disappear; CLI dispatches on spec name).
Fold in the missing Wan `revision=` (see 011) at the same time.

## Investigation / progress / resolution log

- 2026-09-25: found by structure sweep.
- 2026-09-25: repair pass — added `## Why this is an issue`; all 12 function
  lines had drifted +1 since the sweep (fixed); pasted rg output into Evidence.
- Open: table-drive; keep CLI output strings byte-stable via tests.
- 2026-09-25 (batch 4, track A; log written by orchestrator in review):
  FIXED. `MODEL_SPECS` table (`model_registry.py:528`) + generic
  `download_model`/`verify_model` (`:743/:774`); the 12 per-backend
  `download_*`/`verify_*` functions remain as thin wrappers (CLI output
  strings byte-stable). Verified live via def list above; gates green.
