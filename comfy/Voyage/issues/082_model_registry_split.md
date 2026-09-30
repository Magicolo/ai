# 082 — Split `model_registry.py` (1267L, 9 families + download + verify + manifest)

- Severity: HIGH (structure)
- File: `voyage/model_registry.py:1` (1368 lines, 49 fns — sweep said 1267)
- Area: structure — registry decomposition

## Description

One file for 9 model families (longlive/wan/qwen/inspector/acestep/ltxv/sfx/causvid/film/realesrgan) + download + verify + manifest read-modify-write + `_MANIFEST_LOCK` workaround in `models_ensure.py` (`:38`) instead of a fix here. Legacy `_sha256()` alias of `hashing.sha256_file` (`:333-338`, sole pin `tests/test_hashing.py:47`). Comment at `:173` cross-references supervisor/CLI sets (registry is declared single source but consumers hand-copy).

## Rationale

Per-family pins change at different cadences (video weights vs director vs audio vs augment). Single file forces every pin bump through the same review + merge contention. Manifest race fix belongs here, not in the caller.

## Live evidence

- `wc -l voyage/model_registry.py` → 1368
- `:333-338` `_sha256` legacy alias; `:333,484,578` prod `sha256_file` direct uses
- `voyage/models_ensure.py:38` `_MANIFEST_LOCK` + `:142-167` `_repair_manifest` workaround
- `docs/MODELS.md` + `UPSTREAM_*_NOTES` cite registry as truth (triple pin mirror, by design)

## Repro

```bash
wc -l voyage/model_registry.py; grep -n "^def \|^class \|^MODEL_SPECS\|^def download\|^def verify" voyage/model_registry.py | head -n 60
grep -n "_sha256\|_MANIFEST_LOCK\|_repair_manifest" voyage/model_registry.py voyage/models_ensure.py tests/test_hashing.py
```

## Fix candidates

1. Split per-family spec modules (`registry_{ltxv,causvid,qwen,audio,sfx,augment}.py`) + generic `download/verify/manifest` core; `model_registry.py` becomes re-export + `MODEL_SPECS` assembly.
2. Fix manifest read-modify-write race in-core; delete `_MANIFEST_LOCK`/`_repair_manifest` workaround; delete `_sha256` alias + 1 test line.
3. Add mirror-freshness test: every `MODEL_SPECS` key appears in INSTALL/README/MODELS (see 092).
4. Gate: `gates.sh` green + `test_checkpoint_safety.py` + `test_director_models_dir.py` + `test_augment_models.py` green.

## Refs

- Issues 065 (stale install lists), 021-analog hashing canonical (`voyage/hashing.py:30`); `voyage/hashing.py`, `voyage/models_ensure.py`
