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

## Drift note (2026-09-30 pre-work re-verification, live in-container)

- `wc -l voyage/model_registry.py` → **1543** (was 1368 at filing, 1267 at
  sweep — growth, same finding); 62 `def`/`class` (was 49).
- 10 `_record_*` + 10 `_describe_*` confirmed live (film + realesrgan rows
  added since the issue's "9 families" text — now 10 bundles).
- `_sha256` alias still present — but the alias-deletion candidate is
  **withdrawn**: `media._sha256_file` + `workers/video_ltxv.sha256_file`
  form an established 3-alias delegation pattern (all pinned in
  `tests/test_hashing.py`), so deleting one of three adds inconsistency,
  not structure. Kept + pinned instead.
- `_MANIFEST_LOCK`/`_repair_manifest` (models_ensure.py) are load-bearing
  (3 tests in `tests/test_containers_rank2.py` pin repair behavior) — the
  in-core manifest-race rewrite is deferred as a DESIGN proposal; the
  split keeps the race exactly where it is.
- `cli.py`/`model_registry.py` both clean in `git status` (only
  supervisor/director/video_ltxv concurrently modified) — safe to proceed.

## Fix candidates

1. Split per-family spec modules (`registry_{ltxv,causvid,qwen,audio,sfx,augment}.py`) + generic `download/verify/manifest` core; `model_registry.py` becomes re-export + `MODEL_SPECS` assembly.
2. Fix manifest read-modify-write race in-core; delete `_MANIFEST_LOCK`/`_repair_manifest` workaround; delete `_sha256` alias + 1 test line.
3. Add mirror-freshness test: every `MODEL_SPECS` key appears in INSTALL/README/MODELS (see 092).
4. Gate: `gates.sh` green + `test_checkpoint_safety.py` + `test_director_models_dir.py` + `test_augment_models.py` green.

## Progress log (2026-09-30, resolution pass)

- Re-verified live in-container: `model_registry.py` **1543L** / 62
  defs / 10 `_record_*` + 10 `_describe_*` (drift: +175L since filing;
  film + realesrgan rows added). `git status` clean for the file.
- Withdrew two fix candidates with recorded rationale: `_sha256` alias
  deletion (established 3-alias delegation pattern with
  `media._sha256_file` + `video_ltxv.sha256_file`, all pinned in
  `tests/test_hashing.py` — deleting one adds inconsistency); in-core
  manifest-race rewrite (`_MANIFEST_LOCK`/`_repair_manifest` load-bearing,
  3 tests pin repair behavior in `tests/test_containers_rank2.py`).
- Wrote fail-first `tests/test_registry_split.py` (6 tests): collection
  failed pre-impl (`ModuleNotFoundError: voyage.registry_records`).
- Extracted `voyage/registry_records.py` (**767L**: all pins + 10
  `_record_*` + 10 `_describe_*`); `model_registry.py` keeps spec
  dataclasses + `MODEL_SPECS` assembly + download/verify/manifest core +
  re-exports + `__all__`. Dry-run caught and fixed before writing:
  `MODEL_SPECS`.isupper() misrouting the assembly table, decorator names
  (`dataclass`) + class-field annotations (`Callable`) dropped from
  generated imports.
- Goldens: `dir()` surface byte-identical (after the `Callable` fix);
  `MODEL_SPECS` keys unchanged (10 bundles).
- Gates on touched files: `ruff check` + `ruff format --check` clean
  (new `F401` seam ignores in `pyproject.toml`, mirroring the 034
  convention), `mypy strict` clean. New tests 6/6; related
  checkpoint_safety/director_models/augment_models/augment_weight_loading/
  causvid_prep/ltxv/longlive/sfx_contract/vocoder suites green.

## Resolution

- Verdict: **partial**. Pins + builders are single-sourced in
  `registry_records.py`; `model_registry.py` 1543 → **1270L**.
- Residual (open, ordered): (1) per-family `registry_{ltxv,causvid,qwen,
  audio,sfx,augment}.py` split of the `MODEL_SPECS` rows (this pass keeps
  one family module — the table assembly stays whole); (2) in-core
  manifest read-modify-write race fix + `_MANIFEST_LOCK`/`_repair_manifest`
  removal (needs the 3 `test_containers_rank2.py` repair tests re-pointed;
  runtime locking behavior change — separate pass with a GPU-box
  contention probe).
- DESIGN proposal (not applied, see return report): per-family modules +
  manifest-race fix shape.

## Refs

- Issues 065 (stale install lists), 021-analog hashing canonical (`voyage/hashing.py:30`); `voyage/hashing.py`, `voyage/models_ensure.py`
