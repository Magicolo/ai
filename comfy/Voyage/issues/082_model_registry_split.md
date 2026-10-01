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

## Progress log (2026-09-30, batch 12)

- Pre-checks live: `wc -l voyage/registry_records.py` → 767 at pass
  start; `git diff --name-only -- voyage/registry_records.py` empty
  before all four edits (facade import, FILM block deletion,
  EXPECTED line deletion, 2 builder deletions), so the quiet-region
  rule held. Smallest row set confirmed: FILM (8 pins + 1 hash + 2
  builders, zero cross-family deps — `FILM_REPO_PATH` derives only
  from `FILM_SUBDIR`/`FILM_FILE`).
- TDD failing-first: wrote `tests/test_registry_film_split.py`
  (3 agreement tests: pins single-sourced, builders single-sourced,
  MODEL_SPECS row points at family builders) BEFORE the new module —
  in-container collection failed with `ModuleNotFoundError: No module
  named 'voyage.registry_film'` (red), then created the module +
  re-export until green.
- Extraction: new `voyage/registry_film.py` (65L, DESIGN §§84-85)
  owns all FILM pins + `EXPECTED_FILM_SHA256` + `_record_film` +
  `_describe_film` verbatim; `registry_records.py` (767→770L)
  re-exports all 11 names via explicit-`as` self-aliases and carries
  move comments at the three old sites; `model_registry.py` untouched
  (its `from voyage.registry_records import ... FILM_*` chain holds
  through the facade). `sha256_file`/`Path`/`JsonValue` imports stay
  (used by 4+ remaining families, verified via rg).
- Gate evidence (in-container `voyage:latest`, CPU-only): new suite
  3 passed; neighbors `test_registry_split` + `test_registry_pins` +
  `test_augment_models` + `test_checkpoint_safety` 52 passed.
  Per-file gates: `ruff check` + `ruff format --check` + `mypy
  strict` clean on all 3 files (`registry_film.py`,
  `registry_records.py`, `test_registry_film_split.py`).

## Resolution (2026-09-30, batch 12)

- Verdict: **partial** — first per-family split landed;
  `registry_records.py` 767→770L (facade imports outweigh the moved
  block; the payoff compounds as the remaining 9 families follow).
- Files changed: `voyage/registry_film.py` (new, 65L),
  `voyage/registry_records.py` (facade re-export + 3 move comments,
  net +3L), `tests/test_registry_film_split.py` (new, 3 tests).
- Residual (open, ordered): (1) remaining 9 per-family
  `registry_{realesrgan,ltxv,causvid,qwen,audio,sfx,...}.py` splits
  (realesrgan is the next-smallest single-file row — same recipe);
  (2) in-core manifest read-modify-write race fix (unchanged).
- DESIGN proposal (not applied, see return report): per-family
  modules continue the `registry_film.py` pattern (one family per
  pass, facade chain `registry_<family>` → `registry_records` →
  `model_registry`, agreement test per family); manifest-race fix
  shape unchanged from the batch-7 entry.

## Refs

- Issues 065 (stale install lists), 021-analog hashing canonical (`voyage/hashing.py:30`); `voyage/hashing.py`, `voyage/models_ensure.py`

## Progress log (2026-09-30, batch 13 — realesrgan + inspector families)

- Pre-checks live: `wc -l voyage/registry_records.py` → 770 at pass
  start; `git diff --name-only -- voyage/registry_records.py` showed
  only own edits at every step (concurrent flight holds foreign hunks
  in `config.py`, `cli_run_ops.py`, `cli_observe.py`, `augment.py`,
  issues + tests — none inside either extraction region; verified the
  final `git diff` on `registry_records.py` shows exactly the 8 own
  edit sites, 74 insertions / 97 deletions). `model_registry.py`
  untouched both times (facade chain holds); longlive2 lines untouched
  (079 owner).
- TDD failing-first, twice: wrote
  `tests/test_registry_realesrgan_split.py` then
  `tests/test_registry_inspector_split.py` (3 agreement tests each:
  pins single-sourced, builders single-sourced, MODEL_SPECS row points
  at family builders — mirroring `test_registry_film_split.py`)
  BEFORE the new modules — in-container collection failed with
  `ModuleNotFoundError: No module named 'voyage.registry_realesrgan'`
  (red) and `... 'voyage.registry_inspector'` (red), then created
  each module + re-export until green.
- Extraction (1): new `voyage/registry_realesrgan.py` (75L, DESIGN
  §§84-85) owns all 8 `REALESRGAN_*` pins +
  `EXPECTED_REALESRGAN_SHA256` + `_record_realesrgan` +
  `_describe_realesrgan` verbatim (the next-smallest single-file row
  per the batch-12 residual); `registry_records.py` re-exports all 11
  names via explicit-`as` self-aliases (sorted between the film and —
  later — inspector blocks) and carries move comments at the four old
  sites. `sha256_file`/`Path`/`JsonValue` imports stay (used by the 7
  remaining families).
- Extraction (2): new `voyage/registry_inspector.py` (75L, DESIGN
  §§43-44, 100, 132) owns all 7 `QWEN35_*` pins + `_record_inspector`
  + `_describe_inspector` verbatim (next-smallest decoupled row; the
  ltxv/causvid/sfx/director/audio rows are bigger and coupled, the
  longlive+wan rows frozen for 079); same facade + move comments
  (three old sites). Deviation from the film recipe, recorded in both
  docstrings: no EXPECTED hash exists for this row (manifest record
  carries no sha — same open residual as the CausVid checkpoint), so
  the module omits the `sha256_file` import. `cli_observe.py:91`
  resolves `QWEN35_HF_REVISION` through the facade at call time
  (`getattr(model_registry, ...)`), so the benchmark-revisions path is
  unchanged with zero edits to that (foreign-owned, mid-pass
  modified) file.
- Gate evidence (in-container `voyage:latest` 2026-09-30 + bind mount,
  CPU-only — the image rebuild is foreign-broken this pass, see 036):
  new suites 3+3 passed; combined split/agreement 15 passed
  (`test_registry_{inspector,realesrgan,film,records}_split`);
  neighbors `test_augment_models` + `test_augment_weight_loading` +
  `test_checkpoint_safety` + `test_director_models_dir` +
  `test_longlive` 65 passed, 3 foreign-failed (all 079
  longlive2-removal surface: `test_cuda_backends_...[longlive2]` +
  `test_video_worker_module_map` raise `ConfigurationError: unknown
  video backend 'longlive2'`, `test_models_verify_reports...` raises
  `AttributeError: voyage.cli has no attribute
  'verify_longlive2_bf16'` — none import from the touched files).
  Per-file gates: `ruff check` + `ruff format --check` + `mypy
  strict` clean on all 5 touched files (`registry_realesrgan.py`,
  `registry_inspector.py`, `registry_records.py`, both new tests).
  No `pyproject.toml` change (family modules are clean under the base
  rule set — `registry_film.py` carries no per-file entry either).

## Resolution (2026-09-30, batch 13)

- Verdict: **partial** — second and third per-family splits landed;
  `registry_records.py` 770→747L (facade imports outweigh each moved
  block singly; the payoff compounds as the remaining 7 follow).
- Files changed: `voyage/registry_realesrgan.py` (new, 75L),
  `voyage/registry_inspector.py` (new, 75L),
  `voyage/registry_records.py` (2 facades + 7 move comments, net
  −23L), `tests/test_registry_realesrgan_split.py` (new, 3 tests),
  `tests/test_registry_inspector_split.py` (new, 3 tests).
- Residual (open, ordered): (1) remaining 7 per-family splits
  (ltxv, causvid+wan21, sfx triple, director triple, audio pair,
  longlive+wan — same recipe; longlive2 lines stay frozen for the 079
  owner); (2) in-core manifest read-modify-write race fix (unchanged).
- DESIGN proposal (not applied, see return report): per-family
  modules continue the `registry_film.py` pattern (one to two
  families per pass, facade chain `registry_<family>` →
  `registry_records` → `model_registry`, agreement test per family);
  manifest-race fix shape unchanged from the batch-7 entry.
