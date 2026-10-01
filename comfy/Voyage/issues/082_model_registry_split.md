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

## Progress log (2026-10-01, ltxv + audio families)

- Pre-checks live in-container (`voyage:latest`, CPU-only,
  `docker run --rm -v $PWD:/app -w /app voyage:latest ...` from
  `Voyage/`): `wc -l voyage/registry_records.py` → 665 at pass
  start (747 at batch-13 close, minus 079 longlive2 removal);
  `git diff --name-only` on owned targets empty before every edit
  (`Voyage/` quiet except `M Voyage/LTX2.md`, left intact per
  brief); `model_registry.py` untouched (facade chain holds).
- Re-verified remaining families live via
  `grep -n "^QWEN\|^MINILM\|^ACE_\|^LTXV\|^CAUSVID\|^WAN21\|^MMAUDIO\|^def _record_\|^def _describe_"`:
  director QWEN_* + QWEN4B_AWQ_* + shared MINILM_* (coupled via
  both `_record_director` builders), audio ACE pair, LTXV row,
  causvid+WAN21 row, SFX triple — all decoupled except the
  director triple (MINILM shared). Two smallest decoupled rows
  taken this pass (quota convention, max two): LTXV then audio.
- TDD failing-first, twice: wrote
  `tests/test_registry_ltxv_split.py` then
  `tests/test_registry_audio_split.py` (3 agreement tests each:
  pins single-sourced, builders single-sourced, MODEL_SPECS row
  points at family builders — mirroring
  `test_registry_film_split.py`) BEFORE the new modules —
  in-container collection failed with
  `ModuleNotFoundError: No module named 'voyage.registry_ltxv'`
  (red) and `... 'voyage.registry_audio'` (red), then created
  each module + re-export until green.
- Extraction (1): new `voyage/registry_ltxv.py` (84L, DESIGN
  Phase 7, §§84-85) owns all 13 `LTXV_*` pins +
  `EXPECTED_LTXV_DIT_SHA256` + `EXPECTED_LTXV_UPSC_SHA256` +
  `_record_ltxv` + `_describe_ltxv` verbatim; `registry_records.py`
  re-exports all 17 names via explicit-`as` self-aliases (sorted
  between the inspector and realesrgan blocks) and carries move
  comments at the four old sites (pins, EXPECTED hashes, record,
  describe). `video_ltxv.py:42` imports the four used pins via
  `model_registry` (facade-safe, zero edits); `cli_observe.py`
  resolves `LTXV_*_REVISION` via call-time `getattr` (facade-safe).
- Extraction (2): new `voyage/registry_audio.py` (105L, DESIGN
  §§6, 37) owns all 13 `ACE_*` pins + `_ACE_CHECKPOINTS_RELATIVE`
  + `_ACE_LM_RELATIVE` + `_record_audio` + `_describe_audio`
  verbatim (next-smallest decoupled row); same facade + move
  comments (four old sites, audio block sorted before film).
  Deliberate deviation from the film recipe, recorded in both
  docstrings: no EXPECTED hash exists for this row (manifest
  record carries no sha — same open residual as inspector/
  CausVid), so the new module omits the `sha256_file` import
  (ruff F401 would fire). No worker imports ACE pins directly
  (only `model_registry` + `cli_observe` getattr) — facade-safe.
- Gate evidence (in-container `voyage:latest`, CPU-only): new
  suites 3+3 passed; combined split/agreement 15 passed
  (`test_registry_{ltxv,audio,film,realesrgan,inspector}_split`);
  neighbors 69 passed (`test_registry_split` +
  `test_registry_pins` + `test_augment_models` +
  `test_augment_weight_loading` + `test_checkpoint_safety` +
  `test_director_models_dir`) + 60 passed on the worker-adjacent
  set (`test_ltxv` + `test_causvid_prep` + `test_sfx_contract`
  alongside the new suites — LTXV worker facade path covered).
  Per-file gates: `ruff check` + `ruff format --check` + `mypy
  strict` clean on all 5 touched files (`registry_ltxv.py`,
  `registry_audio.py`, `registry_records.py`, both new tests).
  One isort catch on the way (`_ACE_*` constants sort before
  `ACE_*` in the facade) — fixed via in-container
  `ruff check --fix`, all green after. No `pyproject.toml` change
  (family modules are clean under the base rule set).
- Remaining families in `registry_records.py` (646L, live line
  numbers post-pass): director QWEN_* (`QWEN_HF_REPO:216`,
  `QWEN4B_AWQ_HF_REPO:244`, shared `MINILM_HF_REPO:272` +
  `_record_director:485` + `_describe_director:564` +
  `_record_director_awq:576` + `_describe_director_awq:597`);
  causvid+WAN21 (`CAUSVID_COMMIT:308`, `WAN21_HF_REPO:342` +
  `_record_causvid:520` + `_describe_causvid:627`); SFX triple
  (`MMAUDIO_CODE_COMMIT:379`, `MMAUDIO_VOCODER_REPO:414`,
  `MMAUDIO_CLIP_REPO:436` + `_record_sfx:542` +
  `_describe_sfx:619`). No candidate proved coupled mid-pass —
  both extractions landed as planned.

## Resolution (2026-10-01, ltxv + audio families)

- Verdict: **partial** — fourth and fifth per-family splits landed;
  `registry_records.py` 665→646L (net −19L; facade imports
  outweigh each moved block singly; payoff compounds as the
  remaining 3 follow).
- Files changed: `voyage/registry_ltxv.py` (new, 84L),
  `voyage/registry_audio.py` (new, 105L),
  `voyage/registry_records.py` (2 facades + 8 move comments, net
  −19L), `tests/test_registry_ltxv_split.py` (new, 3 tests),
  `tests/test_registry_audio_split.py` (new, 3 tests).
- Residual (open, ordered): (1) remaining 3 per-family splits
  (director triple incl. shared MINILM, causvid+wan21, sfx triple
  — same recipe; supervisor-side derivation stays with its
  owner); (2) in-core manifest read-modify-write race fix +
  `_MANIFEST_LOCK`/`_repair_manifest` removal (needs the 3
  `test_containers_rank2.py` repair tests re-pointed; runtime
  locking behavior change — separate pass, owner-held).
- DESIGN proposal (not applied, see return report): per-family
  modules continue the `registry_film.py` pattern (one to two
  families per pass, facade chain `registry_<family>` →
  `registry_records` → `model_registry`, agreement test per
  family); manifest-race fix shape unchanged from the batch-7
  entry.

## Progress log (2026-10-01, causvid + sfx families)

- Pre-checks live in-container (`voyage:latest`, CPU-only,
  `docker run --rm -v $PWD:/app -w /app voyage:latest ...` from
  `Voyage/`): `wc -l voyage/registry_records.py` → 646 at pass
  start (matches the ltxv+audio close); `git diff --name-only` on
  owned targets empty before every edit (`Voyage/` carries foreign
  concurrent hunks in `LTX2.md`, issues 031/035/081/088/152,
  `scripts/gates.sh`, `tests/test_generation_stack.py` + a
  `test_prefetch_shutdown.py` deletion — all left intact per §9,
  none inside either extraction region); `model_registry.py`
  untouched (facade chain holds).
- Re-verified remaining families live via grep: director QWEN_* +
  QWEN4B_AWQ_* + shared MINILM_* (coupled via both
  `_record_director` builders), causvid+WAN21 row, SFX triple.
  Quota is max TWO families — the two decoupled rows taken this
  pass (causvid, then sfx); the director triple is DEFERRED (see
  Resolution: shared-MINILM single-source decision + largest
  remaining row, needs its own pass).
- TDD failing-first, twice: wrote
  `tests/test_registry_causvid_split.py` then
  `tests/test_registry_sfx_split.py` (3 agreement tests each:
  pins single-sourced, builders single-sourced, MODEL_SPECS row
  points at family builders — mirroring
  `test_registry_film_split.py`) BEFORE the new modules —
  in-container collection failed with
  `ModuleNotFoundError: No module named 'voyage.registry_causvid'`
  (red) and `... 'voyage.registry_sfx'` (red), then created
  each module + re-export until green.
- Extraction (1): new `voyage/registry_causvid.py` (121L, DESIGN
  §5.4) owns all 11 `CAUSVID_*` pins + all 9 `WAN21_*` pins +
  `_record_causvid` + `_describe_causvid` verbatim;
  `registry_records.py` re-exports all 22 names via explicit-`as`
  self-aliases (sorted between the audio and film blocks) and
  carries move comments at the three old sites (pins, record,
  describe). `workers/video_causvid.py:60-69` imports its 7 used
  pins via `model_registry` (facade-safe, zero edits);
  `cli_observe.py` resolves `CAUSVID_*`/`WAN21_*` revisions via
  call-time `getattr` (facade-safe).
- Extraction (2): new `voyage/registry_sfx.py` (135L, SFX slice
  2 three-caption doctrine) owns all 26 `MMAUDIO_*` pins (code +
  vocoder + CLIP) + `_record_sfx` + `_describe_sfx` verbatim
  (largest decoupled row); same facade + move comments (three
  old sites, sfx block sorted after realesrgan). No worker
  imports MMAUDIO pins directly (only `model_registry` +
  `cli_observe` getattr) — facade-safe.
- Deliberate deviation from the film recipe, recorded in both
  docstrings: neither row carries an EXPECTED ingest hash (the
  CausVid DMD file was pruned 2026-09-24, the SFX record carries
  no sha — same open residual as inspector/audio), so
  `registry_sfx.py` omits the `sha256_file` import (ruff F401
  would fire); `registry_causvid.py` KEEPS it (`_record_causvid`
  computes `checkpoint_sha256` live). Consequence: `sha256_file`
  is now unused in `registry_records.py` (causvid was its sole
  user — verified via grep), so the import is deleted there.
- Gate evidence (in-container `voyage:latest`, CPU-only): new
  suites 3+3 passed; combined split/agreement 40 passed
  (`test_registry_{causvid,sfx,ltxv,audio,film,realesrgan,
  inspector}_split` + `test_registry_split` +
  `test_registry_pins`); neighbors 89 passed
  (`test_augment_models` + `test_augment_weight_loading` +
  `test_checkpoint_safety` + `test_director_models_dir` +
  `test_causvid_prep` + `test_ltxv` + `test_sfx_contract` +
  `test_vocoder_allowlist` — CausVid worker + SFX/vocoder paths
  covered). Per-file gates: `ruff check` + `ruff format --check`
  + `mypy strict` clean on all 5 touched files
  (`registry_causvid.py`, `registry_sfx.py`,
  `registry_records.py`, both new tests). No `pyproject.toml`
  change (family modules are clean under the base rule set).
  Full `gates.sh` left to the orchestrator.
- Remaining family in `registry_records.py` (609L, live line
  numbers post-pass): director triple (`QWEN_HF_REPO:365`,
  `QWEN4B_AWQ_HF_REPO:393`, shared `MINILM_HF_REPO:421` +
  `_record_director:490` + `_describe_director:533` +
  `_record_director_awq:545` + `_describe_director_awq:566`).

## Resolution (2026-10-01, causvid + sfx families)

- Verdict: **partial** — sixth and seventh per-family splits
  landed; `registry_records.py` 646→609L (net −37L; the two
  largest decoupled rows — facade imports no longer outweigh the
  moved blocks).
- Files changed: `voyage/registry_causvid.py` (new, 121L),
  `voyage/registry_sfx.py` (new, 135L),
  `voyage/registry_records.py` (2 facades + 6 move comments +
  `sha256_file` import deletion, net −37L),
  `tests/test_registry_causvid_split.py` (new, 3 tests),
  `tests/test_registry_sfx_split.py` (new, 3 tests).
- Residual (open, ordered): (1) director triple
  (`QWEN_*` + `QWEN4B_AWQ_*` + shared `MINILM_*`, 20 pins + 4
  builders — DEFERRED by quota, not by difficulty: MINILM is
  shared by both `_record_director` builders, so the split must
  decide explicitly whether MINILM stays in `registry_records`
  with a comment or moves into `registry_director` with a
  single-source note; silent pin duplication is forbidden);
  (2) in-core manifest read-modify-write race fix +
  `_MANIFEST_LOCK`/`_repair_manifest` removal (needs the 3
  `test_containers_rank2.py` repair tests re-pointed; runtime
  locking behavior change — separate pass, owner-held).
- DESIGN proposal (not applied, see return report): the director
  triple follows the `registry_causvid.py` pattern (one family
  module + facade chain + agreement test, with the MINILM
  coupling decided explicitly in both docstrings); manifest-race
  fix shape unchanged from the batch-7 entry.

## Progress log (2026-10-01, director triple — last family)

- Pre-checks live in-container (`voyage:latest`, CPU-only,
  `docker run --rm -v $PWD:/app -w /app voyage:latest ...` from
  `Voyage/`): `wc -l voyage/registry_records.py` → 609 at pass
  start (matches the causvid+sfx close); `git diff --name-only`
  on owned targets empty before every edit (`Voyage/` carries
  foreign concurrent hunks in `DESIGN.md`, `LTX2.md`,
  `supervisor.py`, issues 031/035/081/089/093/152/166,
  `scripts/gates.sh`, `tests/test_finalize_fastpath.py` +
  `test_stage_a_telemetry.py` + a `test_commit_slice_compensation.py`
  addition — all left intact per §9, none inside the director
  extraction region; verified the final `git diff` on
  `registry_records.py` shows exactly the 9 own edit sites, 93
  insertions / 138 deletions). `model_registry.py` untouched
  (facade chain holds).
- Batch-15 decision (explicit, in both docstrings): the shared
  `MINILM_*` pins MOVE into `voyage.registry_director` alongside
  the QWEN rows — not duplicated, not left behind. Both
  `_record_director` builders resolve the embedding pins from the
  one module; silent pin duplication is forbidden.
- TDD failing-first: wrote
  `tests/test_registry_director_split.py` (3 agreement tests: 20
  pins single-sourced, 4 builders single-sourced, both
  `director-qwen8b` + `director-qwen4b-awq` MODEL_SPECS rows point
  at family builders — mirroring `test_registry_sfx_split.py`,
  extended to two spec rows) BEFORE the new module —
  in-container collection failed with
  `ModuleNotFoundError: No module named 'voyage.registry_director'`
  (red), then created the module + re-export until green.
- Extraction: new `voyage/registry_director.py` (175L, DESIGN §8
  + §140 GPU-director entry) owns all 7 `QWEN_*` pins + all 7
  `QWEN4B_AWQ_*` pins + all 6 shared `MINILM_*` pins +
  `_record_director` + `_describe_director` +
  `_record_director_awq` + `_describe_director_awq` verbatim;
  `registry_records.py` re-exports all 24 names via explicit-`as`
  self-aliases (sorted director block between the causvid and
  film blocks) and carries move comments at the seven old sites
  (QWEN pins, AWQ pins, MINILM pins with the batch-15
  single-source note, 4 builders). `from pathlib import Path` +
  `from voyage.atomic import JsonValue` deleted there (the
  director builders were their sole users — same consequence as
  the causvid+sfx `sha256_file` deletion). `cli_observe.py`
  resolves `QWEN_*`/`MINILM_*` revisions via call-time `getattr`
  (facade-safe, zero edits); no worker imports director pins
  directly (only `model_registry` + `cli_observe` getattr +
  `doctor` verify entry points).
- Deliberate deviation from the film recipe, recorded in both
  docstrings: this row carries no EXPECTED ingest hash (the
  manifest record carries no sha — same open residual as
  inspector/audio/sfx/CausVid), so the new module omits the
  `sha256_file` import (ruff F401 would fire).
- Gate evidence (in-container `voyage:latest`, CPU-only): new
  suite 3 passed; combined split/agreement 43 passed
  (`test_registry_{director,causvid,sfx,ltxv,audio,film,realesrgan,
  inspector}_split` + `test_registry_split` +
  `test_registry_pins`); neighbors 89 passed
  (`test_augment_models` + `test_augment_weight_loading` +
  `test_checkpoint_safety` + `test_director_models_dir` +
  `test_causvid_prep` + `test_ltxv` + `test_sfx_contract` +
  `test_vocoder_allowlist`); live golden probe: MODEL_SPECS keys
  unchanged (9 rows incl. both director rows), all 20 pins +
  4 builders identical across `registry_director` /
  `registry_records` / `model_registry`. Per-file gates:
  `ruff check` + `ruff format --check` + `mypy strict` clean on
  all 3 touched files (`registry_director.py`,
  `registry_records.py`, new test). No `pyproject.toml` change
  (family module clean under the base rule set). Full `gates.sh`
  left to the orchestrator. `registry_records.py` 609→564L and
  now holds zero family definitions (facade + move comments only
  — verified via grep: no `^QWEN`, `^MINILM`, `^def`,
  `^from pathlib`, `^from voyage.atomic`).

## Resolution (2026-10-01, director triple — last family)

- Verdict: **RESOLVED** — all per-family splits landed;
  `registry_records.py` is a pure facade (564L, zero
  pins/builders).
- Files changed: `voyage/registry_director.py` (new, 175L),
  `voyage/registry_records.py` (facade + 7 move comments +
  `Path`/`JsonValue` import deletion, net −45L),
  `tests/test_registry_director_split.py` (new, 3 tests).
- Residual (open, owner-held — recorded, not attempted):
  in-core manifest read-modify-write race fix +
  `_MANIFEST_LOCK`/`_repair_manifest` removal (needs the 3
  `test_containers_rank2.py` repair tests re-pointed; runtime
  locking behavior change — separate pass).
- DESIGN proposal (quoted, for the DESIGN owner — not applied
  here, file is out of scope): "Per-family modules are complete
  (`registry_director.py` closes the set: film, realesrgan,
  inspector, ltxv, audio, causvid, sfx, director);
  `registry_records.py` is now a pure re-export facade. The
  manifest read-modify-write race fix stays owner-held: move the
  lock into the registry core, delete the `models_ensure.py`
  `_MANIFEST_LOCK`/`_repair_manifest` workaround, and re-point
  the 3 `test_containers_rank2.py` repair tests."
