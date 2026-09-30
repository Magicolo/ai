# 038 — Magic-value comparisons dark: `PLR2004` 328 hits, unselected

- Severity: LOW
- File: `Voyage/pyproject.toml:70` (absence of `PLR`); instances across `voyage/`
- Area: standards / naming
- Overlaps with: 031/032/033/034 (ruff/mypy ratchet cluster — this file is the PLR2004 slice, those are the gate/ignore slices)

## Description

`PLR2004 magic-value-comparison` fires ~328 times under ALL but `PLR` is not
selected, so raw numeric comparisons (`== 0`, `> 32`, `< 720`,
second/minute/hour multipliers, HTTP codes) accumulate without named
constants — directly against §12 ("no magic numbers, named constants with
*why*, e.g. sync-frame floor 17").

## Rationale

The codebase already does this right in places (`voyage/seeds.py:14`
`_MAX_SEED = 2**31 - 1` with a *why* comment; `tests/conftest.py:55`
`bounded_counts`, `MAXIMUM_SEED_VALUE` in property tests). Unselected
PLR2004 means new code does not follow the good examples.

## Live evidence (re-verified 2026-09-30)

`Voyage/pyproject.toml:70` read live — `select` contains no `PLR` family:

```toml
select = ["E", "F", "I", "UP", "B", "A", "C4", "DTZ", "W", "BLE", "TRY", "EM", "SIM", "RUF100", "S101", "T201"]
```

Track-C sweep `ALL` stats (preserved Task output
`ses_f10013fc5ffeLLDtqZEwFbf3JR`): `PLR2004 328`. Positive control —
`voyage/seeds.py:14` shows the required pattern with its *why* comment.

## Repro

```bash
ruff check --select PLR2004 --statistics .
ruff check --select PLR2004 voyage/ | head -n 40
```

## Fix candidates

Enable `PLR2004` alone (narrowest useful slice of `PLR`), extract constants
per module with *why* comments; keep test-only thresholds local but named.

## Refs

- Ruff `PLR2004` docs; AGENTS.md §12 naming/constants clause.
- `voyage/seeds.py:14` (good-example pattern).
- Preserved track result: `ses_f10013fc5ffeLLDtqZEwFbf3JR`, §8.

## Progress log (2026-09-30, toolchain track)

- Re-verified live in-container: `ruff check --select PLR2004` now
  reports **387 tree-wide** (was 328). Top-level `voyage/*.py` slice is
  16 hits in 5 files: media.py 7, doctor.py 4, sfx_finalize.py 3,
  config.py 1, supervisor.py 1. Full select enablement is impossible
  (387 red); the narrow slice below is what this track converts.
- Extractions (named constants with *why*, §12 pattern, zero behavior —
  values identical, messages byte-identical):
  - doctor.py: `_SMI_COLUMN_TOTAL/FREE/DRIVER/COMPUTE_CAP/TEMP = 1..5`
    (nvidia-smi --query-gpu CSV column order; index 0 = name stays
    literal, 0/1 are PLR2004-exempt by default) — 4 hits → 0.
  - config.py: `MAX_FINAL_OVERLAP_FRACTION = 0.5` before AudioConfig
    (at most half a segment re-sliced into the overlap, else the joint
    swallows the take); used in the validator + f-string message
    (renders the identical "[0, 0.5]") — 1 hit → 0.
  - media.py: `FPS_MATCH_TOLERANCE = 0.5` (ffprobe fractional rates
    like 30000/1001 vs integer targets; absorbs rounding without
    masking off-by-one-fps) used at 5 sites (:161, :814 ×2, :1018,
    :1183); `MIN_FADE_GRAPH_SECONDS = 0.1` (sub-0.1 s fades inaudible
    — plain concat); `MIN_OVERLAP_BLEND_SECONDS = 0.05` (sub-50 ms
    overlaps can't carry a fade — deliberately distinct from the
    existing `MIN_SLICE_PIECE_SECONDS`, different why);
    `ABSORPTION_EPSILON_SECONDS = 0.01` (sub-10 ms absorption is
    slice-grid rounding noise) — 7 hits → 0.
  - sfx_finalize.py: `SFX_MAX_WORKERS = 2` (only 1- and 2-GPU shapes
    exist; the cuda:0/cuda:1 pairing; issue 158 fail-fast) used at 3
    sites (`not in (1, …)`, both `== …`) — 3 hits → 0.
  - supervisor.py's 1 hit skipped: file is dirty (concurrent agent) —
    recorded as residual, not touched.
- Verification per file in-container: `ruff check` + `ruff format
  --check` + `ruff check --select PLR2004` + `mypy` (where gated)
  green on doctor/config/sfx_finalize; media.py PLR2004-clean and
  format-clean with exactly one remaining error — F401 from the
  concurrent track's half-landed augment import (`from
  voyage.augment import interpolated_frame_count`, unused; their
  lines 21-23 + comment, verified foreign via git blame/diff —
  untouched here, left for their landing).
- TDD/regression: `scripts/gates.sh` gained a scoped in-container
  check `ruff check --select PLR2004 voyage/config.py voyage/doctor.py
  voyage/media.py voyage/sfx_finalize.py` after the format check — new
  magic values in the four converted files fail the gate (the scoped
  form is immune to media.py's foreign F401, which only fires under
  the full select).

## Resolution

- 15 top-level hits extracted to 10 named constants (4 files); the
  four files are PLR2004-clean with a gate tripwire each. Full-PLR
  select stays out (371 hits remain in workers/tests/subpackages —
  owning passes' scope).
- Files changed: `voyage/doctor.py`, `voyage/config.py`,
  `voyage/media.py`, `voyage/sfx_finalize.py` (constants + usages) +
  `scripts/gates.sh` (scoped PLR2004 check). Gate evidence: per-file
  ruff/format/PLR2004/mypy outputs recorded above (all green; media.py
  minus the foreign F401). DESIGN proposals: none. Residuals:
  supervisor.py 1 hit (dirty file, concurrent owner); 371
  workers/tests/subpackage hits; media.py F401 (foreign, in flight).
