# 036 — God modules: `cli.py` 2105 / `supervisor.py` 1902 lines vs §12 ~500-line split signal

- Severity: MEDIUM
- Files: `voyage/cli.py:1` (2105), `voyage/supervisor.py:1` (1902), `voyage/model_registry.py:1` (1368), `voyage/workers/video_longlive.py:1` (1264), `voyage/workers/video_causvid.py:1` (1157), `voyage/media.py:1` (1116), `voyage/tui.py:1` (1078); tests `tests/test_tui_app.py:1` (943), `tests/test_causvid_worker.py:1` (748)
- Area: structure
- Overlaps with: 080/081 (cli/supervisor split execution — this file is the signal, those are the splits)

## Description

AGENTS.md §12 sets a ~500-line module-split soft signal with god-module
watch on `cli/supervisor`/video workers. Seven shipped modules exceed 2-4x
that; `media.py` also carries the open "workers/media surface stays open"
split from batch 4. Large-file reviewability, `PLR0912/0915` and `C901` all
concentrate here.

## Rationale

Size correlates with branch/statement-count violations and merge contention
(AGENTS.md §9 warns concurrent agents collide in `supervisor.py`). The
standardization pass already split once (batch 4: `video_common`, `hashing`,
`BackendName` registry) — the remainder is tracked nowhere.

## Live evidence (re-verified 2026-09-30)

Host `wc -l` today (re-verified 2026-09-30, drifted up from the Track-C sweep by concurrent edits):

```
2105 voyage/cli.py
1902 voyage/supervisor.py
1368 voyage/model_registry.py
1264 voyage/workers/video_longlive.py
1157 voyage/workers/video_causvid.py
1116 voyage/media.py
1078 voyage/tui.py
975 voyage/workers/video_ltxv.py
940 voyage/config.py
```

(Track-C sweep recorded 2030 for `cli.py`; current tree reads 2105 — drift
is growth, same finding.) Sweep `ALL` stats (preserved): `PLR0912 13
too-many-branches, PLR0915 11 too-many-statements, C901 16
complex-structure`.

## Repro

```bash
wc -l voyage/*.py voyage/workers/*.py | sort -rn | head
```

## Fix candidates

Split `cli.py` by verb group (generate/run/finalize vs models/doctor),
extract `supervisor` commit/augment helpers, extract `model_registry`
per-backend record builders; split `test_tui_app.py` /
`test_causvid_worker.py` by area. No behavior change, DESIGN §-ref'd new
modules.

## Fold note (2026-09-30, append-only — this tracker stays open)

080/082/085 landed as the first three executions of this signal; this file
is folded as their tracker (not deleted) until the remaining splits land.

- Landed: `cli.py` 2403 → **735L** (080, verbs in 10 modules ≤365L;
  parsers retained as the seam by design); `model_registry.py` 1543 →
  **1270L** (082, pins/builders in `registry_records.py` 767L;
  per-family tables + manifest race remain); 085 collapses (epsilon,
  fallbacks, reserved leaf, TOML fallbacks) + agreement tests for the
  rest. Evidence: `tests/test_cli_split.py` (13) +
  `tests/test_registry_split.py` (7) + `tests/test_single_source.py` (8);
  15/15 `--help` goldens identical; live `init/status/validate` smoke exit 0.
- Still above the ~500L signal (host `wc -l` today): `supervisor.py`
  2538, `model_registry.py` 1270, `video_longlive.py` 1301,
  `video_causvid.py` 1157, `media.py` 1404, `tui.py` 1080,
  `video_ltxv.py` 1115, `config.py` 1023, `cli.py` 735 (seam, stable
  point), `test_tui_app.py` 943, `test_causvid_worker.py` 748.
- Next in rank order: supervisor commit/augment-helper extraction (needs a
  quiet tree — concurrently modified today), `media.py` workers/media
  surface (open since batch 4), per-family registry tables (082
  residual), video-worker splits, then the two test files by area.
  Procedure per split: re-verify premises live, fail-first surface tests,
  verbatim moves + DAG check + goldens, seam-dispatch rule for patched
  names (see 080 Resolution), gates on touched files only.

## Refs

- AGENTS.md §12 soft signals ("~500-line module split signal, god-module
  watch, dead-code removal on sight").
- Preserved track result: `ses_f10013fc5ffeLLDtqZEwFbf3JR`, §6.
