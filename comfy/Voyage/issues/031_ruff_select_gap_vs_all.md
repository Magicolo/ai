# 031 — Ruff `select` gap vs Zoomy `ALL`: thousands of findings invisible by config

- Severity: LOW (lint scope — green gate is a scope artifact, not a runtime defect)
- Group: standards/lint — Rank: 4/5
- File:line: `Voyage/pyproject.toml:70`
- Overlaps: 032/033/034 cluster (lint/typing scope debt) — same ratchet, not duplicates; fix as one toolchain pass.

## Description

Voyage selects 16 ruff families while Zoomy enforces `ALL` (line-length 100).
Probing Voyage with `--select ALL` yields thousands of un-gated hits, so the
green gate is a scope artifact, not cleanliness. Entire debt classes
(docstrings, annotations, magic values, security `S`, pytest style `PT`,
naming `N`, performance `PERF`, `PLR0913/0917` complexity) stay permanently
dark.

## Rationale

Ruff docs recommend starting small and adding a category at a time with
explicit `select` — but the project norm (AGENTS.md §12) is `ALL` parity with
Zoomy. Without a ratchet plan the current select freezes the gap: new code in
unselected families lands ungated and reviewers assume coverage that does
not exist.

## Live evidence (re-verified 2026-09-30)

`Voyage/pyproject.toml:70` (read live):

```toml
select = ["E", "F", "I", "UP", "B", "A", "C4", "DTZ", "W", "BLE", "TRY", "EM", "SIM", "RUF100", "S101", "T201"]
```

Track-C sweep capture (in-container `ruff check --select ALL --statistics`,
quoted from preserved Task output `ses_f10013fc5ffeLLDtqZEwFbf3JR`):

```
D103 771, COM812 511, TRY003 377, PLR2004 328, SLF001 290, EM102 259,
ANN401 255, CPY001 137, EM101 124, TC003 79, D102 76, ARG001 69,
PLR0913 52, PLR0917 41, FBT001 39, PT011 38 …
```

Gate itself is green on the scoped select (`ruff check .`: All checks passed).

## Repro

From `Voyage/` (needs the gate image):

```bash
docker run --rm -v "$PWD:/app" voyage:latest bash -c "ruff check --select ALL --statistics . 2>&1 | head -n 60"
```

Compare `Voyage/pyproject.toml:70` vs `Zoomy/pyproject.toml` (`select = ["ALL"]`).

## Fix candidates

(a) Ratchet `select` upward family-by-family with per-family autofix passes.
(b) Declare the Zoomy-parity gap explicitly in `pyproject.toml` comment +
`TASK.md` with an ordered adoption list.
(c) At minimum enable `ANN,D,PLR2004,PT,S,PERF,N` in CI as warn-only before
enforcing.

## Refs

- `Voyage/pyproject.toml:70`
- Ruff linter docs: "Use ALL with discretion…" / "Prefer lint.select… Start
  with a small set and add a group at-a-time" (docs.astral.sh/ruff/linter).
- AGENTS.md §12 (Zoomy-parity toolchain posture).
- Preserved track result: `ses_f10013fc5ffeLLDtqZEwFbf3JR`, §1.

## Progress log (2026-09-30, toolchain track)

- Re-verified premises live in-container (`voyage:latest`, CPU-only):
  `ruff check --select ALL --statistics .` now reports (as-read 2026-09-30):
  D103 797, COM812 703, PLC0415 576, TRY003 461, SLF001 457, PLR2004 387,
  ANN401 326, EM102 324, CPY001 167, EM101 143, ARG001 110, TC003 95 —
  every family grew vs the preserved sweep (D103 771→797, COM812 511→703,
  TRY003 377→461, PLR2004 328→387, ANN401 255→326), so counts drifted up
  with tree growth. `pyproject.toml:70` select is unchanged (16 families).
- Probed each ordered-adoption candidate family in-container
  (`ruff check --select FAMILY --statistics .`):
  ANN 339 (ANN401 326 + ANN001 10 + ANN202 3), D ~900+ (D103 797, D102 73,
  D205 35, …), PLR2004 387, PT 85 (PT011 42 + PT018 41 + …), S 77 (S603 31
  + S607 19 + S108 18 + S110 7 + S112 2), PERF 10 (PERF401 5 + PERF203 5),
  N 39 (N806 34 + N802 4 + N818 1). **No candidate family is green** —
  not even the narrowest (PERF: 10 hits across 7 files incl.
  tests/test_tui_app.py, voyage/cli_observe.py, voyage/cli_validate.py,
  voyage/concepts.py, voyage/models_ensure.py, voyage/supervisor.py ×2,
  voyage/workers/director.py ×2, voyage/workers/video_longlive.py;
  fixing them is loop restructuring across dirty/foreign files, out of
  this track's scope).
- Verdict per family (all stay out this pass): ANN — 339 hits, `Any` is
  the default seam type (see 035); D — ~900 hits, undocumented-public
  sweep needs its own pass; PLR2004 — 387 tree-wide, only the 15
  top-level hits convert this pass (see 038; scoped tripwire added to
  gates.sh instead of select); PT — 85 hits, test-style pass belongs to
  the test-structure track (039/040/086/088/089, explicitly out of
  scope); S — 77 hits, subprocess/tempfile hardening is behavior-adjacent
  (needs per-site review, not autofix); PERF — 10 hits but scattered
  across dirty/foreign files; N — 39 hits, naming pass belongs with the
  relevant owners.
- `select` left unchanged (no family added). No pyproject change for 031.

## Resolution

- Document-only: the gap stays policy (per-file ratchet continues via
  032/033/034/038/094), with fresh as-read counts above replacing the
  preserved sweep numbers. Next pass should retry PERF first (smallest:
  10) once the owning files settle, then N (39), then PT with the test
  track.
- Files changed: none for 031. Gate evidence: `select` untouched, so
  `ruff check .` scope is identical (its 7 current errors are all
  foreign in-flight files — see 032 log). DESIGN proposals: none.
  Residuals: entire adoption list (ANN, D, PLR2004-full, PT, S, PERF, N)
  still dark; counts re-baselined above.

## Progress log (2026-09-30, Group D pass)

- Re-verified live in-container (`voyage:latest`): `ruff check --select PERF .`
  → **11 hits** (was 10): `tests/test_tui_app.py:720` (PERF401),
  `voyage/cli_observe.py:352` (PERF203), `voyage/cli_validate.py:223,229`
  (PERF401), `voyage/concepts.py:431` (PERF203),
  `voyage/models_ensure.py:222` (PERF401), `voyage/supervisor.py:591,663`,
  `voyage/workers/director.py:390,517`, `voyage/workers/video_longlive.py:348`.
  `ruff check --select N --statistics .` → **39** (N806 34 + N802 4 + N818 1,
  unchanged shape). `ruff check --select PT --statistics .` → **85**
  (PT011 42 + PT018 41 + PT013 1 + PT012 1, unchanged shape).
- Retry order PERF → N → PT (batch-7 prescription): PERF is red (11 hits
  across cli_observe/cli_validate/concepts/models_ensure/dirty-supervisor/
  workers — loop restructuring across dirty + foreign files, out of scope),
  so N and PT were probed for counts only and the adoption stops: **no
  family is green in isolation, no `select` change**.
- `pyproject.toml:70` select confirmed unchanged (16 families).

## Resolution (2026-09-30, Group D pass)

- Document-only: gap stays policy; as-read counts re-baselined above
  (PERF 10→11, N 39, PT 85). Next pass retries PERF first once the owning
  files settle.
- Files changed: none for 031. Gate evidence: `select` untouched.
  DESIGN proposals: none. Residuals: full adoption list (ANN, D,
  PLR2004-full, PT, S, PERF, N) still dark.

## Progress log (2026-09-30, CLI track — PERF retry, this pass)

- Re-verified live in-container (`voyage:latest`): `ruff check --select
  PERF .` → 12 hits (was 11): `tests/test_issue_citation_gate.py:55`
  (PERF401), `tests/test_tui_app.py:748` (PERF401),
  `voyage/cli_observe.py:756` (PERF203 — the pre-existing `cmd_inspect`
  media-probe try/except, shifted from :362 by this pass's additions),
  `voyage/cli_validate.py:223,229` (PERF401 — file concurrently dirty),
  `voyage/concepts.py:437` (PERF203 — foreign), `voyage/models_ensure.py:227`
  (PERF401 — foreign), `voyage/supervisor.py:613,685` (dirty + out of scope),
  `voyage/workers/director.py:390,517` (foreign),
  `voyage/workers/video_longlive.py:389` (concurrently dirty).
- Retry rule applied: adoption needs EVERY site in clean owned files — three
  sites sit in concurrently-dirty files (`cli_validate.py`,
  `supervisor.py`, `video_longlive.py`, all with uncommitted concurrent
  edits as-read) and seven more in foreign files. Adoption EXCLUDED
  (record, don't force). The one owned-clean site (`cli_observe.py:756`)
  is left untouched: fixing it without the `select` adoption is a bare
  drive-by that would also change per-file probe-error semantics
  (fail-loud vs per-file report). `pyproject.toml:70` select unchanged.

## Resolution (2026-09-30, CLI track — this pass)

- Verdict: DEFERRED (record-only) — PERF 11→12, whole adoption excluded.
- Files changed: none for 031. Gate evidence: `select` untouched.
  DESIGN proposals: none. Residuals: full adoption list still dark; PERF
  unblocks when `supervisor.py` + `cli_validate.py` +
  `video_longlive.py` settle AND the foreign owners (concepts,
  models_ensure, director ×2, tests ×2) clear their sites — the
  `cli_observe.py:756` handoff is this track's only owned site for that
  future pass.

## Progress log (2026-09-30, this pass — PERF → N → PT retry)

- `git diff --name-only` at pass start: clean tree; at pass end a
  concurrent batch-9 + uncommitted foreign edits landed (incl. a 75-line
  foreign `supervisor.py` hunk — never touched here). Adoption needs
  EVERY site, so foreign/dirty holders block regardless of start state.
- Re-verified live in-container (`voyage:latest`, CPU-only):
  `ruff check --select PERF --output-format concise` → **12 hits**
  (7 PERF401 + 5 PERF203): `tests/test_issue_citation_gate.py:55`,
  `tests/test_tui_app.py:748`, `voyage/cli_observe.py:756` (PERF203 —
  intentional per-file probe-error report, restructuring changes
  fail-loud vs per-file semantics),
  `voyage/cli_validate.py:223,229` (conditional appends, not clean
  comprehension targets), `voyage/concepts.py:437`,
  `voyage/models_ensure.py:227`, `voyage/supervisor.py:613,685`
  (other group — record, do not touch),
  `voyage/workers/director.py:390,517`,
  `voyage/workers/video_longlive.py:389`.
- `ruff check --select N --statistics` → **60** (N806 38 + N801 16 +
  N802 5 + N818 1): 38 N806 are `VoyageApp` locals across the Pilot
  suites (mass churn in load-flaky files), N801/N802 are intentional
  test-double names (`_014_FakeTensor`, `Generator`, `Event`), N818
  `ProposalRejected` is a public-exception rename (API break, other
  groups). `ruff check --select PT --statistics` → **111** (PT011 63 +
  PT018 45 + PT017/PT013/PT012 1 each — test-style track owns it).
- No family is green in isolation; PERF is additionally hard-blocked by
  the 2 `supervisor.py` sites (banned file). No `select` change, no
  per-file-ignores added (ignoring supervisor to adopt PERF would be a
  scope artifact of the same kind this issue tracks).
- `pyproject.toml:70` select confirmed unchanged (16 families).

## Resolution (2026-09-30, this pass)

- Verdict: DEFERRED (record-only) — PERF 12, N 60, PT 111 as-read above.
- Files changed: none for 031. Gate evidence: `select` untouched
  (`ruff check .` whole-tree green).
  DESIGN proposals: none. Residuals: full adoption list (ANN, D,
  PLR2004-full, PT, S, PERF, N) still dark; PERF unblocks when the
  supervisor sites + worker/test sites clear under their owners; N
  unblocks with a test-double naming pass + a `ProposalRejected`
  rename decision; PT rides the test-structure track.

## Progress log (2026-09-30, close-out pass — STILL-BLOCKED)

- Handoff-site check first: `git diff HEAD --
  comfy/Voyage/voyage/cli_observe.py` shows a concurrent foreign hunk
  (metrics block extracted to `voyage/cli_inspect_metrics.py` +
  `iter_metric_files` import dropped — another agent's uncommitted
  work). The file is NOT foreign-hunk-free, so per the contract the
  single owned `cli_observe.py` probe-error semantics site was NOT
  touched — no edit, no drive-by. The media-probe `try/except` region
  itself is untouched by their hunk (theirs is the `metrics` block
  below it), but the file-level condition fails so the attempt stops.
- Fresh counts re-probed live in-container (`voyage:latest`, CPU-only,
  no host pip): `ruff check --select PERF --output-format concise` →
  **12 hits** (7 PERF401 + 5 PERF203):
  `tests/test_issue_citation_gate.py:55`,
  `tests/test_tui_app.py:748`, `voyage/cli_observe.py:728` (PERF203 —
  the owned handoff site, intentional per-file probe-error report),
  `voyage/cli_validate.py:223,229`, `voyage/concepts.py:437`,
  `voyage/models_ensure.py:227`, `voyage/supervisor.py:560,632`
  (banned/foreign file), `voyage/workers/director.py:390,517`,
  `voyage/workers/video_longlive.py:389` (concurrently dirty).
  `ruff check --select N --statistics` → **60** (N806 38 + N801 16 +
  N802 5 + N818 1). `ruff check --select PT --statistics` → **111**
  (PT011 63 + PT018 45 + PT017/PT013/PT012 1 each).
- `Voyage/pyproject.toml:70` select confirmed unchanged (16 families:
  `E,F,I,UP,B,A,C4,DTZ,W,BLE,TRY,EM,SIM,RUF100,S101,T201`).
  No family is green in isolation; PERF is additionally hard-blocked by
  the 2 `supervisor.py` sites plus the concurrent `cli_observe.py`
  hunk. No `select` change, no per-file-ignores added.

## Resolution (2026-09-30, close-out pass)

- Verdict: DEFERRED (record-only) — PERF 12, N 60, PT 111 as-read above.
- Files changed: none for 031 (this issue file only). Gate evidence:
  `select` untouched. DESIGN proposals: none.
- Residuals: full adoption list (ANN, D, PLR2004-full, PT, S, PERF, N)
  still dark; the `cli_observe.py:728` handoff stays this track's only
  owned PERF site for the future pass — it unblocks when the file is
  foreign-hunk-free AND the supervisor/worker/test sites clear under
  their owners (extract the loop body to a helper to preserve per-file
  probe-error semantics; do NOT hoist the try outside the loop).

## Progress log (2026-09-30, batch 12)

- Pre-flight: `git diff --name-only` showed a clean Voyage tree at
  pass start (only the 035 `scoreboard.py` hunk of this same pass was
  dirty when the 031 leg began). The batch-11 concurrent
  `cli_observe.py` hunk (metrics block → `cli_inspect_metrics.py`)
  has landed upstream — the file is foreign-hunk-free, so the owned
  handoff was actionable. `supervisor.py` (banned) and
  `video_longlive.py` (hot — concurrent `director.py` hunk from
  another agent landed mid-pass, same worker-track family) were never
  touched.
- Retry order PERF → N → PT, all re-probed live in-container
  (`voyage:latest`, CPU-only, no host pip):
  `ruff check --select PERF --output-format concise .` → **12 hits**
  at pass start (7 PERF401 + 5 PERF203), same shape as batch 11
  (only line numbers drifted: `cli_observe.py:727`,
  `supervisor.py:539,611`).
- Owned handoff LANDED: `cli_observe.py:722-728` media-probe
  try/except extracted to `_probe_media_line(name: Path) -> str`
  (`:692`; try lives in the helper, the loop is call+print —
  try was never hoisted outward, per-file `MediaError` report
  preserved verbatim). First shape tripped scoped TRY300
  (return-in-try) — reworked to try/except-then-compute so
  `ruff check` on the scoped select stays green. Behavior probe
  in-container (mocked `media_probe` success + `MediaError`
  branches): success line `000001/video.mp4: duration=7.5` and
  failure line `<path>: PROBE FAILED (no ffprobe)` both identical
  to the inline version.
- PERF after the handoff: **11 hits** (7 PERF401 + 4 PERF203) —
  every remaining site sits in a foreign/dirty/banned file, so
  family adoption stays blocked (no `select` change, no
  per-file-ignores — ignoring our way to green would be the scope
  artifact this issue tracks):
  `tests/test_issue_citation_gate.py:55`,
  `tests/test_tui_app.py:748` (both PERF401, test-track owned),
  `voyage/cli_validate.py:223,229` (conditional appends, not clean
  comprehension targets — needs its owner),
  `voyage/concepts.py:437` (foreign),
  `voyage/models_ensure.py:227` (foreign),
  `voyage/supervisor.py:539,611` (banned file),
  `voyage/workers/director.py:390,517` (foreign),
  `voyage/workers/video_longlive.py:389` (hot track).
- N → **60** (`N806 38 + N801 16 + N802 5 + N818 1`, unchanged):
  38 N806 are `VoyageApp` locals across the Pilot suites, N801/N802
  intentional test-double names, N818 `ProposalRejected` is a
  public-exception rename (API break, other groups). Mass churn —
  not attempted, recorded.
- PT → **111** (`PT011 63 + PT018 45 + PT017/PT013/PT012 1 each`,
  unchanged): test-style track owns it. Not attempted, recorded.
- `Voyage/pyproject.toml:70` select confirmed unchanged (16
  families).

## Resolution (2026-09-30, batch 12)

- Verdict: PARTIALLY RESOLVED — one owned PERF site cleared
  (12 → 11); family adoption stays DOCUMENTED (no `select`
  change); N/PT record-only.
- Files changed: `voyage/cli_observe.py` only (`_probe_media_line`
  helper + 6-line loop-body replacement). No test files changed
  (behavior-preserving refactor; mocked-branch probe above instead
  of TDD — no behavior to drive). Concurrent
  `workers/director.py:594-596` hunk left intact (disjoint region).
- Gate evidence (in-container `voyage:latest`, CPU-only):
  `ruff check voyage/cli_observe.py voyage/scoreboard.py
  voyage/model_registry.py voyage/rpc.py` — All checks passed;
  `ruff format --check voyage/cli_observe.py voyage/scoreboard.py`
  — 2 files already formatted; `mypy voyage/cli_observe.py
  voyage/scoreboard.py` + `mypy voyage` — clean (67 files);
  `pytest -p no:cacheprovider -q tests/test_scoreboard.py
  tests/test_cli_split.py tests/test_cli_inspect_metrics.py` —
  22 passed (covers the `cmd_inspect` seam +
  `cli.cmd_inspect is cli_observe.cmd_inspect`).
  DESIGN proposals: none.
- Residuals: PERF 11 (list above); N 60; PT 111; full adoption
  list (ANN, D, PLR2004-full, PT, S, PERF, N) still dark.
  PERF unblocks when the supervisor sites + worker/test sites
  clear under their owners.

## Progress log (2026-09-30, batch 13 — re-probe counts only)

- Re-probed live in-container (`voyage:latest`, CPU-only, no host
  pip — record-only, no source edits outside owned files):
  `ruff check --select PERF --output-format concise .` → **11 hits**
  (7 PERF401 + 4 PERF203):
  `tests/test_issue_citation_gate.py:55`,
  `tests/test_tui_app.py:748` (both PERF401, test-track owned),
  `voyage/cli_validate.py:223,229`,
  `voyage/concepts.py:437`, `voyage/models_ensure.py:227`,
  `voyage/supervisor.py:505,577` (banned file),
  `voyage/workers/director.py:390,517`,
  `voyage/workers/video_longlive.py:389` (hot track — the other
  group's 079 surface, never touched). The batch-12 owned handoff
  (`cli_observe.py` probe helper) holds — no `cli_observe.py` PERF
  hit remains.
- `ruff check --select N --statistics` → **60** (N806 38 + N801 16 +
  N802 5 + N818 1, unchanged shape). `ruff check --select PT
  --statistics` → **112** (PT011 63 + PT018 46 + PT017/PT013/PT012 1
  each — PT018 +1 vs batch 12). `ruff check --select ALL
  --statistics` top: D103 934, COM812 873, PLC0415 788, SLF001 587,
  TRY003 531, PLR2004 452, ANN401 421, EM102 384 (drifted up with
  tree growth; full 40-line capture in the run output).
- `Voyage/pyproject.toml:70` select confirmed unchanged (16
  families). No family is green in isolation; no `select` change, no
  per-file-ignores added.

## Resolution (2026-09-30, batch 13)

- Verdict: DEFERRED (record-only) — PERF 11, N 60, PT 112 as-read
  above. Files changed: none for 031 (this issue file only). Gate
  evidence: `select` untouched. DESIGN proposals: none.
- Residuals: full adoption list (ANN, D, PLR2004-full, PT, S, PERF,
  N) still dark; PERF unblocks when the supervisor sites +
  worker/test sites clear under their owners.

## Progress log (2026-10-01, verification-only pass — re-probe counts)

- Read-only probes live in-container (`voyage:latest`, CPU-only, no host
  pip — no source edits for 031): `ruff check --select ALL --statistics .`
  top rows: D103 889, COM812 825, PLC0415 767, SLF001 583, TRY003 509,
  PLR2004 419, EM102 374, ANN401 345 (full capture in run output;
  drifted with tree growth, consistent with prior passes).
- `ruff check --select PERF --output-format concise .` → **10 hits**
  (6 PERF401 + 4 PERF203): `tests/test_issue_citation_gate.py:55`,
  `tests/test_tui_app.py:748`, `voyage/cli_validate.py:223,229`,
  `voyage/concepts.py:437`, `voyage/models_ensure.py:226`,
  `voyage/supervisor.py:499,571` (banned file),
  `voyage/workers/director.py:390,517` (foreign). Down 11→10 vs batch 13:
  the `video_longlive.py` site vanished with the 079 file deletion, and
  the batch-12 `cli_observe.py` handoff holds (no `cli_observe.py` hit).
- `ruff check --select N --statistics` → **51** (N806 38 + N801 10 +
  N802 2 + N818 1; was 60 — the deleted longlive test modules carried
  the retired test-double names). `ruff check --select PT --statistics`
  → **111** (PT011 63 + PT018 46 + PT013 1 + PT012 1; was 112 — one
  PT017 gone with the deleted modules).
- No family is green in isolation; every remaining PERF site sits in a
  test-track / foreign / banned file. No `select` change, no
  per-file-ignores added.

## Resolution (2026-10-01, verification-only pass)

- Verdict: DEFERRED (record-only) — PERF 10, N 51, PT 111 as-read above.
- Files changed: none for 031 (this issue file only). Gate evidence:
  `select` untouched (`Voyage/pyproject.toml:70` confirmed 16 families).
  DESIGN proposals: none.
- Residuals: full adoption list (ANN, D, PLR2004-full, PT, S, PERF, N)
  still dark; PERF unblocks when the supervisor sites + worker/test
  sites clear under their owners.

## Progress log (2026-10-01, re-probe pass — fresh counts, no adoption)

- Pre-flight: `git diff --name-only HEAD -- comfy/Voyage/voyage/
  comfy/Voyage/tests/ comfy/Voyage/pyproject.toml` EMPTY (only
  pre-existing `comfy/Voyage/LTX2.md` modified at repo root, out of
  scope) — tree clean, but adoption still needs EVERY site in an
  owned-clean file, and none qualifies (below).
- Fresh counts live in-container (`voyage:latest` image 2026-09-30,
  CPU-only, `docker run --rm -v $PWD:/app -w /app`, no host pip):
  `ruff check --select PERF --output-format concise .` → **10 hits**
  (6 PERF401 + 4 PERF203): `tests/test_issue_citation_gate.py:55`,
  `tests/test_tui_app.py:748` (both test-track owned),
  `voyage/cli_validate.py:223,229`, `voyage/concepts.py:437`,
  `voyage/models_ensure.py:226`, `voyage/supervisor.py:502,574`
  (banned file), `voyage/workers/director.py:390,517` (foreign).
  The batch-12 `cli_observe.py` handoff holds (no hit remains).
- `ruff check --select N --statistics` → **51** (N806 38 + N801 10 +
  N802 2 + N818 1, unchanged). `ruff check --select PT --statistics`
  → **111** (PT011 63 + PT018 46 + PT013 1 + PT012 1, unchanged).
  `ruff check --select ALL --statistics` top rows: D103 889, COM812
  826, PLC0415 767, SLF001 604, TRY003 509, PLR2004 419, EM102 374,
  ANN401 345 (SLF001 583→604 and COM812 825→826 drifted with tree
  growth; rest match the prior pass).
- Adoption re-checked site-by-site (regions read live, no edits):
  `cli_validate.py:223,229` are conditional appends inside nested
  loops with sibling statements (not clean comprehension targets —
  needs the owner); `concepts.py:437` / `models_ensure.py:226` same
  conditional-append shape in foreign files; supervisor/director
  PERF203 are try-except-in-loop restructures (banned/foreign);
  tests are test-track owned. No owned-clean site exists, so no
  family is green in isolation and no `select` change was made.
- `Voyage/pyproject.toml:70` select confirmed unchanged (16 families).

## Resolution (2026-10-01, re-probe pass)

- Verdict: DEFERRED (record-only) — PERF 10, N 51, PT 111 as-read above.
- Files changed: none for 031 (this issue file only). Gate evidence:
  `select` untouched. DESIGN proposals: none.
- Residuals: full adoption list (ANN, D, PLR2004-full, PT, S, PERF, N)
  still dark; PERF unblocks when the supervisor sites + worker/test
  sites clear under their owners.
