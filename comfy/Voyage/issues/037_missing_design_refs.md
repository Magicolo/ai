# 037 — Seven modules lack DESIGN § refs in their docstrings (§12 violation)

- Severity: LOW
- Files: `voyage/augment.py:1`, `voyage/bench.py:1`, `voyage/hashing.py:1`, `voyage/__init__.py:1`, `voyage/models_ensure.py:1`, `voyage/sfx_finalize.py:1`, `voyage/tui.py:1`
- Area: standards / docs
- Overlaps with: 091 (docs SFX/augment follow-ups may add the missing refs as a side effect — not a duplicate)

## Description

AGENTS.md §12 requires "every module docstring states purpose + DESIGN §
ref". All 42 modules have a docstring, but the first-15-lines scan shows
these 7 carry no `DESIGN` reference (contrast `cli.py:1` "DESIGN §§58, J",
`rpc.py:1` "DESIGN §§45-46, D", `seeds.py:1` "DESIGN §62",
`supervisor.py:1` "DESIGN §73"). `augment.py` additionally carries a
`TODO (unify)` in shipped code.

## Rationale

DESIGN refs are the project's traceability mechanism (gates comments cite
§83/§87/§92). Unref'd modules rot first — reviewers cannot tell which spec
section governs them.

## Live evidence (re-verified 2026-09-30)

Host scan today (`head -n 15` per file):

- `voyage/augment.py:1` — "GPU augment runner orchestration … (Track D
  spike). Pure orchestration — stdlib only, never torch (supervisor section
  12 GPU ban) … TODO (unify): if media.py ever gains chunk/frame-count plan
  math, move `interpolated_frame_count` there…" — no `DESIGN` token.
- `voyage/bench.py:1` — "Benchmark + soak helpers: timing stats, §104
  reports…" — bare `§104`, no `DESIGN` token.
- `voyage/hashing.py:1` — "Shared SHA-256 helpers (issue 021)…" — no
  `DESIGN` token.
- `voyage/__init__.py:1` — "Voyage — autonomous infinite audiovisual voyage
  (Phase 0 skeleton)…" — no `DESIGN` token.
- `voyage/models_ensure.py:1` — "Inline model ensure for `voyage generate`
  (selective + parallel)…" — no `DESIGN` token.
- `voyage/sfx_finalize.py:1` — "Finalize-time SFX pass (slice 3,
  three-caption doctrine)…" — no `DESIGN` token.
- `voyage/tui.py:1` — "Interactive launcher TUI…" — no `DESIGN` token in the
  first 15 lines.

Track-C sweep loop (preserved): `for f in voyage/*.py; do head -n 15 "$f" |
grep -q DESIGN || echo "NO-DESIGN-REF: $f"; done` → the same 7.

## Repro

```bash
for f in voyage/*.py; do head -n 15 "$f" | grep -q DESIGN || echo "NO-DESIGN-REF: $f"; done
rg -n "TODO|FIXME" voyage/
```

## Fix candidates

One-line docstring amendments pinning each module to its DESIGN section +
converting the `augment.py` TODO into a tracked follow-up or deleting it.

## Refs

- AGENTS.md §12 ("every module docstring states purpose + DESIGN § ref").
- Preserved track result: `ses_f10013fc5ffeLLDtqZEwFbf3JR`, §7.

## Progress log (2026-09-30, toolchain track)

- Scope check vs 151: 151 owns workers/ + audio/ subpackages — this
  track touched `voyage/*.py` top level only. Re-ran 037's sweep
  (`for f in voyage/*.py; do head -n 15 "$f" | grep -q DESIGN ...`):
  the same 7 files failed before the fix, **zero after** (sweep clean).
- One-line docstring amendments (traced to live DESIGN sections, read
  not edited): augment.py → §§56-57 + §140
  finalize-augmentation-floors as-built (§140 entry 2026-09-30; the
  module feeds the finalize path); bench.py → §104 (Benchmarking
  protocol — docstring already said "§104", added the DESIGN prefix);
  hashing.py → §§29-30, 56 (segment-dir/commit checksums the module's
  sha256_file serves: users in media.py/supervisor.py/workers);
  __init__.py → §§11, 83 (package root; docstring states the §83 GPU
  ban); models_ensure.py → §§84-85 + §140 generate-ensure as-built
  (§140 entry 2026-09-29); sfx_finalize.py → §7 + §56 + §140 SFX-slice
  as-built (three-caption doctrine, §140 entry 2026-09-29);
  tui.py → §140 launcher-TUI as-built (mirrors tui_state.py:14-17,
  the already-conforming sibling).
- E501 tripwire fired mid-pass (TDD working as designed): the first
  one-liner form pushed 5 docstring openers past line-length 100 —
  caught immediately by in-container `ruff check`, rewrapped to
  two-line form, re-verified.
- The augment.py `TODO (unify)` stays (residual): a concurrent agent
  is mid-flight on exactly that unification (media.py just gained
  `from voyage.augment import interpolated_frame_count`, currently
  F401-unused — their in-progress state, not touched here). Deleting
  the TODO while their migration is half-landed would destroy the
  pointer; converting it to a follow-up file risks numbering collision
  with concurrent agents. Revisit once their migration lands.
- TDD/regression: added a host-side DESIGN-ref check to `scripts/gates.sh`
  (fail-fast before the build): loops `voyage/*.py`, greps first 15
  lines for DESIGN, exits 1 listing offenders. Top-level only, so
  151's subpackage scope can never trip it. Verified: snippet passes
  on the fixed tree; `bash -n` clean.

## Resolution

- All 7 modules now carry DESIGN refs; the sweep is clean; the gate
  enforces it (new regressions fail gates.sh before the build).
- Files changed: the 7 docstrings (one logical line each) +
  `scripts/gates.sh` (host-side check + comment). Gate evidence:
  sweep output empty; in-container `ruff check` + `ruff format
  --check` green on all 7 files; `bash -n` green; snippet logic
  traced (grep-fail is always `||`-guarded, substitution always exits
  0, `[ -n ]` carries the verdict — set -e safe). DESIGN proposals:
  none (this change only points at existing sections). Residuals:
  augment.py TODO (concurrent migration in flight); subpackages belong
  to 151.

### Postscript (same day, tripwire's first catch)

- The new gates.sh DESIGN check fired within the hour on
  `voyage/cli_core.py` (concurrent issue-080 cli-split migration,
  created after this fix): its docstring describes the split but
  carries no DESIGN token, while its ten siblings all comply. Left
  for the owning track — one-line fix: append `(DESIGN §58, task
  group J)` (mirroring `cli.py:1`) to the docstring opener. This is
  the tripwire working as designed, not a regression from this pass.

### Closure (batch-7 commit, same day)

- Owning track (batch 7, issue 080 scope) applied the one-line fix:
  `cli_core.py:1` opener now reads `DESIGN §58 — leaf module of the
  issue-080 split …`. Full `gates.sh` green after the fix.
