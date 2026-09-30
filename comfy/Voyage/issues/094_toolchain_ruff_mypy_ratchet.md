# 094 — Toolchain ratchet: ruff ALL gap, per-file-ignores burn-down, mypy scope unify

- Severity: LOW (toolchain — policy, not drift)
- Files: `pyproject.toml:51-181` (`[tool.ruff]` 12L, `[tool.ruff.lint]` 17L `select E,F,I,UP,B,A,C4,DTZ,W,BLE,TRY,EM,SIM,RUF100,S101,T201`, `per-file-ignores` 35L ~18 entries, `[tool.mypy]` + 3 overrides 67L), `Voyage/issues/*.md` (never `ruff format` — evidence fences corrupt)
- Area: toolchain — lint/type ratchet

## Description

`select` (~16 families) vs `ALL` (+~35: `D` docstrings, `N` naming, `ANN` annotations beyond mypy strict, `S` bandit beyond S101, `C90/PLR0913` complexity, `PL`, `RUF` beyond RUF100, `PGH/TID/TCH/PERF/RET/ARG/SLF/FBT/Q/INP/YTT/EXE/CP/COM/DJ`). Stance documented (`:64-69` issue 034, no-mass-fix) — gap is policy. `ALL` would flag `getattr` shims (086), bare `except Exception: pass` (`tui.py:168,190,202`, `supervisor.py:373,570,599,1331`), `T201` prints (exempted). `per-file-ignores` (~18: `RUF100` in tui/supervisor/doctor/logrotate/concepts/persistence/video_longlive/video_ltxv/director; `BLE` in tui/supervisor/cli/doctor/models_ensure/loop/director/video_causvid; `SIM105/108/103/117/300`; `S101`; `T201`) is the 034 ratchet — each line dated follow-up, delete entry not rule as passes convert. mypy: `strict` + `warn_unused_ignores` + py310; 3 overrides (tomllib/tomli, heavy torch/transformers/numpy `follow_imports=skip`, config/tui_state `unused-ignore` ratchet); inline ignores dominated by GPU-worker `type: ignore[import-not-found]` (load-bearing under skip) + `UP017 datetime.UTC` noqas (deliberate py310) + `PLC0415` lazy + `E402` post-sys.path. Only 087-6's two tomli ignores are stale. Lesson (do not regress): NEVER `ruff format issues/*.md` — ruff 0.16.9 rewrites md python fences; fix is top-level `exclude=["issues/"]` (already), not reformatting evidence.

## Rationale

Ratchet without ownership re-grows; scope in 3 scripts drifts; md-format incident recurs without the pointer here.

## Live evidence

- `sed -n '51,181p' pyproject.toml`; `grep -c "per-file-ignores" -A 40 pyproject.toml`
- `grep -rn "noqa: BLE001\|type: ignore" voyage/ | head -n 20`

## Repro

```bash
ruff check voyage/ --select ALL --statistics | head -n 20  # gap measure only, do NOT mass-fix
grep -n "RUF100\|BLE\|SIM\|S101\|T201" pyproject.toml
```

## Fix candidates

1. Keep `select`, burn down `per-file-ignores` oldest-first (RUF100 + BLE cheapest); each deletion its own owning-pass commit.
2. Unify mypy invocations behind `scripts/mypy-scope.sh` (see 089/090); delete 2 stale tomli ignores + override block (keep fallback import).
3. Add 034 lesson pointer (no-md-format) to scripts headers.
4. Gate: `gates.sh` green + `ruff check` + `ruff format --check` clean at current select.

## Refs

- Issues 031 (select gap), 032 (ignores debt), 033 (mypy gate), 034 (stale ignores); AGENTS.md §12 toolchain

## Progress log (2026-09-30, toolchain track — the pass that implements this policy)

- As-read baselines re-verified live (all drifted up since filing):
  ALL-stats D103 797 / COM812 703 / PLC0415 576 / TRY003 461 / SLF001 457
  / PLR2004 387 / ANN401 326; `mypy tests` 140 errors in 38 files
  (124 checked); Any in voyage/ 387; per-file-ignores block unchanged.
- What this pass ratcheted (each step is its own regression tripwire —
  that is the policy working, not overhead):
  1. `select` untouched (031: no adoption family green; counts
     re-baselined, retry order PERF → N → PT documented).
  2. per-file-ignores burn-down, one entry (032: console.py T201 —
     stale via ruff 0.16.9 `file=` exemption; removal itself trips
     future bare-print regressions).
  3. mypy gate 4 → 82 test modules (033: 78 newly gated, exact-command
     verification `Success: no issues found in 133 source files`, set
     equality proof; untracked/in-flight + 150-owned + error files
     excluded with reasons).
  4. Stale-ignore atomic fix (034: 2 tomli shims + override block;
     isolated-tree probe first, then live-tree gate verification;
     `warn_unused_ignores` now guards both files).
  5. ANN401 tripwire stays out (035: 326 hits; alias migration
     proposed for the owning pass).
  6. DESIGN refs completed top-level + gated host-side fail-fast
     (037: 7 docstrings, sweep clean, E501 tripwire fired mid-pass).
  7. PLR2004 top-level slice converted + scoped gate check (038: 15
     hits → 10 named constants in 4 files; full select stays out).
  8. Coverage ratchet (041: see that file — measurement in flight at
     write time).
- Incidents that become policy (do not regress): (a) never `ruff
  format` issues/*.md (existing exclude honored — all 9 files edited
  via exact-string appends only); (b) mount-root discipline for probes
  (`-v $PWD:/app -w /app` from the Voyage dir — one all-zero round
  discarded); (c) override-block surgery must target the exact block
  (rindex picked the heavy-deps block first — caught by probe noise);
  (d) one-liner docstring refs trip E501 (wrap to two lines);
  (e) gate-context verification beats full-tree verification
  (test_repaint_similarity_gate passed `mypy tests`, failed the gate
  invocation — concurrent edit mid-flight); (f) concurrent tracks own
  their hunks (media.py F401, cli-split I001/SIM300/format drift,
  pytest failures — all verified foreign via blame/diff, none
  touched); (g) untracked files never enter the gate.

## Resolution (ratchet policy, binding until superseded)

- Ratchet without ownership re-grows: every future toolchain change
  follows this pass's shape — re-verify premises live (counts drift),
  convert the smallest green slice (one entry / one file / one family),
  verify in the EXACT gate configuration (not a proxy), make the
  conversion itself the regression tripwire (no separate test files —
  test structure belongs to 039/040/086/088/089), and record as-read
  numbers in the owning issue file.
- Scope fences that held: pyproject.toml + scripts/gates.sh +
  voyage/*.py docstrings/constants only; tests/ via gate-membership
  only; no commits; no 000_INDEX.md/DESIGN.md/AGENTS.md edits; no
  foreign-hunk contact (re-read + blame before every edit).
- Gate delta of this pass: +78 mypy test modules, +1 scoped PLR2004
  check (4 files), +1 host-side DESIGN-ref check, −1 per-file-ignore
  entry, −1 mypy override block; `select` and `fail_under` per their
  own issues (031/041).
- Files changed: pyproject.toml, scripts/gates.sh, 11 voyage/*.py
  files (7 docstrings, 4 constant extractions, 2 shim deletions —
  config.py carries both kinds), 9 issue files (this log pattern).
  DESIGN proposals: none.

## Progress log (2026-09-30, re-verification pass)

- Policy text verified present: `## Resolution (ratchet policy,
  binding until superseded)` above is the record — no code needed.
- Live re-verification (in-container `voyage:latest`): `[tool.ruff.lint]
  select` still the 16-family list
  (`E,F,I,UP,B,A,C4,DTZ,W,BLE,TRY,EM,SIM,RUF100,S101,T201`);
  `per-file-ignores` block intact; `exclude = ["issues/"]` intact;
  mypy `strict` + `warn_unused_ignores` + py310 target + tomli /
  heavy-deps overrides intact.
- Gates named by fix candidate 4 (`gates.sh` green + `ruff check` +
  `ruff format --check` clean at current select): `ruff check .` →
  "All checks passed!"; per-file gates on this pass's touched files
  green; full `gates.sh` exits red ONLY on foreign in-flight files
  (untracked `tests/test_augment_contract_166.py` format violation,
  untracked `tests/test_supervisor_proposal_helpers.py` collection
  error) + 3 `test_worker_perf_rank2` load-flakes green in isolation
  — none in toolchain scope, none touched per §9.

## Resolution (2026-09-30, re-verification pass)

- Verdict: closed — the policy text IS the deliverable and it is on
  record; no code changed. Files changed: none (this appendix only).
  DESIGN proposals: none.
- Residuals: none in this issue's scope; the two foreign
  gate-blockers belong to their owners (augment-contract format,
  supervisor-proposal migration).
