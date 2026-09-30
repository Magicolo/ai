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
