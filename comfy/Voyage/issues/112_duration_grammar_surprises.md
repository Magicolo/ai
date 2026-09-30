# 112 — `parse_duration` grammar surprises: per-component signs do subtractive arithmetic, bare trailing numbers bind to seconds, interior whitespace rejected

- **Severity:** LOW-MEDIUM (silent mis-parse of plausible human input; the worst case understates the run by 30 minutes with exit 0)
- **Track:** second-pass TUI/CLI edges (`cli.py` duration math)
- **Verified live:** 2026-09-30 in-container (`voyage:latest`; lines as-read)

## File:line (live-verified)

- `voyage/cli.py:1108` (`_DURATION_PATTERN`: each of hours/minutes/seconds carries its own `-?`, trailing `s?` makes the seconds unit optional)
- `voyage/cli.py:1115-1135` (`parse_duration`: sums `float(value) * scale` per component; only the TOTAL is positivity-checked)
- Consumers: `cli --duration` (`:1831-1836`, `type=parse_duration`), TUI `field_errors`/`plan_counts`/`plan_summary` (`tui_state.py:178, 365-380, 396-416`) — one grammar, both surfaces

## Description

Three grammar behaviors, all verified live, none documented in `--help` (which lists only `_DURATION_EXAMPLES` = `'5s', '90', '1m30s', '2m', '1h', '1h2m3.5s'`):

1. **Subtractive durations.** Each component accepts its own sign, but only the total is range-checked. `2m-30s` → 90.0, `1h-30m` → 1800.0. The docstring says signs exist "so negatives reach the positivity error below instead of the regex error" — true for all-negative input (`-5s` → positivity error), but mixed-sign input performs silent subtraction no human intends. `1h-30m` reads as a typo for `1h30m` and yields half the intended length.
2. **Bare trailing numbers bind to seconds, not to the next unit down.** `1m30` → 90.0 (correct by luck), but `1h30` → 3630.0 — one hour plus thirty SECONDS. A user typing `1h30` for ninety minutes gets 60.5 minutes: a 30-minute shortfall with exit 0, planned/announced as `~60.5s`-style output the user has no reason to distrust. `segments_for_duration` then rounds UP, so the video never runs short of the (wrong) plan — the error is self-consistent and invisible.
3. **Interior whitespace is rejected** (`1m 30s` → `invalid duration`), while leading/trailing whitespace is stripped. Rejecting is defensible, but the error message lists examples without stating the no-spaces rule, so a user pasting `1h 30m` from a chat message retries blind.

## Rationale

- Duration is the one CLI/TUI input that scales cost (segments × GPU minutes). A mis-parse that shortens the run wastes a planning cycle; one that lengthens it (`2m-30s` typed for `2m30s` shortens — but `-` in the wrong place can also ADD: `1h+-30m`? rejected; still, sign arithmetic is undirected) wastes GPU budget. Either direction violates "never runs short" expectations asymmetrically.
- The TUI amplifies it: `plan_summary` derives segments/frames/seconds from the same parse, so the live plan solemnly confirms the wrong duration.
- Fix cost is trivial (reject any `-` unless it is the first character; either reject bare trailing digits after a unit or document the seconds-binding loudly). Only 022 and 080 mention `parse_duration` in-tree (forwarding and epsilon-literal notes) — the grammar itself is uncovered.

## Live evidence (container, 2026-09-30)

```
'2m-30s'  -> 90.0      # 120 - 30, silent subtraction
'1h-30m'  -> 1800.0    # 3600 - 1800, half the likely intent
'1h30'    -> 3630.0    # 3600 + 30s, not 5400
'1m30'    -> 90.0      # correct by accident (seconds-binding happens to match)
'1m 30s'  -> ValueError: invalid duration '1m 30s' (examples: '5s', '90', '1m30s', '2m', '1h', '1h2m3.5s')
'1h2m3.5s'-> 3723.5    # canonical form fine
```

## Repro

```bash
docker run --rm -v "$PWD/Voyage:/app" -w /app voyage:latest python3 -c "
from voyage.cli import parse_duration
print(parse_duration('1h30'))    # 3630.0 — expect 5400 if you meant 1h30m
print(parse_duration('2m-30s'))  # 90.0 — sign arithmetic, not a typo error
"
```

## Fix candidates

- Reject component signs except a single leading `-` (preserve the documented "negatives reach the positivity error" behavior: `-5s`, `-1h30m` → positivity error; `1h-30m`, `2m-30s` → invalid-duration error).
- Decide the bare-trailing rule explicitly: either reject `1h30` ("trailing number without a unit after h/m — did you mean 1h30m?") or document seconds-binding in `--help` + `FIELD_HELP["duration"]`. Rejection is safer; the did-you-mean hint keeps it actionable.
- Optionally accept single interior spaces (`1h 30m`) — GNU-style humanizers do; if not, add "no spaces" to the error message alongside the examples.
- Tests: subtractive/mixed-sign, bare-trailing-after-unit, interior-space cases for both `parse_duration` and TUI `field_errors` (same grammar, both surfaces).

## Refs

- `voyage/cli.py:1076-1095` (docstring documents the sign rationale — the mixed-sign hole is its unconsidered consequence); `tui_state.py:69-70` (`FIELD_HELP["duration"]` — examples live here too, update both or single-source them).

## Progress log

- 2026-09-30 (Group A): all three behaviors re-verified live
  (`2m-30s` → 90.0, `1h-30m` → 1800.0, `1h30` → 3630.0,
  `1m 30s` → invalid without naming the rule).
- Wrote failing tests first (three tests: mixed signs, bare trailing,
  interior space): all red.
- Fixed with rejection + actionable hints (no silent re-interpretation
  anywhere; interior spaces stay rejected but named).

## Resolution: FIXED

- `voyage/cli_planning.py:28-78` (`parse_duration`): a sign anywhere
  except a single leading `-` is now `invalid duration (mixed signs are
  not supported; …)` — leading-negative input still reaches the
  positivity error, preserving the documented rationale and the
  existing `-5s` test; a trailing bare number after h/m is now
  `invalid duration (a trailing number after h/m needs its own unit,
  e.g. '1h30m'; …)` with a did-you-mean hint; interior whitespace is
  now `invalid duration (no spaces allowed; …)`.
- One grammar, both surfaces: TUI `field_errors`/`plan_counts` call the
  same function, so the fix covers CLI and TUI with no second edit.
- Test evidence: `test_parse_duration_rejects_mixed_signs`,
  `test_parse_duration_rejects_bare_trailing_number`,
  `test_parse_duration_interior_space_names_the_rule` green alongside
  the pre-existing documented-forms/positivity tests.
- Gates: `ruff check` + `ruff format --check` + `mypy strict` green.
- Residuals: none. (`FIELD_HELP["duration"]` examples remain valid;
  no help-text change needed since nothing newly accepted exists.)
