# 043 — TUI swallows everything: 13 bare `except Exception` + silent fallbacks (failure paths untested)

- Status: open
- Severity: medium (untested failure modes in the launcher; corrupt-state
  handling asserts nothing about *which* corruption)
- Area: standards/UX — `voyage/tui.py`, `voyage/tui_state.py:25`
- Rank rationale: each swallow is defensible alone (launcher must not crash) but
  collectively they hide 8+ untested paths; draw-path coverage is good, failure
  coverage is not.

## Technical description

`voyage/tui.py:165-166` (load→defaults), `:186-188` (warning→`""`), `:198-199`
(save fail→`pass`), `:228-230` (yield→post+raise, ok),
`:642-643,652-653,690-691,697-698,709-710,728-729,735-736` (query/focus/refresh→
`pass`/`return`/fallback), plus `voyage/tui_state.py:25`
(`import tomli as tomllib # type: ignore[import-not-found, no-redef]` — py310/312
dual path). Corrupt `~/.config/voyage/tui-last.toml` → silent defaults
(`test_tui_state.py:67` covers corrupt→defaults, but not *which* corruption);
save failure → silent `pass` (covered only as "never raises",
`test_tui_state.py:119`); 8× UI-refresh swallows have no failing-widget tests.
Only `tui.py:791,818` use `as exc`. Draw-path coverage itself is good
(`test_tui_app.py:27` via `run_test()` Pilot, `test_tui.py:140` headless
fake-backend run); CLI surface (`cli.py` 30+ `print`s, `benchmark/soak/inspect/
scoreboard` verbs) is exercised only through a handful of capsys tests
(`test_console:12`, `test_benchmark:9`, `test_scoreboard:3`).

## Why this is an issue

Each swallow is defensible alone — a launcher must not crash — but thirteen of them collectively hide eight-plus untested failure paths, so corrupt configs, failed saves, and dead widgets all degrade silently instead of telling the user what broke. Corrupt-state handling that asserts "falls back to defaults" without pinning *which* corruption was tolerated cannot distinguish a typo from a schema change. The launcher is the first thing new users touch; silent failure there erodes trust before anything renders.

## Evidence

`rg -n -A2 "except Exception" Voyage/voyage/tui.py | head -n 60` (sweep output).

Verified live 2026-09-25 — `rg -n "except Exception" voyage/tui.py` now reports
16 sites (was 13 at sweep time; lines drifted, e.g. :166,187,199,235,685,695,
733,740,752,771,778,834,875,894,902,914). The in-text line numbers above are
stale by a few lines each; the site count grew. `tui_state.py:25` tomli shim and
`test_tui_state.py:67,119` refs confirmed current.

## Reproduction

Corrupt `tui-last.toml` in each distinct way (bad TOML, wrong types, unknown
backend — see 024's causvid fallback); kill individual widget queries in a Pilot
test.

## Source references

- `voyage/tui.py` lines above; `voyage/tui_state.py:25`;
  `Voyage/tests/test_tui_state.py:67,119`, `test_tui_app.py:27`.

## Resolution candidates

Select `BLE+TRY+EM` (see 034); convert swallows to `except (AttributeError,
KeyError, ...)` or `log.debug(..., exc_info=True)`; add failing-widget Pilot
tests (query raises → fallback path asserts the warning line).

## Investigation / progress / resolution log

- 2026-09-25: found by standards sweep.
- 2026-09-25: repair pass — added `## Why this is an issue`; re-verified live:
  `tui.py` `except Exception` count is now 16 (was 13) and all in-text line
  numbers drifted a few lines — in-text enumeration left as-is (historical),
  fresh `rg -n` output pasted into Evidence; `tui_state.py:25` +
  `test_tui_state.py:67,119` refs current.
- Open: narrow the excepts + add failure-path Pilot tests.
