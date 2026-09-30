# 184 — DESIGN §140 frontpage entry inventories the TUI form without the two augment-floor fields

- **Severity:** LOW (docs — the handoff record new agents read first; stale inventory misdirects the next form change, none affects runtime)
- **Track:** E (§140/DESIGN drift — below 133/135)
- **Verified live:** 2026-09-30 host read (tree as-read; no edits in these ranges)

## File:line (live-verified)

- `Voyage/DESIGN.md:6865` (frontpage-rework entry: "Style (`TextArea` h3), Name, Duration, Backend/Director/Quantization `Select`s, Blocks, Take seconds, Beats, Drift, Seed, 5 checkboxes; run-id + output + final-video merged into a single `name` ...")
- `voyage/tui.py:638-657` (live form rows: `Min fps` at `:638-647` with `tooltip=FIELD_HELP["min_fps"]`, `Min resolution` at `:648-657` with `tooltip=FIELD_HELP["min_resolution"]` — both between Drift and Seed)
- `voyage/tui_state.py:66-86` (`FIELD_HELP`: `min_fps` at `:83`, `min_resolution` at `:84-85`), `:112-113` (`GenerateFormState.min_fps/min_resolution` defaults `"32"`/`"1280x720"`), `:226-243` (validation)
- `Voyage/tests/test_tui_app.py` (field-row count/geometry tests enumerate the live rows including the two floors)

## Description

The §140 frontpage-rework entry inventories every form row — Style, Name, Duration, three Selects, Blocks, Take seconds, Beats, Drift, Seed, 5 checkboxes — and never mentions Min fps / Min resolution. Live the form has both (wired end to end: help keys, state defaults matching the `[augment]` TOML defaults, per-field validation, namespace pass-through at `tui_state.py:297-298`, `_read_form` reads at `tui.py:776-777`). The entry postdates neither: the augment-floor slice (Track A) landed after the frontpage rework and updated the code + tests + `FIELD_HELP` but never amended the §140 inventory sentence two entries above it. Any agent scoping "add a field to the form" from the entry undercounts the rows (and misses that Drift↔Seed are no longer adjacent).

## Rationale

- §140 is the chronological handoff log — its entries are read as "what the form looks like as of this date". An inventory sentence that silently predates two rows is precisely the drift 133 was chartered to catch (its 17-claim sweep verified console flags, verb counts, and augment wiring, but never the TUI field inventory).
- The fix is one clause; the cost of leaving it is a miscounted form on the next refactor-the-row-list task (row-count/geometry tests assert exact heights per row).
- Docs-only: no runtime behavior changes, no test changes.

## Live evidence (host read, 2026-09-30)

```
DESIGN.md:6865:
  "... Backend/Director/Quantization `Select`s, Blocks,
  Take seconds, Beats, Drift, Seed, 5 checkboxes; run-id + output + ..."

voyage/tui.py:638-657 (live — two rows the entry omits):
  638:             "min_fps",
  641:             Input(... id="field-min-fps", tooltip=FIELD_HELP["min_fps"],),
  648:             "Min resolution",
  655:             Input(... id="field-min-resolution", tooltip=FIELD_HELP["min_resolution"],),

$ rg -c "field-min-fps|field-min-resolution" Voyage/voyage/tui.py Voyage/tests/test_tui_app.py
Voyage/voyage/tui.py:4
Voyage/tests/test_tui_app.py:0  # (ids referenced via FIELD_WIDGET_IDS mapping, not literals)
```

`FIELD_WIDGET_IDS` (`tui.py:84-98`) lists `min_fps`/`min_resolution` at `:95-96` — the mapping the entry predates.

## Repro

```bash
sed -n '6860,6870p' Voyage/DESIGN.md  # field inventory without the floors
sed -n '638,657p' Voyage/voyage/tui.py  # the two live rows
rg -n "min_fps|Min fps" Voyage/voyage/tui_state.py | head -8
```

## Fix candidates

1. Amend the `:6865` inventory clause to "... Beats, Drift, Min fps, Min resolution, Seed, 5 checkboxes ..." with a dated note (augment-floor slice added the two rows after this entry).
2. Optionally point at `FIELD_WIDGET_IDS` (`tui.py:84-98`) as the live inventory single source so the next row addition updates one mapping, not prose.
3. No test change (row-count tests already enumerate the live form).

## Refs

- `Voyage/DESIGN.md:6865` vs `voyage/tui.py:638-657,84-98` + `voyage/tui_state.py:83-85,112-113,226-243`.
- Not-a-duplicate: 133 (17 claims incl. TUI `min_fps` help/validation as VERIFIED-HOLD — inventory prose never checked); 135 (§5.3 geometry addendum — different section); 042 (README geometry vs floors — different doc); 159/160 (BACKENDS/OPERATIONS tables — different files).

## Progress log (Group C, 2026-09-30)

- Verdict: CONFIRMED live (DESIGN.md not owned — verify-only, proposal
  below as quoted text). Live numbers drifted from the filing: the
  inventory sentence is now at `DESIGN.md:7136` ("... Blocks, Take
  seconds, Beats, Drift, Seed, 5 checkboxes; run-id + output + ...");
  the two live rows are at `voyage/tui.py:638-657` (`Min fps` with
  `FIELD_HELP["min_fps"]`, `Min resolution` with
  `FIELD_HELP["min_resolution"]`), state defaults at
  `tui_state.py:112-113` (`"32"`/`"1280x720"`). Omission confirmed.
- Adjacent fix in owned docs: the same omission existed in
  `docs/OPERATIONS.md:96` (form-fields list) — amended to "Blocks, Take
  seconds, Beats, Drift, Min fps, Min resolution, Seed ..." with the
  floor defaults noted.
- Files changed: `docs/OPERATIONS.md` (DESIGN.md untouched).

## Resolution

- OPERATIONS form-field list fixed. Proposed DESIGN text for the owner:
  "Amend the `:7136` inventory clause to '... Beats, Drift, Min fps, Min
  resolution, Seed, 5 checkboxes ...' with a dated note (augment-floor
  slice added the two rows after this entry). Optionally point at
  `FIELD_WIDGET_IDS` (`tui.py:84-98`) as the live inventory single
  source."
- Residual: none in this file's scope (no test change — row-count tests
  already enumerate the live form).
