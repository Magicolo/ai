# 116 — Run-name whitespace: TUI strips, CLI does not — same input produces different run dirs

- **Severity:** LOW (cosmetic divergence today; same input → different on-disk state depending on surface)
- **Track:** second-pass TUI/CLI edges (generate-vs-run/TUI-vs-CLI resolution parity)
- **Verified live:** 2026-09-30 in-container (`voyage:latest`; lines as-read)

## File:line (live-verified)

- `voyage/tui_state.py:271` (`to_generate_namespace`: `name = state.name.strip()` — output, run_id, final_video all derive from the stripped value)
- `voyage/cli.py:127-132` (`_effective_run_id`: returns `args.name` verbatim when non-empty — no strip)
- `voyage/cli.py:108-113` (`is_flat_folder_name` validates `value.strip()` — so `" foo "` passes validation) then `cmd_init`/`cmd_generate` use the UNstripped id for `output/<id>/` and the stored charter (`:169-184`, `:1209-1225`)

## Description

`is_flat_folder_name(" foo ")` is True (it strips for the check), so the CLI accepts a padded name and then uses it verbatim: `generate --name " foo "` creates `output/ foo /` and charters the run `" foo "`. The TUI strips the same input to `foo` before building the namespace. Both surfaces accept; they produce different directories, different `run_id`s, and different `final.mp4` paths for byte-identical keystrokes. All-whitespace (`" "`) is rejected on both (strip → empty → invalid), so the divergence is confined to leading/trailing padding — typically pasted, not typed.

## Rationale

- Two frontends for one command should agree on name normalization; the check (`is_flat_folder_name`) and the use (`_effective_run_id`) disagree on whether whitespace is significant, and the TUI picked the third behavior (strip). Whichever is canonical, it should hold in all three places.
- Practical edge: a padded CLI name creates a directory with spaces that later `--run` references must quote exactly; tab-completion and scripts silently diverge from the TUI-created twin.

## Live evidence (container, 2026-09-30)

```
_effective_run_id(Namespace(name=' foo ', run_id='voyage'))  -> ' foo '   # CLI: kept
to_generate_namespace(GenerateFormState(style='x', name=' foo ')).name   -> 'foo'  # TUI: stripped
```

## Repro

```bash
docker run --rm -v "$PWD/Voyage:/app" -w /app voyage:latest python3 -c "
import argparse
from voyage import cli
from voyage.tui_state import GenerateFormState, to_generate_namespace
print(repr(cli._effective_run_id(argparse.Namespace(name=' foo ', run_id='voyage'))))
print(repr(to_generate_namespace(GenerateFormState(style='x', name=' foo ')).name))
"
```

## Fix candidates

- Strip in `_effective_run_id` (one line; makes check-use-TUI all agree). Residual risk: a stored run chartered `" foo "` (pre-fix) vs post-fix lookup `foo` — document as a one-way rename, or strip at `cmd_init`/`cmd_generate` output derivation instead of the accessor. Either is a 1-line change + test.
- Test: padded name through both surfaces resolves to the same run dir and charter id.

## Refs

- Issue 008 (traversal guard — the check this asymmetry sits inside); issue 023 (TUI/CLI absent-vs-default parity — same family of "two surfaces, two behaviors" findings).

## Progress log

- 2026-09-30 (Group A): premise re-verified live —
  `_effective_run_id(name=' foo ')` → `' foo '` while the TUI produced
  `'foo'`.
- Wrote failing test first
  (`tests/test_cli_group_a.py::test_effective_run_id_strips_padding_on_both_surfaces`):
  red on the CLI side.
- Fixed with the one-line candidate, plus missing-attribute tolerance.

## Resolution: FIXED

- `voyage/cli_paths.py:70-84` (`_effective_run_id`): strips the winning
  value, so check (`is_flat_folder_name`, which strips), use, and TUI
  all agree — `" boba "` lands in `output/boba/` on both surfaces.
  Also tolerates a missing `run_id` attribute (returns `""` so
  `_check_run_id` reports exit 2 instead of `AttributeError` — half of
  185's missing-attribute class closed here).
- Pre-existing chartered `" foo "` runs resolve post-fix to `foo`:
  accepted one-way rename, documented here (no migration — padded
  charters were never reachable from the TUI).
- Test evidence: CLI+TUI parity pinned in the new test;
  `test_generate_name_*` routing tests still green.
- Gates: `ruff check` + `ruff format --check` + `mypy strict` green.
- Residuals: none.
