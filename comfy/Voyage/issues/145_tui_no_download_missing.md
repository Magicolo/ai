# 145 — TUI `to_generate_namespace` drops `--no-download`: TUI runs can never verify-only

- Severity: LOW (TUI always allows model downloads; no offline/verify-only path)
- Area: config/CLI/TUI — namespace parity
- Files (as-read 2026-09-30; concurrent uncommitted edits in `voyage/tui_state.py`, `voyage/cli.py` — lines as-read):

## File:line

- `voyage/tui_state.py:252-310` (`to_generate_namespace`: builds the `cmd_generate` namespace; keys are backend/duration/style/run_id/name/output/seed/force/final_video/skip_bad/draft/director/blocks/take_seconds/quantization/beats_per_segment/drift_every_n/min_fps/min_resolution/no_augment/verbose/no_color/no_sfx/sfx_* — NO `no_download`)
- `voyage/tui_state.py:98-115` (`GenerateFormState` defaults: backend/duration/style/name/seed/force/skip_bad/draft/director/blocks/take_seconds/quantization/beats_per_segment/drift_every_n/min_fps/min_resolution/verbose/no_color — NO download field)
- `voyage/cli.py:1912-1916` (the `--no-download` flag on the generate parser: "fail instead of downloading missing models (verify only)")
- `voyage/cli.py:1260-1280` (`cmd_generate`: `init_args` build + `cmd_init` + `_load_run` + `_require_cuda_stack` + full-stack comment)

## Description

The `generate` parser offers `--no-download` (`cli.py:1912-1916`: "fail instead of downloading missing models (verify only)"), and `cmd_generate` honors it at the ensure gate via `allow_download=not bool(getattr(args, "no_download", False))` (`cli.py:1334`). The TUI namespace built at `tui_state.py:273-310` never sets `no_download` — and the form at :98-115 has no corresponding field. `getattr(..., False)` then silently resolves the missing attribute to "downloads allowed", so every TUI Generate unconditionally permits downloads. There is no way to do a verify-only / offline run from the TUI, and the divergence is invisible (no error, no warning — the flag just doesn't exist on that path).

## Rationale

- Namespace parity is the TUI contract: every `cmd_generate`-meaningful flag the CLI exposes should either be represented in the form or explicitly defaulted in `to_generate_namespace` with a comment. `no_sfx`/`sfx_*` ARE explicitly defaulted (:304-309); `no_download` is simply absent.
- The `getattr(..., False)` defensive read (correct for hand-built test namespaces) masks the gap — the TUI path looks intentional when it is actually an omission.
- Low severity only because auto-download is the desired default; the missing piece is the opt-OUT for metered/offline boxes.

## Live evidence

```
tui_state.py:273-310  Namespace(... no_augment=False, verbose=..., no_color=...,
                       no_sfx=False, sfx_backend=None, ... sfx_workers=1)  # no no_download
cli.py:1912-1916  gen.add_argument("--no-download", action="store_true", ...)
cli.py:1334  allow_download=not bool(getattr(args, "no_download", False)),
```

`grep no_download voyage/` hits only `cli.py:1334` (read) + the parser flag — nothing in `tui_state.py`/`tui.py`.

## Repro

```bash
docker run --rm -v "$PWD/Voyage:/app" -w /app voyage:latest python3 -c "
from voyage.tui_state import GenerateFormState, to_generate_namespace
ns = to_generate_namespace(GenerateFormState(style='x', name='t'))
print('no_download' in vars(ns), getattr(ns, 'no_download', '<MISSING→False>'))
"
# Expect: False <MISSING→False> — indistinguishable from an explicit allow.
```

## Fix candidates

- Add an explicit `no_download=False` (+ comment, mirroring the `no_sfx` block) if the TUI intentionally always allows downloads — turns a silent gap into a documented default.
- Better: add a form checkbox (default off) + validation + namespace pass-through, so offline users get the CLI's verify-only path.
- Test: TUI namespace carries the documented value; offline run with missing models fails instead of downloading.

## Refs

- `voyage/tui_state.py:252-310`; `voyage/cli.py:1912-1916` + `:1334`; `_add_sfx_args` explicit-default precedent.

## Progress log

- 2026-09-30 (Group A): premise re-verified live — `no_download`
  absent from the TUI namespace, `getattr(..., False)` silently
  allowing downloads.
- Wrote failing test first
  (`tests/test_cli_group_a.py::test_tui_namespace_carries_no_download_opt_out`):
  red (`TypeError: unexpected keyword argument 'no_download'`).
- Fixed with the "Better" candidate (real checkbox, not just an
  explicit default), jointly with 182's opt-out.

## Resolution: FIXED

- `voyage/tui_state.py`: new `GenerateFormState.no_download`
  (default `False`), `FIELD_HELP["no_download"]`, save/load round-trip
  entries, and `to_generate_namespace` pass-through (`:110`, `:156`,
  `:358`, `:529`, `:594`).
- `voyage/tui.py`: new `flag-no-download` checkbox
  (`Verify only (fail instead of downloading models)`) wired in
  `_form_fields` + `_read_form` (`:684-689`, `:794`).
- The Pilot-pinned seam test (`tests/test_cli_tui_split.py`, in
  scope) updated: 7 flags, count 25, both new ids pinned.
- Test evidence: default-False + True-passthrough pinned; save/load
  round-trip suite still green.
- Gates: `ruff check` + `ruff format --check` + `mypy strict` green.
- Residuals: none.
