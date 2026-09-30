# 182 — TUI has no SFX fields: every TUI run finalizes with default SFX, no opt-out, no overrides

- **Severity:** LOW-MEDIUM (namespace parity: six CLI flags collapsed to hardcoded defaults; metered/offline/GPU-poor boxes cannot skip or retarget the SFX pass from the default interface)
- **Track:** B (TUI→CLI namespace parity — below 145/022/154/158)
- **Verified live:** 2026-09-30 in-container (`voyage:latest`, tree as-read; concurrent uncommitted edits in `voyage/tui_state.py`, `voyage/cli.py` — lines as-read)

## File:line (live-verified)

- `voyage/tui_state.py:89-115` (`GenerateFormState`: 18 fields — backend/duration/style/name/seed/force/skip_bad/draft/director/blocks/take_seconds/quantization/beats_per_segment/drift_every_n/min_fps/min_resolution/verbose/no_color — NO `no_sfx`/`sfx_*`)
- `voyage/tui_state.py:302-309` (`to_generate_namespace`: `no_sfx=False, sfx_backend=None, sfx_caption=None, sfx_device=None, sfx_model_size=None, sfx_workers=1` — all hardcoded, mirroring the generate-parser defaults)
- `voyage/tui_state.py:66-86` (`FIELD_HELP`: no `no_sfx`/`sfx_*` keys)
- `voyage/cli.py:1773-1811` (`_add_sfx_args`: six flags — `--no-sfx`, `--sfx-backend`, `--sfx-caption`, `--sfx-device`, `--sfx-model-size`, `--sfx-workers` — on generate/run-finalize/finalize/stop/sfx)
- `voyage/cli.py:1015-1036` (`cmd_finalize`: `if not getattr(args, "no_sfx", False) and sfx_backend != "fake"` — the TUI namespace always takes the render branch with `[sfx]` backend)

## Description

The `generate` parser offers six SFX flags (skip the pass, switch backend, pin the caption, move devices, pick the variant, shard workers). The TUI form offers zero: `to_generate_namespace` hardcodes `no_sfx=False` and default overrides, so every TUI Generate unconditionally runs the finalize-time SFX pass with the run-config backend. Consequences: (a) no offline/verify-only SFX path from the TUI (compounds 145's `--no-download` gap — a TUI run on a metered box downloads MMAudio weights with no way to decline); (b) no single-GPU retarget (a 2060-class box that needs `small_44k`/CPU cannot ask from the TUI); (c) runs committed before SFX captions existed cannot supply `--sfx-caption` from the TUI (the exact case `sfx_caption`'s help names: "required for runs committed before SFX captions existed, e.g. poulah"). The `getattr(..., default)` defensive reads in `cmd_finalize` mask the gap — the TUI path looks intentional when it is an omission (same masking as 145).

## Rationale

- Namespace parity is the TUI contract (145's rationale): every `cmd_generate`-meaningful flag should be represented in the form or explicitly defaulted with a comment. The SFX block IS explicitly defaulted (`:302-309` comment cites "the generate parser defaults") — but unlike `no_download` (absent entirely), the six SFX keys are present-yet-frozen: the namespace satisfies `cmd_finalize` while the user can influence none of it.
- The default interface should not force the heaviest finalize stage: SFX needs MMAudio weights (~13 GB, CC-BY-NC-4.0) + a worker slot; CLI users opt out with one flag, TUI users cannot.
- Severity is LOW-MEDIUM (not HIGH like 022's silent caption drop) because the behavior is at least the documented default, not a contradicted pin — but the missing opt-out bites the same boxes 145 names.

## Live evidence (container, 2026-09-30)

```
$ docker run --rm -v "$PWD:/app" -w /app voyage:latest python3 -c "
import dataclasses
from voyage.tui_state import GenerateFormState, to_generate_namespace
print('form fields:', [f.name for f in dataclasses.fields(GenerateFormState)])
ns = to_generate_namespace(GenerateFormState(style='x', name='t'))
print('no_sfx:', getattr(ns,'no_sfx','<MISSING>'), '| sfx_backend:', ns.sfx_backend,
      '| sfx_caption:', ns.sfx_caption, '| sfx_device:', ns.sfx_device,
      '| sfx_model_size:', ns.sfx_model_size, '| sfx_workers:', ns.sfx_workers)"

form fields: ['backend', 'duration', 'style', 'name', 'seed', 'force', 'skip_bad',
  'director', 'blocks', 'take_seconds', 'quantization', 'beats_per_segment',
  'drift_every_n', 'min_fps', 'min_resolution', 'verbose', 'no_color']
  # (18 fields — no no_sfx/sfx_* among them)
no_sfx: False | sfx_backend: None | sfx_caption: None | sfx_device: None | sfx_model_size: None | sfx_workers: 1
```

`grep -n "sfx" voyage/tui_state.py` hits only the six hardcoded namespace lines — no field, no help, no reader.

## Repro

```bash
docker run --rm -v "$PWD:/app" -w /app voyage:latest python3 -c "
from voyage.tui_state import GenerateFormState, to_generate_namespace
ns = to_generate_namespace(GenerateFormState(style='x', name='t'))
print(ns.no_sfx, ns.sfx_backend, ns.sfx_workers)  # False None 1 — user cannot change any
"
# Contrast: voyage generate --help lists --no-sfx/--sfx-backend/--sfx-caption/--sfx-device/--sfx-model-size/--sfx-workers.
```

## Fix candidates

1. Minimum (parity-by-default): add an explicit `no_sfx` checkbox (default off) + pass-through, mirroring the `skip_bad` flag pattern — TUI users get the opt-out; overrides stay CLI-only and documented as such.
2. Full: add SFX override fields (backend/caption/device/model-size/workers) or an "advanced SFX" collapsed section; keep `to_generate_namespace`'s explicit defaults with a comment per key (the existing `:302-309` comment already does this — extend it to name the TUI's frozen scope).
3. Test: TUI namespace carries the documented values; offline TUI run with missing MMAudio models + checked skip fails instead of downloading (pairs with 145's verify-only test).

## Refs

- `voyage/tui_state.py:89-115,302-309`; `voyage/cli.py:1773-1811` (six flags), `:1015-1036` (finalize gate the TUI always takes).
- Not-a-duplicate: 145 (`--no-download` absent — different flag, same parity class; coordinate the help text); 022 (caption pins silently dropped between generate→run — different handoff, music/video family); 154/163 (benchmark/soak SFX axes — harness, not TUI); 158 (two-worker cuda:1 hardcode — worker placement, not form surface).
