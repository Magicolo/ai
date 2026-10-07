# 203 — `configure --sfx-*` flags accepted but silently dropped; `generate` hardcodes them away (HIGH)

## Technical description
`cli.py:_add_sfx_args` defines 7 flags. `cli_configure.cmd_configure` only
forwards `**_sfx_dual_pan_overrides(args)` into `resolve_config`; the other
five (`--sfx-backend/--sfx-caption/--sfx-device/--sfx-model-size/--sfx-workers`)
are never read. `resolve_config` has no such parameters at all
(args: config, backend, director, director_device, blocks, take_seconds,
quantization, beats_per_segment, definition, drift_every_n_segments,
scene_cut_every_n_segments, music_caption, video_caption, prompt_enhance,
upscale, interpolate, presentation_fps, interp_backend, sfx_dual_pan).
`_commit_manifest`/`build_manifest` persist only
`final_video/skip_bad/no_sfx`. Then `cli_generate._finalize_run_dir`
synthesizes `sfx_backend=None, sfx_caption=None, sfx_device=None,
sfx_model_size=None, sfx_workers=1, sfx_dual_pan=None`
(`cli_generate.py:154-159`) — even a stored value could never reach
`cmd_finalize`.

## Rationale
Silent option loss is the worst CLI failure mode: exit 0, manifest looks
configured, behavior unchanged. `--sfx-workers` additionally uses `default=1`
while every sibling uses `default=None`, so absent-vs-explicit-1 are
indistinguishable.

## Live evidence
```
=== _add_sfx_args defaults ===
"--sfx-backend", default=None, choices=["fake","mmaudio"]
"--sfx-caption", default=None
"--sfx-device", default=None
"--sfx-model-size", default=None, choices=[...]
"--sfx-workers", type=int, default=1, choices=[1,2]
```

## Repro
`configure NAME --sfx-backend mmaudio --sfx-model-size small_44k --sfx-workers 2`
→ `manifest.json` has no such keys (`grep sfx manifest.json` shows only
backend-preset `backend/device/model_size` + `dual_pan`); `generate NAME`
finalizes with `sfx_workers=1` regardless.

## Source refs
- `Voyage/voyage/cli.py:141-196`
- `Voyage/voyage/cli_configure.py:285-302,334-353` (both `resolve_config` calls)
- `Voyage/voyage/cli_generate.py:128-169`
- `Voyage/voyage/config.py:761-781`
- `Voyage/voyage/persistence.py:18-49`

## Online sources
- `https://docs.python.org/3/library/argparse.html` (flags must mean
  something); argparse UX discussion
  https://stackoverflow.com/questions/46719811/best-practices-for-writing-argparse-parsers

## Fix candidates
1. Delete the five dead flags if intentionally preset-only.
2. Or add `sfx_backend/sfx_device/sfx_model_size/sfx_caption/num_workers`
   to `SfxConfig` + `resolve_config` + `build_manifest` + `_finalize_run_dir`
   forwarding; normalize `--sfx-workers` to `default=None`.
3. Add a test asserting every `configure` flag round-trips through
   `read_effective_config`.

## Log
- Track B sweep, 2026-10-07. Read-only; nothing fixed.

## Evaluation (2026-10-07, live re-check before fix)

Claim is CURRENT, not stale — every load-bearing assertion re-verified
against the live tree (no concurrent agent touched these lines):
- `cli.py:_add_sfx_args` (lines 141-196) still defines 7 flags;
  `--sfx-workers` still `default=1` while all six siblings default None.
- `cli_configure.cmd_configure` (both `resolve_config` calls, ~285-302 and
  ~336-353) still forwards only `**_sfx_dual_pan_overrides(args)`; the
  other five flags are never read.
- `resolve_config` (`config.py:753-886`) still has no
  sfx_backend/device/model_size/caption/workers parameters.
- `cli_generate._finalize_run_dir` (`cli_generate.py:147-169`) still
  synthesizes `sfx_backend/caption/device/model_size=None, sfx_workers=1,
  sfx_dual_pan=None`.
- `SfxConfig` (`config.py:513-529`) still carries only
  backend/device/models_dir/model_size/dual_pan — no caption, no workers.
- `build_manifest` (`persistence.py:18-49`) needs no change for this fix:
  it dumps the whole config, so new SfxConfig fields persist automatically
  (only video/audio caption pins are stripped).
- Nothing changed since the sweep: no flag got wired in the meantime.

## Progress log (2026-10-07)

- Decision: fix candidate (b) wire-through (NOT (a) delete). Rationale:
  (1) help texts promise stored-override semantics
  ("default: [sfx] backend/device/model_size"), identical to the wired
  `--upscale/--interpolate` siblings via `_augment_overrides`; deleting
  would break the `test_augment_config.py` parser-parity pins AND remove
  the only surface for the documented `--sfx-caption` repair use case
  (pre-SFX-caption runs). (2) `cmd_finalize` already resolves
  backend/device/model_size as `args … or config.sfx.…`, and
  `_finalize_run_dir` None sentinels already mean "stored rules" (same as
  sfx_dual_pan + augment multipliers) — storing needs zero finalize-logic
  change. (3) Caption doctrine nuance: music/video captions are
  in-memory-only (stripped) because they evolve per segment via the
  director; `--sfx-caption` documents whole-timeline semantics
  ("single SFX caption for the whole timeline"), a run-level choice, so
  it persists in SfxConfig. (4) `--sfx-workers` normalized to
  `default=None` as mandated: with `default=1`, `is_provided(1)` is
  always true, so every update would clobber a stored 2 back to 1.
- Implemented (voyage/config.py, cli_core.py, cli_configure.py, cli.py,
  cli_finalize.py, cli_generate.py; tests: new
  tests/test_sfx_configure_roundtrip.py + test_configure.py fixture):
  SfxConfig gains `sfx_caption: str | None = None` + `num_workers: int = 1`
  (before-validator rejects bool/non-{1,2} — an `after` validator sees the
  parsed int, so stored `true` coerced to 1 silently; caught by the new
  rejection test). `resolve_config` gains the five SFX params applied
  AFTER the backend preset (explicit wins). `cli_core._sfx_overrides`
  mirrors `_augment_overrides`; both `cmd_configure` calls spread it.
  `cmd_finalize` resolves workers + caption from
  `args … or config.sfx.…` on both SFX paths. `_finalize_run_dir` passes
  `sfx_workers=None` (stored rules).
- Verified in-container (image voyage:latest, bind-mounted tree, no GPU):
  ruff check + format --check clean on all 8 touched files; mypy strict
  clean on all 8; pytest 88 passed (new 12-test module + configure, sfx,
  resolution, registry, manifest neighbors); wider sweep 140 passed
  (definition-tiers, scene-cut, enhancer, cli-group-a, three-captions,
  sfx-finalize neighbors).
- KNOWN COLLATERAL (out of scope, needs owning agent):
  `tests/test_augment_config.py::test_configure_carries_sfx_flags` pins
  the OLD parser default (`sfx_workers: 1`) and now fails
  (`None != 1`) — the mandated normalization requires it. File is outside
  this fix's touch scope; one-line update (`"sfx_workers": 1` → `None`
  in the expected dict) heals it.
- Stale-note flag: AGENTS.md §11 says sfx_caption/sfx_workers are
  "not inheritable by construction" (`--from` entry) — true when they
  were non-stored; both now persist via SfxConfig and inherit through
  `--from` like every other stored field. AGENTS.md is outside this
  fix's touch scope; left for the orchestrator.

## Resolution (2026-10-07)

RESOLVED via candidate (b). All five `configure --sfx-*` flags now
round-trip through `read_effective_config` (proven by
`tests/test_sfx_configure_roundtrip.py`: create-persist, update-inherit,
single-flag update, absent-vs-explicit workers distinction, rejection of
workers 3/True + unknown backend/size, generate-sentinel pins); `generate`
finalizes under the stored SFX config. Full `gates.sh` deliberately not
run (concurrent agents' in-flight changes per mandate); scoped gates on
touched files are green. One out-of-scope pin left red (see collateral
above). No GPU work performed.
