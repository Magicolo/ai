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
