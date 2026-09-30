# 022 — `generate --music-caption/--video-caption` silently dropped: inner `cmd_run` reloads from disk without the pins

- Severity: HIGH (silent intent-drop: exit 0, wrong behavior)
- Group: CLI/handoff — Rank: 1/5
- File:line: `voyage/cli.py:1289-1295` (`cmd_generate` builds `effective`), `voyage/cli.py:1355+` (inner `cmd_run` namespace omits pins), `voyage/cli.py:484-500` (`cmd_run` pin support)

## Description

`cmd_generate` builds `effective` with the caption pins, uses it for planning/model-ensure, then calls `cmd_run` with a namespace that forwards every numeric/behavioral override — but not `music_caption`/`video_caption`. `cmd_run` re-loads the config from `run_dir/voyage.toml` (written at `init` without pins) and re-resolves from its args. The pins exist only in the discarded `effective`. Result: the flags parse, appear in `--help`, affect planning display, then affect nothing in the run phase — the director drives instead, with no warning.

This is structural, not a typo: the two-stage `init→run` split means any in-memory-only pin added later falls into the same hole unless forwarded or persisted.

## Rationale

- Silent intent-drop defeats the CLI contract: the user stated intent, the tool acknowledged it (no error), the tool ignored it.
- The inner-call slice is otherwise complete (blocks/takes/quantization/beats/drift/floors all forwarded), so the omission is invisible in review — only a forwarding-contract test catches it.
- `cmd_run` already honors both pins; the bug is purely the handoff, which makes the fix trivial and the continued silence embarrassing.

## Live evidence (read + grep, 2026-09-30)

Effective build (`voyage/cli.py:1236-1249`):

```python
        effective = apply_draft_overrides(
            config,
            draft=args.draft,
            director=director,
            blocks=args.blocks,
            take_seconds=args.take_seconds,
            quantization=args.quantization,
            beats_per_segment=args.beats_per_segment,
            drift_every_n_segments=args.drift_every_n,
            music_caption=getattr(args, "music_caption", None),
            video_caption=getattr(args, "video_caption", None),
            **_augment_overrides(args),
        )
```

Inner call (`voyage/cli.py:1316-1333`) — note the absence:

```python
    cmd_run(
        argparse.Namespace(
            run=str(run_dir),
            segments=segments,
            draft=args.draft,
            director=director,
            blocks=args.blocks,
            take_seconds=args.take_seconds,
            quantization=args.quantization,
            beats_per_segment=args.beats_per_segment,
            drift_every_n=args.drift_every_n,
            min_fps=getattr(args, "min_fps", None),
            min_resolution=getattr(args, "min_resolution", None),
            no_augment=bool(getattr(args, "no_augment", False)),
            verbose=console.verbose,
            no_color=getattr(args, "no_color", False),
            progress_sink=sink,
        )
    )
```

Grep (2026-09-30):

```
$ grep -n "music_caption\|video_caption" voyage/cli.py
427:def cmd_run(args: argparse.Namespace) -> int:
447:        or is_provided(getattr(args, "music_caption", None))
448:        or is_provided(getattr(args, "video_caption", None))
461:                music_caption=getattr(args, "music_caption", None),
462:                video_caption=getattr(args, "video_caption", None),
1248:            music_caption=getattr(args, "music_caption", None),
1249:            video_caption=getattr(args, "video_caption", None),
```

Zero hits inside the 1316-1333 inner-namespace block. Sweep probe confirms `default_config_toml` output contains neither key (`False False`).

## Repro

```bash
docker run --rm -v "$PWD:/app" -w /app voyage:latest python3 -c "
import inspect
from voyage import cli
gen = inspect.getsource(cli.cmd_generate)
tail = gen.split('cmd_run(')[1][:1500]
print('music_caption forwarded?', 'music_caption' in tail)
print('video_caption forwarded?', 'video_caption' in tail)"
# Buggy: both False
```

End-to-end: `voyage generate --music-caption "X" ...` then inspect the run-phase effective settings line — no caption override is ever applied (director captions drive).

## Fix candidates

1. Forward both pins into the inner `cmd_run` namespace (two lines mirroring `cmd_run`'s own `getattr` reads). Minimal, matches every sibling override.
2. Or persist pins into `voyage.toml` at init (stored-config source of truth) — larger, but closes the hole for all future in-memory pins.
3. Add a namespace-forwarding contract test: assert every `apply_draft_overrides`/`resolve_config` parameter accepted by `cmd_generate` reaches the inner `cmd_run` namespace (guards the next pin too).
4. Log the winning caption source at plan time so a future drop is visible, not silent.

## Refs (with links/quotes)

- "The first source that supplies a value wins … Encode it once and log the winning source so drift is visible." A dropped layer violates the contract silently. — https://python-config-secrets-hub.com/core-configuration-patterns-file-formats/configuration-precedence-rules/
