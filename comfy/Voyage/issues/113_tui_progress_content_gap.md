# 113 — `TuiProgress` drops the console's decision-relevant lines despite its mirror-the-console docstring

- **Severity:** LOW-MEDIUM (observability: the TUI run log omits exactly the lines an operator watches during a long render)
- **Track:** second-pass TUI/CLI edges (TUI progress sink vs console sink)
- **Verified:** 2026-09-30 by source comparison (no import needed; lines as-read; concurrent edits in `cli.py` do not touch these methods)

## File:line (live-verified)

- `voyage/tui.py:211-275` (`TuiProgress`: `segment_start :227`, `stage :230-241`, `segment_plan :243-256`, `segment_done :258-275`)
- `voyage/tui.py:212-217` (docstring: "Plain-text lines mirror the console wording … so behavior stays recognizable across both displays")
- `voyage/console.py:204-258` (`segment_plan`: drift line with backend+hold, video-geometry line, verbose seeds/transitions, audio line with backend+energy, texture/environment/notes) and `:260-287` (`segment_done`: take ids+action, verbose reason/blend, prefetch, elapsed total)

## Description

`TuiProgress.segment_plan` prints destination/phase/novelty + video prompts + `music: caption (beats @ BPM)`. Compared against the console implementation it claims to mirror, it drops:

- the **video-geometry line** (`video · {backend} · {geometry} @{fps}fps · {frames}f ≈ {duration}s · {blocks} block(s){scene-cut flag}`) — a TUI operator never sees what geometry/fps the segment actually rendered at, nor scene-cut flags;
- the **backend labels** (director/video/audio) and the **drift-hold flag** — the drift line keeps destination/phase/novelty but loses `(backend)` and `· drift hold`;
- the **energy/texture/environment/notes** audio detail (partly verbose-gated on console — acceptable to drop — but `energy` rides the non-verbose audio line);
- in `segment_done`: **take ids + take action** (`{takes} ({take_action})` — keep/render/repaint, the audio-fit judgment the whole slow loop exists to produce), the **prefetch-hit marker**, and the **elapsed total**.

`stage` matches (both sinks print `▸ head ...` / `✓ label in Ns` / `✗ failed`), and `segment_start` matches — the divergence is confined to the two content methods. Not covered by 028/063, which file stream-sink/teardown/timing contracts, not content parity.

## Rationale

- The TUI is the default interface for long renders; the run log is the only monitor. Take-action (did the slow loop keep, render, or repaint this segment's music?) and scene-cut/geometry are the lines that answer "is this run healthy?" mid-flight. Their absence is felt precisely during the multi-minute silent stretches the progress layer was built to cover (console.py:1-16).
- The docstring promises mirroring; the help-panel/tests (`test_mid_run_progress_reaches_log_before_completion` asserts only that a segment line lands, not its content) lock in the weaker behavior. Either the docstring or the method is wrong — pick one.
- Cheap to verify going forward: a content-parity test asserting the TUI history contains the same key tokens (backend, geometry, take action) the console prints for one canned `segment_plan`/`segment_done` pair.

## Evidence (source, 2026-09-30)

TUI `segment_plan` (`tui.py:243-256`) emits 3 shapes: `◆ drift → …`, `🎬 prompt…`, `🎵 music: …`. Console (`console.py:204-258`) emits `◆ drift → … (backend)`, `🎬 video · backend · geometry @fps …`, prompts, `🎵 audio · backend · beats @ BPM · energy …`, `music: …`. TUI `segment_done` (`tui.py:258-275`) emits `✓ SEGMENT … · Nf ≈ Ss · B beats @ P BPM` + stages; console (`console.py:260-287`) emits the same plus `takes (action)`, `prefetch hit`, `total Ns`.

## Repro

```python
# No runtime needed — compare the two methods:
# sed -n '243,275p' Voyage/voyage/tui.py    # 3 line shapes per segment
# sed -n '204,287p' Voyage/voyage/console.py  # 6+ line shapes per segment
```

Pilot-level: drive `TuiProgress.segment_plan` with a full supervisor-style `info` dict (destination/phase/novelty/video_prompts/geometry/fps/blocks/scene_cuts/director_backend/video_backend/audio_backend/energy) and assert which keys reach `app.run_history` — geometry/backends/take-action never do.

## Fix candidates

- Add the dropped non-verbose lines to `TuiProgress` verbatim (geometry/video line, backend labels, take ids+action, prefetch, elapsed total); keep verbose-gated detail (seeds/transitions/texture/notes) console-only and say so in the docstring.
- Or narrow the docstring to the actual contract ("segment headers, prompts, music caption, commit summaries") and file the dropped lines as an accepted TUI/console difference. Either way the promise and the code must agree.
- Test: one canned `segment_plan` + `segment_done` through both sinks, asserting token parity on the non-verbose subset.

## Refs

- DESIGN §59 (console progress contract); `tests/test_tui_app.py:812-860` (liveness test — extend to content); issues 028/063 (sink/teardown/timing — explicitly not content parity).
