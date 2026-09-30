# 161 — `sfx_caption` reaches the console plan dict but neither renderer prints it

- **Severity:** MEDIUM (observability — the third caption family's per-segment drift is computed, persisted, and then invisible in both live surfaces, contradicting the DESIGN promise)
- **File:line:** `Voyage/voyage/supervisor.py:1372-1429` (plan dict; `audio_sfx_caption` at `:1435`) → `Voyage/voyage/console.py:204-258` (music-only at `:238-248`) + `Voyage/voyage/tui.py:243-256` (music-only at `:253-256`); `Voyage/docs/OPERATIONS.md:23-25` (music-caption-only description); `Voyage/DESIGN.md:7112-7113` (promise)
- **Area:** console/TUI render gap (three-caption doctrine, slice 1 done, renderers still two-caption)

## Description

The supervisor builds the console plan dict with the SFX caption
present:

```python
# supervisor.py:1435 (inside _segment_plan_info, :1380-1436)
"audio_sfx_caption": decision.audio.sfx_caption,
```

But `ConsoleReporter.segment_plan` (`console.py:204-258`) renders the
audio section from `audio_caption`/`audio_beats`/`audio_bpm`/
`audio_energy` only (`:238-248` — the `🎵` line plus the `music:` line
and verbose texture/environment), and never reads
`audio_sfx_caption`. The TUI sink (`tui.py:243-256`) is the same shape
— destination, video prompts, then one `🎵 music:` line at `:253-256`
— with no SFX line. So the per-segment SFX drift the director
computes (and `transition.json` persists) is invisible in both live
surfaces: a user watching captions evolve sees video + music move and
has no signal for what the finalize-time SFX pass will condition on.

`docs/OPERATIONS.md:23-25` documents the two-caption status quo
("the full video prompt(s) and the music caption with the beat grid"),
so the runbook matches the code — but `DESIGN.md:7112-7113` (SFX
slice 1 log) promises the three-caption surface:

```
- Console: `_segment_plan_info` surfaces `audio_sfx_caption` alongside
  the music caption so the drift is visible per segment.
```

The dict half of that promise landed (`supervisor.py:1435`); the
render half never did in either renderer.

## Rationale

SFX captions are the only pre-finalize visibility into the MMAudio
conditioning that will dub the shipped video. An SFX caption that
drifts off-brief (wrong objects, stuck on an old concept, empty on
pre-SFX runs) is currently undetectable until after the finalize pass
renders stems — the most expensive point to discover it. Printing the
already-computed string costs one line per segment in each renderer
and fulfills an explicit DESIGN commitment.

## Live evidence

- `sed -n '1380,1436p' voyage/supervisor.py` — plan dict carries
  `video_prompts`, `audio_caption` (`:1429`), and `audio_sfx_caption`
  (`:1435`); `transition_mechanism` at `:1427` anchors the cited range.
- `sed -n '204,258p' voyage/console.py` — audio block `:238-248`
  reads `audio_caption`/`audio_energy`/`audio_beats`/`audio_bpm`;
  `audio_sfx_caption` appears nowhere in the file (`grep -n
  "sfx" voyage/console.py` → no hits).
- `sed -n '243,256p' voyage/tui.py` — `:253-256` posts beats/bpm +
  `audio_caption` as `🎵 music:`; `sfx` appears nowhere in the method
  (`grep -n "sfx_caption" voyage/tui.py` → no hits).
- `sed -n '23,25p' docs/OPERATIONS.md` — console section names
  "the music caption with the beat grid" only.
- `sed -n '7112,7113p' DESIGN.md` — "`_segment_plan_info` surfaces
  `audio_sfx_caption` alongside the music caption so the drift is
  visible per segment."
- Overlap check: 028 is the console contract (shape/keys — predates
  the third caption); 061 is the status-vs-config gap (status verb,
  not the segment-plan renderers). Neither names the missing SFX line.

## Repro

Static (deterministic): call `segment_plan` with a plan dict
containing `audio_sfx_caption: "rain on canvas"` — neither the
console reporter nor the TUI sink emits the string. Dynamic: run two
segments with evolving SFX captions and watch both surfaces — video
prompts and `🎵 music:` update, no SFX line ever appears.

## Fix candidates

1. Console: after the `music:` line (`console.py:248`), emit
   `self.line(f"     sfx: {info.get('audio_sfx_caption', '')}")`
   (empty → print a `-` placeholder or skip, matching the texture/
   environment verbose convention; prefer always-print so a missing
   caption on pre-SFX runs is itself visible).
2. TUI: after `:256`, `self._post(f"  🔔 sfx: {caption_sfx}")` with
   the same fallback rule (bell emoji distinguishes the SFX family
   from the `🎵` music line; any stable marker works).
3. Docs: extend `OPERATIONS.md:23-25` to "the music caption with the
   beat grid and the SFX caption" once both renderers print it.
4. Tests: plan-dict → rendered-text pins for both renderers
   (sfx caption present/absent/empty × verbose/compact), mirroring
   `tests/test_console.py` style.

## Refs

- `Voyage/voyage/supervisor.py:1380-1436`; `Voyage/voyage/console.py:204-258`;
  `Voyage/voyage/tui.py:243-272`; `Voyage/docs/OPERATIONS.md:21-28`;
  `Voyage/DESIGN.md:7105-7118` (three-caption slice log).
- Adjacent, not overlapping: 028 (console contract shape); 061
  (status verb gaps); 113 (TUI progress content — progress bar, not
  caption families).
