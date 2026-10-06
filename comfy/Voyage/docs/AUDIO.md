# AUDIO — ACE-Step slow loop and final mix

## ACE-Step backend

`audio_acestep` runs ACE-Step 1.5 (turbo config) + the 0.6 B planner LM.
The LM initializes with `offload_to_cpu=True` — otherwise it stays
resident and the 45 s DiT preflight OOMs. A readiness gate converts
load-time OOM into a late, actionable error. First real segment verified
at 1280×704; the 2060 cannot hold the DiT — audio renders on the 4060 Ti
after evicting video (see ARCHITECTURE).

## Audio state and the slow loop

Music is **not** rendered per segment. A ledger (`takes.jsonl`) tracks
takes of `take_seconds = 45 s`; each commit extends coverage only when
the buffered audio ahead of the video timeline drops within
`ahead_seconds = 20 s` (invariant: `take_seconds > ahead_seconds`, pinned
by test). Takes land as WAV on disk (`voyage/supervisor.py:1095`,
`take_file = audio_dir / f"{take.take_id}.wav"`; the ACE worker renders
FLAC staging then converts to the requested WAV shape,
`voyage/workers/audio_acestep.py:66-80`, so downstream never branches on
backend); segment slices are WAV too. Take filenames must end `.wav`
to match their codec (`pcm_s16le`) — the flac muxer rejects `pcm_s16le`
mislabeled as `.wav`.

## Music continuation

- **keep**: an existing take already covers the segment — slice it.
- **render**: coverage is short — render a new take from the music style
  + energy prompt.
- **repaint**: the style changed — re-render aligned to the source
  timeline (inherits the source take's `covers_from`; anchoring at video
  time duplicated the preserved head) with a 1 s WAV crossfade.

## Crossfade and slicing

`slice_take` uses `-ss`/`-t` with six decimals (3-decimal rounding
drifted ~0.33 ms/segment). Joins use manual fades + delay, never
`acrossfade` (it collapses on short tails). Crossfades clamp to half the
shortest slice; every assembly step verifies non-empty outputs.

## Final mix

`finalize` re-slices each segment window (extended half the overlap per
side) from the takes ledger and blends pairwise with manual fades
(`voyage/media.py: _blend_pair` — afade out/in + adelay + amix,
2 inputs per ffmpeg call, never `acrossfade`: a 31-input chain
  deadlocks the filter scheduler and long-first pairs collapse), then
  encodes the concatenated segment videos in one libx264 pass
  (defaults `-preset slow -crf 30`, 2026-10-06 compression change —
  the old stream-copy publish is gone) and muxes the mix
  (`-shortest` safety; AAC 128k at 48 kHz) with checksum, frame-range,
and A/V-alignment checks (§56 steps 4–6). Config: 48 kHz stereo
throughout.
