# ARCHITECTURE — processes and ownership

```text
┌─ supervisor (voyage run) ──────────────────────────────┐
│ owns: state.json, segments/, novelty/, logs/, config   │
│ single writer of all run state; workers own nothing    │
│ persistent across the whole run                        │
└──┬──────────────┬──────────────┬───────────────────────┘
   │ JSONL-RPC    │ JSONL-RPC    │ JSONL-RPC
   │ (stdin/out)  │ (stdin/out)  │ (stdin/out)
┌──▼──────┐  ┌────▼─────┐  ┌─────▼──────┐
│ video   │  │ audio    │  │ director   │
│ worker  │  │ worker   │  │ worker     │
└─────────┘  └──────────┘  └────────────┘
```

## Process ownership

- **Supervisor** (`voyage/supervisor.py`): the only process that reads or
  writes `state.json`, `run_manifest.json` (`voyage/paths.py:29`),
  segment dirs, the concept store, the audio ledger, and `metrics.jsonl`. Workers receive file paths in
  RPC payloads and write only the media files they are told to.
- **Workers**: stateless across segments except the video stream session
  (KV caches persist across blocks *and* segments by design — that is the
  continuity mechanism, rebuilt from `recovery.pt` after evict/restart).
  A restarted worker replays `init` + resume from the latest tape.
- **One writer rule**: never run two supervisors on the same run dir.
  State advances are atomic file replaces; concurrent writers would
  interleave segments.

## Commit pipeline (per segment)

[inspect previous segment + amendments →]
director `decide` → staged prompt plan → video `generate_blocks` →
audio coverage (slow loop: keep/render/repaint takes) → `validate_video`
+ `validate_audio` + A/V drift check (±0.6 s) → metadata + `sha256.json`
→ `DONE` → state advance → `resource_gauges` event.

The bracketed inspect stage runs only when `[experimental]
visual_inspector` is on: it samples the previous segment's committed
video, merges measured metrics into the director context, and applies
`feedback_amendments` post-validation pre-style-check — amendments
invalidate the prefetched proposal. Failures degrade to
`inspect_skipped` and never block the commit (see `docs/BACKENDS.md`
inspector backend and DESIGN §44).

Validation failures abort the commit *before* `DONE` exists, so the next
run reuses the same segment number and overwrites the media in place.

## GPU time-sharing

`ltxv`/`causvid` video + `acestep` cannot co-reside on 16 GB. The supervisor
sequence is: evict video → render audio take → evict audio → rebuild
video from tape. `del` alone frees nothing — eviction is
`del` + `gc.collect()` + `torch.cuda.empty_cache()`. The `ltx25`/`ltx23`
backends skip this dance entirely: their audio is joint (committed with
the video, no audio worker), so no GPU swap ever runs for them.

## What lives where in a run dir

`voyage.toml` (`voyage/paths.py:28`), `run_manifest.json` (`:29`),
`state.json` (`:30`), `concepts.jsonl` (`:31`), `novelty/` (vectors +
index + jsonl), `segments/NNNNNN/` (video.mp4, audio.wav,
world/transition/prompt-plan/audio-state/metrics.json, sha256.json,
recovery.pt on GPU backends, `video_tail.mp4` tail anchor on
tail-chained backends, DONE), `audio/` (takes ledger + slices,
`sfx/` stems + `sfx.jsonl` effects ledger — see `docs/SFX.md`),
`logs/` (metrics.jsonl, *-worker.log, bench outputs), `final.mp4`
after finalize (plus `final-sfx.mp4` when the SFX pass runs and the
augmented presentation output — see `docs/AUGMENT.md`).

2-GPU pairing: video-augment chunks run on cuda:0 while the MMAudio
SFX stack renders on cuda:1 (serial on 1-GPU boxes;
`voyage/augment.py:14-18`).
