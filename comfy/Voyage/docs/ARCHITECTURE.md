# ARCHITECTURE — processes and ownership

```text
┌─ supervisor (voyage generate) ──────────────────────────┐
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
  writes `state.json`, `manifest.json` (`voyage/paths.py:29`),
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
video-only cover (conditioning-tail derive on streaming backends — no
`audio.wav` is ever written at commit, `voyage/supervisor.py:2359-2397`)
→ `validate_video` (manifest checksum over `video.mp4` only,
`REQUIRED_CHECKSUM_ARTIFACTS = ("video.mp4",)` at
`voyage/segment_manifest.py:41`, plus frame range and duration —
video-only, no A/V gate, `voyage/media_audio.py:451-461`) →
`manifest.json` → `DONE` → state advance → `resource_gauges` event.

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
`del` + `gc.collect()` + `torch.cuda.empty_cache()`. All backends
(`fake`/`ltxv`/`causvid`/`ltx25`/`ltx23`) defer ACE music to finalize:
commit is video-only and takes render once at finalize from the takes
ledger, so no per-segment audio swap ever runs.

## What lives where in a run dir

`manifest.json` (`voyage/paths.py:32` — carries the effective config,
CLI-is-config, no TOML), `state.json` (`:34`), `concepts.jsonl` (`:35`), `novelty/` (vectors +
index + jsonl), `segments/NNNNNN/` (`video.mp4`,
`manifest.json` — transition/prompt-plan/audio-state/world-state/
metrics/checksums in one file, checksums video-only
(`voyage/segment_manifest.py:41-48`; legacy `sha256.json` only via the
fallback reader for old runs), `recovery.pt` on GPU backends,
`video_tail.mp4` tail anchor on
tail-chained backends, DONE), `audio/` (takes ledger — takes render at
finalize, never at commit — plus
`sfx/` stems + `sfx.jsonl` effects ledger — see `docs/SFX.md`),
`logs/` (metrics.jsonl, *-worker.log, bench outputs), `final.mp4`
after finalize (plus the augmented presentation output — see
`docs/AUGMENT.md`).

2-GPU pairing: video-augment chunks run on cuda:1 while the MMAudio
SFX stack renders on cuda:0 (serial on 1-GPU boxes;
`voyage/augment.py:18-26` contract, `model_pass_devices`/
`upscale_pass_devices`/`interp_pass_devices` at
`voyage/augment.py:514-571`).
