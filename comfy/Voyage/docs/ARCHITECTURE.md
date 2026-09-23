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
  writes `state.json`, `manifest.json`, segment dirs, the concept store,
  the audio ledger, and `metrics.jsonl`. Workers receive file paths in
  RPC payloads and write only the media files they are told to.
- **Workers**: stateless across segments except the video stream session
  (KV caches persist across blocks *and* segments by design — that is the
  continuity mechanism, rebuilt from `recovery.pt` after evict/restart).
  A restarted worker replays `init` + resume from the latest tape.
- **One writer rule**: never run two supervisors on the same run dir.
  State advances are atomic file replaces; concurrent writers would
  interleave segments.

## Commit pipeline (per segment)

director `decide` → staged prompt plan → video `generate_blocks` →
audio coverage (slow loop: keep/render/repaint takes) → `validate_video`
+ `validate_audio` + A/V drift check (±0.6 s) → metadata + `sha256.json`
→ `DONE` → state advance → `resource_gauges` event.

Validation failures abort the commit *before* `DONE` exists, so the next
run reuses the same segment number and overwrites the media in place.

## GPU time-sharing

`longlive2` + `acestep` cannot co-reside on 16 GB. The supervisor
sequence is: evict video → render audio take → evict audio → rebuild
video from tape. `del` alone frees nothing — eviction is
`del` + `gc.collect()` + `torch.cuda.empty_cache()`.

## What lives where in a run dir

`config.toml`, `manifest.json`, `state.json`, `concepts.json` (legacy),
`novelty/` (vectors + index + jsonl), `segments/NNNNNN/` (video.mp4,
audio.wav, world/transition/prompt-plan/audio-state/metrics.json,
sha256.json, recovery.pt on GPU backends, DONE), `logs/` (metrics.jsonl,
*-worker.log, bench outputs), `final.mp4` after finalize.
