# 013 — Sequential video↔audio GPU swap on every take (full evict + full rebuild)

- Status: open
- Severity: major (dominant wall-clock + VRAM cliff on 16 GiB)
- Area: performance — `_with_audio_gpu` swap path
- Rank rationale: the single biggest production cost; any caption change/repaint/
  short take forces minutes of teardown per segment.

## Technical description

CUDA backends are `STREAMING_VIDEO_BACKENDS` (`voyage/supervisor.py:94`). Whenever
`audio.backend == acestep`, `_with_audio_gpu` (`voyage/supervisor.py:718-753`)
does `video.evict_gpu → audio.generate_audio → audio.evict_gpu →
video.rebuild(recovery_path)` **synchronously inside the commit**. Both stacks are
~11–14 GiB resident (LongLive fp8 peak 14.28 GiB; ACE ~12.3 GiB per
`voyage/audio/acestep.py:13` + Dockerfile comment "30s take 4.3s peak 12.84GB").
They never co-reside by design — every take pays two full teardowns + two full
loads + two CUDA context re-inits + `gc+empty_cache`.

Steady-state hides it only via the `take_seconds=45 / ahead_seconds=20` invariant
(`Voyage/reports/video-backends.md:52`: seg0 ~98 s incl. first take; steady ~0.1 s
via take keep). Any caption change / repaint / short `take_seconds` forces the full
swap per segment. `handle_rebuild`'s own comment admits a `14.97GB resume OOM`
without teardown-first ordering.

## Why this is an issue

This is the dominant wall-clock cost on a 16 GiB card: every take pays two
full stack teardowns plus two full loads with CUDA re-inits and cache flushes,
synchronously inside the commit, and the two stacks can never co-reside by
design. The `take_seconds=45 / ahead_seconds=20` invariant hides the cost in
steady state only — any caption change, repaint, or shortened take forces
minutes of teardown per segment. On single-GPU hosts there is no overlap
available, so the commit tail becomes swap-bound and sets the throughput
ceiling for the whole voyage. Every long music-backed run pays in GPU hours,
not seconds.

## Evidence

- `rg -n "evict_gpu|rebuild" Voyage/voyage/supervisor.py`; read
  `_ensure_audio_coverage:755-819` → `_with_audio_gpu`.
- `Voyage/reports/video-backends.md:52` steady-state table; `grep -n
  "load_seconds|vram_free" voyage/workers/video_*.py`.

Re-verified 2026-09-25 (swap sites unchanged; `_ensure_audio_coverage` now
`supervisor.py:755-868`):

```
$ rg -n "evict_gpu|rebuild" voyage/supervisor.py
737:            self._call_with_restart(self._video, "video", segment_id, "evict_gpu", {})
744:                self._call_with_restart(self._audio, "audio", segment_id, "evict_gpu", {})
746:                    raise MediaError(f"segment {segment_id}: no recovery tape for video rebuild")
751:                    "rebuild",
```

Worker evict entries confirmed: `video_longlive.py:701` (`def evict`),
`video_ltxv.py:602` (`def evict`), `video_causvid.py:827` (`def evict`).

## Reproduction

Run `acestep+longlive2` with `take_seconds <= ahead_seconds` (or force a repaint
per segment); observe per-segment full swap wall vs the invariant-holding baseline.

## Source references

- `voyage/supervisor.py:718-753`; `voyage/workers/video_longlive.py:701-716,
  1054-1084`; `voyage/workers/video_ltxv.py:602-612,821-833`;
  `voyage/workers/video_causvid.py:827-835`;
  `voyage/audio/acestep.py:46-84,145-157`.

## Resolution candidates

1. Enforce `take_seconds >> ahead_seconds` in config validation (today only an
   invariant test, no validator) — prevents the pathological per-segment swap.
2. Split devices: video `cuda:0` + audio `cuda:1` (retest the 2060 ACE-DiT OOM
   verdict `5.30 vs 5.60 GiB` with `offload_to_cpu` planner LM + fp8 DiT; if it
   fits, the swap disappears on 2-GPU hosts).
3. Overlap: start async audio render during `vae_decode_ms` (~16 s, DiT parked on
   CPU per `video_longlive.py:800-802`) — needs 2-process GPU sharing; measure first.

## Investigation / progress / resolution log

- 2026-09-25: found by perf sweep; numbers from `reports/video-backends.md`.
- Open: (1) is cheap and should land first; (2)/(3) need idle-GPU measurement.
- 2026-09-25 (repair pass): added `## Why this is an issue`; Evidence enriched
  (swap + evict sites re-verified); `_ensure_audio_coverage` range updated to
  755-868.
