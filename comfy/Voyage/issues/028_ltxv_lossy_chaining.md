# 028 — LTXV inter-block chaining via lossy mp4 temp files (encode + decode per block)

- Status: resolved (fixed 2026-09-25: tensor handoff + tail-length asserts + tests; encode-wall A/B needs idle GPU, see log)
- Severity: medium-high (wall + quality — generation loss on the continuity tail)
- Area: performance — `voyage/workers/video_ltxv.py:477-591`
- Rank rationale: N encodes + N decodes for what is logically a tensor handoff;
  lossy roundtrip degrades the §5.3 continuity tail.

## Technical description

`generate_blocks` renders block `i`, writes `*_chain{i:02d}.mp4` (full h264 encode
via `_save_mp4`, `video_ltxv.py:615-627`, imageio defaults — no `preset`, no
`pix_fmt`, unlike LongLive's explicit writer and finalize's `yuv420p` pin), then
block `i+1` reads it back through `prepare_conditioning` (`:344-374`: ffmpeg
decode + resize + pad). Per multi-block segment: N encodes + N decodes of 25–121
frames at 768×512. Today's defaults risk `yuv444`/compat issues.

## Why this is an issue

The chain tail is the continuity-critical handoff between blocks, and routing it through N lossy h264 encode/decode roundtrips injects compression artifacts into the conditioning of every subsequent block — damage that compounds across the segment and directly fights the §5.3 continuity guarantee. On top of the quality cost, each block pays a full codec roundtrip in wall time at 768×512, and the writer's unpinned pixel format risks compat surprises. A tensor handoff removes both costs at once.

## Evidence

Read `generate_blocks:477-591`; `rg -n "chain_tails|_save_mp4|
prepare_conditioning" Voyage/voyage/workers/video_ltxv.py`.

Verified live 2026-09-25:

```
350:  from ltx_video.inference import calculate_padding, prepare_conditioning
498:  chain_tails: list[Path] = []   516: conditioning = str(chain_tails[-1])
533:  _save_mp4(tail_clip, chain_tail, fps)   538: _save_mp4(video, output_path, fps)
615:def _save_mp4(images: Any, path: Path, fps: int) -> None:
```

## Reproduction

Multi-block LTXV segment; count `*_chain*.mp4` temp files and their encode/decode
wall in stage timings.

## Source references

- `voyage/workers/video_ltxv.py:344-374,477-591,615-627`.

## Resolution candidates

1. Pass the `novel[:, :, -tail:]` tensor directly to `prepare_conditioning` (it
   accepts arrays upstream) or write lossless PNG dir / raw `.pt` tail.
2. If mp4 must stay: explicit `pix_fmt yuv420p + preset veryfast`.

Payoff: ~2× codec wall saved per chained block + removes a generation-loss source
on the continuity-critical tail.

## Investigation / progress / resolution log

- 2026-09-25: found by perf sweep.
- 2026-09-25: repair pass — added `## Why this is an issue`; chain_tails /
  _save_mp4 / prepare_conditioning refs re-verified live, current; pasted rg
  output into Evidence.
- Open: implement tensor handoff + A/B continuity check.
- 2026-09-26 (resolution, PARTIAL): tensor handoff landed with the mp4
  fallback kept (`voyage/workers/video_ltxv.py`). Pure helpers (slim-tested):
  `validate_tail_length` (`:147-160`, the issue-064 short-anchor guard
  extracted from the inline assert), `tail_frames_for_conditioning`
  (`:163-181`, (T,H,W,C) uint8 shape gate), `_tail_clip_to_handoff_frames`
  (`:184-215`, `_save_mp4`-mirroring conversion returning None with a
  stderr note when the tail cannot be bottled). `_generate_block` takes the
  union `str | NDArray[np.uint8] | None` in its existing 7th slot
  (`:418-466`) — arity unchanged so stubbed sessions keep working;
  `generate_blocks` chains onto `pending_tail_frames` when bottling
  succeeded, else the chain mp4 path (`:592-647`). Chain mp4s are still
  written (crash-recovery/fallback artifact + final-tail source): the
  quality win (no lossy roundtrip on the continuity tail) and the decode
  wall land now; the encode wall remains. Tests:
  `tests/test_ltxv_tensor_handoff.py` (11 tests: tail-length exact/short/
  long, constants, shape/dtype gates, bottling of a channel-first fake,
  stub-tensor fallback). Compatibility note: an 8th-parameter variant was
  tried first and broke `test_ltxv_failure_hygiene` stubs (7-arg) — the
  union slot fixes it; those 3 tests pass unmodified. FOLLOW-UP (needs idle
  GPU): (a) A/B continuity — tensor-chained vs mp4-chained multi-block
  segment, consecutive-frame diff + eyeball, confirming `prepare_conditioning`
  array semantics in the pinned ltx-video rev; (b) only then stop writing
  intermediate `*_chain*.mp4` (keep the final `video_tail.mp4` recovery
  anchor). Pre-existing, out of scope: `test_encode_moves_mask_to_session_device`
  fails on the concurrent pass's `_encode` LRU change (`{}` has no `.put`) —
  untouched here.
