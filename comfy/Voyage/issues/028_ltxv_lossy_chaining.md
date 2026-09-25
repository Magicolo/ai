# 028 — LTXV inter-block chaining via lossy mp4 temp files (encode + decode per block)

- Status: open
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
