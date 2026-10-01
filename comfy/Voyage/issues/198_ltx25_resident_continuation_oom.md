# 198 — ltx25 resident continuation OOMs on 16 GB (GGUF dequant spike), fresh-process rebuild works

## Technical description
`video_ltx25` keeps a resident ComfyUI session across segments. Every
continuation block (`seg000001-b0`, `seg000002-b0`, …) rendered in the SAME
worker process as the previous segment OOMs inside GGUF dequant
(`ComfyUI-GGUF/ops.py get_weight` → `dequantize_tensor`, tried to allocate
3.75 GiB with 3.09 GiB free; 12.47 GiB already in use on a 15.57 GiB card),
then the retry path produces 0 frames → `INVALID_PAYLOAD … produced 0
frames, expected 121` → run FAILED. Restarting the run (fresh worker
process, rebuild from `recovery.pt`) renders the identical block fine
(~144–148 s). Pattern reproduced 3/3 times on the ltx25-compare run: seg0
OK in process A, seg1 OOMs in process A, seg1 OK in fresh process B, seg2
OOMs in process B, seg2 OK in fresh process C.

## Rationale
Steady-state ltx25 streaming is unusable in a single `voyage run`
invocation on a 16 GB card: each continuation needs a manual resume. ltx23
(same geometry, smaller unsloth stack) completed 3 segments in one process
with no OOM, so this is ltx25-stack-specific memory pressure, not geometry.

## Live evidence
- `Voyage/output/ltx25-compare/logs/video-worker.log` (OOM traceback +
  `ValueError: LTX25 block seg000001-b0 produced 0 frames, expected 121`).
- `Voyage/output/ltx25-compare/logs/metrics.jsonl`: `segment_commit_failed`
  events for 000001 (first attempt), 000002 (second process), 000003
  (overrun attempt); `state.json` ended FAILED with `committed_segments: 3`.
- Per-segment video wall-s: 158.5 / 144.5 / 147.5 s (fresh processes).

## Repro
`run.sh generate --backend ltx25 --duration 10s --style <boba> --name
ltx25-compare --seed 1846428821` on idle RTX 4060 Ti 16 GB → seg1
continuation OOMs ~11 s into the block.

## Fix candidates
- Evict + rebuild the video session between segments (tape resume already
  works — automate the fresh-process path: restart the worker or session
  after each commit).
- Reduce resident pressure before continuation blocks (offload TE after
  encode, as VAE-offload-for-generate does for longlive2).
- Shrink the dequant spike (per-layer dequant already? check GGUF ops —
  the 3.75 GiB alloc suggests a whole-tensor materialization).
- At minimum: supervisor should auto-retry a 0-frame block after a worker
  restart instead of FAILING the run (one restart budget, tape resume).

## Source refs
- `Voyage/voyage/workers/video_ltx25.py` `generate_blocks` (~720–762),
  `_read_block_images` (~654).
- `Voyage/voyage/backends.py` `transport_from_restarting_call` (restart
  exists at RPC level but the 0-frame ValueError surfaces as
  INVALID_PAYLOAD, not a restart trigger).
