# Video backend qualification (DESIGN §137A)

Stream B owns the **longlive2** leg; Stream A owns the LTXV leg. A backend
becomes operationally preferable only on measured numbers — no static
ranking here overrides benchmark results. Every GPU-gated field below is
either measured or marked PENDING; nothing is invented.

Method (§137A legs, in run order): smoke → resolution/FPS → VRAM/RAM →
steady state → crash recovery → 3-segment visual review. The CPU-runnable
harness (`tests/test_qualification.py`) encodes the stage list, the metric
math, and the continuity gate; `scripts/qualify.sh` drives the GPU stages.

Continuity gate (provisional): sample evenly spaced frames from each
committed `segments/<id>/video.mp4`, take mean absolute gray diffs within
segments vs across boundaries; PASS when
`boundary_mean < 3x within_mean` (the continuity investigation measured ~6x
jumps for fresh-scene failures, so 3x is a conservative gate).

## Environment (measured 2026-09-24, host)

| Field | Value | Source |
|---|---|---|
| GPU 0 | NVIDIA GeForce RTX 4060 Ti, 16380 MiB, PCI 00000000:01:00.0, cc 8.9 | `nvidia-smi` |
| GPU 1 | NVIDIA GeForce RTX 2060, 6144 MiB | `nvidia-smi` |
| Driver | 595.84 | `nvidia-smi` |
| CUDA runtime / PyTorch / video-image digest | PENDING (read from the video image at GPU-run time) | — |
| Host RAM / OS-kernel | PENDING (record at GPU-run time) | — |

## longlive2 leg (Stream B)

Fixed configuration (pins quoted from `voyage/model_registry.py`, not measured):

| Field | Value |
|---|---|
| Code | NVlabs/LongLive @ `6b36d20ec6f7958d29d11a704dfa64611a9f2572` |
| Generator | Efficient-Large-Model/LongLive-2.0-5B rev `8521079b863720a57c1a8d9b19c8d9e6ccb04c0f`, file `model_bf16.pt` |
| Base weights | Wan-AI/Wan2.2-TI2V-5B (`wan_models/` subdir) |
| Native geometry | 1280x704 @ 24 fps, 8 latents/block, 4 sampling steps, guidance 1.0 |
| Quantization | fp8 default (bf16 alt derives its own recovery profile) |
| Attention window | local_attn 16 / sink 8 (capacity floor: sink + one block) |
| Recovery class (§137C) | `persistent_kv` (replay clean latents from recovery.pt, never CUDA tensors) |

Results:

| Stage | Metric | Value |
|---|---|---|
| smoke (1 block) | time_to_first_output_s | PENDING (GPU blocked, see below) |
| resolution/FPS | committed WxH @ fps | PENDING |
| VRAM/RAM | peak_vram_bytes / peak_cpu_ram_bytes | PENDING / PENDING |
| steady state (3 segs) | seconds_generated / wall_seconds / steady_state_ratio | PENDING / PENDING / PENDING |
| crash recovery | kill -9 video worker mid-segment → resume → VALID | PENDING |
| visual review (3 segs) | within_mean / boundary_mean / boundary_ratio / verdict | PENDING / PENDING / PENDING / PENDING |

Blocked reason (2026-09-24 03:46–04:31 UTC, full ~45 min back-off window):
GPU 0 never dropped under the 2 GiB threshold — Stream A cycled
`python3 -m voyage.workers.video_ltxv` workers the whole window (PIDs
2551866 6702 MiB → 2683618 6184 MiB → 2706744 6184 MiB → 2714161 6664 MiB,
plus its `audio_acestep` worker peaking at ~14.6 GiB mid-window).
Per the repo GPU-contention rule the GPU leg was not attempted under
contention and no contention OOM/transient death is recorded as a result.
To complete: wait for an idle GPU, then run `scripts/qualify.sh` (idle
gate + `benchmark video` + 3-segment run + validate + JSON summary) and
fill the table above plus the environment PENDINGs.

Harness dry-run (measured 2026-09-24, CPU, **fake backend — NOT longlive2
numbers**): `run.sh init/run --segments 3` on an in-container absolute run
dir, then `validate`, `finalize`, and `summarize_run` from the harness:

| Metric | Value |
|---|---|
| committed | 3 segments, 48f each, `validate` VALID 144f |
| segment_elapsed_s | 0.518 / 0.497 / 0.286 (warm-up excluded from steady mean by construction) |
| time_to_first_output_s | 0.518 |
| steady_state_ratio | 1.32 |
| continuity within_mean / boundary_mean / ratio | 0.0182 / 0.0394 / 2.16 → PASS (< 3.0 gate) |
| finalize | h264 768x432@24 + AAC, 6.0s |
| kill-test | `inject_worker_crash("video")` → commit 000000 → DONE → 1 restart event → VALID |

These numbers measure harness overhead on testsrc media, never model speed.
Procedural note (out of Stream B scope, not fixed here): `run`/`finalize`
with a host-relative `--run` dir doubles the path (take WAV landed under
`<run>/output/<run>/...`, commit stalled at 0 segments); in-container
absolute paths (`/app/output/...`) work — the codebase invariant is
absolute voyage paths. The `test_fake_*_dry_run` tests use absolute
`tmp_path` dirs and are unaffected.

## LTXV leg (pending Stream A)

Empty on purpose — Stream A fills this leg with its measured numbers
(time_to_first_output, seconds_generated, wall_seconds, steady_state_ratio,
peak_vram_bytes, peak_cpu_ram_bytes, boundary-diff continuity stats,
kill-test outcome) or its own blocked-reason. Stream B does not touch it.
