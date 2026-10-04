# Video backend qualification (DESIGN §137A)

Stream A owns the LTXV leg. A backend
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
| CUDA runtime / PyTorch | 12.8.0 / 2.8.0+cu128 | `benchmark video` header |
| Video image | voyage-video:latest (local build, ID `6a55a3691940`, no registry digest) | `docker images` |
| Host RAM / OS-kernel | 62 GiB / 6.8.0-139-generic | `free` / `uname -r` |

Harness dry-run (measured 2026-09-24, CPU, **fake backend** numbers): `run.sh init/run --segments 3` on an in-container absolute run
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
