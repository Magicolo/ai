# Video backend qualification (DESIGN §137A)

> HISTORICAL MARKER (issue 079): the `longlive2` leg below is dated
> evidence from 2026-09-24 — the backend was full-deleted (worker,
> registry, Dockerfile, tests). Nothing below is a live path; the
> live legs are LTXV/CausVid. Kept verbatim for the numbers.

Stream B owned the **longlive2** leg (historical); Stream A owns the LTXV leg. A backend
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

## longlive2 leg (Stream B — HISTORICAL, backend removed by issue 079)

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

Results (measured 2026-09-24, run `output/qual-longlive2`, GPU idle per the
contention rule — only host-compositor residue ~137 MiB):

| Stage | Metric | Value |
|---|---|---|
| smoke (1 block) | time_to_first_output_s | 98.2 (incl. model load + first take render; video stage alone 81.7) |
| resolution/FPS | committed WxH @ fps | 1280x704 @ 24 (native; 29f ≈ 1.21s per 1-block segment) |
| VRAM/RAM | peak_vram_bytes / peak_cpu_ram_bytes | 14.28 GiB (`benchmark video`, 3 measured blocks) / not instrumented |
| steady state (3 segs) | seconds_generated / wall_seconds / steady_state_ratio | 1.21 / 38.385 / 2.56 (seg elapsed 98.2 / 38.335 / 38.435; video stages 81.7 / 38.1 / 38.3; audio 97.6 / 0.1 / 0.1 via take keep) |
| benchmark detail | 29 frames/block @ 38.485 / 38.448 / 38.457 s/block, 0.026 blk/s | `benchmark video --warmup 1 --measured 3` |
| crash recovery | kill -9 video worker mid-segment → resume → VALID | NOT re-run on this leg; covered by prior Phase-2 GPU evidence (SIGKILL → restart → resume → VALID 186f, DESIGN §140) |
| visual review (3 segs) | within_mean / boundary_mean / boundary_ratio / verdict | 0.0246 / 0.1134 / 4.61 → FAIL vs the provisional <3.0 gate |

Continuity FAIL analysis (same run, extra probe): per-boundary gray diffs are
`000000[28]->000001[0] = 0.0957` vs `000001[28]->000002[0] = 0.1062`. The
second boundary crosses NO rebuild (resident stream, audio take kept, no
evict) yet is identical to the post-rebuild one — so the resume path is
exonerated and the jump is native inter-block drift for this prompt. Every
boundary here IS a block boundary by construction (1 block/segment), which is
exactly the Phase-2 pass criterion (seg-boundary == block-boundary); the
provisional 3x gate compares against tiny within-segment frame diffs
(0.0246 at 24fps slow motion) and is miscalibrated for this content. Whether
0.10 reads as a cut needs the manual eyeball review (still open — see
`scripts/qualify.sh` header). `validate` → VALID 3 segments, 87f.

Bugs found and fixed by this leg (all verified in the same run):

1. Post-audio `rebuild` OOM (deterministic, 3/3 attempts in fresh worker
   processes, NOT contention): `resume_from_tape` replayed the DiT forward
   with the 1.31 GiB VAE resident, peaking at 14.40 GiB + audio residue over
   the 15.57 GiB budget. Fix: `pipe.vae.to("cpu")` + `empty_cache()` around
   the replay with `finally` restore in
   `voyage/workers/video_longlive.py:resume_from_tape` (mirrors the existing
   VAE-offload-for-generate). Post-fix rebuild succeeds; seg0 committed.
2. `init --backend longlive2` could never commit: the preset left the default
   768x432 geometry while the worker always renders native 1280x704
   (latent_shape x16), failing the commit-time resolution check. Fix:
   `voyage/config.py` longlive2 preset now pins `longlive2-704p` /
   1280x704 (+ comment), and
   `tests/test_generate.py::test_longlive2_preset_pins_native_geometry`
   replaces the old keeps-default-geometry assertion. Full suite 291 passed.
3. Hit live (first attempt, pre-existing, NOT fixed here): host-relative
   `--run` doubles paths inside the worker (`output/.../segments/...`
   missing → generate_blocks circuit-breaker, 3 restarts logged in
   metrics.jsonl). Workaround: absolute in-container path
   (`/app/output/qual-longlive2` + explicit `VOYAGE_IMAGE`/`VOYAGE_GPUS`).
   Still open — see TASK §30.2.

Superseded blocked-reason (2026-09-24 03:46–04:31 UTC back-off window): GPU 0
never dropped under the 2 GiB threshold then — Stream A cycled
`python3 -m voyage.workers.video_ltxv` workers the whole window. This leg ran
once the card idled; no contention OOM/transient death is recorded as a result.

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
