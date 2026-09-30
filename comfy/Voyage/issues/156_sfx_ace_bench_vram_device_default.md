# 156 — SFX/ACE benchmarks call CUDA peak APIs unguarded while worker and config defaults disagree on device

- **Severity:** MEDIUM (CPU-box crash + default mismatch — benchmark dies where it should report, defaults invite it)
- **File:line:** `Voyage/voyage/workers/sfx_mmaudio.py:301` (`torch.cuda.reset_peak_memory_stats()`) + `:317` (`torch.cuda.max_memory_allocated()`) + `:77-101` (`handle_init` defaults `device "cuda:0"`, `model_size "large_44k_v2"`) → `Voyage/voyage/workers/audio_acestep.py:220` (reset) + `:226` (max) + `:79-97` (init defaults `device "cuda:0"`) → `Voyage/voyage/sfx_finalize.py:290-292` (dual `sizes`/`devices` override) vs `Voyage/voyage/config.py:369-372` (`SfxConfig.device = "cpu"`)
- **Area:** SFX/audio benchmark VRAM accounting + device defaults (below both previous windows; 051 covers dropped VRAM fields in reports, 115 covers missing CUDA preflight — neither covers unguarded CUDA calls or the cpu-vs-cuda:0 default split)

## Description

Two coupled edges:

1. **Unguarded CUDA peak calls.** Both GPU benchmarks bracket every window with `torch.cuda.reset_peak_memory_stats()` (`sfx_mmaudio.py:301`, `audio_acestep.py:220`) and read `torch.cuda.max_memory_allocated() / BYTES_PER_GIB` (`:317`, `:226`). Neither checks `torch.cuda.is_available()` first. On a CPU-only box (slim image + torch-CPU, or `CUDA_VISIBLE_DEVICES=""`), `reset_peak_memory_stats` raises (no CUDA context / uninitialized), so `voyage benchmark audio` and any future `voyage benchmark sfx` (154) die with a torch RuntimeError instead of reporting "no CUDA" or CPU walls. The video workers already guard (`video_ltxv.py:772`, `video_longlive.py:1038`, `video_causvid.py:958` all branch on `is_available()`; `augment_worker._resolve_device` at `:127-138` falls back to CPU) — the audio/SFX benchmarks are the only peak reporters without a guard. `_require_torch` (`sfx_mmaudio.py:46-52`, `audio_acestep.py:43-55`) guards *importability*, not *CUDA presence*, so it does not cover this.
2. **Device default split.** `SfxConfig.device` defaults to `"cpu"` (`config.py:370`) while both workers' `handle_init` default to `"cuda:0"` when the payload omits `device` (`sfx_mmaudio.py:93`, `audio_acestep.py:96`). A headless caller that inits a worker without threading the config through (benchmark probes, e2e drivers, tests) silently gets CUDA placement the config never asked for; conversely the dual-shard override (`sfx_finalize.py:290-292`: `devices = ["cuda:0", "cuda:1"]`) assumes two GPUs without consulting either default. The model-size defaults agree (`large_44k_v2` both sides), so device is the lone drifter — exactly the field that decides whether edge (1) fires.

## Rationale

Benchmarks must be runnable everywhere (§104: "honest unknowns off-GPU" is already `_benchmark_env`'s doctrine at `cli.py:1425-1440`). A benchmark that tracebacks on CPU is worse than no benchmark — it teaches operators the probe is GPU-only, so the 2060-ladder and CI boxes stop running it. And disagreeing defaults are the classic source of "works in finalize, crashes in benchmark" reports: finalize threads `SfxConfig.device` explicitly (`render_sfx_bed:262-264,301-314`), benchmark probes rely on worker fallbacks.

## Live evidence

- `sed -n '300,318p' voyage/workers/sfx_mmaudio.py` — reset at `:301`, `max_memory_allocated` at `:317`, no `is_available` between `:254-332`; `sed -n '219,227p' voyage/workers/audio_acestep.py` — same unguarded pair.
- `rg -n "is_available" voyage/workers/sfx_mmaudio.py voyage/workers/audio_acestep.py` → no hits in either file (compare 3+ hits each in the video workers + `audio/mmaudio_sfx.py:299` + `audio/acestep.py:202`).
- `sed -n '77,101p' voyage/workers/sfx_mmaudio.py` — `_device = "cuda:0"` default at `:39`, fallback at `:93`; `sed -n '79,97p' voyage/workers/audio_acestep.py` — same `"cuda:0"` at `:37/:96`; `sed -n '369,372p' voyage/config.py` — `device: str = "cpu"`.
- `sed -n '290,292p' voyage/sfx_finalize.py` — dual shard hardcodes `cuda:0/cuda:1` regardless of either default.
- Overlap check: 051 is report *shape* (percentiles/VRAM columns once measured); 115 is soak/benchmark skipping CUDA *preflight* (CLI gate); 021 is `_CUDA_BACKENDS` membership. None names the unguarded `torch.cuda.*` calls or the cpu-vs-cuda:0 default split.

## Repro

CPU-only (torch-CPU build, no GPU): call either `handle_benchmark` with `warmup=0, measured=1` → `torch.cuda.reset_peak_memory_stats()` raises before the first window renders (SFX needs the ffmpeg probe first at `:276-299`, ACE renders immediately at `:222-224` — both die at the reset, not at model load). Defaults: `handle_init({})` → `device == "cuda:0"` while `SfxConfig().device == "cpu"` — same "default" answers differently depending on which layer is asked.

## Fix candidates

1. Guard both pairs: `if torch.cuda.is_available(): reset…` and `peak = max_memory_allocated()/GIB if is_available() else 0.0/None`, reporting `vram_peak_gib: null` + `cuda_available: false` off-GPU (extends `_benchmark_env`'s honest-unknown doctrine into the worker report).
2. Unify defaults: workers default `device` from the same constant the config uses (or require it — `handle_init` raises when `device` is absent, forcing callers through `SfxConfig`/`AudioConfig`); at minimum change the worker fallback to `"cpu"` so an unthreaded init is safe and the GPU path is always explicit.
3. Gate the dual shard (`sfx_finalize.py:290-292`) on `augment_devices`-style visibility (see 158) so `cuda:1` is never assumed from defaults alone.
4. Tests: CPU-stubbed `torch.cuda.is_available() == False` → both benchmarks return 0 with null VRAM; init-default test pins `handle_init({})["device"] == SfxConfig().device`.

## Refs

- `Voyage/voyage/workers/sfx_mmaudio.py:37-52,77-101,254-332`; `Voyage/voyage/workers/audio_acestep.py:35-55,79-97,194-241`; `Voyage/voyage/config.py:360-372`; `Voyage/voyage/sfx_finalize.py:280-314`; DESIGN §104 (benchmarks), §40 (residency).
- Adjacent, not overlapping: 051 (report content); 115 (CLI preflight); 154 (missing sfx/augment targets — the CLI surface this file's guarded probes would serve); 158 (dual-shard cuda:1 gating).
