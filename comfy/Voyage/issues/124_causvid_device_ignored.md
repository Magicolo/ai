# 124 — CausVid session ignores its `device` param (hardcoded `"cuda"` ×3) and all three video workers report GPU 0's stats

- **Severity:** Medium-Low (multi-GPU correctness + observability — `cuda:1` sessions encode on the wrong GPU; VRAM telemetry lies)
- **File:line:**
  - `Voyage/voyage/workers/video_causvid.py:550-551` (`InferencePipeline(config, device="cuda")`, `pipeline.to(device="cuda", ...)` — `self._device` unused), `:608` / `:612` (`pipeline.text_encoder.to("cuda")` / `.to("cpu")` shuttle in `_encode_conditionals` — `self._device` stored at `:531` but never used for placement).
  - Telemetry, all three workers: `video_causvid.py:973-974`, `video_ltxv.py:777-778`, `video_longlive.py:1051-1052` (`torch.cuda.get_device_name(0)` + `torch.cuda.mem_get_info()` — hardcoded index 0 while the session device may be `cuda:1`); `handle_health` variants (`causvid:996-999`, `ltxv:795-798`, `longlive:1067-1070`) call `mem_get_info()` with no device arg (current-device 0).
- **Area:** workers-internals tail — video worker device plumbing (below pass-1 coverage; 074 fixed the ltxv *embed* device, not this cluster)

## Description

`CausvidSession.__init__` accepts `device`, stores `self._device`, and uses it in exactly one place (`_fresh_noise`, `:581`). Everything else is literal `"cuda"`:

- The pipeline is constructed and moved on `"cuda"` (≡ `cuda:0`), and the 11 GB T5 shuttle in `_encode_conditionals` parks/encodes on `"cuda"` — so a `device="cuda:1"` session builds its DiT on GPU 0, shuttles T5 to GPU 0, then draws noise on GPU 1. Best case the `device` param is a lie and everything runs on 0 anyway; worst case cross-device tensors fault deep in `pipeline.inference` (the 2-GPU `video-aug-cuda:0/SFX-cuda:1` pairing in the tree shows non-zero devices are a real configuration).
- Contrast ltxv, which threads `self._device` through encode/offload paths (the 074 fix) — causvid never got the same pass.
- Independently, `handle_init` in all three workers reports `get_device_name(0)` / `mem_get_info()` regardless of the requested device, and `handle_health` reports default-device memory. On a `cuda:1` worker, `init` returns GPU 0's name and free-VRAM — the supervisor's VRAM-fit reasoning and every benchmark `vram_peak_gib` attribution (via `max_memory_allocated()`, also device-default) silently describe the wrong card.

## Rationale

Single-GPU boxes never notice (0 == 0), which is why this survived: every E2E in the log ran `cuda:0`. But the config schema *offers* per-backend `device` fields, the TUI exposes them, and the 2-GPU augment/SFX pairing proves multi-device is a supported direction. A device knob that is accepted, stored, and then ignored in 3 of 4 placement sites is worse than no knob — operators debugging a `cuda:1` OOM will be reading GPU 0's telemetry.

## Evidence (verified live 2026-09-30, tree reads)

```
video_causvid.py:531:  self._device = device
video_causvid.py:550:  pipeline = InferencePipeline(config, device="cuda")
video_causvid.py:551:  pipeline.to(device="cuda", dtype=torch.bfloat16)
video_causvid.py:581:  generator = torch.Generator(device=self._device)...  # the ONLY use
video_causvid.py:608:  pipeline.text_encoder.to("cuda")
video_causvid.py:612:  pipeline.text_encoder.to("cpu")
video_causvid.py:973-974 / video_ltxv.py:777-778 / video_longlive.py:1051-1052:
  name = torch.cuda.get_device_name(0); free_gib, total_gib = torch.cuda.mem_get_info()
```

`grep -n "self._device" voyage/workers/video_causvid.py` → `:531` (store), `:581` (noise), `:722` (`scaled.device`, read-only) — zero placement uses.

## Repro

1. Static: the grep above (no GPU needed).
2. Live (needs 2 GPUs): `init {models_dir, device: "cuda:1"}` → `nvidia-smi` shows the ~11 GB T5 + DiT resident on GPU 0, and the `init` response names GPU 0.

## Fix candidates

1. Thread the session device through causvid placement: `InferencePipeline(config, device=self._device)`, `.to(device=self._device)`, T5 shuttle `.to(self._device)` (mirror the ltxv 074 fix); fail fast in `handle_init` if the index is out of range.
2. Device-indexed telemetry everywhere: `torch.cuda.get_device_name(torch.cuda.device(device))` / `mem_get_info(device)` in all three `handle_init`s; pass the device to `reset_peak_memory_stats`/`max_memory_allocated`-class reads in benchmarks (or document they are device-0).
3. Test without GPUs: fake-torch assertion that every `.to(` site in the session receives the init device (the ltxv session already passes this shape of test — port it).

## Refs

- `Voyage/voyage/workers/video_causvid.py:516-576,589-614,973-999`; `Voyage/voyage/workers/video_ltxv.py:757-799`; `Voyage/voyage/workers/video_longlive.py:1016-1071`; `Voyage/voyage/config.py:100-114` (per-backend `device` fields).
- Adjacent, not overlapping: 021 (CLI *registry* CUDA sets — this file is worker *placement* strings); 074 (augment loader formats — untouched); the ltxv `_encode` device fix noted in `video_ltxv.py:403-405` (the precedent to port).
