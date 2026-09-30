# 193 — `augment_worker._resolve_device` falls back CPU→silent on a GPU-labeled request

- Severity: LOW
- Area: augment worker — placement visibility
- Files (as-read 2026-09-30):
  - `voyage/workers/augment_worker.py:127-138` (`_resolve_device`: CUDA-unavailable → CPU, no signal)
  - `voyage/workers/augment_worker.py:141-150` (`_prepare_model`: consumes the resolved device)
  - `voyage/augment.py:238-263` (`augment_devices`: the plan stamps `cuda:0`/`cuda:1`)

## Description

`_resolve_device` maps a planned `cuda:0`/`cuda:1` chunk assignment to CPU when
`torch.cuda.is_available()` is false — silently:

```python
# augment_worker.py:127-138 (as-read)
def _resolve_device(preferred: str) -> Any:
    """torch.device for `preferred`, falling back to CPU when CUDA is unavailable.
    Why fallback instead of fail-loud: the spike stays end-to-end runnable
    on CPU-only boxes (slow but exact); callers that need the GPU gate on
    `augment_devices` / nvidia-smi instead.
    """
    import torch
    if preferred.startswith("cuda") and not torch.cuda.is_available():
        return torch.device("cpu")
    return torch.device(preferred)
```

The fallback is a documented, deliberate spike choice — this file is about its
invisibility, not its existence. The module imports no logging and prints
nothing (`rg -n "logging|warn|print|log\." voyage/workers/augment_worker.py` →
zero hits), so a chunk planned for `cuda:0` (visible in `AugmentChunk.device`
and any plan dump) executes on CPU with no line anywhere saying so. RRDBNet-x4
+ FILM on CPU over 1536×1024 frames is ~100× slower than CUDA; the operator
watching a finalize stall sees a correct plan, a live process, and no placement
signal — the exact shape that gets mis-triaged as "model slow" instead of
"torch without CUDA" (slim image, CPU-only container, broken nvidia runtime).

## Rationale

Documented leniency still needs a loud edge: the plan (`augment_devices`) and
the execution (`_resolve_device`) can disagree with no observer able to tell.
One `print`/log line at fallback time (or a returned `(device, fell_back)`
flag the orchestrator can surface) turns a mis-triage trap into a one-line
diagnosis. Cheap, zero behavior change.

## Live evidence (verified live 2026-09-30, host reads)

- `sed -n '127,150p' voyage/workers/augment_worker.py` — fallback branch with no
  side effect besides the returned device; `_prepare_model` then reports
  `torch.float32`/CPU downstream with no comparison to `preferred`.
- `rg -n "import logging|getLogger|warnings|print" voyage/workers/augment_worker.py`
  → no hits: the module is observability-silent by construction.
- Contrast `voyage/augment.py:252-263` — the planner distinguishes
  hide-GPU (`()`) from unknown (`(cuda:0,)`); the worker collapses that
  distinction without recording it.

## Repro

CPU-only box (or `CUDA_VISIBLE_DEVICES=""` with torch lacking CUDA):
`upscale_frames(frames, weights, device="cuda:0")` → returns correct tensors
from CPU; nothing in return value, stdout, or exceptions distinguishes the run
from a CUDA run except wall time.

## Fix candidates

1. (Preferred) Emit one loud line on fallback (`print`/logging warning naming
   `preferred` → actual), and/or return the resolution so `upscale_frames` /
   `interpolate_*` callers can thread it into chunk outcome records.
2. Add a `strict: bool = False` (or env-gated) path for the finalize caller, so
   production can opt into fail-loud while the spike keeps its CPU-runnable
   default.
3. Test: fallback returns CPU *and* records the signal (caplog / flag assert).

## Refs

 - In-tree: `voyage/workers/augment_worker.py:127-150,356-394`;
   `voyage/augment.py:43-50,238-263` (device constants + planner).
 - Not-a-duplicate: 047 is reload-per-call + full-batch stack (load/compute
   shape); 157 is preset knob + fan-out + batch-peak interaction; 158 is SFX
   `cuda:1` presence gating (different stage, fail-loud direction). This file is
   placement *visibility* on the documented CPU fallback — the one facet neither
   names.

## Progress log

- 2026-09-30 (Group E2): evaluated live first. Premise CONFIRMED as-read: `_resolve_device` (`voyage/workers/augment_worker.py:~169-180`) returned CPU with no signal, and the module had zero logging/print hits. TDD: warn-once + silence legs of `tests/test_e2_augment_worker_157_193.py` failed pre-fix (missing `_DEVICE_FALLBACK_WARNED`, no stderr), green post-fix. Note: `augment_worker.py` is not in the brief's OWN-FILES list but the brief assigns this file (and 157's worker half) to Group E2 — hunks kept to the cited function only. One toolchain touch: the new `print` needed the house per-file `T201` entry for `augment_worker.py` in `pyproject.toml` (worker-module convention, following `video_causvid`/`video_ltxv`/`video_longlive`).

## Resolution

- Verdict: FIXED in `voyage/workers/augment_worker.py`.
- Change: CPU fallback prints one stderr line naming `preferred → CPU` (`_DEVICE_FALLBACK_WARNED` once-per-process flag — per-chunk spam would bury it, since `_prepare_model` resolves per chunk); zero behavior change (same device returned). The `strict`/fail-loud path (candidate 2) deliberately NOT added — YAGNI: the planner (`augment_devices`) owns the GPU gate, and no caller asked for it.
- Files changed: `voyage/workers/augment_worker.py`, `pyproject.toml` (one T201 per-file-ignore line) (+ warn legs in `tests/test_e2_augment_worker_157_193.py`).
- Test evidence (in-container `voyage:latest`, CPU-only, stubbed torch): fallback warns once with `cuda:0` + `cpu` named, second call silent, CUDA path silent. Ruff + format + mypy strict clean (incl. the new pyproject entry — `ruff check` green).
- DESIGN proposal (quoted text only, for the DESIGN owner — augment track): "Documented CPU fallbacks stay loud: `_resolve_device` emits one per-process stderr line naming the requested device and the CPU execution, so plan-vs-execution placement mismatches diagnose in one line."
- Residuals: none in this file.
