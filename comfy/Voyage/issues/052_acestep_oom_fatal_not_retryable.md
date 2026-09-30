# 052 — ACE-Step OOMs are wrapped as non-retryable `VoyageError` → instant Fatal, no restart

**Severity:** HIGH

**File:line:** `voyage/audio/acestep.py:172-183` (`render_take`), `voyage/workers/loop.py:103-110` (taxonomy), `voyage/supervisor.py:465-521` (`_call_with_restart`), `voyage/supervisor.py:1011-1082` (audio swap)

**Description:**
`render_take` catches all `Exception` (including CUDA OOM) and re-raises `VoyageError(f"ACE-Step render failed: ...")`. `loop.serve` maps any `VoyageError` subclass to `retryable=False` (deterministic/fatal), so a transient DiT OOM — the expected event on a 16 GiB card during the 45 s take render — bypasses the supervisor restart budget entirely and fails the run. Contrast: raw `RuntimeError: CUDA out of memory` from MMAudio/LTXV paths stays retryable (`WORKER_ERROR`) and gets a restart + resume. The supervisor additionally performs zero `gc`/cache hygiene around the audio swap (0 `gc.collect`/`empty_cache` hits in `supervisor.py`) despite owning the evict→render→evict→rebuild sequence where the 14.5 GiB ACE stack vs video DiT residency is the known OOM site.

**Rationale:**
Error-taxonomy branch-on-class is correct, but the ACE compat layer erases the recoverable/fatal distinction by wrapping everything as the base class. OOM must stay recoverable (evict + retry once, ideally with shorter take or fp downgrade), never fatal on first hit.

**Live evidence (current tree):**
```python
acestep.py:180-181: except Exception as failure:
    raise VoyageError(f"ACE-Step render failed: {failure}") from failure
loop.py:103-110: except VoyageError as exc: ... failure(..., retryable=False)
supervisor.py:486: except RecoverableWorkerError as exc:  # only Recoverable restarts; Fatal propagates
supervisor.py:1029: _call_with_restart(video, evict_gpu) → 1033: _call_with_restart(audio, generate_audio) → 1046/1064 evict/rebuild
rg "gc.collect|empty_cache" voyage/supervisor.py → 0 hits
acestep.py:199-203: evict() itself does gc.collect() + empty_cache — worker-side only, supervisor-side missing
```

**Repro (CPU, no weights):**
```python
from voyage.workers.loop import serve
from voyage.errors import VoyageError, RecoverableWorkerError
# Any torch.cuda.OutOfMemoryError raised inside render_take exits as:
#   VoyageError("ACE-Step render failed: CUDA out of memory...")
# → wire code = "VoyageError", retryable=False → supervisor raises FatalWorkerError, no restart.
```

**Fix candidates:**
- Don't wrap `torch.cuda.OutOfMemoryError` / `RuntimeError("...out of memory...")` as `VoyageError`; let it propagate as retryable, or re-raise as `RecoverableWorkerError` explicitly.
- Add an `is_oom()` predicate in `video_common` or `errors.py` and an OOM branch in `_call_with_restart` (evict opposite stack + `gc` + retry once before consuming restart budget).
- Supervisor-side `gc.collect()` discipline around the audio GPU swap (already done worker-side; missing supervisor-side).

**Refs:** `voyage/errors.py:19-24` taxonomy; PyTorch OOM FAQ (recover outside `except` — the wrapped exception pins frames); audio-eviction lesson (`del` alone frees nothing).

**Overlaps with:** 049 (LTXV narrow OOM catch — shared `is_oom` predicate would fix both; not duplicates).
