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

## Progress log (2026-09-30)

Premise re-verified against live code before any change (all hold):
- `voyage/audio/acestep.py:180-181`: `except Exception` → `VoyageError("ACE-Step render failed: ...")` — wraps CUDA OOM (confirmed). Same shape in `initialize` (`:108-109`, `"ACE-Step stack init failed"`).
- `voyage/workers/loop.py:103-110`: `except VoyageError` → `retryable=False` (confirmed).
- `voyage/rpc.py:116`: `failure(..., retryable=True)` default; `:125-127` generic `except Exception` → `WORKER_ERROR` retryable; `:372-376` wire `retryable=True` → supervisor `RecoverableWorkerError` (restart), `False` → `FatalWorkerError`.
- `voyage/supervisor.py:510`: only `RecoverableWorkerError` restarts (read-only; file out of scope).
- No shared `is_oom()` exists in-tree (grep: zero hits) → local predicate per the issue's fallback.
- `loop.py` deliberately UNTOUCHED: branch-on-class is correct; the bug is class erasure in `acestep.py`. Raising `RecoverableWorkerError` there would NOT fix it (it is a `VoyageError` subclass → same fatal branch); raw propagation is the correct route.

TDD (ephemeral `/tmp/issue052_oom_repro.py`, host stdlib-only, never committed):
- Pre-fix: 1/4 — both OOM shapes (`RuntimeError("CUDA out of memory...")`, `OutOfMemoryError`-typed) plus init OOM all wrapped as `VoyageError` (bug reproduced); non-OOM still wrapped (guard).
- Post-fix: 4/4 PASS.

Wire proof (ephemeral `/tmp/issue052_wire.py`, in-container via `loop.serve` + fake `acestep.inference`):
- OOM → `ok=False, code=WORKER_ERROR, retryable=True` (supervisor restarts).
- Non-OOM → `code=VoyageError, retryable=False` (no over-correction).

## Resolution (2026-09-30) — FIXED

`voyage/audio/acestep.py` only (+30/-2): new `is_oom()` (torch-free: class-name `OutOfMemoryError` OR `out of memory` substring, both documented) + bare `raise` guards in `render_take` and `initialize` so OOMs propagate unwrapped to the worker loop's retryable branch. Non-OOM failures still wrap as `VoyageError`.

Evidence: fail-first/pass-after above; related audio suites 87 passed (`test_acestep_contract`, `test_audio_*`, `test_failure_policy`, `test_commit_hardening`, `test_rpc_*`); ruff + format + mypy-strict green on touched files; full suite 1129 passed with 5 failures all foreign (`test_paths`/`test_run_relative_consumer`/`test_state_integrity` — concurrently-modified `voyage/paths.py`, `MediaError: stored path escapes the run dir`, outside this scope) and one foreign unformatted file (`tests/test_supervisor_hardening.py`, left for its owner).

Residual / follow-ups (need files outside this scope): supervisor-side `gc`/evict hygiene around the audio swap (issue's 3rd candidate; `supervisor.py` untouched); plain `MemoryError` (host RAM) still wraps as fatal — CUDA-only per the issue; shared predicate for 049 stays a future import of `acestep.is_oom` by its owning pass.
