# 025 — God modules + quadruple backend registries with no compiler-enforced unity

- Severity: MEDIUM (structure: agreement without enforcement)
- Group: structure/registries — Rank: 4/5
- File:line: `voyage/cli.py` (2051 lines), `voyage/supervisor.py:88-96` (`VIDEO_WORKER_MODULES`/`STREAMING_VIDEO_BACKENDS`), `voyage/backends.py:74-89` (derived views), `voyage/config.py:117` (`BACKEND_REGISTRY`), `voyage/cli.py:1174` (`_CUDA_BACKENDS` — see 021)

## Description

`cli.py` mixes argparse construction, duration math, planning, disk guards, orchestration (`cmd_generate` = init+run+validate+finalize), and view rendering. Three separate modules restate "which backends stream / exist / need CUDA":

| set | location | derivation |
|-----|----------|------------|
| `BACKEND_REGISTRY` | `config.py:117` | source |
| `BACKEND_STATE_MODES` / `_STREAMING_BACKENDS` | `backends.py:74/82` | derived from registry ✓ |
| `VIDEO_WORKER_MODULES` / `STREAMING_VIDEO_BACKENDS` | `supervisor.py:88/96` | hand-maintained ✗ |
| `_CUDA_BACKENDS` | `cli.py:1135` | hand-maintained ✗ (diverged — issue 021) |

Today the streaming sets agree (all `['causvid','longlive2','ltxv']`), but agreement is coincidental — nothing fails at import if `supervisor.STREAMING_VIDEO_BACKENDS` drops `causvid` while `backends._STREAMING_BACKENDS` keeps it. `VIDEO_WORKER_MODULES` keys duplicate `BACKEND_REGISTRY` keys by hand.

## Rationale

- The codebase already learned this lesson once (issues 022/025 unified presets into `BACKEND_REGISTRY`) but left three derived sets hand-maintained. The one set that could diverge already has (021).
- God modules raise the cost of every future unification: the streaming/CUDA/worker-module knowledge is scattered across CLI, supervisor, and adapter layers instead of projected from the registry.
- An import-time assertion is cheap; a runtime divergence on a GPU box (wrong payload shape, missed CUDA preflight) is expensive.

## Live evidence (grep, 2026-09-30)

```
$ grep -n "STREAMING_VIDEO_BACKENDS\|_STREAMING_BACKENDS\|VIDEO_WORKER_MODULES\|BACKEND_REGISTRY" voyage/*.py
voyage/model_registry.py:173: # in supervisor VIDEO_WORKER_MODULES + STREAMING_VIDEO_BACKENDS, CLI via
voyage/supervisor.py:88:VIDEO_WORKER_MODULES = {
voyage/supervisor.py:96:STREAMING_VIDEO_BACKENDS = ("longlive2", "ltxv", "causvid")
voyage/config.py:117:BACKEND_REGISTRY: dict[VideoBackendName, BackendRecord] = {
voyage/backends.py:74:BACKEND_STATE_MODES: dict[...] = {name: record.state_mode ...}
voyage/backends.py:82:_STREAMING_BACKENDS: frozenset[...] = frozenset(name ... if record.streaming)
voyage/cli.py:1135:_CUDA_BACKENDS = frozenset({"ltxv", "longlive2", "causvid", "acestep"})
```

`backends.py:85-89` docstring (the codebase knows):

```python
"""Backends taking the multi-block payload (mirrors `supervisor.STREAMING_VIDEO_BACKENDS`).

Derived from config.BACKEND_REGISTRY (issue 022): the supervisor track
owns unifying this with supervisor.STREAMING_VIDEO_BACKENDS.
"""
```

## Repro

```bash
wc -l voyage/cli.py voyage/supervisor.py voyage/tui.py voyage/model_registry.py voyage/config.py voyage/backends.py
grep -n "STREAMING" voyage/supervisor.py voyage/backends.py
docker run --rm -v "$PWD:/app" -w /app voyage:latest python3 -c "
from voyage import backends
from voyage import supervisor
print(sorted(backends._STREAMING_BACKENDS), sorted(supervisor.STREAMING_VIDEO_BACKENDS))"
# Today equal — the point is nothing enforces it
```

## Fix candidates

1. Derive all three sets from `BACKEND_REGISTRY` (streaming flag, worker-module map, CUDA set); delete the hand-maintained literals.
2. Split `cli.py` into `cli_{parse,plan,run_ops,inspect}.py` behind the existing `build_parser` seam (mechanical, no behavior change).
3. Add an import-time assertion/test that the supervisor/adapter/CUDA sets equal the registry projection (fails the gate on drift, not on a GPU box at midnight).

## Refs

- Single-source precedent in-tree: `BackendRecord` docstring at `voyage/config.py:88-97` ("every preset dict, state-mode map, and streaming set below derives from the registry instead of restating these values") — the fix is extending that rule to the two modules that opted out.

## Progress log

- 2026-09-30 (surface-rank2 track): premise re-verified live — `backends._STREAMING_BACKENDS`, `supervisor.STREAMING_VIDEO_BACKENDS`, and the registry streaming projection all equal `{'causvid','longlive2','ltxv'}`; `VIDEO_WORKER_MODULES` keys equal BACKEND_REGISTRY keys; `cli._CUDA_*` sets are already registry-derived (021 landed). No unity enforcement exists (no import-time assert, no test). Verdict: CONFIRMED (agreement coincidental, not enforced).
- Fix (test-only by ownership: supervisor module-header region is owned by concurrent groups, so no source derivation here): import-time unity test in `tests/test_surface_rank2.py` — streaming triple-equality, worker-module key equality, and CUDA-set-vs-device projections. Fails the gate on drift instead of on a GPU box at midnight. Source derivation of the two supervisor literals stays open for the supervisor owner.
- Evidence: unity tests pass (agreement holds today); ruff + format-check + mypy strict clean. Note: ruff 0.16.9 SIM300 treats SCREAMING_CASE module attrs as constants — asserts read `expected == module.SET` per the linter.

## Resolution

- GUARDED 2026-09-30 (not restructured): registry unity is now gate-enforced by test; the `cli.py` split + supervisor derivation (fix candidates 1-2) remain open and owned elsewhere.
