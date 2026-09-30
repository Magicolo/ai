# 158 — SFX two-worker mode hardcodes cuda:1 with no visibility gate, and the CUDA preflight does not know SFX exists

- **Severity:** MEDIUM (single-GPU crash + preflight blind spot — explicit `--sfx-workers 2` on a 1-GPU box fails at worker init, not at parse)
- **File:line:** `Voyage/voyage/sfx_finalize.py:280-292` (worker-count validation + dual override), esp. `:283-287` (only a model-size guard) → `Voyage/voyage/augment.py:238-263` (`augment_devices`: the visibility probing SFX never does) → `Voyage/voyage/cli.py:1193-1205` (`_cuda_offenders`: video/audio only) + `:1175` (`_CUDA_BACKENDS` without sfx) + `:1757-1762` (caption pins that change SFX work without touching placement)
- **Area:** SFX device placement + CUDA gating (below both previous windows; 054 covers the two-worker ledger race, 021 covers `_CUDA_BACKENDS` pollution — this file is the missing cuda:1 presence check + the missing sfx membership)

## Description

`render_sfx_bed` accepts `num_workers in (1, 2)` (`sfx_finalize.py:280-281`) and, for 2, unconditionally pins `devices = ["cuda:0", "cuda:1"]` with `sizes = [small, small]` (`:290-292`). The only guard is a model-size check (`:283-287`: `device == "cuda:1" and model_size != "small_44k"` → error). There is no check that cuda:1 *exists*: no `CUDA_VISIBLE_DEVICES` inspection, no `nvidia-smi` probe, no `torch.cuda.device_count() >= 2`. On a single-GPU box (4060 Ti 16 GB, the reference card) `--sfx-workers 2` starts the second `SubprocessWorker` with `device: "cuda:1"`, whose lazy `initialize` fails at first-window render — after the first worker already rendered half the windows and appended ledger lines, so a retry inherits a half-populated ledger (see 153's duplicate-line edge). The codebase already owns the correct probing: `augment_devices` (`augment.py:238-263`) prefers `CUDA_VISIBLE_DEVICES`, then a test seam, then live `nvidia-smi -L`, honors explicitly-emptied visibility with `()`, and caps at two devices (`MAX_PARALLEL_DEVICES`). SFX duplicates the "two devices" conclusion without any of the premises.

The preflight cannot catch it either: `_CUDA_BACKENDS` (`cli.py:1175`) is `{"ltxv", "longlive2", "causvid", "acestep"}` — no `"mmaudio"`/`"sfx"` member — and `_cuda_offenders` (`:1194-1205`) inspects `config.video.backend` and `config.audio.backend` only. An `mmaudio`-SFX run on a CPU-only container (`SfxConfig.backend = "mmaudio"`, video/audio fake) reports zero offenders and proceeds past `_require_cuda_stack` (`:1218` checks video-or-audio membership), dying at SFX worker init instead of the parse-adjacent fast-fail every CUDA video/audio backend gets. Caption pins (`--sfx-caption` at `cli.py:1794-1799`, music/video pins at `:1757-1762`) change SFX render *content* without touching placement, so a user iterating on captions re-hits the same ungated placement each retry.

## Rationale

Two-GPU sharding is an explicit user flag (`--sfx-workers`, `choices=[1, 2]` at `cli.py:1811-1817`), so the failure mode is opt-in — but the error arrives late (post-render, half-ledger) instead of at startup, and the message blames CUDA/model init rather than the flag. The single-GPU box is the common case (reference hardware is one 4060 Ti; CI is CPU-only); a flag that only works on 2-GPU boxes must fail fast with "needs 2 GPUs, saw N" naming the flag. And every CUDA backend belongs in the preflight set — the TUI already imports `_CUDA_BACKENDS` as its single source (`tui_state.py:546-548`), so the omission propagates to the launcher's GPU warning too (156's default-split file covers the cpu-vs-cuda:0 half; this file covers membership + presence).

## Live evidence

- `sed -n '280,292p' voyage/sfx_finalize.py` — `:280-281` count check, `:283-287` size-only guard, `:290-292` unconditional `cuda:0/cuda:1`; `rg -n "device_count|VISIBLE|nvidia-smi|is_available" voyage/sfx_finalize.py` → no hits.
- `sed -n '238,263p' voyage/augment.py` — full visibility cascade SFX skips (explicit-empty → `()`, cap at 2, `smi_count` seam for tests).
- `sed -n '1175,1205p' voyage/cli.py` — `_CUDA_BACKENDS` without sfx/mmaudio; `_cuda_offenders` reads video+audio only; `rg -n "sfx|mmaudio" voyage/cli.py | sed -n '1,20p'` shows SFX flags/verbs but no preflight membership.
- `sed -n '1757,1762p' voyage/cli.py` — caption pins adjacent to SFX flags (content changes, placement unchanged).
- Overlap check: 054 is the two-worker *ledger* race (needs threads+lock; this file is placement/presence, reproduces serially); 021 is `_CUDA_BACKENDS` pollution/completeness from the video side (this file is the sfx-membership + runtime-presence remainder); 156 is benchmark VRAM guards + cpu-vs-cuda:0 defaults (this file is the cuda:1 existence gate).

## Repro

Single-GPU box: `voyage sfx --run <dir> --sfx-workers 2` (or finalize with `--sfx-workers 2`) → worker slot 1 inits with `device cuda:1`, fails at first `generate_sfx` with a CUDA device error after slot 0 already appended ledger lines. CPU-only container: `SfxConfig(backend="mmaudio")` + fake video/audio → `_cuda_offenders(config)` returns `[]`, `_require_cuda_stack` passes, SFX init dies late. Static: `python3 -c "from voyage.cli import _CUDA_BACKENDS; assert 'mmaudio' not in _CUDA_BACKENDS and 'sfx' not in _CUDA_BACKENDS"`.

## Fix candidates

1. Presence gate in `render_sfx_bed`: when `num_workers == 2`, resolve visibility via `augment_devices()` (single source for "how many GPUs are visible") and raise `MediaError("--sfx-workers 2 needs 2 visible GPUs, saw …")` before starting any worker — same fail-fast shape as `:280-281`.
2. Preflight membership: add `"mmaudio"` (and any sfx backend name) to the CUDA set or extend `_cuda_offenders` with an `config.sfx.backend` branch, so CPU-container + mmaudio-SFX fast-fails with the standard torch/image pointer (`_cuda_stack_error`); update the TUI GPU warning via the same single source.
3. Keep the `:283-287` size guard (it encodes the measured 6 GB ladder result) and document the pairing matrix: 1 GPU → `--sfx-workers 1` any size; 2 GPUs → `--sfx-workers 2` forces small/small on cuda:0/cuda:1.
4. Tests: single-GPU stub (`smi_count=0`/`CUDA_VISIBLE_DEVICES=0`) + `num_workers=2` → `MediaError` before any worker start, no ledger lines appended; preflight test pins mmaudio-SFX as an offender on a torch-free box.

## Refs

- `Voyage/voyage/sfx_finalize.py:255-314`; `Voyage/voyage/augment.py:216-263`; `Voyage/voyage/cli.py:1175-1260,1773-1817`; `Voyage/voyage/tui_state.py:542-548`; DESIGN §40 (residency/pairing), §104 (ladders).
- Adjacent, not overlapping: 054 (two-worker ledger race — same flag, threading half); 021 (CUDA set from the video side); 156 (benchmark guards + device defaults); 153 (half-ledger duplicate lines the late failure leaves behind).

## Progress log

- 2026-09-30: re-verified live in slim `voyage:latest`: `render_sfx_bed` with `num_workers=2` resolves visibility via `augment_devices()` and raises `MediaError("--sfx-workers 2 needs 2 visible GPUs…")` before any worker starts (no ledger writes); `_cuda_offenders` lists `sfx 'mmaudio'` for a fake/fake/mmaudio config (preflight membership present); `--sfx-workers` help names the 2-visible-GPU requirement. Verdict: premise confirmed, fix present.
- TDD: `tests/test_worker_perf_rank2.py` (4 tests: single-GPU fail-fast with no ledger, two-GPU fake shard end-to-end, preflight offender pin, help-text pin) failed on base HEAD (red) and passes with the fix. `voyage/cli.py` touched only at the `--sfx-workers` help line; validate/finalize/models regions belong to concurrent groups and were not restructured.

## Resolution

- Fixed in `voyage/sfx_finalize.py` (presence gate via `augment_devices()`, fail fast at plan time) + `voyage/cli.py` help line only (`--sfx-workers` help states the 2-visible-GPU need). The `:283-287` cuda:1 size guard is kept (6 GB ladder result). Tests: `tests/test_worker_perf_rank2.py` (158 block). Residual: `_CUDA_SFX_BACKENDS`/`_cuda_offenders` membership lives in the cli validate region owned by a concurrent group (021) — pinned by test but not modified here; the 1-GPU/2-GPU pairing matrix (1 worker any size, 2 workers small/small) is documented in help + plan error, full prose belongs in a docs follow-up.
