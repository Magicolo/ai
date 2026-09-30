# 128 — CausVid `_vae_encode_slice` builds an empty window when `overlap_frames=1` (negative-slice-to-empty)

- **Severity:** Low (narrow reachability — custom configs only — but a silent empty tensor into the VAE encode path)
- **File:line:** `Voyage/voyage/workers/video_causvid.py:466-479` (`_vae_encode_slice`: `end = -4 * (overlap_frames - 1)`, `start = end - 1`, `window = video[:, start:end, :]`); validators `:125-135` (`validate_overlap_frames`: multiple-of-block only, no floor above 0)
- **Area:** workers-internals tail — causvid tail-slice re-encode (below pass-1 coverage)

## Description

For `overlap_frames=1`: `end = -4*0 = 0`, `start = -1`, and `video[:, -1:0, :]` is **empty** (a negative start with a 0 stop and positive step selects nothing). Verified live with the exact slice expression: shape `(1, 0, 16, 60, 104)` vs `(1, 1, 16, 60, 104)` for overlap 3. The empty window flows into `_vae_encode_window` → `wrapper.model.encode` with `T=0`, where it fails deep inside the VAE (or worse, returns a degenerate encoding that `torch.cat`s into `start_latents` and trips the shape guard in `_advance_start_latents` far from the cause).

Meanwhile the sibling accounting disagrees: `reencode_window_frames(1)` returns `dropped_tail_frames(1) == 1` (`:168-175`) — the rest of the module believes a 1-frame window exists. So the slice and the window-size function contradict each other at overlap 1, and nothing reconciles them.

Reachability is narrow but real: `handle_init` validates overlap against the constant `NUM_FRAME_PER_BLOCK=3` (`:946-947`), which rejects 1 — but `CausvidSession.__init__` validates against the *config file's* `num_frame_per_block` (`:543-544`). A custom `wan_causal_dmd.yaml` with `num_frame_per_block: 1` + `overlap_frames: 1` passes session validation and then builds empty re-encode windows on every in-session rollout advance (plus `select_tail_window`, which *would* correctly take 1 frame — the two paths diverge).

## Rationale

Empty-tensor-into-encoder is the kind of fault that either crashes obscurely (best case) or produces a degenerate latent that poisons `start_latents` for the rest of the run (worst case). The fix is a one-line guard at the function that owns the slicing arithmetic, where the author already centralized the "mirror the script" logic. The overlap=1 configuration is also *legitimate* (cheapest continuity anchor) — it should either work or be refused, not silently empty.

## Evidence (verified live 2026-09-30, `voyage:latest`, numpy slice semantics identical to torch here)

```
overlap=1 window shape: (1, 0, 16, 60, 104)   # video[:, -1:0, :] — EMPTY
overlap=3 window shape: (1, 1, 16, 60, 104)   # video[:, -9:-8, :] — correct
```

- `video_causvid.py:475-477` verbatim arithmetic; `:125-135` validator has no `overlap >= 2` (or equivalent) floor.
- `:543-546` session validation vs `:946-947` init validation use different block values (config vs constant) — the hole overlap=1 walks through.

## Repro

```bash
docker run --rm -v "$PWD:/app" -w /app -e PYTHONPATH=/app/Voyage voyage:latest python3 -c "
import numpy as np
video = np.zeros((1, 81, 16, 60, 104))
for overlap in (1, 2, 3):
    end, start = -4*(overlap-1), -4*(overlap-1)-1
    print(overlap, video[:, start:end, :].shape)
"
# 1 → (1, 0, ...) empty; 2 → (1, 1, ...); 3 → (1, 1, ...)
```

## Fix candidates

1. Guard in `_vae_encode_slice`: special-case `overlap_frames == 1` (window = last frame, `video[:, -1:, :]`) or reject `overlap < 2` in `validate_overlap_frames` with a message — either way the slice/window-size functions must agree.
2. Unify the block value: `handle_init` should read the config's `num_frame_per_block` for its early check (or drop the early check and let the session's post-load check own it) so init and session can't disagree.
3. Assert `window.shape[1] > 0` at the slice site (fail-loud invariant, same spirit as the `_advance_start_latents` shape guard at `:694-698`).

## Refs

 - `Voyage/voyage/workers/video_causvid.py:125-135,166-176,449-479,688-699`.
 - Adjacent, not overlapping: 124 (same file, device placement — different subsystem); 064-era tail-length loudness (short-*anchor* detection — this file is the *slice arithmetic* producing emptiness, not the anchor length).

## Progress log

- 2026-09-30 (Group E2): evaluated live first. Premise CONFIRMED as-read: `end = -4*(1-1) = 0`, `start = -1`, `video[:, -1:0, :]` empty; `reencode_window_frames(1) == 1` contradicts the slice. TDD: `tests/test_e2_causvid_overlap_128.py` (numpy-backed fake tensor — basic slicing identical in torch — plus stub VAE asserting non-empty windows) — 2 failed pre-fix (empty-window encode, no rejection of 0), 4/4 green post-fix.

## Resolution

- Verdict: FIXED in `voyage/workers/video_causvid.py::_vae_encode_slice` (the function that owns the slicing arithmetic).
- Change: `overlap_frames < 1` → `ValueError` (fail-loud, no asserts per the issue-045 discipline); `end == 0` (overlap 1) takes `video[:, -1:, :]` explicitly so the slice agrees with `reencode_window_frames(1) == 1`; any still-empty window raises `ValueError` at the slice site (same spirit as the `_advance_start_latents` shape guard). Overlap 3+ paths byte-unchanged.
- Files changed: `voyage/workers/video_causvid.py` (+ new `tests/test_e2_causvid_overlap_128.py`, incl. a marker test proving the last frame's values reach the encoder).
- Test evidence (in-container `voyage:latest`, CPU-only): new file 4 passed; `test_causvid_worker` (43) green. Ruff + format + mypy strict clean.
- DESIGN proposal (quoted text only, for the DESIGN owner — §5.4 causvid continuation): "The tail-slice re-encode never builds an empty window: overlap 1 takes the last frame explicitly (agreeing with the window-size accounting), non-positive overlaps and empty slices fail loud at the slice site."
- Residuals (logged, not implemented): fix candidate 2 (unify the block value — `handle_init` validates against constant `NUM_FRAME_PER_BLOCK` while the session validates against the config file's `num_frame_per_block`) is left as-is: reading the config at init would need omegaconf on the RPC boundary, and the session check already owns correctness. An `overlap=1` + custom `num_frame_per_block=1` config now works instead of silently emptying (this fix); any other init/session disagreement still surfaces at session build, fail-loud.
