# 065 — `video_ltxv._save_mp4` missing `np.clip`: out-of-range floats wrap modulo 256

- Status: resolved (fixed 2026-09-25)
- Severity: medium (silent visual corruption — bright pixels flip to black)
- Area: LTXV worker — `voyage/workers/video_ltxv.py:615-627` line 624
- Rank rationale: pass-2 worker finding; the sibling worker already does it
  right, so the fix is a one-line mirror.

## Technical description

```python
frames = (frames * 255).astype("uint8")  # ltxv.py:624 — wraps
```

vs the correct sibling `voyage/workers/video_causvid.py:600-602`:

```python
clipped: NDArray[np.uint8] = np.clip(frames_any * 255.0, 0, 255).astype(np.uint8)
```

Verified live by orchestrator 2026-09-25 (`sed -n` output above shows both
sites). No clip anywhere in `video_ltxv.py`.

## Why this is an issue

VAE decoders routinely emit values slightly outside `[0,1]` (overshoot at high
saturation), and without clipping those wrap modulo 256 — near-white becomes
near-black. That is silent visual corruption in every LTXV render, while the
sibling worker already does it right, so the fix is a proven one-line mirror
with a pure-numpy unit test and no GPU needed.

## Evidence (code experiment, host numpy, verified by orchestrator)

```
$ python3 -c "import numpy as np; arr=np.array([1.01,-0.01,1.0]); ..."
# sweep probe: wrapped (ltxv, no clip): [1, 254, 255]; clipped (causvid): [255, 0, 255]
```

1.01 (VAE overshoot) becomes 1 (near-black) under LTXV, 255 under CausVid.

## Reproduction

Any LTXV decode emitting slightly out-of-`[0,1]` values (common VAE overshoot at
high saturation — cf. slice-4 blowout notes) writes wrapped pixels.

## Source references

- `voyage/workers/video_ltxv.py:615-627`; `voyage/workers/video_causvid.py:598-602`.

## Resolution candidates

`frames = np.clip(frames*255.0, 0, 255).astype("uint8")` mirroring causvid; add a
unit test with `[1.01, -0.01]` inputs (pure numpy, no GPU).

## Investigation / progress / resolution log

- 2026-09-25: found by pass-2 worker sweep; both sites re-verified live.
- 2026-09-25: issue-file repair — added `## Why this is an issue`; re-verified
  cited lines live (`video_ltxv.py:615-627`, wrap at `:624`;
  `video_causvid.py:598-602`, clip at `:601` — both match).
- Open: one-line fix + test.
- 2026-09-25: FIXED. `voyage/workers/video_ltxv.py:657` now mirrors the
  causvid sibling exactly: `np.clip(frames * 255.0, 0, 255).astype("uint8")`
  (top-level `numpy` import at `:36`). Test:
  `test_save_mp4_clips_overshoot_instead_of_wrapping` in
  `Voyage/tests/test_ltxv_failure_hygiene.py` (pure numpy + stubbed
  imageio: 1.01→255, -0.01→0, 1.0→255, 0.5→127, dtype uint8). Gates: `ruff
  check` clean, `ruff format --check voyage tests` clean, `mypy voyage`
  strict clean (40 files), `pytest` 472 passed.
