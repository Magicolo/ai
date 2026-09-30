# AUGMENT — finalize-time presentation floors

Every shipped video is ≥32 fps and ≥1280×720 by default. The floors
lift low-native backends (CausVid 832×480 @ 16, fake testsrc) to
presentation size at finalize; sources already above the floors pass
through untouched ("minimal upscale"). `0` disables a floor — the 24 fps
`PRESENTATION_MIN_FPS` shipped-video guarantee still applies.

## Floors, flags, TOML, TUI

Defaults (`voyage/config.py:375`, `AugmentConfig`):

- `min_fps = 32` (0 disables the fps floor);
- `min_width = 1280`, `min_height = 720` (`0x0` disables the resolution
  floor; a half-disabled pair like `0x720` is rejected — both-zero or
  both-positive).

```toml
[augment]
min_fps = 32
min_width = 1280
min_height = 720
```

Flags (shared helper `_add_augment_args`, `voyage/cli.py:2117` — every
finalizing verb carries the same three, so they cannot drift apart):

- `--min-fps N` (default: `[augment] min_fps 32`; 0 disables);
- `--min-resolution WxH` e.g. `"1280x720"` (default: `[augment]`
  1280×720; `"0"` disables);
- `--no-augment` (wins over explicit floors — both to 0).

Carried by `finalize` (owns the floors — defaults ride the run's
`[augment]` TOML), forwarded by `generate`/`run` overrides into
`resolve_config`, and `stop --finalize` forwards into it. TUI fields
`min_fps` (`"32"`) / `min_resolution` (`"1280x720"`)
(`voyage/tui_state.py:125-126`, help at `:84-85`) validate the same way
(non-negative integer / `WxH`-or-`0`) and stay `Unset` when untouched so
the stored config wins.

## `plan_augmentation` contract (fast-path vs re-encode)

`voyage/media.py:772`, pure (no I/O, fully unit-tested):

```text
effective_fps = max(requested, min_fps or 0, PRESENTATION_MIN_FPS)
out box     = max(target, min) per axis (never stretched — downstream
              scale-to-fit + pad preserves aspect)
```

- Floors only ever upscale: a target already above them is kept as-is.
- `needs_minterpolate` is True only for an fps lift
  (`source + 0.5 < out` — motion interpolation); an fps drop uses the
  plain fps filter.
- `needs_reencode` covers any pixel/timing change (dims differ, fps
  differs past 0.5 either way, lift, or unknown source fps) and gates
  the stream-copy fast path off. Unknown geometry (`0×0` from a failed
  probe) forces re-encode — callers never stream-copy blind.

Examples (live `plan_augmentation`):

- ltxv native 768×512 @ 24 → out 1280×720 @ 32
  (`needs_reencode=True, needs_minterpolate=True`);
- CausVid native 832×480 @ 16 → out 1280×720 @ 32 (same flags —
  ~2.7× pixels plus 2× fps, the cost note below).

## Chunked runner and 2-GPU pairing

`voyage/augment.py` (stdlib only — never torch, supervisor §12 GPU ban):

- `DEFAULT_CHUNK_FRAMES = 32` (mirrors VHS `frames_per_batch`),
  `DEFAULT_INTERP_MULTIPLIER = 4` (the 4× video-export recipe),
  `DEFAULT_UPSCALE_FACTOR = 2` (4× model + Lanczos 0.5 downscale);
- `AUGMENT_DEVICE_PRIMARY = "cuda:0"`,
  `AUGMENT_DEVICE_SECONDARY = "cuda:1"`.

`augment_plan` tiles the timeline into chunk windows with per-chunk
output counts and round-robin devices. Chunk outputs do NOT sum to the
unchunked `(n−1)*m+1` total — each chunk interpolates independently, so
one boundary pair per chunk joint is skipped (a tiny hitch per 32
frames; raise the chunk size for fewer hitches, lower it for less
memory). `run_augment_chunks` owns the serial-vs-`ThreadPoolExecutor(2)`
switch.

Parallelism contract (`voyage/augment.py:14-18`): augment chunks run on
cuda:0 while the MMAudio SFX stack (when present) renders on cuda:1 —
the two stages share nothing but chunk boundaries, so a 2-GPU box runs
them side by side and a 1-GPU box runs chunks serially on cuda:0.
`--sfx-workers 2` shards `small_44k` across both GPUs instead (needs
2 visible GPUs, fails fast otherwise).

## FILM stand-in caveat and weight-port follow-up

`voyage/workers/augment_worker.py:15-19` (spike, not the full port):

- `RRDBNet` is the vendored Real-ESRGAN ×4 residual-in-residual net
  (state-dict shapes match the ×4plus family; the compact
  `RealESRGAN_x4plus_anime_6B` SRVGG variant needs its own loader —
  follow-up, not this spike).
- `FilmNetMini` is a spike stand-in flow blender with FILM's semantic
  contract (two frames + time give the mid frame), batched as
  `(B, 2, C, H, W)` so OOM-halving applies. Official `film_net`
  weights will NOT load here — shape mismatch raises
  `ModelCompatibilityError` (`augment_worker.py:471-484`); the full
  upstream FILM port is follow-up.

Torch loads only inside functions behind a `find_spec` guard (never at
module scope); weight checks run before any torch import, so missing
weights raise torch-free. Resident nets are cached by (weights path,
device) and dropped by `evict_augment_models` on the GPU hand-off.
Precision is fp16 on CUDA, fp32 elsewhere; batch inference starts full
and halves on OOM down to single items (Comfy `FrameInterpolate`
recipe).

Weights (`models download film` ~66 MB, `realesrgan-anime` ~18 MB;
`docs/MODELS.md:45-65`): FILM fp16 lands in
`<models>/frame_interpolation/`, the anime upscaler in
`<models>/realesrgan/`. `generate` ensures both when augmentation is
enabled; `fake` runs still exercise the planning path (fast-path vs
re-encode) without weights.

## Cost note and benchmark effect

Lifting costs pixels × fps. CausVid (832×480 @ 16) and fake testsrc
(768×432 fake-432p) both ship as 1280×720 @ 32 — ~2.7× pixels plus a 2×
fps lift through minterpolate + upscale + re-encode. Expect finalize
to dominate e2e wall time on those backends; ltxv (768×512 @ 24) pays
a smaller lift. `BENCHMARKING.md` e2e numbers predate the floors —
compare GPU runs to GPU runs at the same floor settings, and pass
`--no-augment` (or `min_*=0`) when you need native-geometry timings.
Soak trends stay comparable run-to-run only when the floors match.
