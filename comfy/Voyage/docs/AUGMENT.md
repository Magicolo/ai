# AUGMENT — finalize-time explicit quality

There are no minimum quality floors: quality is specified explicitly
per run. The shipped box is `source × --upscale` per axis and the
shipped rate is `--presentation-fps`, else `round(source_fps ×
--interpolate)`. A CausVid 16fps source ships at 16fps unless pinned
otherwise; `upscale=1/interpolate=1` (defaults) ships the source
geometry as-is via ffmpeg.

## Multipliers, flags, stored config

Defaults (`voyage/config.py`, `AugmentConfig`):

- `upscale = 1`, `interpolate = 1` (1 = no work on that axis; both
  accept 1, 2, or 4 — the worker/poller vocabulary, validated at
  `configure` time so a bad value dies fast instead of deep in the
  worker);
- `presentation_fps = None` (unset — ships `round(source_fps ×
  interpolate)`; pins the shipped rate when set, e.g. 24fps ×2
  content presented at 32fps stretches the timeline 1.5× slow motion
  instead of lifting the frame rate).

Stored in the manifest's `[augment]` section at `configure` time:

```toml
[augment]
upscale = 1
interpolate = 1
# presentation_fps unset unless pinned
```

Flags (shared helper `_add_augment_args` on `configure`):

- `--upscale N` (doubles width and height at 2);
- `--interpolate N` (doubles the frame count at 2 — the same
  `(n−1)*m+1` FILM math);
- `--presentation-fps N` (pins the shipped rate).

Old manifests carrying `min_fps`/`min_width`/`min_height`/
`min_resolution`/`use_model_pass`/`no_augment`/`interp_multiplier`
fail loud with a re-configure hint (clean break, no migration).

## `plan_augmentation` contract (fast-path vs re-encode)

`voyage/media.py`, pure (no I/O, fully unit-tested):

```text
out box = source × upscale per axis (exact integer multiply — the
          model SRVGG leg upscales 2x exactly, any residual lands
          in the vf)
out fps = presentation_fps, else round(source_fps × interpolate)
```

- Multipliers only ever scale up from the probed source: 1/1 ships
  the source untouched ("native ship").
- `needs_minterpolate` is True only for an fps lift the model pass
  did not already produce (`model_interpolate <= 1` and source + 0.5
  < out — ffmpeg motion interpolation); an fps drop uses the plain
  fps filter. When the model pass interpolates, the lift is measured
  against the interpolated rate.
- `needs_reencode` covers any pixel/timing change (dims differ, fps
  differs past 0.5 either way, lift, or unknown source fps) and gates
  the stream-copy fast path off. Unprobable dims fail loud
  (`MediaError`); unprobable fps fails loud unless
  `presentation_fps` is set.
- The model pass (Real-ESRGAN upscale + FILM interpolate via
  `resolve_augment_weights` when provisioned) runs exactly when the
  multipliers demand work (`upscale > 1 or interpolate > 1`) and the
  legs are present; otherwise finalize ships via ffmpeg. Absent legs
  with demanded work fall back to the ffmpeg vf path (scale /
  minterpolate) — never an error.

Examples (live `plan_augmentation`):

- ltx25 native 1216×704 @ 24, 1/1 → out 1216×704 @ 24
  (`needs_reencode=False` — stream-copy fast path);
- same source, `--upscale 2` → out 2432×1408 @ 24;
- same source, `--interpolate 2` → 24fps ×2 content (model FILM mids),
  shipped at 48fps unless `--presentation-fps 32` pins slow motion;
- CausVid native 832×480 @ 16, 1/1 → out 832×480 @ 16 (ships native).

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

- The anime upscaler leg loads `realesr-animevideov3.pth` (native 4x
  SRVGGNetCompact XS) strict via its own key-sniffed builder
  (`_is_srvgg_compact_state` / `_build_srvgg_net`); RRDB-shaped
  state goes to the classic 23-block builder instead.
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

Weights (`models download film` ~66 MB, `realesrgan-anime` ~2.5 MB;
`docs/MODELS.md:45-65`): FILM fp16 lands in
`<models>/frame_interpolation/`, the anime upscaler in
`<models>/realesrgan/`. `generate` ensures both when the multipliers
demand work; `fake` runs still exercise the planning path (fast-path
vs re-encode) without weights.

## Cost note and benchmark effect

Lifting costs pixels × fps. `--upscale 2 --interpolate 2` on a
1216×704 @ 24 source ships 2432×1408 @ 48 — 4× pixels plus the FILM
mids; expect finalize to dominate e2e wall time whenever the
multipliers are raised. Defaults (1/1) ship native, so `BENCHMARKING.md`
e2e numbers are comparable run-to-run only when the multipliers
match — record them in the setup block (`presentation_setup_facts`).
