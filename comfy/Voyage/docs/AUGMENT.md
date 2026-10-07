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
  instead of lifting the frame rate);
- `interp_backend = "rife"` (the interpolation model: `rife`
  default, `film` opt-in for hero/archival renders — see the RIFE
  section below; rides the manifest `augment` object and the
  freshness key, so a switch re-renders interp by design).

Stored in the manifest's `augment` object at `configure` time:

```json
{
  "augment": {
    "upscale": 1,
    "interpolate": 1,
    "interp_backend": "rife"
  }
}
```

(`presentation_fps` unset unless pinned — `null` default ships
`round(source_fps × interpolate)`.)

Flags (shared helper `_add_augment_args` on `configure`):

- `--upscale N` (doubles width and height at 2);
- `--interpolate N` (doubles the frame count at 2 — the same
  `(n−1)*m+1` blend math);
- `--interp-backend {rife,film}` (default `rife`; a switch
  re-renders interp outputs and prunes the other backend's stale
  plan dirs by design);
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

- Multipliers only ever scale up from the probed source: 1/1 keeps
  the source geometry ("native" — since the 2026-10-06 compression
  change publish still encodes it in one libx264 pass, `-preset slow
  -crf 30` by default; the old stream-copy publish is gone).
- `needs_minterpolate` is True only for an fps lift the model pass
  did not already produce (`model_interpolate <= 1` and source + 0.5
  < out — ffmpeg motion interpolation); an fps drop uses the plain
  fps filter. When the model pass interpolates, the lift is measured
  against the interpolated rate.
- `needs_reencode` covers any pixel/timing change (dims differ, fps
  differs past 0.5 either way, lift, or unknown source fps) and feeds
  the native-vs-presentation publish label — both encode since the
  2026-10-06 compression change retired the stream-copy fast path.
  Unprobable dims fail loud
  (`MediaError`); unprobable fps fails loud unless
  `presentation_fps` is set.
- The model pass (Real-ESRGAN upscale + configured-backend interpolate via
  `resolve_augment_weights` when provisioned) runs exactly when the
  multipliers demand work (`upscale > 1 or interpolate > 1`) and the
  legs are present; otherwise finalize ships via ffmpeg. Absent legs
  with demanded work fall back to the ffmpeg vf path (scale /
  minterpolate) — never an error.

Examples (live `plan_augmentation`):

- ltx25 native 1216×704 @ 24, 1/1 → out 1216×704 @ 24
  (`needs_reencode=False` — single libx264 encode at publish, no vf work);
- same source, `--upscale 2` → out 2432×1408 @ 24;
- same source, `--interpolate 2` → 24fps ×2 content (model mids),
  shipped at 48fps unless `--presentation-fps 32` pins slow motion;
- CausVid native 832×480 @ 16, 1/1 → out 832×480 @ 16 (native geometry, encoded at publish).

## Chunked runner and 2-GPU pairing

`voyage/augment.py` (stdlib only — never torch, supervisor §12 GPU ban):

- `DEFAULT_CHUNK_FRAMES = 32` (mirrors VHS `frames_per_batch`),
  `DEFAULT_INTERP_MULTIPLIER = 4` (the 4× video-export recipe),
  `DEFAULT_UPSCALE_FACTOR = 2` (4× model + Lanczos 0.5 downscale);
- `AUGMENT_DEVICE_PRIMARY = "cuda:0"`,
  `AUGMENT_DEVICE_SECONDARY = "cuda:1"`.

`augment_plan` tiles the timeline into overlapping chunk windows with
per-chunk output counts and round-robin devices. Consecutive windows
share exactly one source frame (stride `size − 1` in
`chunk_windows`, `voyage/augment.py:161-187`), and every chunk after
the first drops its duplicated boundary frame
(`interpolated_chunk_frame_count`, `voyage/augment.py:136-158`) — so
chunked output sums to the exact unchunked `(n−1)*m+1` total with no
skipped boundary pair and no intra-segment hitch. Chunk size now trades
memory against throughput only (raise for fewer chunks, lower for less
memory). Exception: `size == 1` windows cannot overlap, so they keep the
legacy stride-1 tiling, still lossy versus unchunked but byte-identical
to before. `run_augment_chunks` owns the serial-vs-`ThreadPoolExecutor(2)`
switch.

Parallelism contract (two streams on a 2-GPU box, sequential
otherwise): Thread-A runs one interleaved model pass on cuda:1
(each segment's upscale immediately followed by its interp — RIFE
peaks ~0.65 GiB at 2048×1152, so both legs fit the 6 GB card beside
the llama director sidecar, gated by a 1.0 GiB free-VRAM floor;
the fix stage renders every boundary's 4 source-res morph bridges
first, then segments AND joint units run the same up→ip loop —
joints re-interpolate by design) while Thread-B renders ACE
music takes then the MMAudio SFX bed sequential on cuda:0; the
join is followed by mix and publish, all reporting live bars on one
shared display. Background pre-warm interleaves the same fix-then-
segments+joints up→ip loop on cuda:1 during generation when the same
headroom allows, gated by director idleness. A 1-GPU box runs everything
sequential on cuda:0.

Joint order (seam fix -> upscale -> interpolate, DESIGN §140): the fix
renders morph 2+2 bridges at SOURCE resolution only (anchors `A[-3]`/
`B[+2]`, no anchor upscale, no extra interp — `voyage/augment_joints.py`)
into one 4-frame joint video per boundary (content key
`jointfix2x2|a|b|WxH@fps`, ledger-hit resume); the joint video then
flows through the SAME upscale + interp pollers as committed segments
(the pollers' `sources=` override — standard ChunkKey ledger shapes in
the joint's own plan dir, `(4-1)*m+1` frames out), and the drain
assembles [trimA, joint, trimB] with source-derived trims (A keeps
through model frame `(a-3)*m`, B drops its first `2m` — anchors owned
exactly once, no duplicates, no gaps). The retired mids-insert seam
path (`augment_seam.py`, `seam-v2` keys) is deleted; old seam sidecars
age out via prune grace. The no-model-pass native morph assembly
(`augment_morph.py`, ltx gate) is unchanged. Prune liveness = segment
plan dirs + joint plan dirs derived from joint videos on disk (never
rendered during prune).

Bidirectional model pass (audio-first, `voyage/augment_parallel.py`):
when `parallel_model_pass_armed` holds (RIFE-only backend, both cuda:1
and cuda:0 visible, tensor path with both legs present) this branch
takes precedence over the model∥music fork — audio runs first
(music takes then SFX bed on cuda:0, mix inputs ready), then the model
pass runs bidirectional across both cards: a shared in-process deque
holds every (segment, chunk) task, worker A pops from the front on
cuda:1 while worker B pops from the back on cuda:0 (meet-in-the-middle;
the same worker renders a chunk's upscale immediately followed by its
interp, so no cross-worker waiting). Tasks dispatch as per-chunk
poller calls (`segment_ids=[seg]`, `chunk_ids=[idx]`); pruning runs
once upfront per segment and the workers pass `prune_partials=False`
(a per-pass whole-dir sweep would delete the other worker's live
`.partial`); ledger appends are `flock`-guarded and settle is verified
on the global ledger (fail-loud `MediaError`). Joint units flow as
joint tasks through the same deque (the `sources=` override honors
`segment_ids`/`chunk_ids` scoping); the drain assembles
[trimA, joint, trimB]. After publish, resident nets
are evicted via `evict_augment_models()`.
`--sfx-workers 2` shards `small_44k` across both GPUs instead (needs
2 visible GPUs, fails fast otherwise).

## Interpolation backends (RIFE default) and weight contract

`voyage/workers/augment_worker.py` carries the full upstream FILM port:

- `_FilmNet` (7 flow-predictor levels, shared pyramid extractor, style
  curve + fuse decoder) strict-loads the 82 pinned keys from
  `film_net_fp16.safetensors` (`ModelCompatibilityError` on shape
  mismatch); there is no stand-in blender anymore.
- `interpolate_mids` evaluates every adjacent pair at all blend moments
  with one flow computation per pair (via `forward_multi_timestep`) in
  pair-major order, batching `FILM_PAIR_BATCH` (1) pairs per forward
  with whole-pair OOM halving down to single pairs. `pair_batch=1`
  matches the old per-pair `interpolate_pair` loop bit-exactly; larger
  batches are deterministic but shift pixels slightly (cudnn picks
  different kernels per batch shape — measured 2026-10-05 on the 4060 Ti:
  mean abs 1.9e-4, ~4% of pixels flip after PNG rounding — with zero
  speedup, so the default stays 1).
  `interpolate_pair` / `interpolate_triplet` stay for single-pair use.
- `_RifeNet` ports Comfy's IFNet (RIFE v4.25-heavy, 86.7 MB,
  strict-loads the pinned keys; `ModelCompatibilityError` on shape
  mismatch). RIFE has no flow-once factorization — each blend moment
  is a full forward — yet it stays ~10-15x faster per pair than FILM
  at 2048x1152 (measured 2026-10-06 on the 4060 Ti: 0.05-0.10 s/pair
  vs ~0.85 s, fp16; heavy is the sharpest variant and the closest to
  FILM, at an identical ~0.65 GiB peak, so it is the pin).
- `interpolate_rife_mids` mirrors the `interpolate_mids` contract
  (validation order/messages, `timings`, `on_pair`, float32 CPU
  outputs): reflect-pad to 64 + crop, per-moment forwards with a
  shared per-pair encode cache. No MIN_SIDE floor (padding covers
  all sizes); OOM propagates (the peaks make halving pointless).
- `AugmentConfig.interp_backend` (`rife` default, `film` opt-in)
  selects the leg at every render site (fix, pollers, native morph). The sidecar weights key keeps the legacy `sha|sha` shape,
  so pre-knob FILM ledgers keep hitting with zero re-render, while a
  backend switch misses by construction (the interp-leg sha differs)
  and finalize prunes the other backend's stale plan dirs
  (`prune_stale_interp_plans` — whole-dir deletion, conservative
  keep on unknown/unparseable records, upscale-only dirs untouched).
- The anime upscaler leg loads `realesr-animevideov3.pth` (native 4x
  SRVGGNetCompact XS) strict via its own key-sniffed builder
  (`_is_srvgg_compact_state` / `_build_srvgg_net`); RRDB-shaped
  state goes to the classic 23-block builder instead.

Torch loads only inside functions behind a `find_spec` guard (never at
module scope); weight checks run before any torch import, so missing
weights raise torch-free. Resident nets are cached by (weights path,
device) and dropped by `evict_augment_models` on the GPU hand-off.
Precision is fp16 on CUDA, fp32 elsewhere; batch inference starts full
and halves on OOM down to single items (Comfy `FrameInterpolate`
recipe).

Weights (ensured at `configure` time via
`model_registry.download_model(models_dir, "rife")` ~87 MB heavy,
`model_registry.download_model(models_dir, "realesrgan-anime")`
~2.5 MB, `model_registry.download_model(models_dir, "film")` ~66 MB stays
provisionable for the opt-in backend;
`docs/MODELS.md:45-65`): the interp leg lands in
`<models>/frame_interpolation/`, the anime upscaler in
`<models>/realesrgan/`. `generate` ensures the configured backend
(+ upscaler) when the multipliers demand work; `fake` runs still
exercise the planning path (fast-path vs re-encode) without weights.

## Cost note and benchmark effect

Lifting costs pixels × fps. `--upscale 2 --interpolate 2` on a
1216×704 @ 24 source ships 2432×1408 @ 48 — 4× pixels plus the interp
mids; expect finalize to dominate e2e wall time whenever the
multipliers are raised. With RIFE the interp leg is the cheap part
(~2 s per 32-frame chunk at 2048×1152 vs ~26 s FILM on the 4060 Ti)
— upscale + PNG/ffmpeg I/O dominate the model pass (the PNG bridge
is threaded at fast zlib level 1: ~2–3 s per 63-frame chunk save at
2048×1152 vs ~18 s serial level 6, pixel-identical). Defaults (1/1)
ship native, so `BENCHMARKING.md` e2e numbers are comparable
run-to-run only when the multipliers match — record them in the
setup block (`presentation_setup_facts`).
