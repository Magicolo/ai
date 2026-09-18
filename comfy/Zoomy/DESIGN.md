# Zoomy — Design Document

> Maintained alongside the code. Any behavior change must update this file in
> the same change. See also the wrapper doc `../AGENTS.md` §10.
>
> Standalone revision (2026-09-16): the ComfyUI backend is gone. Zoomy renders
> with an in-process engine (diffusers + ACE-Step + MMAudio + RIFE) inside its
> own container; models live in the `zoomy_models` volume. Comfy is kept only
> as an untouched backup (see `../AGENTS.md`).

## 1. Overview

Zoomy is a small Gradio web application that renders the two verified
infinite-zoom loops (Ernie Image Turbo, Juggernaut Z) without any external
backend. The studio layout pairs a left canvas (latest frame, filmstrip,
finished video, Finish/refresh/clear, interrupt, session log) with a right
stepped rail: pick a model card, choose LoRA style cards with strengths,
shape the prompt, then **Grow** the zoom (one frame, loop to a duration, or
loop until stopped); **Finish video** turns the accumulated frames into a
music video with sound effects. Every frame draws a fresh random seed —
there is no manual seed control. Resolution is preset-only (official
vendor buckets plus a 512 draft card, no width/height boxes).

Deliberate non-goals:

- No workflow JSON, no graph nodes. Switch/condition/duration logic lives in
  Python; the heavy lifting (diffusion, interpolation, audio, video encode)
  runs in-process via vendored libraries.
- No multi-user support, no auth. Single local user assumed; the UI listens
  on all interfaces (no auth — never expose the host beyond a trusted
  network).
- No host-side dependencies. Everything (runtime and dev tools) runs inside
  the `zoomy` container; models live in a Docker volume, never on the host.

## 2. Runtime topology

```
browser (host) ──http──> zoomy:7861 ──in-process──> LocalEngine ──files──> /output + /models
      localhost:7861      container        diffusers/ACE/MMAudio/RIFE   bind mount + volume
```

- `zoomy` service (`Zoomy/Dockerfile`, CUDA base — see §3): Gradio app, port
  `7861:7861`, volumes `./Zoomy/output:/output` (bind mount, gitignored) and
  `zoomy_models:/models` (named volume, ~26 GB provisioned). No `depends_on`,
  no Comfy address anywhere. GPU reservations are kept — frames and finalize
  both need CUDA.
- Lifecycle: `./zoomy.sh` launches zoomy itself
  (`run --build --rm --detach --service-ports zoomy`) as an ephemeral
  `comfy-zoomy-run-*` container; `docker stop` removes it. It never touches
  `serve.sh` or the `comfy` service.
- First launch ever needs one provisioning run (models volume starts empty):
  `CIVITAI_API_KEY=$(cat civit-ai-api-key) docker compose run --rm
  -v ./Comfy/input:/seed-source:ro zoomy python scripts/download_models.py
  --seed-source /seed-source` (full command lives in `zoomy.sh`).

## 3. Container (`Zoomy/Dockerfile`, one file for run/tests/tools)

- Base `pytorch/pytorch:2.10.0-cuda12.8-cudnn9-devel` (torch 2.10.0+cu128,
  torchvision 0.25.0 — matches the proven spike recipe). `PIP_BREAK_SYSTEM_PACKAGES=1`
  (Debian-derived base would otherwise refuse).
- `apt`: `git curl libgl1 libglib2.0-0 libxcb1` — the last three are headless
  `cv2` support.
- Vendors cloned to `/opt`: `ACE-Step-1.5` (music) and
  `ComfyUI-MMAudio@8eaeb72` (SFX; pinned — the host checkout that proved the
  recipe). `PYTHONPATH=/opt/ACE-Step-1.5:/opt/ComfyUI-MMAudio:/application/vendor`.
- `pip install` covers `requirements.txt` (runtime: gradio, httpx, …) +
  `requirements-gpu.txt` (diffusers/peft/hf_hub/imageio, ccvfi, librosa,
  torchdiffeq/timm/omegaconf/open_clip/ftfy/soundfile) + dev tools
  (ruff/mypy/pytest/hypothesis). Requirements are COPYed and installed
  *before* the tree, so rebuilds after code edits reuse the pip layer.
- Build-time smoke imports (frames/interp/acestep/mmaudio) fail the build
  early if a vendor breaks. `CMD ["python", "-m", "zoomy"]`.
- ACE import warnings (LyCORIS/Lightning/bitsandbytes) are training-only and
  harmless.

## 4. Repository layout (`Zoomy/`)

```
Zoomy/
  DESIGN.md               # this file (excluded from the image via .dockerignore)
  pyproject.toml          # metadata + ruff (ALL, line-length 100) + mypy strict + pytest
  requirements.txt        # runtime
  requirements-gpu.txt    # CUDA/vision/audio model deps
  requirements-dev.txt    # dev: ruff/mypy/pytest/hypothesis
  Dockerfile              # single GPU image (see §3)
  .dockerignore
  output/                 # bind-mounted to /output (gitignored, holds .gitkeep)
  scripts/
    download_models.py    # one-shot provisioner filling /models (see §5)
    quality-gates.sh      # full gates in-container (see §11)
    run-tests.sh          # pytest-only iteration (see §11)
  vendor/
    comfy/                # ProgressBar stand-in (MMAudio imports it)
  zoomy/                  # the application package
    __init__.py           # version only
    __main__.py           # `python -m zoomy` entry
    main.py               # wiring: settings -> engine/repository -> app -> launch
    settings.py           # Settings.from_environment (ZOOMY_* variables)
    errors.py             # ZoomyError hierarchy (engine-flavored)
    family_catalog.py     # frozen FamilyDefinition/LoraDefinition catalog of 3
    engine_protocol.py    # FrameRenderRequest/FinalizeRequest, duration math,
                          # segmentation, EngineProtocol, EngineStatistics
    local_engine.py       # LocalEngine: the whole render backend (see §7)
    vendor_compat.py      # transformers-5 compatibility shim
    final_assembly.py     # segmented-finalize ffmpeg assembly
    frame_repository.py   # frame/video discovery + clearing under the output dir
    rendering.py          # generate_video + render_next_frame/finalize_video/
                          # render_loop generators
    interface.py          # Gradio layout + event wiring
  tests/                  # one module per source module (see §11)
```

## 5. Provisioning (`scripts/download_models.py` + `zoomy_models` volume)

The script is idempotent (skips files already present), stdlib `urllib` +
`huggingface_hub`, and provisions a Comfy-style tree: `diffusion_models/`
(ernie-image-turbo + 2 Juggernaut Z), `vae/ae.safetensors`, `loras/` (c64,
chalkboard, clay-art), `mmaudio/` (4 files + BigVGAN snapshot minus the
1.5 GB discriminator training state), `seed/` (both cold-start PNGs via
`--seed-source`).

Deliberately excluded (~30 GB of Comfy-format dead weight the engine never
opens): ministral/qwen text encoders, flux2-vae, Comfy ACE weights, prompt
enhancer — those components come from official HuggingFace repositories at
first render into `HF_HOME=/models/.hf-cache`.

Three provisioning gotchas, all encoded in the script + tests:

- **CivitAI needs a browser User-Agent.** Plain `urllib` gets HTTP 403 with
  body `error code: 1010` (Cloudflare browser-integrity block) at
  `civitai.com` itself. `CIVITAI_USER_AGENT` + `build_civitai_request`
  carry it (regression-tested).
- **Never forward the API token to the CDN.** The download URL 302-redirects
  to a signed file URL; the CDN rejects foreign `Authorization` headers
  (403) and receiving it would leak the token. `_CivitaiRedirectHandler`
  strips it on redirect (regression-tested; override signature mirrors the
  stdlib base exactly — `req/fp/msg/newurl`, `http.client.HTTPMessage`).
- **Pin the real HF sub-path.** `Comfy-Org/z_image` nests blobs under
  `split_files/`, so the VAE is `split_files/vae/ae.safetensors` (a bare
  `vae/` path 404s mid-provision; pinned by test).

Juggernaut variant choice (user-approved 2026-09-16): the bare version URL
serves each version's PRIMARY file — the fp8-pruned variant (5.7 GB, pure
`F8_E4M3`) — rather than the full bf16/fp16 files Comfy ran (11.5 GB).
Deliberate: half the VRAM on the 16 GB card, and e2e frames confirm the zoom
still looks right. Full-precision fileIds are recorded in a script comment
for a future swap (fast v3011968: fp16 = 2891192, bf16 = 2891189; quality
v2921151: bf16 = 2799849, fp16 = 2804116). Total volume after provisioning:
~26 GB.

## 6. Module reference

- `settings.Settings.from_environment()`: `ZOOMY_MODELS_DIRECTORY`
  (`/models`), `ZOOMY_SEED_DIRECTORY` (`/seed` — compose overrides to
  `/models/seed`), `ZOOMY_MUSIC_PROJECT_DIRECTORY` (`/music-project` —
  compose overrides to `/models/music-project`),
  `ZOOMY_OUTPUT_DIRECTORY` (`/output`), `ZOOMY_INTERFACE_ADDRESS`
  (`0.0.0.0` — inside the container only), `ZOOMY_INTERFACE_PORT`
  (`7861`), `ZOOMY_CUDA_DEVICE` (`cuda:0`). Blank values fall back to
  defaults; garbage integers raise `ZoomyError`. (`ZOOMY_COMFY_ADDRESS` is
  gone — no backend to address.)
- `errors`: `ZoomyError` base; `EngineConfigurationError` (missing models /
  seeds — the music project directory is auto-created, seed inputs are not),
  `EngineExecutionError` (carries the failing `stage`), plus the retained
  `RenderInterruptedError`, `EmptyFrameSequenceError`, `AssemblyError`. The
  interface catches `ZoomyError` and shows it in the status line.
- `family_catalog`: `FAMILY_CATALOG` holds `ernie_turbo`, `z_fast`,
  `z_quality`. `find_family(catalog, key)` raises `ZoomyError` on miss.
  `LoraDefinition.selected_by_default` mirrors the source workflows (Ernie
  C64 on; Z styles off = photorealistic default). Each family carries
   `resolution_presets`: Ernie offers the seven official Baidu sizes
   (1024x1024, 1264x848, 848x1264, 1376x768, 768x1376, 1200x896,
   896x1200); both Z variants offer the eleven official Comfy-Org
   1024-buckets (1024x1024, 1152x896, 896x1152, 1152x864, 864x1152,
   1248x832, 832x1248, 1280x720, 720x1280, 1344x576, 576x1344). Each
   list appends `DRAFT_RESOLUTION_PRESET` (Draft 512x512) LAST for
   quick generation — appended, never prepended, so the official
   defaults selected by `_default_preset_name` never shift. The
   non-official Z sizes (1376x768, 1536x864) were dropped in the 2026-09-17
   revamp — off-bucket sizes risk RoPE artifacts. Every preset is a positive
   multiple of 16. Ernie panels default to the 1376x768 HD entry (which also
   matches `FRAME_WIDTH/HEIGHT_PIXELS`); Z panels default to Square
   1024x1024 (the first bucket, since 1376x768 no longer exists for Z).
- `engine_protocol`: `FrameRenderRequest(family, prompt, negative_prompt,
  frame_count, lora_selections, seed, frame_width = 1376,
  frame_height = 768, denoise_strength = 0.60)` / `FinalizeRequest` / `SegmentWindow`
  / `ProgressUpdate` / `EngineStatistics`; pure duration math
  (`compute_audio_seconds` = `max(interp / 32, 1.0)` — the ACE
  `seconds >= 1.0` floor; `compute_frames_for_seconds` inverts the ×4
  interpolation: `(frames - 1) * 4 + 1` interp frames at 32 fps, so 10 s →
  81 frames); requested sizes must be positive multiples of
  `FRAME_SIZE_ALIGNMENT_PIXELS` (16) and segmentation (`needs_segmentation`,
  `compute_segment_windows`, 48-frame windows); `EngineProtocol` is the
  structural contract rendering/interface depend on (tests use fakes);
  `MINIMUM_SYNC_FRAMES = 17` feeds the SFX padder (§7).
- `frame_repository.FrameRepository(output_directory)`: `frame_paths` globs
  `<sequence>/frame_*.png` sorted (counters are zero-padded, so name
  order is render order — the layout is flat since 2026-09-16: no `Zoomy/`
  nesting, old nested artifacts were orphaned without migration);
  `latest_video_path` returns the newest `<sequence>*.mp4`, **preferring
  the `-audio` twin** (see §8) and excluding segment twins (which could
  otherwise win as newest); `sequence_statistics()` summarizes count,
  bytes, recent paths, and video details in one pass for the stats panel
  (tolerates files vanishing mid-read).
- `local_engine.LocalEngine`: see §7. Constructor
  `(models_directory, seed_directory, repository, device,
  music_project_directory)`; `render_frame` / `finalize_sequence` /
  `request_interrupt` (a `threading.Event`, cleared per render) /
  `is_ready` / `engine_statistics` (live RAM/VRAM).
- `vendor_compat`: transformers-5 compatibility shim (attribute replacement
  via a documented `setattr` helper — co-located `noqa` + `type: ignore`
  pragmas are not honored on multi-line statements, so the helper carries
  the suppression once).
- `rendering`: `generate_video(family, request_factory, environment,
  options)` is the one call making a whole video — the duration loop with
  auto-finalize (see below); `render_next_frame` / `finalize_video` /
  `render_loop` are generators yielding `ProgressUpdate`; all take a
  `RenderEnvironment` bundle (engine + repository). `generate_video` sizes
  the loop with `compute_frames_for_seconds` minus existing frames (0 new
  → an "already covers" note), renders until a stop request when
  `target_seconds` is `None`, always finalizes on target-reached but only
  with `finalize_on_stop` on manual stop, and yields "nothing to finalize"
  instead of raising on zero frames. `VideoGenerationOptions` bundles
  `target_seconds` / `finalize_on_stop` / `max_attempts_per_frame` (a
  dataclass because the lint cap is 5 params per function). Interrupts
  surface as `RenderInterruptedError` from the engine flag; `render_loop`
  renders until a frame target, a stop request (module-level
  `threading.Event`, single-user justification documented), or exhausted
  per-frame retries (3 attempts, transient errors only — interrupts
  propagate immediately).
- `interface.build_application(...)`: see §9.
- `main.main()`: builds settings/engine/repository, queues the app with
  `default_concurrency_limit=1` (one render job at a time), and launches
  with `allowed_paths=[output_directory]`, `show_error=True`,
  `inbrowser=False`, `ssr_mode=False` (no Node.js in the image).

## 7. Frame backend (`local_engine.py`)

Cold start (`frame_count == 0`) loads the family seed image from the seed
directory; otherwise the newest sequence frame is cropped to the request
size minus a 20 px border at `(10, 10)` → bicubic-rescaled back to the
request size (the border is `min(10, width // 4, height // 4)`, so HD is
`1356×748 → 1376×768`, zoom `1376 / 1356 ≈ 1.0147` per frame, ~1.5% dive;
`512×512` uses a `492×492` crop, `512 / 492 ≈ 1.0407`, the old 512-era
  dive) → encoded → img2img with the family recipe at the request's
  denoise strength (the UI coherence slider maps `denoise = 1 - coherence`,
  coherence 0.05-0.90, default 0.40 = denoise 0.60; the complement is
  clamped so float dust at the edges never fails engine validation) →
  saved as `<sequence>/frame_<counter>.png`. Requested sizes come from
`FrameRenderRequest.frame_width/frame_height` (UI-editable, blank falls
back to 1376×768) and must be positive multiples of 16
(`_validate_frame_size`, else `EngineConfigurationError`); the sampler
call uses the request dims, so families render at any aligned size.

One frame pipeline stays resident per family (evicted on family change —
each holds ~20 GB of CPU weights under sequential offload). `_apply_lora_selection`
loads newly selected LoRAs once via `load_lora_weights` and activates
exactly the selection with `set_adapters(names, weights)`; deselecting is a
weight swap, no reload. (Empty selections skip `set_adapters` entirely, so
an e2e A/B must use a fresh engine or sequence — a stale adapter would stay
active. Caught once during verification, now documented.)

- **Ernie** (`_load_ernie_pipeline`): `ErnieImagePipeline`; local fp8 DiT
  via `from_single_file` + official text encoder, VAE, tokenizer, scheduler
  from `baidu/ERNIE-Image-Turbo` (only `transformer/config.json` downloads —
  never the official bf16 weights); euler/simple, 8 steps, cfg 1.0;
  sequential CPU offload.
- **Z** (`_load_z_pipeline`): `ZImageImg2ImgPipeline`; local fp8 DiT via
  `from_single_file`; text encoder, tokenizer, scheduler from
  `Tongyi-MAI/Z-Image-Turbo` (fast) or `Tongyi-MAI/Z-Image` (quality).
  Fast: DDIM, 6 steps, cfg 1.0. Quality: `res_multistep`, `beta`,
  22 steps, cfg 4.0. The scheduler is never reconfigured — the repo default
  is used. Sequential CPU offload. The local `ae.safetensors` is tried via
  `from_single_file` but **rejected** (its `conv_out` is 32-channel vs the
  8-channel config → `ValueError`), so the Turbo family renders with the
  official VAE through the fallback chain — the fallback is load-bearing,
  not dead code (verified live).
- **Interpolation**: RIFE via `ccvfi` (recursive bisection, depth 2 = ×4),
  replacing the old FILM node.
- **Music**: ACE-Step 1.5 (in-image clone), instrumental tags per family,
  seed `31 + window.index` so long sequences evolve.
- **SFX**: MMAudio (in-image clone @8eaeb72) conditioned on the interpolated
frames, seed 7 fixed. Three load-bearing guards: the interp batch is tiled
to ≥ `MINIMUM_SYNC_FRAMES` (17) by `_pad_frames_to_minimum` — the
synchformer needs ≥ 16 sync frames and short renders crash with
`torch.stack([])` (the old `BatchPadToMin` node, ported as a helper);
`_render_music` / `_render_sound_effects` **evict their stack after
the stems land** (`_unload_music_stack` / `_unload_effects_stack` over the
shared `_collect_free_video_memory`: `gc` + `cuda.empty_cache`) — without
eviction the resident 14.5 GiB ACE stack OOMs the MMAudio load (verified
live; pinned by test); and **every fallible stage runs under
`run_stage_with_retries`** (3 attempts, evict-all-stacks + empty-cache
between tries — OOM is matched on the error shape without importing
torch, so the slim image stays light; interrupts, config errors, and
non-OOM failures propagate immediately; a budget below 1 raises
`ValueError`).

## 8. Finalize (`local_engine.py` + `final_assembly.py`)

Short sequences finalize in one window: interpolate → music + SFX →
per-stem FLAC stems + twin videos → mux. Long ones — anything interpolating
past what 48 source frames produce (`needs_segmentation`) — finalize window
by window and assemble in Python, so VRAM per job depends only on the fixed
48-frame window, never on the total frame count. The segmented path is
covered by unit tests (`_finalize_window`, `compute_segment_windows`
including a Hypothesis tiling property, crossfade assembly) but
deliberately **not run live** (a 49-frame verification render is uneconomical);
only the single-window path is verified live (see §12).

Assembly (`final_assembly.py`, ffmpeg via the `imageio-ffmpeg` binary —
`imageio`/`moviepy` are not installed): stems are joined with manual
fades/delays from known durations (music 0 dB + SFX −6 dB) → video streams
concatenated (stream copy) → AAC twin muxed. Never `acrossfade`: it
collapses on short tails. Every step verifies non-empty outputs, raising
`AssemblyError` at the culprit step; segment intermediates are removed after
a successful mux. The silent main mp4 + `-audio` twin convention is kept —
the twin (muxed AAC stereo) is the real artifact and what
`latest_video_path` prefers.

Known litter: the ACE-Step handler drops tiny UUID-named `.flac` droppings
into the output directory on each finalize (third-party behavior, a few
hundred KB); safe to delete, not referenced by any artifact.

## 9. Interface behavior (`interface.py`)

Creator-studio layout (2026-09-17 revamp, user-approved): a header row
(reachability badge backed by `engine.is_ready()` / statistics, slim
statistics line — frames · disk · last/avg durations · video · VRAM/RAM,
no timestamp — plus a short status line), then a main row with the canvas
on the left and a stepped rail on the right.

Left canvas (`zoomy-canvas` column): a status pill (Idle green / Rendering
violet / Finalizing cyan, via `_render_status_pill`), the latest frame,
an 8-slot filmstrip gallery (4 columns × 2 rows), the finished video, a
button row (Finish video / Refresh previews / Clear frames), an Interrupt
button, and the session log (append-only, last 8 of 200 lines) inside a
collapsed accordion.

Right rail (`zoomy-rail` column), four numbered steps. Step 1 Style: a
model `Radio` (display name → family key) with a shared-sequence note —
Z Fast and Z Quality continue one zoom sequence, switching cards keeps
every frame. The per-family panel is drawn by
`@gr.render(inputs=[family_selector])`. Style cards are a LoRA
`CheckboxGroup` (defaults mirror the source workflows: Ernie C64 on, Z
styles off = photorealistic default) plus one strength slider (0–2, step
0.05) per LoRA, each wrapped in its own `gr.Group` that reveals itself
only while its card stays selected (`selected_loras.change` →
`_lora_slider_visibility`). The toggle targets the wrapper group, never the
Slider itself: a Slider that doubles as a `gr.update()` target has its
frontend value clobbered by the update response, so the next Grow submits
`[{'type': 'update', ...}]` for it and Gradio's own `Slider.preprocess`
crashes before `grow_frames` runs. Step 2 Prompt: a
large prompt box (12 lines, up to 20) applying to every grown frame, and
the negative prompt inside a collapsed accordion only for families that
define one (Ernie shows a zeroed-conditioning note instead). Step 3 Grow:
a unified grow-mode `Radio` (Single frame / Loop to duration / Loop until
stopped), a target-duration number (blank = run until stopped; carries no
`minimum=`, for the same preprocess-rejection reason as the old frame
target: Gradio validates minimums before the handler runs and would
reject every blank input outright), a resolution-preset radio (one card per official vendor bucket plus a
Draft 512x512 quick-generation card last — sizes
the next grown frames, preset-only, no width/height boxes), a temporal coherence slider (0.05-0.90, step
0.05, default 0.40 — higher keeps more of the previous frame; maps to
img2img denoise as `denoise = 1 - coherence`; garbage restores the
default), and the **Grow** button (primary). Step 4 Finish: an
auto-finish checkbox (default on — auto-finish runs after loops; the
Finish button under the canvas runs it by hand).

Seeds are fully automatic: every frame draws a fresh random seed via
`_fresh_seed` (`MAXIMUM_SEED = 2**48`); there is no seed box and no lock.
`_coerce_seed` / `_resolve_frame_seed` survive only as tested pure
helpers for compatibility. `_coerce_frame_size` / `_apply_resolution_preset`
likewise survive only as tested helpers — the revamp panel passes the
preset name straight into the grow request instead of filling geometry
boxes. `_preset_dimensions` falls back to the family's own default preset
on garbage names so the geometry is never empty.

Events: the Grow button carries TWO `click` listeners because generators
must queue while stops must not wait: the queued generator (renders one
frame, or dogfoods `generate_video` with a request factory carrying the
panel preset, denoise, and a fresh random seed per frame for loops — also
surfacing the finalized video preview, so the UI loop and the
programmatic API can never drift apart) and an unqueued plain toggle
(`_grow_toggle_button`, no-op when idle, sets the stop flag when running —
this is what ends a running grow). A `grow_state` (`gr.State`) morphs the
button label between Grow and Stop; the final yield resets it
programmatically, which fires no event, so no phantom grow can start.
Gallery/preview refresh only on frame completion to avoid flicker;
durations accumulate in session `State`. The finalize generator streams
the Finalizing pill while the mux runs. Health badge + stats refresh on a
10 s `Timer` and on model change/refresh (`queue=False` so they never
block behind a render); Interrupt sets the engine flag (checked between
stages, so it lands promptly) and also sets the loop-stop flag so it ends
grows even between frames; clear-frames uses a two-click confirm
(`gr.State` armed flag + button label showing the live frame count, e.g.
"Confirm: clear 12 frames") and refreshes disarm it (no cross-family
accidents); `blocks.queue(default_concurrency_limit=1)` serializes render
jobs. Wiring lives in `FamilyPanelWiring` / `InterfaceContext` dataclasses
with small `_bind_*`/`_create_*` factories plus pure, unit-tested
format/parse helpers; the 8-10 grow-helper arguments travel as a `_GrowRun`
mutable dataclass to keep every function under the complexity budget.
Every panel input carries explicit `interactive=True`: listeners registered
inside `@gr.render` do not flip the frontend's inferred interactivity, so
without it the whole panel arrives disabled (verified live 2026-09-17 —
the server sent `interactive: null` for all 10 render-block inputs and the
browser showed a disabled cursor).

Theme: `STUDIO_THEME` (`gr.themes.Soft`, violet primary / cyan secondary)
plus `STUDIO_CSS` (dark cinematic container, canvas, and rail styling) are
applied in `main.py`'s `launch()` call — Gradio 6 takes `theme=`/`css=`
at launch, not on `Blocks()`. Every technical control carries an `info=`
hover tooltip behind a short label (Model/Style/Prompt/Negative/Mode/
Duration (s)/Resolution/Coherence/Auto-finish — the verbose explanations,
e.g. `denoise = 1 - coherence`, live in the tooltip, not the label);
`.zoomy-tip` CSS hides each `block-info` until its own control is hovered,
when it floats as a tooltip card. Gradio `Button` has no `info=` parameter
(verified against 6.26), so buttons keep plain labels.

Gradio 6.26 specifics (all verified against the runtime, not assumed):
`Dropdown` tuple choices are `(display, key)`; `Timer` takes
`value=<seconds>` (not `seconds=`); `gr.State(value=...)` keyword form;
`gr.render` is decorator-only (no direct-call form); `Checkbox` exposes
`change`/`input`/`select` but NO `unselect` (assuming it broke page load
once — the construct test never executes render functions, so only the
headless draw-all-panels test guards this); event methods
(`tick`/`change`/`click`/`select`) exist at runtime but are generated
dynamically, so the stubs omit them — each call site carries a targeted
`# type: ignore[attr-defined]` (see §11). Launch URLs carry a trailing
slash — strip it before joining paths in tests, or every probe 404s.

## 10. Configuration

| variable | default | used by |
|----------|---------|---------|
| `ZOOMY_MODELS_DIRECTORY` | `/models` | engine loaders |
| `ZOOMY_SEED_DIRECTORY` | `/seed` (compose: `/models/seed`) | cold starts |
| `ZOOMY_MUSIC_PROJECT_DIRECTORY` | `/music-project` (compose: `/models/music-project`) | ACE-Step + stems |
| `ZOOMY_OUTPUT_DIRECTORY` | `/output` | repository, frames, videos |
| `ZOOMY_INTERFACE_ADDRESS` | `0.0.0.0` (container-local) | launch |
| `ZOOMY_INTERFACE_PORT` | `7861` | launch |
| `ZOOMY_CUDA_DEVICE` | `cuda:0` | engine device |

`HF_HOME=/models/.hf-cache` and `HF_HUB_ENABLE_HF_TRANSFER=1` are set on
the service (the latter is deprecated upstream — expect a `FutureWarning`,
transfers still work). `CIVITAI_API_KEY` is only needed for provisioning.

## 11. Quality loop (all in-container, zero host deps)

```bash
./Zoomy/scripts/quality-gates.sh           # full gates: ruff fix + format + ruff + mypy + pytest
./Zoomy/scripts/run-tests.sh [-q path]     # test-only iteration, args pass through to pytest
```

Both scripts run the gates in-container against the host tree via the
`./Zoomy` bind mount (image files are throwaway copies) and export
`RUFF_CACHE_DIR`, `MYPY_CACHE_DIR`, and `HYPOTHESIS_STORAGE_DIRECTORY` to
`/tmp/...` so no tool residue ever lands in the tree. `quality-gates.sh`
additionally mounts the named `zoomy_mypy_cache` volume at `/tmp/mypy-cache`,
so repeat runs reuse type-check results (~8 s warm vs ~44 s cold; verified
2026-09-17, 245 tests). The cache is content+mtime keyed — safe to drop any
time (`docker volume rm zoomy_mypy_cache`).

Config summary (`pyproject.toml`): ruff `select = ["ALL"]`, line-length 100,
`target py312`, pydocstyle google; ignores are documented inline (`ANN401`
untyped JSON payloads, `CPY001` no header policy, `COM812`/`ISC001` formatter
conflicts, `D203`/`D213` convention conflict, `S104` container bind,
`TRY003`/`EM101`/`EM102` user-facing messages, `T201` prints (scripts only,
via per-file ignore); tests additionally ignore `S101`/`PLR2004`).
mypy `strict = true` plus `warn_unused_ignores = true`, with
`mypy_path = ["scripts"]` (the provisioner is type-checked through its test
imports). pytest `testpaths = ["tests"]`, `pythonpath = [".", "scripts"]`,
`addopts = ["-p", "no:cacheprovider", "--strict-markers"]`. Decision on the
`scripts/` seam: the provisioner stays unpackaged — `pythonpath` already
puts `scripts/` on `sys.path`, so the test modules import `download_models`
directly with no path hacks. Coverage (`pytest-cov`, exact pin in
`requirements-dev.txt`) runs in the full gate only
(`--cov=zoomy --cov-report=term-missing` in `quality-gates.sh`, floor
`fail_under = 67` ratcheting upward as GPU-independent coverage grows);
targeted `run-tests.sh` invocations stay floor-free so iteration never fails
on a partial run. Coverage data lands in `/tmp` via `data_file`, never the
bind-mounted tree.

Test layout mirrors the package (`test_settings/engine_protocol/
family_catalog/final_assembly/frame_repository/local_engine/rendering/
interface/vendor_compat/download_manifest/download_redirect`, plus
`conftest.py` with the Hypothesis profile disabling the example database
(failures are reported, not replayed; the `constants`/`unicode_data` cache
still needs `HYPOTHESIS_STORAGE_DIRECTORY=/tmp/...`, which the scripts
export). Rendering/engine tests
use fakes over `EngineProtocol`; workflow-equivalent tests assert
cold/warm starts, LoRA load-once/activate-exactly, crop math, the duration
formula, and segmentation tiling; interface tests cover the pure
format/parse helpers. Two rigor gates go beyond construction: headless
`_draw_family_panel` for every family (executes all component constructors
and event-method bindings) and an ephemeral launch serving HTTP 200.

Extra discipline carried over: property-based tests over arbitrary-value
examples (Hypothesis generators with domain constraints, `st.data()` over
6+-argument `@given`, fresh `TemporaryDirectory` per filesystem example);
UI changes need draw-path coverage; every Gradio API assumption verified
against the installed runtime; TDD for behavior/bugfix work.

## 12. End-to-end procedure (verified 2026-09-16, standalone)

```bash
./zoomy.sh                      # rebuild + launch, UI at http://localhost:7861/
# e2e drivers are ephemeral /tmp scripts (RenderEnvironment + generate_video /
# render_next_frame / finalize_video through docker exec, never committed):
# frame contract is (family, prompt, negative_prompt, frame_count,
# lora_selections, seed, frame_width, frame_height), engine ctor is
# (models_directory, seed_directory, repository, device,
# music_project_directory), and exec needs the FULL image PYTHONPATH extended
# with /application — replacing it (``-e PYTHONPATH=/application``) hides the
# /opt ACE/MMAudio clones and finalize dies with ModuleNotFoundError. Drivers
# must also run with the host tree shadowed over the image
# (``-v ./Zoomy:/application``); otherwise they import the stale baked code
# (e.g. the pre-flatten nested-layout lookup → phantom EmptyFrameSequence).
#   z_fast frame 1 cold start (~140 s incl. Qwen/Z hub fetch, then ~33 s warm)
#     -> /output/Zoomy/z_image/frame_00001_.png, 1376x768, coherent zoom
#     (continuity: mean|f2-zoom(f1)|=35.29 < mean|f2-f1|=39.32)
#   z_quality frame (~202 s res_multistep/beta/22/cfg 4 vs Comfy 65 s — slower)
#   ernie_turbo frame 1 + C64 LoRA @1.0 (~234 s cold): full C64 pixel-art style
#   finalize 2 frames -> Zoomy_z_image_00001.mp4 (0.16 s, 5 interp frames,
#     h264 32 fps) + AAC stereo -audio twin, RMS 0.27/peak 1.0 (real mix)
#   10 s demo (flat layout): 81 z_fast frames @512x512 (~12 s/frame) +
#     finalize -> z_image_00001.mp4 (9.94 s, 512x512 h264 32 fps, silent
#     main) + AAC stereo -audio twin (RMS -12 dB, real full-length mix);
#     frame 41 viewed: coherent rainbow-path zoom, no artifacts
# cleanup inside the container: rm -f /output/cold_* /output/ab2_* /output/*.flac
#   + any test-sequence frames beyond the real chain
```

LoRA verification (all Z-Image, all viewed): Comfy-format
`diffusion_model.*` keys attach exactly (chalkboard 240/3097 transformer
modules, C64 control 252/3078, no unused-key warnings, healthy B@A norms).
Chalkboard @1.0 from seed renders a strong black-board + chalk-linework
look; @0.85 (the ported catalog default) it is partial — a possible mild
sensitivity gap vs the Comfy confirmation (whose exact strength/seed are
lost to history; not a bug: attachment, norms, and dose-response all check
out). Clay-art @1.0 reads sculptural but moderate, not full claymation.
Style flips mid-sequence transition slowly: at denoise 0.60 the predecessor
dominates, so a new style needs a fresh sequence or several consecutive
frames (same mechanics as Comfy — document, don't fix).

## 13. Future work

- More families = more catalog entries (nothing else changes).
- Optional per-frame prompt history in the UI. (Manual seed control was
  deliberately dropped in the 2026-09-17 revamp — every frame draws fresh
  random; do not re-add without a user decision.)
- Z LoRA default strengths: consider 1.0 (chalkboard renders partial at the
  ported 0.85; user decision).
- If the UI is ever exposed beyond localhost, add auth in front of it.
