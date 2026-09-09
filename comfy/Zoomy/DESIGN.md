# Zoomy — Design Document

> Maintained alongside the code. Any behavior change must update this file in
> the same change. See also the wrapper doc `../AGENTS.md` §10.

## 1. Overview

Zoomy is a small Gradio web application that unifies the two verified
infinite-zoom ComfyUI loop workflows (Ernie Image Turbo, Juggernaut Z) into
one control panel. The user picks a model family, selects LoRA styles with
strengths, edits the prompt, and clicks **Render next frame** once per frame;
**Finalize video** turns the accumulated frames into a music video with
sound effects.

Deliberate non-goals:

- No workflow JSON is stored or edited. Graph logic that used to live in
  `ComfySwitchNode` / `ComfyMathExpression` nodes (switches, conditions,
  duration math) now lives in Python; only the heavyweight ComfyUI nodes
  (loaders, samplers, interpolation, audio models, video encode) remain in
  the submitted graphs.
- No multi-user support, no auth. Single local user assumed; like ComfyUI
  itself, the UI listens on all interfaces (no auth — never expose the host
  beyond a trusted network).
- No host-side dependencies. Everything (runtime and dev tools) runs inside
  the `zoomy` container.

## 2. Runtime topology

```
browser (host) ──http──> zoomy:7861 ──REST──> comfy:8188 ──files──> Comfy/output
      localhost:7861      container              container           bind mount
```

- `zoomy` service (`Zoomy/Dockerfile`, `python:3.12-slim`): Gradio app,
  port `7861:7861`, volume `./Comfy/output:/comfy/output`. Declares
  `depends_on: [comfy]`; the healthcheck is `curl --fail
  http://localhost:7861/` (same tool/pattern as the comfy service).
- The `comfy` container carries a fixed `--name comfy` (see `serve.sh`), so
  the name resolves on the shared `comfy_default` bridge and zoomy reaches
  ComfyUI at `http://comfy:8188` with no host-network access and no dependence
  on published ports. Do not restart comfy or edit `serve.sh` beyond the name
  for zoomy networking.
- Lifecycle: `./zoomy.sh` spawns comfy via `serve.sh` when no container
  named `comfy` is running, then launches zoomy itself
  (`run --build --rm --detach --service-ports --no-deps zoomy`) as an
  ephemeral `comfy-zoomy-run-*` container; `docker stop` removes it. The
  `depends_on` in compose declares the relationship, but `--no-deps`
  deliberately skips compose-managed startup so a second, port-conflicting
  comfy instance is never auto-started.

## 3. Repository layout (`Zoomy/`)

```
Zoomy/
  DESIGN.md               # this file (excluded from the image via .dockerignore)
  pyproject.toml          # metadata + ruff (ALL, line-length 100) + mypy strict + pytest
  requirements.txt        # runtime: gradio==6.26.0, httpx==0.28.1
  requirements-dev.txt    # dev: ruff==0.16.6, mypy==2.3.1, pytest==9.1.1
  Dockerfile              # python:3.12-slim, /application, CMD ["python", "-m", "zoomy"]
  .dockerignore
  zoomy/                  # the application package
    __init__.py           # version only
    __main__.py           # `python -m zoomy` entry
    main.py               # wiring: settings -> connection/repository -> app -> launch
    settings.py           # Settings.from_environment (ZOOMY_* variables)
    errors.py             # ZoomyError hierarchy
    family_catalog.py     # frozen FamilyDefinition/LoraDefinition catalog of 3
    graph.py              # ComfyWorkflow builder + NodeReference ([key, slot] links)
    frame_repository.py   # frame/video discovery + clearing under the output dir
    comfy_connection.py   # typed ComfyUI REST client + ConnectionProtocol
    frame_workflow.py     # per-frame render graph builder
    finalize_workflow.py  # frames-to-music-video graph builder + duration math
    rendering.py          # queue -> poll -> progress generators
    interface.py          # Gradio layout + event wiring
  tests/                  # one module per source module (see §11)
```

## 4. Module reference

- `settings.Settings.from_environment()`: `ZOOMY_COMFY_ADDRESS`
  (default `http://comfy:8188`), `ZOOMY_OUTPUT_DIRECTORY`
  (`/comfy/output`), `ZOOMY_INTERFACE_ADDRESS` (`0.0.0.0` — inside the
  container only), `ZOOMY_INTERFACE_PORT` (`7861`),
  `ZOOMY_OPERATION_TIMEOUT_SECONDS` (`1800`), `ZOOMY_POLL_INTERVAL_SECONDS`
  (`2`). Blank values fall back to defaults; garbage numbers raise
  `ZoomyError`.
- `errors`: `ZoomyError` base; `ComfyConnectionError` (transport/unexpected),
  `ComfyRejectedWorkflowError` (HTTP 400, carries server `summary` +
  `node_errors`), `ComfyExecutionError` (node_type + exception_message from
  the `execution_error` status message), `RenderInterruptedError`,
  `OperationTimeoutError`, `EmptyFrameSequenceError`. The interface catches
  `ZoomyError` and shows it in the status line.
- `family_catalog`: `FAMILY_CATALOG` holds `ernie_turbo`, `z_fast`,
  `z_quality`. `find_family(catalog, key)` raises `ZoomyError` on miss.
  `LoraDefinition.selected_by_default` mirrors the source workflows (Ernie
  C64 on; Z styles off = photorealistic default).
- `graph.ComfyWorkflow.add(key, class_type, inputs)`: converts
  `NodeReference` values (including inside lists) to `[key, slot]` arrays;
  rejects duplicate keys; `build()` returns a deep copy.
- `frame_repository.FrameRepository(output_directory)`: `frame_paths` globs
  `Zoomy/<sequence>/frame_*.png` sorted (ComfyUI zero-pads counters, so name
  order is render order); `latest_video_path` returns the newest
  `Zoomy_<sequence>*.mp4`, **preferring the `-audio` twin** (see §7).
- `comfy_connection.ComfyConnection`: persistent `httpx.Client`;
  `queue_workflow` posts `{"prompt": workflow, "client_id": "zoomy"}` and
  parses 400 bodies into `ComfyRejectedWorkflowError`;
  `fetch_history` parses `GET /history/<id>` into `HistoryEntry(outputs,
  status_messages, is_completed)`; `interrupt` posts `/interrupt`;
  `is_reachable` probes `/system_stats`. `ConnectionProtocol` is the
  structural contract rendering/interface depend on (lets tests use fakes).
- `rendering`: `render_next_frame` / `finalize_video` are generators yielding
  `ProgressUpdate(message, frame_path, video_path, frame_count)`. The poll
  loop fetches history first and checks the timeout after, so fast jobs never
  trip a zero budget; completed entries are scanned once for
  `execution_error` / `execution_interrupted`.
- `interface.build_application(...)`: see §8.
- `main.main()`: builds settings/connection/repository, queues the app with
  `default_concurrency_limit=1` (one ComfyUI job at a time), and launches
  with `allowed_paths=[output_directory]`, `show_error=True`,
  `inbrowser=False`, `ssr_mode=False` (no Node.js in the slim image).

## 5. Family catalog

| key | display | sequence | base model | text encoder | shift | sampler |
|-----|---------|----------|------------|--------------|-------|---------|
| `ernie_turbo` | Ernie Image Turbo | `ernie_turbo` | ernie-image-turbo | ministral-3-3b (`flux2`) | none | euler/simple/8/cfg 1.0 |
| `z_fast` | Juggernaut Z (Fast) | `z_image` | juggernautZ_v10FastBy | qwen_3_4b_fp8_mixed (`lumina2`) | 3.0 | ddim/normal/6/cfg 1.0 |
| `z_quality` | Juggernaut Z (Quality) | `z_image` | juggernautZ_v10ByRundiffusion | qwen_3_4b_fp8_mixed (`lumina2`) | 3.0 | res_multistep/beta/22/cfg 4.0 |

Shared: 1376×768 frames, img2img denoise 0.60, seed images
`ernie_zoom_seed.png` / `z_zoom_seed.png` from ComfyUI's input dir, ACE-Step
music + MMAudio SFX prompts per family (verbatim in `family_catalog.py`),
SFX negative `speech, voice, vocals, singing, music, melody, drums, beat`.
Ernie has no negative prompt (uses `ConditioningZeroOut`); Z uses a real
negative `CLIPTextEncode`. Both Z variants share sequence `z_image`, so
switching speed mid-sequence continues the same zoom.

**Adding a family**: append one `FamilyDefinition` to `FAMILY_CATALOG` (new
`sequence_key` for a new sequence, existing one to join it). The interface
picks it up with zero UI changes.

## 6. Frame workflow (`frame_workflow.py`)

Cold start (`frame_count == 0`) loads `LoadImage(family.cold_start_image)`;
otherwise `VHS_LoadImagesPath(directory, image_load_cap=1,
skip_first_images=count-1, select_every_nth=1)` reads only the newest frame.
Then: loaders → chained `apply_lora_<n>` (model slot 0 + clip slot 1 through
every selected LoRA in order; skipped entirely when none) → optional
`apply_model_shift` (only when `model_shift` is set) → prompt encode (+
negative encode or zeroing) → `crop_previous_frame` (1356×748 at 10,10) →
`rescale_cropped_frame` (bicubic to 1376×768, crop disabled) → `VAEEncode` →
`KSampler` (fresh `SystemRandom` seed `< 2**48` per frame + family recipe) →
`VAEDecode` → `SaveImage` prefix `Zoomy/<sequence>/frame` (lands as
`frame_00001_.png`, …).

Zoom math: `1376 / 1356 ≈ 1.0147` per frame (~1.5% dive); offset rule
`(1376 − 1356) / 2 = 10` keeps the dive centered. Geometry constants
(`FRAME_WIDTH_PIXELS`, `FRAME_HEIGHT_PIXELS`, `CROP_BORDER_PIXELS`) are
module-level and covered by tests.

## 7. Finalize workflow (`finalize_workflow.py`)

Python computes `interpolated = (frame_count − 1) * 4 + 1` and
`audio_seconds = max(interpolated / 32, MINIMUM_AUDIO_SECONDS)`; the graph
holds only the heavy nodes: full-frame loader → FILM `×4` → `BatchPadToMin`
(17, for MMAudio's ≥16 sync-frame requirement) → ACE-Step music chain
(seed 31, 8 steps, cfg 1.0, euler/simple, empty lyrics = instrumental) →
MMAudio SFX chain (conditioned on the *padded* frames, 25 steps, cfg 4.5,
seed 7, mask_away_clip, force_offload, −6 dB) → `AudioMerge(add)` →
`VHS_VideoCombine` (images ← the **unpadded** interpolation, audio ← merge,
32 fps, `Zoomy_<sequence>` prefix, h264/yuv420p/crf 19).

Audio floor rationale: `EmptyAceStep1.5LatentAudio.seconds` enforces
**min 1.0** (live-server truth; the ACE text encoder allows ≥ 0.0 and
MMAudio has no minimum), so `MINIMUM_AUDIO_SECONDS = 1.0`. Overshoot is safe:
the VHS mux always applies `-shortest`, trimming floored audio to the video
length (verified: 2-frame finalize → 0.157s video, no 1.0s stretch).

**VHS twin behavior** (read from the installed VHS source): with audio
connected, the node writes a silent main file `Zoomy_<seq>_00001.mp4` *plus*
an `-audio` twin `Zoomy_<seq>_00001-audio.mp4` carrying the muxed AAC track
(plus a preview PNG). The twin is the real artifact — `latest_video_path`
prefers it.

## 8. Interface behavior (`interface.py`)

Header row (family dropdown, reachability badge, frame counter) → status
line → previews column (latest frame, finalized video) + per-family panel
column. The panel is drawn by `@gr.render(inputs=[family_dropdown])`: LoRA
`CheckboxGroup` + one strength slider (0–2, step 0.05) per family LoRA,
prompt box, negative box only for families that define one, and the render
button. Components the panel wiring touches are created *before* the render
block (the decorator executes immediately at build).

Events: render/finalize buttons run the rendering generators (streamed
status); health badge + counter refresh on a 10 s `Timer` and on dropdown
change/refresh (`queue=False` so they never block behind a render); interrupt
posts `/interrupt`; clear-frames uses a two-click confirm (`gr.State` armed
flag + button label change); `blocks.queue(default_concurrency_limit=1)`
serializes ComfyUI jobs. Wiring lives in `FamilyPanelWiring` /
`InterfaceContext` dataclasses with small `_bind_*` factories to keep every
function under the complexity budget.

Gradio 6.26 specifics (all verified against the runtime, not assumed):
`Dropdown` tuple choices are `(display, key)`; `Timer` takes
`value=<seconds>` (not `seconds=`); `gr.State(value=...)` keyword form;
`gr.render` is decorator-only (no direct-call form); event methods
(`tick`/`change`/`click`) exist at runtime but are generated dynamically, so
the stubs omit them — each call site carries a targeted
`# type: ignore[attr-defined]` (see §11).

## 9. ComfyUI API contract

- `POST /prompt` `{"prompt": <api-format dict>, "client_id": "zoomy"}` →
  `{"prompt_id": ...}`; HTTP 400 body `{"error": {...}, "node_errors": {...}}`
  becomes `ComfyRejectedWorkflowError` (a spec bug, never retried).
- `GET /history/<id>` → `{id: {"outputs": {...}, "status": {"completed":
  bool, "messages": [[name, payload], ...]}}}`. `execution_error` payload
  carries `node_type`/`exception_message`; `execution_interrupted` means the
  user (or someone) stopped the job.
- `POST /interrupt`, `GET /system_stats` (reachability probe).
- Artifacts are resolved through the repository (mtime/name), not through
  history outputs — SaveImage/VHS naming is deterministic.

## 10. Configuration

| variable | default | used by |
|----------|---------|---------|
| `ZOOMY_COMFY_ADDRESS` | `http://comfy:8188` | connection |
| `ZOOMY_OUTPUT_DIRECTORY` | `/comfy/output` | repository, loaders, save prefixes |
| `ZOOMY_INTERFACE_ADDRESS` | `0.0.0.0` (container-local) | launch |
| `ZOOMY_INTERFACE_PORT` | `7861` | launch |
| `ZOOMY_OPERATION_TIMEOUT_SECONDS` | `1800` | poll loops |
| `ZOOMY_POLL_INTERVAL_SECONDS` | `2` | poll loops |

## 11. Quality loop (all in-container, zero host deps)

```bash
# fix + verify host files through a bind mount (image files are throwaway copies):
docker compose run --rm --no-deps -v ./Zoomy:/workspace \
  -e RUFF_CACHE_DIR=/tmp/ruff-cache -e MYPY_CACHE_DIR=/tmp/mypy-cache zoomy sh -c \
  "cd /workspace && ruff check --fix zoomy tests && ruff format zoomy tests \
   && ruff check zoomy tests && mypy zoomy tests && python -m pytest -p no:cacheprovider"
```

Config summary (`pyproject.toml`): ruff `select = ["ALL"]`, line-length 100,
`target py312`, pydocstyle google; ignores are documented inline (`ANN401`
untyped JSON payloads, `CPY001` no header policy, `COM812`/`ISC001` formatter
conflicts, `D203`/`D213` convention conflict, `S104` container bind,
`TRY003`/`EM101`/`EM102` user-facing messages; tests additionally ignore
`S101`/`PLR2004`). mypy `strict = true`. pytest `testpaths = ["tests"]`,
`pythonpath = ["."]`.

Test layout mirrors the package (`test_settings/graph/family_catalog/
frame_workflow/finalize_workflow/frame_repository/rendering/interface`;
rendering uses a scripted `ConnectionProtocol` fake; workflow tests assert
cold/warm starts, LoRA chaining, shift/negative conditionals, crop math, the
duration formula, and that every `[key, slot]` link resolves).

Known stub gaps (gradio 6.26 ships incomplete types for its dynamically
generated API): `Timer`/`Dropdown`/`Button` event methods
(`# type: ignore[attr-defined]` per call site), `@gr.render`
(`# type: ignore[untyped-decorator]`), `Blocks` context-manager return
(`typing.cast`). Each is verified present at runtime.

## 12. End-to-end procedure (verified 2026-09-09)

```bash
./zoomy.sh                      # rebuild + launch, UI at http://localhost:7861/
# inside the zoomy container (docker exec <zoomy-container> python):
#   z_fast frame 1 cold start (~15 s) -> Zoomy/z_image/frame_00001_.png 1376x768 RGB
#   z_fast frame 2 warm start (~9 s)
#   finalize 2 frames (~20 s) -> Zoomy_z_image_00001.mp4 (0.157 s) + AAC -audio twin
#   ernie_turbo frame 1 with C64 LoRA @1.0 (~23 s, exercises the LoRA chain live)
# cleanup inside the container: rm -rf /comfy/output/Zoomy /comfy/output/Zoomy_<seq>*
```

The 400-path was verified live too: the first finalize attempt (0.7 s floor)
was rejected with per-node detail, which exposed the ACE `seconds >= 1.0`
minimum and led to the 1.0 floor (§7).

## 13. Future work

- More families = more catalog entries (nothing else changes).
- Optional seed control / per-frame prompt history in the UI.
- Frame gallery strip (currently only the latest frame previews).
- If the UI is ever exposed beyond localhost, add auth in front of it.
