# comfy/ AGENTS.md — Project Wrapper (host + container)

> This file lives at **`comfy/AGENTS.md`** (i.e. `/home/goulade/Projects/ai/comfy/AGENTS.md`), **not** `Comfy/AGENTS.md` which is upstream ComfyUI's own engineering guide. Keep both. After each significant amount of work, update this file if new non-obvious facts became known — that's a hard requirement for this wrapper.

## Quick Map

```
comfy/                          # wrapper repo (this file's scope)
  serve.sh                      # ONLY entrypoint — `docker compose run --build --rm --detach --service-ports comfy`
  docker-compose.yml            # single service `comfy`, init:true, healthcheck, ports 8188/7860/9000, volumes ./Comfy:/comfy ./input:/input comfy:/root/.cache
  Dockerfile                    # FROM pytorch/pytorch:2.9.0-cuda12.8-cudnn9-devel, bakes comfy-cli/mcp + custom nodes + requirements.txt
  entry.sh                      # dual-process: ComfyUI :8188 + comfy-mcp bridged :9000 (Streamable HTTP /mcp), handles set-default, req installs, patches
  AGENTS.md                     # THIS FILE — wrapper facts + update discipline
  Comfy/                        # upstream checkout (mounted as /comfy inside container), NOT rebuilt from scratch
    main.py                     # `python /comfy/main.py --listen 0.0.0.0 --port 8188` (invoked by entry.sh)
    models/
      diffusion_models/         # ernieImageTurboQT_fp8V2.safetensors (7.7G, Civitai 3028150 fp8 inference requested, symlink ernie-image-turbo.safetensors) + put_*here placeholders
      text_encoders/            # ministral-3-3b.safetensors 7.2G, ernie-image-prompt-enhancer.safetensors 6.5G (currently NOT wired — see GPU note)
      vae/                      # flux2-vae.safetensors 321M (HF Comfy-Org/ERNIE-Image)
    input/  -> ../input host mount (see docker-compose)
    output/                     # ALL generations confined here, including loopback cache `output/loopback_ernie_zoom/` — `rm -rf Comfy/output/*` cleans everything
      loopback_ernie_zoom/ernie_zoom_fixed/ernie_zoom/{history/*.png, cached_img.png, ...}
      Ernie_Zoom_Loop_*.mp4     # native CreateVideo/SaveVideo outputs
      Ernie-Image-Turbo_*.png   # earlier single-frame tests
    temp/image_loopback/        # LEGACY location, migrated on boot to output/ — do NOT rely on it
    custom_nodes/
      comfyui-manager           # 3.41, package manager
      comfy-loopback-buffer     # holo-q/comfy-loopback-buffer 1.0.0 MIT — Store/Load/Configure Loopback (history append via disk)
      ComfyUI-VideoHelperSuite  # Kosinkadink 1.7.9 — VideoCombine etc. (kept QoL, not used after native CreateVideo)
      cg-use-everywhere         # chrisgoringe 7.8 Apache2 — Anything Everywhere broadcast (replaces manual spaghetti)
    user/default/
      workflows/                # browser-visible workflows (auto-formatted, see below)
        ernie_turbo_qt.json               # adapted official Ernie Turbo template (PreviewAny fixed, enhancer removed)
        ernie_turbo_qt_simple.json        # minimal variant
        ernie_turbo_qt_enhanced.json      # backup with enhancer chain (OOM on 16GB — don't use without >24GB)
        ernie_infinite_zoom.json          # unrolled 12-frame fixed batch, N=12 768px, CreateVideo fps8 (legacy, per-queue rebuilds all frames)
        ernie_infinite_zoom_loop.json     # ✅ incremental: one new frame per Queue, 512px, scale 0.85 Pad 38/39 feather20, history_limit 0 unlimited, Preview + CreateVideo fps8
      comfy.settings.json       # frontend settings — *must* contain `"Comfy.LinkRenderMode":"Straight"` for square right-angle edges
```

## Non-Obvious Facts (read before touching anything)

### 1. Host vs Container Boundary
- **opencode runs on host only, comfy MUST stay in container** via `serve.sh` (`docker compose run --build --rm --detach ...`). Never `comfy launch` on host; host has no comfy binaries by design (`pip uninstall comfy-cli/comfy-mcp`, `~/.config/comfy-cli` removed, opencode `mcp: { comfy-mcp: { type: remote, url: http://localhost:9000/mcp, oauth:false }}`).
- `serve.sh` uses **`run` not `up`** intentionally — keep it that way. `run --service-ports` publishes 8188/9000 and creates an ephemeral container name like `comfy-comfy-run-<hash>`. `docker ps` shows `comfy-comfy-run-... Up (healthy)`.
- `COMFY_BIN` / `COMFYUI_URL` env passthrough is via `mcp-proxy` style `--pass-environment` (now inlined bridge). Don't set them manually.

### 2. Dockerfile Rebuild Safety
- Rebuilds are frequent (`--build` on every `serve.sh`). Workflows MUST survive rebuild.
- Pattern: `pip install ... --requirement https://raw.githubusercontent.com/comfyanonymous/ComfyUI/master/requirements.txt --requirement https://raw.githubusercontent.com/Comfy-Org/ComfyUI-Manager/main/requirements.txt` — **follow same pattern for custom nodes** (see Dockerfile).
- Dockerfile **bakes** the three QoL packs into the image (`git clone` to `/comfy/custom_nodes/...` at build) **and** `pip install -r` for each `requirements.txt` found (VHS has `opencv-python imageio-ffmpeg`; loopback has no req; UE has none). This covers fresh clones where host volume `./Comfy:/comfy` overlays image's `/comfy/custom_nodes` — pip deps survive overlay, code comes from host mount.
- `entry.sh` additionally loops `for req in /comfy/custom_nodes/*/requirements.txt` at **runtime** to install host-side fresh clones not baked (covers `docker compose run --build` after `git clone` on host without rebuilding image).
- Never add host-side `pip install` — all installs belong inside image / entry.sh runtime.

### 3. Entry.sh Dual-Process & Patches
- PID1 is `init` (docker `init: true` handles zombie reaping); entry.sh `wait -n` then kills the other pid for clean `docker stop`.
- Must run `comfy set-default /comfy` every boot because config at `/root/.config/comfy-cli/config.ini` is not volume-persisted.
- **Loopback fixed key patch**: upstream `nodes.py:_resolve_workflow_key` hashes workflow JSON, so `comfy run` (cli) vs UI Queue produce different workflow hashes → different cache dirs → history lost. We patch to return fixed `"ernie_zoom_fixed"` when `cache_path` is `/comfy/output/loopback_ernie_zoom` (or legacy `"ernie_zoom"`). Patch is idempotent and applied at runtime via `entry.sh` Python heredoc + at build via host file. If you clone loopback fresh, entry.sh re-patches.
- **Square edges**: entry.sh stamps `Comfy/user/default/comfy.settings.json` to `{"Comfy.LinkRenderMode":"Straight"}` (options: Hidden/Linear/Spline/Straight — Straight = square right angles). Also creates file if missing. Workflows carry `_linkRenderMode:Straight` in `extra` as documentation, but real setting is global.
- **Cache migration**: on boot, if `/comfy/temp/image_loopback/ernie_zoom_fixed` exists and new `/comfy/output/loopback_ernie_zoom/ernie_zoom_fixed` does not, `cp -a` migrates. Manual migration already done for current history (4 frames).

### 4. Loopback Confinement to `output/`
- Old `cache_path="ernie_zoom"` → `_cache_dir` returned `/comfy/temp/image_loopback/<workflow_key>/ernie_zoom` (relative path, trapped under `get_temp_directory()`). Escaping with `../../output` is blocked by `candidate.startswith(base_root)`.
- New `cache_path="/comfy/output/loopback_ernie_zoom"` → `_cache_dir` returns `/comfy/output/loopback_ernie_zoom/<workflow_key>` (absolute branch, `os.path.isabs`). With fixed key, final dir is `/comfy/output/loopback_ernie_zoom/ernie_zoom_fixed/ernie_zoom/` containing `history/`, `cached_img.png`, etc. **All generation is now under `Comfy/output/`** — cleanup is `rm -rf Comfy/output/*` (also clears history).
- Workflow widgets: `Image-Loopback-Load/Cache/Configure` all have `cache_path` widget 0 = new absolute path. Formatting script enforces this.

### 5. GPU & Model Notes
- Host reports `RTX 4060 Ti 16GB` (17175674880 bytes → 15.57 GiB usable) + secondary 2060. comfy-mcp `server_info` confirms same.
- Ernie Turbo QT: base ~8B DiT + ministral-3b + flux2-vae, official template `image_ernie_image_turbo` (subgraph 19 nodes, shift 3.1, 8 steps euler simple cfg1). Civitai 2660020 fp8-v2 id 3028150 hash 3A2FC7B902.
- **Why enhancer removed**: `TextGenerate` (prompt enhancer) loads second LLM ~6.5G on top of ministral 7.2G + DiT 7.7G + VAE 0.32G → OOM (`15.57 GiB` limit). Logs showed `nodes_textgen.py:64 clip.generate OOM` even when Switch false (node still executes). Fix: delete nodes 93,95,96,97,98 and wire `94->67` directly via link 134; keep 9 inputs with empty linkIds for slot indices. Kept `ernie_turbo_qt_enhanced.json` as backup for >24GB VRAM.
- Resolution dropped to **512** (PrimitiveInt) from original 1024/768 to keep img2img 0.85 pad math cheap and fit VRAM: pad = (512-435)//2 ≈ 38/39 with feather 20, grow_mask_by 6, denoise 0.65, steps 8 cfg1 euler simple. Few sec/frame on 4060 Ti.
- Models symlinked: `ernieImageTurboQT_fp8V2.safetensors` ↔ `ernie-image-turbo.safetensors` for CLIPLoader compatibility.

### 6. Workflows & Validation
- Browser location is `Comfy/user/default/workflows/` (not `Comfy/workflows/`).
- All workflows are auto-formatted via `/tmp/format_workflows.py` (layered grid: `X_STEP 380, Y_STEP 150, X0 -600, Y0 -200`, topological depth via links, groups centered). Re-run formatter after manual edits to keep positions tidy — requirement is **always use auto-formatting tool**, never hand-tweak pos.
- Validate via `docker exec <cid> comfy workflow validate --workflow /comfy/user/default/workflows/<name>.json` (or comfy-mcp `validate_workflow`). Must be `valid: true, 0 errors` before queue. Current: `ernie_infinite_zoom_loop.json` valid 25 nodes 15 levels; `ernie_infinite_zoom.json` 67 nodes 63 levels (legacy unrolled).
- `ernie_infinite_zoom_loop.json` flow: `EmptyFlux2LatentImage(512) -> KSampler(seed random) -> VAEDecode -> Sample Loopback(1) -> ScaleBy 0.85 bicubic -> PadForOutpaint 38/39 feather20 -> VAEEncodeForInpaint -> KSampler denoise 0.65 random -> VAEDecode -> Store Loopback -> Sample Full history 1-1000 -> Preview + CreateVideo fps8 -> SaveVideo Ernie_Zoom_Loop`. `Configure history_limit 0` = unlimited, `update_from_cache true`, `history_indices 1` and `1-1000`.

### 7. Custom Nodes Audit (QoL / Safety)
- **comfy-loopback-buffer** (holo-q, 1.0.0 MIT, deps torch/pillow/numpy/aiohttp): replaces filesystem Save/Load reimplementation for incremental accumulation. Small, tested, maintained — **keep** (required for perpetual zoom). Not hugely starred but functionally unique, no known CVEs, no network.
- **ComfyUI-VideoHelperSuite** (Kosinkadink, 1.7.9, ~4k★): `VideoCombine` etc. Thousands of dependents, active, deps already baked. Native `CreateVideo/SaveVideo` now covers our use, so VHS is not strictly needed but harmless QoL for future video tasks — **keep**, but don't rely on it for core workflow.
- **cg-use-everywhere** (chrisgoringe, 7.8 Apache2, 1k★, frontend-only): `Anything Everywhere` broadcast reduces link spaghetti, no python deps. Lightweight, widely used — **keep** (replaces manual wiring reimplementation).
- No removal recommended; all are maintained, trustworthy, popular. If a new dep is added, check stars, commit recency, license, and whether it replaces a reimplementation.

### 8. Confinement & Cleanup Discipline
- Everything generated goes under `Comfy/output/` — verify after each workflow edit: `ls -R Comfy/output` should contain only intended files. Loopback history counts as generation.
- `Comfy/temp/` should stay empty of persistent history; entry.sh migration ensures legacy temp history moves to output.
- To reset loopback history: `rm -rf Comfy/output/loopback_ernie_zoom` or queue with `Store` disabled once. To clean videos: `rm Comfy/output/Ernie_Zoom*.mp4`.

### 9. Update Discipline
- **After each significant amount of work, update this AGENTS.md** if new relevant info/facts became known (new model, new workflow, new custom node, new volume, new port, new patch, perf numbers, VRAM lessons). Keep this file as the single source of truth for the wrapper; don't let facts live only in chat.

## Commands Cheat Sheet

```bash
./serve.sh                                   # build + run (ephemeral) — the ONLY way to start
docker ps --format "{{.Names}} {{.Status}}"   # comfy-comfy-run-... Up (healthy)
docker logs <cid> --tail 100                 # ComfyUI logs
docker exec <cid> comfy workflow validate --workflow /comfy/user/default/workflows/ernie_infinite_zoom_loop.json
docker exec <cid> bash -c 'ls /comfy/output/loopback_ernie_zoom/ernie_zoom_fixed/ernie_zoom/history | wc -l'
curl -s http://localhost:9000/status | jq    # comfy-mcp bridge health
curl -s http://localhost:8188 | head         # ComfyUI health
# Format workflows after manual edits:
python3 /tmp/format_workflows.py
```

## Known Gotchas
- Host `comfy` binary absent by design — use `docker exec <cid> comfy ...` instead.
- Changing `cache_path` to an absolute under `/comfy/output` is the ONLY way to confine loopback; relative paths are jailed under temp.
- Workflow `id` changes on save can drift hash key — fixed key patch prevents history loss.
- 512px is deliberate for speed/VRAM; don't bump to 1024 without testing OOM.
- `Comfy.LinkRenderMode` must be `Straight`; if edges look curved, re-check `comfy.settings.json`.
