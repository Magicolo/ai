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
      loopback_ernie_zoom/ernie_zoom_fixed/{history/*.png, cached_img.png, current_img.png, loopback_config.json}
      Ernie_Zoom_Loop_*.mp4     # native CreateVideo/SaveVideo outputs
      Ernie-Image-Turbo_*.png   # earlier single-frame tests
    temp/image_loopback/        # LEGACY location, migrated on boot to output/ — do NOT rely on it (now empty)
    custom_nodes/
      comfyui-manager           # 3.41, package manager
      comfy-loopback-buffer     # holo-q/comfy-loopback-buffer 1.0.0 MIT — Store/Load/Configure Loopback (history append via disk)
      ComfyUI-VideoHelperSuite  # Kosinkadink 1.7.9 — VideoCombine etc. (kept QoL, not used after native CreateVideo)
      cg-use-everywhere         # chrisgoringe 7.8 Apache2 — Anything Everywhere broadcast (replaces manual spaghetti)
      comfyui-workflow-prettier # deepme987/comfyui-workflow-prettier — auto-positioning (Sugiyama Layered DAG layout)
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
- Dockerfile **bakes** the four QoL packs into the image (`git clone` to `/comfy/custom_nodes/...` at build) **and** `pip install --requirement` for each `requirements.txt` found (VHS has `opencv-python imageio-ffmpeg`; loopback/UE/prettier have none/empty). This covers fresh clones where host volume `./Comfy:/comfy` overlays image's `/comfy/custom_nodes` — pip deps survive overlay, code comes from host mount. Total 6 `--requirement` flags (2 remote + 4 local).
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
- New `cache_path="/comfy/output/loopback_ernie_zoom"` → `_cache_dir` returns `/comfy/output/loopback_ernie_zoom/<workflow_key>` (absolute branch, `os.path.isabs`). With fixed key, final dir is `/comfy/output/loopback_ernie_zoom/ernie_zoom_fixed/` containing `history/`, `cached_img.png`, etc. **All generation is now under `Comfy/output/`** — cleanup is `rm -rf Comfy/output/*` (also clears history).
- Workflow widgets: `Image-Loopback-Load/Cache/Configure` all have `cache_path` widget 0 = new absolute path. Prettier formatting preserves this.

### 5. GPU & Model Notes
- Host reports `RTX 4060 Ti 16GB` (17175674880 bytes → 15.57 GiB usable) + secondary 2060. comfy-mcp `server_info` confirms same.
- Ernie Turbo QT: base ~8B DiT + ministral-3b + flux2-vae, official template `image_ernie_image_turbo` (subgraph 19 nodes, shift 3.1, 8 steps euler simple cfg1). Civitai 2660020 fp8-v2 id 3028150 hash 3A2FC7B902.
- **Why enhancer removed**: `TextGenerate` (prompt enhancer) loads second LLM ~6.5G on top of ministral 7.2G + DiT 7.7G + VAE 0.32G → OOM (`15.57 GiB` limit). Logs showed `nodes_textgen.py:64 clip.generate OOM` even when Switch false (node still executes). Fix: delete nodes 93,95,96,97,98 and wire `94->67` directly via link 134; keep 9 inputs with empty linkIds for slot indices. Kept `ernie_turbo_qt_enhanced.json` as backup for >24GB VRAM.
- Resolution dropped to **512** (PrimitiveInt) from original 1024/768 to keep img2img 0.85 pad math cheap and fit VRAM: pad = (512-435)//2 ≈ 38/39 with feather 20, grow_mask_by 6, denoise 0.65, steps 8 cfg1 euler simple. Few sec/frame on 4060 Ti.
- Models symlinked: `ernieImageTurboQT_fp8V2.safetensors` ↔ `ernie-image-turbo.safetensors` for CLIPLoader compatibility.

### 6. Workflows & Validation
- Browser location is `Comfy/user/default/workflows/` (not `Comfy/workflows/`). All 5 custom workflows are listed below; 3 are subgraph-based (Ernie Turbo QT), 2 are flat DAGs (infinite zoom).
- All workflows are auto-formatted via **comfyui-workflow-prettier** (Sugiyama Layered DAG: topoSort → assignLayers → 6-iter barycenter minimizeCrossings → median assignCoordinates, Left-to-Right, `hGap 100 vGap 100`). Never hand-tweak `pos` — re-run prettier (or `python3 /tmp/prettify.py` port) after manual edits. Current bbox: `ernie_infinite_zoom_loop.json` 25 nodes 15 layers 5540×1330; `ernie_infinite_zoom.json` 67 nodes 63 layers 21690×840 (linear chain inherently wide, legacy).
- Validate via `docker exec <cid> comfy workflow validate --workflow /comfy/user/default/workflows/<name>.json` (or comfy-mcp `validate_workflow`). Must be `valid: true, 0 errors` before queue. Current: `ernie_infinite_zoom_loop.json` 25 nodes valid; `ernie_infinite_zoom.json` 67 nodes valid; `ernie_turbo_qt*.json` 3 top nodes + subgraph (9 or 14 inner nodes) valid.

#### 6.1 Ernie Turbo QT — Subgraph Family (3 files, `definitions.subgraphs[0]` id `03921aea-a70e-44b4-bc77-f6bda10f2120`)

Top-level DAG is identical in all 3 files (3 nodes, 1 link):

| id | type | role | widgets |
|----|------|------|---------|
| 88 | `03921aea…` (subgraph) | Text-to-Image engine | proxyWidgets expose 9 inner widgets (see below) |
| 73 | `SaveImage` | output | `["Ernie-Image-Turbo"]` prefix |
| 89 | `MarkdownNote` | docs | Model links + storage tree |

Link: `88:0 IMAGE --101--> 73:0`.

Subgraph definition `definitions.subgraphs[0]` carries `name: "Text to Image (Ernie Turbo - Simple)"`, 4 groups (Text to Image, Image Size, Prompt, Model, Prompt Enhancement), `inputNode -10` with 9 inputs, `outputNode -20` with 1 output.

**Subgraph inputs (order matters — slot indices preserved even when empty):**

| slot | name | type | linkIds | label | wired to |
|------|------|------|---------|-------|----------|
| 0 | `value` | STRING | `[128]` | `prompt` | `94:value` (PrimitiveStringMultiline) |
| 1 | `value_1` | BOOLEAN | `[]` in simple / `[127]` in enhanced | `prompt_enhancement` | `96:value` via 127 if present |
| 2 | `width` | INT | `[104]` | — | `71:width` (EmptyFlux2LatentImage) |
| 3 | `height` | INT | `[105]` | — | `71:height` |
| 4 | `seed` | INT | `[108]` | — | `70:seed` (KSampler) |
| 5 | `unet_name` | COMBO | `[109]` | — | `66:unet_name` (UNETLoader) |
| 6 | `clip_name` | COMBO | `[110]` | — | `62:clip_name` (CLIPLoader ministral) |
| 7 | `clip_name_1` | COMBO | `[]` in simple / `[132]` in enhanced | `prompt_enhancer` | `98:clip_name` if present |
| 8 | `vae_name` | COMBO | `[133]` | — | `63:vae_name` (VAELoader) |

Empty `linkIds: []` for slots 1 and 7 in the simple variant is intentional: keeps indices stable so `vae_name` stays slot 8. Do not renumber.

**Inner nodes — Simple (9 nodes, `lastLinkId 134`):**

| id | type | widgets / notes |
|----|------|-----------------|
| 71 | `EmptyFlux2LatentImage` | `[1024,1024,1]` — latent 1024², flux2 family |
| 66 | `UNETLoader` | `["ernie-image-turbo.safetensors","default"]` — 7.7G fp8 QT; symlink `ernieImageTurboQT_fp8V2.safetensors` ↔ `ernie-image-turbo.safetensors` |
| 62 | `CLIPLoader` | `["ministral-3-3b.safetensors","flux2","default"]` — 7.2G |
| 63 | `VAELoader` | `["flux2-vae.safetensors"]` — 321M |
| 67 | `CLIPTextEncode` | `[""]` — text from 94 via link 134, clip from 62 via 79 |
| 91 | `ConditioningZeroOut` | — negative = zeroed conditioning |
| 70 | `KSampler` | `[423299999918804,"randomize",8,1,"euler","simple",1]` — 8 steps, cfg 1.0, denoise 1.0, euler/simple |
| 65 | `VAEDecode` | — samples 73 + vae 74 → IMAGE to outputNode |
| 94 | `PrimitiveStringMultiline` | prompt string (see below) — title "String (Multiline - Prompt)" |

Inner links (15): `66:0 MODEL --85--> 70:0`, `62:0 CLIP --79--> 67:0`, `94:0 STRING --134--> 67:1`, `67:0 COND --76--> 70:1`, `67:0 COND --112--> 91:0 --113--> 70:2`, `71:0 LATENT --80--> 70:3`, `70:0 LATENT --73--> 65:0`, `63:0 VAE --74--> 65:1`, `65:0 IMAGE --84--> -20:0`, plus 5 ` -10:* --104/105/108/109/110/133--> *` input links and `-10:0 --128--> 94:0`.

Prompt in 94 (identical in all 3 files): `A stylized cinematic side-profile medium shot portrait of a young European woman with sleek dark hair in a tight low bun, wearing a crisp white ruffled-collar shirt, eyes closed in serene contemplation, standing against a moody, dark gradient deep indigo-blue twilight sky with layered misty mountain silhouettes in the background, extreme high-contrast split neon lighting: 95% of the scene bathed in deep, saturated cool cyan-blue ambient light (dim, moody, low-key), with a sharp, intense, vivid neon pink-orange rim light tracing her facial profile, neck, and collar, creating bold color blocking and a surreal, artistic aesthetic, minimalist composition, high-fashion editorial, 8K, ultra-sharp focus on subject, moody desaturated blue tones, dramatic contrast, atmospheric depth, tranquil introspective vibe, dark atmospheric background, no overexposure, stylized color grading, neon rim light glow, low-key cool fill light.`

File identity: `ernie_turbo_qt.json` and `ernie_turbo_qt_simple.json` are **byte-identical** (md5 `9bc6bfd9e3c3b9716aa56af4eefe6e33`, 1175 lines, subgraph 9 nodes). `ernie_turbo_qt.json` is kept as canonical adapter name matching official template; `ernie_turbo_qt_simple.json` is the minimal alias — edit one, copy to the other.

**Inner nodes — Enhanced (+5 nodes, 14 total, `lastLinkId 135`):**

Adds LLM prompt-enhancer chain (requires `ernie-image-prompt-enhancer.safetensors` 6.5G):

| id | type | widgets / notes |
|----|------|-----------------|
| 98 | `CLIPLoader` (title "Load CLIP (PE)") | `["ernie-image-prompt-enhancer.safetensors","flux2","default"]` — second LLM |
| 93 | `StringReplace` | `["<s>[SYSTEM_PROMPT]你是一个专业的文生图 Prompt 增强助手。你将收到用户的简短图片描述，请据此扩写为一段内容丰富、细节充分的视觉描述，以帮助文生图模型生成高质量的图片。仅输出增强后的描述，不要包含任何解释或前缀。[/SYSTEM_PROMPT][INST]{\"prompt\": \"{prompt}\"}[/INST]", "{prompt}", ""]` — wraps raw prompt |
| 95 | `TextGenerate` | `["",2048,"on",0.6,64,0.8,0.05,1.05,0,0,False,True]` — `clip.generate` with max 2048 tokens, temp 0.6 |
| 96 | `PrimitiveBoolean` | `[True]` — "Enable prompt enhancement?" |
| 97 | `ComfySwitchNode` | `[False]` — selects `94` raw vs `95` enhanced |

Wiring deltas vs simple: `94:0 --115--> 93:0` and `93:0 --117--> 95:4 (STRING)`, `98:0 CLIP --116--> 95:0`, `94:0 --118--> 97:0`, `95:0 --119--> 97:1`, `96:0 BOOL --120--> 97:2`, `97:0 STRING --131--> 67:1` (replaces direct `94--134-->67`), plus input links `-10:1 --127--> 96:0` and `-10:7 --132--> 98:0`. File is 1587 lines, md5 `2c5c11d6d3a373522be7397928fae967`. **OOM on 16GB** (ministral 7.2G + PE 6.5G + DiT 7.7G + VAE 0.32G > 15.57 GiB usable on 4060 Ti; `nodes_textgen.py:64 clip.generate` fails even when Switch false — node still executes). Keep as backup for >24GB VRAM only.

#### 6.2 Infinite Zoom — Flat DAGs (2 files, no subgraph)

**`ernie_infinite_zoom.json` (legacy unrolled, 67 nodes, 144 links, 63 levels, bbox 21690×840):**
- 3 loaders (`UNETLoader` ernie-image-turbo, `CLIPLoader` ministral-3-3b flux2, `VAELoader` flux2-vae) fanned to 12 stages.
- Prompt `PrimitiveStringMultiline` → `CLIPTextEncode` → `ConditioningZeroOut` fanout to all 12 samplers.
- `EmptyFlux2LatentImage 768×768×1` → `KSampler seed 1234 denoise 1.0` → `VAEDecode` (frame 0).
- Linear chain ×11: `ImageScaleBy bicubic 0.85` (768→652.8, border ~57.6px) → `ImagePadForOutpaint 58,58,58,58 feather 20` → `VAEEncodeForInpaint grow_mask_by 6` → `KSampler 8 steps cfg 1 euler simple denoise 0.65 seed 1235..1245` → `VAEDecode`. Final scale `0.85^11 ≈ 0.167`.
- `BatchImagesNode` with 12 explicit `images.image0..11` inputs tapped from each `VAEDecode` → `CreateVideo fps 8` → `SaveVideo prefix Ernie_Zoom format auto`. Produces 12f @8fps = 1.5s per queue, rebuilds all frames every queue — **legacy, do not extend**.

**`ernie_infinite_zoom_loop.json` (✅ incremental, 25 nodes, 29 links, 15 levels, bbox 5540×1330):**

| id | type | widgets / inputs | role |
|----|------|------------------|------|
| 10 | `UNETLoader` | `["ernie-image-turbo.safetensors","default"]` | MODEL → 33,53 |
| 11 | `CLIPLoader` | `["ministral-3-3b.safetensors","flux2","default"]` | CLIP → 21 |
| 12 | `VAELoader` | `["flux2-vae.safetensors"]` | VAE → 34,52,54 |
| 20 | `PrimitiveStringMultiline` | psychedelic prompt (ultra fractal cathedral, neon mandala, …) | STRING → 21 |
| 21 | `CLIPTextEncode` | `[""]` text 100 + clip 101 | COND → 22,33,53 |
| 22 | `ConditioningZeroOut` | — | COND → 33,53 (negative) |
| 30 | `PrimitiveInt` | `[512]` | width → 32 |
| 31 | `PrimitiveInt` | `[512]` | height → 32 |
| 32 | `EmptyFlux2LatentImage` | `[512,512,1]` + links 112/113 | LATENT → 33 |
| 33 | `KSampler` | `[42,"randomize",8,1.0,"euler","simple",1.0]` | init txt2img, denoise 1.0 |
| 34 | `VAEDecode` | vae 109 | IMAGE 116 → 40:starting_image |
| 40 | `Image-Loopback-Load` | `["/comfy/output/loopback_ernie_zoom",True,"1",-1,"skip","off",512,512]` | Sample last frame (history_indices "1") |
| 50 | `ImageScaleBy` | `["bicubic",0.85]` | 512→435 (scale 0.85) |
| 51 | `ImagePadForOutpaint` | `[38,38,39,39,20]` | pad back to 512, feather 20, outputs image+mask |
| 52 | `VAEEncodeForInpaint` | `[6]` grow_mask_by | LATENT for inpaint |
| 53 | `KSampler` | `[43,"randomize",8,1.0,"euler","simple",0.65]` | inpaint, denoise 0.65 |
| 54 | `VAEDecode` | — | IMAGE 123 → 55 |
| 55 | `Image-Loopback-Cache` | `["/comfy/output/loopback_ernie_zoom",True]` | Store Loopback (appends history) → 56,60 |
| 56 | `Image-Loopback-Load` | `[..., "1-1000", ...]` | Sample full history for video |
| 57 | `Image-Loopback-Configure` | `["/comfy/output/loopback_ernie_zoom",0]` | history_limit 0 = unlimited |
| 60 | `PreviewImage` | — | preview latest frame (124) |
| 61 | `PreviewImage` | — | preview accumulated stack (126) |
| 62 | `CreateVideo` | `[8]` fps | VIDEO from 127 |
| 63 | `SaveVideo` | `["Ernie_Zoom_Loop","auto","auto"]` | mp4 under `output/` |
| 70 | `Anything Everywhere` | `[]` | broadcast MODEL/CLIP/VAE/COND to reduce spaghetti |

Flow per Queue: `32→33→34→40(starting_image) —40:117→50→51→52→53→54→55→56(1-1000)→61+62→63`, with `55:124→60` for per-frame preview. `57` configures unlimited history. Pad math: `512*0.85=435.2`, border `(512-435)/2≈38.5` → `38,38,39,39`. Seeds `randomize` per queue. History under `/comfy/output/loopback_ernie_zoom/ernie_zoom_fixed/history/*.png` (fixed key patch, see §3), video `output/Ernie_Zoom_Loop_*.mp4`.

- Straight edges enforced globally via `comfy.settings.json` `Comfy.LinkRenderMode=Straight` and per-workflow `extra._linkRenderMode` doc hint (see §3).

### 7. Custom Nodes Audit (QoL / Safety)
- **comfy-loopback-buffer** (holo-q, 1.0.0 MIT, deps torch/pillow/numpy/aiohttp): replaces filesystem Save/Load reimplementation for incremental accumulation. Small, tested, maintained — **keep** (required for perpetual zoom). Not hugely starred but functionally unique, no known CVEs, no network.
- **ComfyUI-VideoHelperSuite** (Kosinkadink, 1.7.9, ~4k★): `VideoCombine` etc. Thousands of dependents, active, deps already baked (`opencv-python imageio-ffmpeg`). Native `CreateVideo/SaveVideo` now covers our use, so VHS is not strictly needed but harmless QoL for future video tasks — **keep**, but don't rely on it for core workflow.
- **cg-use-everywhere** (chrisgoringe, 7.8 Apache2, 1k★, frontend-only): `Anything Everywhere` broadcast reduces link spaghetti, no python deps. Lightweight, widely used — **keep** (replaces manual wiring reimplementation).
- **comfyui-workflow-prettier** (deepme987, Sugi­yama Layered/Compact/Linear/SortByType, 1103-line JS, zero python deps): auto-positions DAGs properly (barycenter crossing minimization, median coordinates). Successor to hand-rolled `/tmp/format_*.py` scripts — **keep** (required tool for all position edits, avoids horizontal explosion).
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
docker exec <cid> bash -c 'ls /comfy/output/loopback_ernie_zoom/ernie_zoom_fixed/history | wc -l'
curl -s http://localhost:9000/status | jq    # comfy-mcp bridge health
curl -s http://localhost:8188 | head         # ComfyUI health
# Format workflows after manual edits (Sugiyama Layered, straight edges):
python3 /tmp/prettify.py --all               # or workflow_prettier UI: select Layered → Save
```

## Known Gotchas
- Host `comfy` binary absent by design — use `docker exec <cid> comfy ...` instead.
- Changing `cache_path` to an absolute under `/comfy/output` is the ONLY way to confine loopback; relative paths are jailed under temp.
- Workflow `id` changes on save can drift hash key — fixed key patch prevents history loss.
- 512px is deliberate for speed/VRAM; don't bump to 1024 without testing OOM.
- `Comfy.LinkRenderMode` must be `Straight`; if edges look curved, re-check `comfy.settings.json`.
