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
      frame_interpolation/      # film_net_fp16.safetensors 66M (FILM, default) + rife_v4.25/4.26/4.26_heavy 22M each (HF Comfy-Org/frame_interpolation)
    input/  -> ../input host mount (see docker-compose)
      ernie_zoom_seed.png       # cold-start seed frame for loop workflow ImageReceiver (rendered once via throwaway txt2img run)
    output/                     # ALL generations confined here — `rm -rf Comfy/output/*` cleans everything
      Ernie_Zoom_Frames/frame_*.png  # current loop: one frame per Queue via SaveImage (the sequence; video loader reads this dir)
      Ernie_Zoom_*.mp4          # current loop video via VHS VideoCombine (only when video group unbypassed; save_output toggle)
      loopback_ernie_zoom/      # LEGACY (pre-Send/Receive) — no longer produced; remove if it reappears
      Ernie_Zoom_Loop_*.mp4     # LEGACY (pre-VHS) — no longer produced
      Ernie-Image-Turbo_*.png   # earlier single-frame tests
    temp/
      image_loopback/           # LEGACY location, migrated on boot to output/ — do NOT rely on it (now empty)
    custom_nodes/
      comfyui-manager           # 3.41, package manager
      comfy-loopback-buffer     # holo-q/comfy-loopback-buffer 1.0.0 MIT — Store/Load/Configure Loopback (LEGACY: no workflow uses it since Send/Receive redesign; keep installed)
      ComfyUI-VideoHelperSuite  # Kosinkadink 1.7.9 — VHS_LoadImagesPath + VHS_VideoCombine drive the loop video group (see §6.2)
      comfyui-impact-pack       # ltdrdata — ImageSender/ImageReceiver cross-Queue feedback (see §6.2); host git clone, NOT baked in Dockerfile; sam2 trimmed from requirements.txt
      cg-use-everywhere         # chrisgoringe 8.0 (50ae9f8) — Anything Everywhere broadcast; ⚠️ BROKEN with frontend 1.51.9 — do not add UE nodes to workflows (see §7)
      comfyui-workflow-prettier # deepme987/comfyui-workflow-prettier — auto-positioning (Sugiyama Layered DAG layout; canvas-menu JS only, no CLI — hand-position new nodes carefully, see §6)
    user/default/
      workflows/                # browser-visible workflows (auto-formatted, see below)
        ernie_turbo_qt.json               # adapted official Ernie Turbo template (PreviewAny fixed, enhancer removed)
        ernie_turbo_qt_simple.json        # minimal variant
        ernie_turbo_qt_enhanced.json      # backup with enhancer chain (OOM on 16GB — don't use without >24GB)
        ernie_infinite_zoom.json          # unrolled 12-frame fixed batch, N=12 768px, CreateVideo fps8 (legacy, per-queue rebuilds all frames)
        ernie_infinite_zoom_loop.json     # ✅ incremental IN-zoom: one new frame per Queue, crop 492@10,10 → rescale 512 bicubic (1.041×/frame), img2img denoise 0.60, ImageReceiver/ImageSender (Impact Pack) cross-Queue feedback, frames → output/Ernie_Zoom_Frames/, bypassed video group (VHS loader + 4× FILM + VideoCombine) + bypassed audio group (ACE-Step 1.5 music + MMAudio SFX, mixed → VideoCombine audio) — see §6.2
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
- Resolution kept at **512** (PrimitiveInt) from original 1024/768 to fit VRAM and stay fast. Zoom mechanics (since 2026-09-02 redesign): crop 492² centered (10 px shaved/side) → bicubic rescale back to 512² (= 1.041× dive per frame), full-frame img2img denoise 0.60, steps 8 cfg1 euler simple. Few sec/frame on 4060 Ti.
- Models symlinked: `ernieImageTurboQT_fp8V2.safetensors` ↔ `ernie-image-turbo.safetensors` for CLIPLoader compatibility.

### 6. Workflows & Validation
- Browser location is `Comfy/user/default/workflows/` (not `Comfy/workflows/`). All 5 custom workflows are listed below; 3 are subgraph-based (Ernie Turbo QT), 2 are flat DAGs (infinite zoom).
- All workflows are auto-formatted via **comfyui-workflow-prettier** (Sugiyama Layered DAG: topoSort → assignLayers → 6-iter barycenter minimizeCrossings → median assignCoordinates, Left-to-Right, `hGap 100 vGap 100`). Never hand-tweak `pos` — re-run prettier (canvas-menu JS only, no CLI; the `/tmp/prettify.py` host port is ephemeral and may not exist — hand-position new nodes carefully and verify zero overlaps) after manual edits. Current bbox: `ernie_infinite_zoom_loop.json` 38 nodes 5870×2280 (20 active + 18 bypassed: video group 89/98/90/99 + audio group 100-114); `ernie_infinite_zoom.json` 67 nodes 63 layers 21690×840 (linear chain inherently wide, legacy).
- Validate via `docker exec <cid> comfy workflow validate --workflow /comfy/user/default/workflows/<name>.json` (or comfy-mcp `validate_workflow`). Must be `valid: true, 0 errors` before queue. Current: `ernie_infinite_zoom_loop.json` 38 nodes valid (1 expected warning: 90 unreachable while video group bypassed); `ernie_infinite_zoom.json` 67 nodes valid; `ernie_turbo_qt*.json` 3 top nodes + subgraph (9 or 14 inner nodes) valid.

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

**`ernie_infinite_zoom_loop.json` (✅ incremental IN-zoom, 38 nodes, 42 links, 2 groups, bbox 5870×2280):**

| id | type | widgets / inputs | role |
|----|------|------------------|------|
| 10 | `UNETLoader` | `["ernie-image-turbo.safetensors","default"]` | MODEL → 53 |
| 11 | `CLIPLoader` | `["ministral-3-3b.safetensors","flux2","default"]` | CLIP → 21 |
| 12 | `VAELoader` | `["flux2-vae.safetensors"]` | VAE → 85 (137), 54 (111) |
| 20 | `PrimitiveStringMultiline` | psychedelic prompt (ultra fractal cathedral, neon mandala, …) | STRING → 21 |
| 21 | `CLIPTextEncode` | `[""]` text 100 + clip 101 | COND → 22,53 |
| 22 | `ConditioningZeroOut` | — | COND → 53 (negative) |
| 95 | `ImageReceiver` | `["ernie_zoom_seed.png",1,False,"",False]` link_id 1 | last frame from previous Queue (cold start: seed PNG from `/comfy/input`) → 83 |
| 80 | `PrimitiveInt` | `[492]` "Zoom crop size" | width+height → 83 (links 131/132) |
| 81 | `PrimitiveInt` | `[10]` "Crop offset X" | x → 83 (link 133) |
| 82 | `PrimitiveInt` | `[10]` "Crop offset Y" | y → 83 (link 134) |
| 83 | `ImageCrop` | `[492,492,10,10]`, all 4 crop widgets wired to 80/81/82 | center-crop 492² @ (10,10) — title "Zoom In: Crop" |
| 84 | `ImageScale` | `["bicubic",512,512,"disabled"]` | rescale crop back to 512² bicubic — title "Zoom In: Rescale" |
| 85 | `VAEEncode` | pixels←136, vae←137 | encode zoomed frame — title "Zoom In: Encode" |
| 53 | `KSampler` | `[<seed>,"randomize",8,1.0,"euler","simple",0.60]` | img2img restore detail, denoise 0.60 |
| 54 | `VAEDecode` | — | IMAGE → 60,96,97 (links 150/151/152) |
| 60 | `PreviewImage` | — | preview new frame (150) |
| 96 | `ImageSender` | `["Ernie_Zoom_Send",1]` link_id 1 | holds frame for next Queue's Receiver (img-send event) |
| 97 | `SaveImage` | `["Ernie_Zoom_Frames/frame"]` | persists frame → `output/Ernie_Zoom_Frames/frame_*.png` (the sequence) |
| 89 | `FrameInterpolationModelLoader` | `["film_net_fp16.safetensors"]` mode 4 (bypassed) | INTERP_MODEL → 90 — FILM 66M, alt rife_v4.26/heavy 22M |
| 90 | `FrameInterpolate` | `[4]` multiplier 4 | images ←98, interp ←89 — title "4x Interpolation (FILM)" |
| 98 | `VHS_LoadImagesPath` | `["/comfy/output/Ernie_Zoom_Frames",0,0,1]` mode 4 (bypassed) | loads all saved frames → 90 |
| 99 | `VHS_VideoCombine` | `[32,0,"Ernie_Zoom","video/h264-mp4",False,True]` mode 4 (bypassed) | 32fps mp4 + preview; save_output toggle; AUDIO ←114 (link 173, muxes `-audio.mp4` twin) |
| 87 | `MarkdownNote` | docs: flow, params (offset=(512−size)/2 centered), reset cmd | reference card in-editor |

**Audio group (nodes 100-114, frontend group id 2 `Audio: ACE music + MMAudio SFX, mixed`, bbox [2060,700,1280,900], ALL mode 4 bypassed — unbypass together with video group to render):**

| id | type | widgets / inputs | role |
|----|------|------------------|------|
| 103 | `PrimitiveFloat` | `[0.657,"fixed"]` "Duration" | SINGLE KNOB for music+SFX length (seconds) → 104 (155), 105 (156), 111 (157) |
| 100 | `UNETLoader` | `["acestep_v1.5_turbo.safetensors","default"]` | ACE-Step 1.5 2B turbo DiT 4.79G → 106 (159) |
| 106 | `ModelSamplingAuraFlow` | `[3]` shift 3.0 | → 107 (160) |
| 101 | `DualCLIPLoader` | `["qwen_0.6b_ace15.safetensors","qwen_1.7b_ace15.safetensors","ace","default"]` | ACE text encoders (3.71G + small) → 105 (158) |
| 105 | `TextEncodeAceStepAudio1.5` | ambient-instrumental tags, lyrics `""` (instrumental), seed 31 fixed, bpm 90, timesig 4, lang en; duration widget stale 5.0, live value ←103 via 156 | COND → 107 (161) |
| 104 | `EmptyAceStep1.5LatentAudio` | seconds widget stale 5.0, live ←103 via 155 | LATENT → 107 (163) |
| 107 | `KSampler` | `[31,"fixed",8,1,"euler","simple",1]` + negative ←22 (162, zeroed COND reused) | music latent → 108 (164) |
| 102 | `VAELoader` | `["ace_1.5_vae.safetensors"]` 337M | → 108 (165) |
| 108 | `VAEDecodeAudio` | — | AUDIO music → 112 (169) |
| 109 | `MMAudioModelLoader` | `["mmaudio_large_44k_v2_fp16.safetensors","fp16"]` 2.06G | MODEL → 111 (166) |
| 110 | `MMAudioFeatureUtilsLoader` | `[vae_44k_fp16 611M, synchformer_fp16 475M, DFN5B-CLIP_fp16 1.97G, "44k","fp16"]` (bigvgan auto-downloads at runtime — no VoCoderLoader needed) | FEATUREUTILS → 111 (167) |
| 111 | `MMAudioSampler` | `[5.0(stale,←103 via 157),25,4.5,7,rumble/whoosh prompt,vocal-negative,True,True]` steps 25 cfg 4.5 seed 7 fixed, mask_away_clip True, force_offload True | images ←90 interp frames (168) → synced SFX AUDIO → 113 (170) |
| 112 | `AudioAdjustVolume` | `[0]` dB (INT, 0 = unchanged) | music level → 114 (171) |
| 113 | `AudioAdjustVolume` | `[-6]` dB | SFX level → 114 (172) |
| 114 | `AudioMerge` | `["add"]` | mix → 99.audio (173) |

**2026-09-08: ADDED full-mix audio (music + SFX) to the bypassed video group** (user request: local/free/SOTA/instrumental, wired into node 99). Music bed = ACE-Step 1.5 2B turbo (native nodes, instrumental via empty lyrics, quality-max that fits 16GB alongside MMAudio — XL 4B rejected); SFX = MMAudio large fp16 in-workflow (interp frames → video-synced rumble/whoosh, mask_away_clip True so CLIP text doesn't fight the video conditioning). Mix = music 0dB + SFX −6dB via `AudioMerge add` → `VHS_VideoCombine.audio` (slot 11 — node 99 has 11 inputs 0-10, so audio is 11 not 12; installed VHS exposes the optional slot, disk JSON needed the input entry appended). Duration = single knob 103 (linked everywhere; stale 5.0 widget fallbacks on 104/105/111 are inert while linked — update 103 to match video length `(n-1)*4+1/32`s; each video-render Queue also appends 1 frame, so exact match drifts by design). Verified end-to-end from scratch: full reset → 6 bypassed Queues (warm cache ≈5s each) → 1 unbypassed render (`Ernie_Zoom_00004.mp4` 21f + `-audio.mp4` with AAC, 0.657s) → 103 set 5.0→0.657 → final render 7 frames → `Ernie_Zoom_00005.mp4` 25f @32fps + `-audio.mp4` AAC, 0.782s, ≈10-20s render on 4060 Ti sequential (Ernie models stay loaded; audio only runs on video render). Stable Audio Open 1.0 kept as backup SFX source; TangoFlux rejected (no ComfyUI nodes); API audio (Seed/ElevenLabs) excluded — not signed in.

**2026-09-02: node 70 (`Anything Everywhere`) REMOVED.** It was completely disconnected (input `link:null`, zero links — broadcast nothing) but its mere presence made every browser Queue fail with `res is undefined`: UE 8.0's `find_duplicate_broadcasted_types` crashes on `node.inputs === undefined` (frontend 1.51.9), the exception is swallowed by `call_function_with_modified_graph` which returns `undefined`, and VHS's outer `graphToPrompt` wrapper then dereferences `res.workflow` → error dialog. All MODEL/CLIP/VAE/COND wiring was already explicit links, so removal required no rewiring; verified end-to-end (queue OK, history 6→7, new mp4). Do not re-add UE nodes (see §7).

**2026-09-02: REDESIGNED from out-zoom (scale 0.85 + pad + inpaint) to true IN-zoom** (user request). Old nodes 50 `ImageScaleBy`/51 `ImagePadForOutpaint`/52 `VAEEncodeForInpaint` deleted; new 80-85 (crop→rescale→encode) + 87 docs; node 53 now full-frame img2img denoise 0.60; node 56 range flipped `"1-1000"`→`"1000-1"` (ascending range = newest→oldest batch order — WRONG for playback; descending = chronological). History was RESET (old 18 out-zoom frames purged, `rm -rf` must run INSIDE container — host lacks perms on root-owned files). Verified end-to-end: 4 sequential headless Queues → history 2→5, 4 mp4s; direction confirmed numerically (each frame closer to 1.04× magnified predecessor than to predecessor itself, all 4 pairs ~0.126 vs ~0.18 mean abs diff @128²). Later superseded by Send/Receive redesign below (loopback nodes 40/55/56/57 deleted).

**2026-09-02: REDESIGNED from loopback history to Send/Receive + saved frames** (user request: simplify). Deleted seed branch (30-34), all loopback nodes (40/55/56/57), old video tail (62/88/91-94) and the `ComfySwitchNode` toggle. Cross-Queue feedback is now Impact Pack `ImageSender` 96 / `ImageReceiver` 95 (link_id 1): `img-send` frontend event rewrites Receiver's image widget to the Sender's temp file (`Ernie_Zoom_Send_temp_*.png [temp]`), so each Queue's crop chain starts from the previous Queue's frame — memory-resident, lost on page reload → falls back to `/comfy/input/ernie_zoom_seed.png` (rendered once via throwaway txt2img run, 512²). Frames persist via `SaveImage` 97 to `output/Ernie_Zoom_Frames/frame_*.png` (subfolder keeps Turbo PNGs / AnimateDiff leftovers out of the sequence). Video group (89/98/90/99, frontend group id 1) is **bypassed by default**: unbypass to render — `VHS_LoadImagesPath` 98 loads all frames → `FrameInterpolate` 90 4× FILM (`(n-1)*4+1` frames, `<2` frames pass through) → `VHS_VideoCombine` 99 @32fps with preview + optional save (`Ernie_Zoom_*.mp4`, save_output toggle). Zoom math unchanged: `512/492 = 1.0407` per frame (≈4% dive); offset rule `(512−size)/2 = 10` keeps it centered. Verified end-to-end: cold Q1 12.1s → Receiver rewritten to temp PNG; Q2 6.0s; zoom continuity mean|f2−zoom(f1)|=0.085 vs mean|f2−f1|=0.133; 4 dir frames → `Ernie_Zoom_00001.mp4` 13 frames @32fps = (4−1)*4+1, no pingpong. History under loopback is gone — `rm -rf` of `Ernie_Zoom_Frames/` + videos (INSIDE container) resets.

Flow per Queue: `95(ImageReceiver: last frame or seed) —148→83(crop 492²@10,10, params from 80/81/82)→84(rescale 512 bicubic)→85(encode)→53(img2img 0.60)→54→60 PreviewImage (150) + 96 ImageSender (151, feeds next Queue) + 97 SaveImage (152, persists sequence)`. Zoom math: `512/492 = 1.0407` per frame (≈4% dive; user's "10px each side"); offset rule `(512−size)/2 = 10` keeps it centered. Bypassed video group (unbypass to render): `98 VHS_LoadImagesPath(dir) —153→90 FrameInterpolate 4× FILM (interp_model from 89) —154→99 VHS_VideoCombine @32fps` with preview + optional save toggle; `(n-1)*4+1` frames (4 dir frames → 13 @32fps verified, no pingpong). **GOTCHA — VHS_VideoCombine `widgets_values` must be the 6 declared widgets only** `[32,0,"Ernie_Zoom","video/h264-mp4",False,True]` (format sub-widgets default: crf19/yuv420p/meta-true/trim-false); a 10-element array with sub-widget values misaligns positionally → `pingpong` got truthy `"yuv420p"` → pingponged videos (2 frames→8, 3→16). VHS counter quirk: its metadata PNG matches the `Ernie_Zoom_(\d+)` counter regex → numbering can skip (harmless).

- Straight edges enforced globally via `comfy.settings.json` `Comfy.LinkRenderMode=Straight` and per-workflow `extra._linkRenderMode` doc hint (see §3).

### 7. Custom Nodes Audit (QoL / Safety)
- **comfy-loopback-buffer** (holo-q, 1.0.0 MIT, deps torch/pillow/numpy/aiohttp): replaces filesystem Save/Load reimplementation for incremental accumulation. Small, tested, maintained — **keep installed** but LEGACY: no workflow uses it since the Send/Receive redesign (§6.2).
- **ComfyUI-VideoHelperSuite** (Kosinkadink, 1.7.9, ~4k★): `VideoCombine` etc. Thousands of dependents, active, deps already baked (`opencv-python imageio-ffmpeg`). `VHS_LoadImagesPath` + `VHS_VideoCombine` drive the loop workflow's bypassed video group — **keep**, now load-bearing for video rendering.
- **comfyui-impact-pack** (ltdrdata, ~1k★, active): `ImageSender`/`ImageReceiver` cross-Queue feedback drives the loop workflow (link_id pair; `img-send` frontend event rewrites Receiver widget to Sender's temp file — memory-resident, falls back to seed PNG on reload). Host git clone (NOT baked in Dockerfile — code persists via `./Comfy:/comfy` mount, pip deps via entry.sh runtime loop); `requirements.txt` trimmed of `git+.../sam2` (SAM nodes unneeded → faster boot; `[Impact Pack] SAM2 functionality unavailable` warning + `impact-sam-editor.js` 404 are expected/harmless). **keep** (required for perpetual zoom).
- **comfyui-mmaudio** (kijai, 574★ MIT, pushed 2026-02): `MMAudioModelLoader`/`MMAudioFeatureUtilsLoader`/`MMAudioSampler` drive the loop workflow's bypassed audio group (video-synced SFX from interp frames). Host git clone (same pattern as impact-pack — NOT baked in Dockerfile); light deps (librosa/torchdiffeq/einops/timm/omegaconf/open_clip/accelerate/ftfy). Models (~4.6G: large fp16 2.06G + synchformer 475M + vae 611M + CLIP 1.97G) under `models/mmaudio/`. **keep** (required for SFX).
- **cg-use-everywhere** (chrisgoringe, checkout 50ae9f8 = upstream main HEAD 2026-07-20, "8.0", 1k★, frontend-only): `Anything Everywhere` broadcast reduces link spaghetti, no python deps. **⚠️ INCOMPATIBLE with frontend 1.51.9** — any workflow containing a UE node fails at browser Queue: UE 8.0 assumes `node.inputs`/`node.outputs` always defined (`find_duplicate_broadcasted_types` use_everywhere_utilities.js:365 `reading 'length'`; also `fix_unconnected_inputs` `reading 'filter'` in afterConfigureGraph on load), the throw is swallowed by `call_function_with_modified_graph` → returns `undefined` → VHS's outer wrapper dereferences `res.workflow` → `res is undefined` error dialog. No newer upstream fix exists as of 2026-09-02. **Keep installed** (workflows without UE nodes queue fine — wrappers pass through cleanly) but **do not add UE nodes to any workflow** until upstream/compatible-frontend resolves; wire explicitly instead.
- **comfyui-workflow-prettier** (deepme987, Sugi­yama Layered/Compact/Linear/SortByType, 1103-line JS, zero python deps): auto-positions DAGs properly (barycenter crossing minimization, median coordinates). Successor to hand-rolled `/tmp/format_*.py` scripts — **keep** (required tool for all position edits, avoids horizontal explosion).
- **Native FrameInterpolate** (`comfy_extras.nodes_frame_interpolation`, no pack): `FrameInterpolationModelLoader` + `FrameInterpolate` (multiplier 2-16, align-aware, OOM-batch halving, feature caching across pairs). Models under `models/frame_interpolation/` — `film_net_fp16.safetensors` 66M (FILM, default, best for large-motion/zoom) + `rife_v4.25/4.26/4.26_heavy` 22M (RIFE, faster). FILMNet has no pad_align, IFNet pad_align 64; both load via same loader. Greedy `prev_frame` + `feat_cache` reuse across pairs. No extra pip deps — **keep**; do not install Fannovel16/ComfyUI-Frame-Interpolation pack (duplicate).
- No removal recommended; all are maintained, trustworthy, popular. If a new dep is added, check stars, commit recency, license, and whether it replaces a reimplementation.

### 8. Confinement & Cleanup Discipline
- Everything generated goes under `Comfy/output/` — verify after each workflow edit: `ls -R Comfy/output` should contain only intended files. Saved frames (`Ernie_Zoom_Frames/`) count as generation.
- `Comfy/temp/` should stay empty of persistent history; entry.sh migration ensures legacy temp history moves to output.
- To reset the loop sequence: `rm -rf Comfy/output/Ernie_Zoom_Frames` (+ videos `rm Comfy/output/Ernie_Zoom*.mp4`) — **must run INSIDE the container** (host lacks perms on root-owned files). Next Queue cold-starts from `input/ernie_zoom_seed.png`.
- `Comfy/output/loopback_ernie_zoom/` is LEGACY — remove if it reappears.

### 9. Update Discipline & Task Alignment
- **After each significant amount of work, update this AGENTS.md** if new relevant info/facts became known (new model, new workflow, new custom node, new volume, new port, new patch, perf numbers, VRAM lessons). Keep this file as the single source of truth for the wrapper; don't let facts live only in chat.
- **When defining/planning a task, go through rounds of Q&A to further specify the solution and ensure proper alignment** before building: ask clarifying questions (node choices, naming, durability, toggle mechanics, scope), present a researched plan, and only proceed once the user confirms. Never jump from a vague request straight to implementation.

## Commands Cheat Sheet

```bash
./serve.sh                                   # build + run (ephemeral) — the ONLY way to start
docker ps --format "{{.Names}} {{.Status}}"   # comfy-comfy-run-... Up (healthy)
docker logs <cid> --tail 100                 # ComfyUI logs
docker exec <cid> comfy workflow validate --workflow /comfy/user/default/workflows/ernie_infinite_zoom_loop.json
docker exec <cid> bash -c 'ls /comfy/output/Ernie_Zoom_Frames | wc -l'
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
