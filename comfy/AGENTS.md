# comfy/ AGENTS.md — Project Wrapper (host + container)

> This file lives at **`comfy/AGENTS.md`** (i.e. `/home/goulade/Projects/ai/comfy/AGENTS.md`), **not** `Comfy/AGENTS.md` which is upstream ComfyUI's own engineering guide. Keep both. After each significant amount of work, update this file if new non-obvious facts became known — that's a hard requirement for this wrapper.

## Quick Map

```
comfy/                          # wrapper repo (this file's scope)
  serve.sh                      # ONLY entrypoint for comfy — `docker compose run --build --rm --detach --service-ports comfy`
  zoomy.sh                      # ONLY entrypoint for zoomy — spawns comfy via serve.sh when missing, then `run --build --rm --detach --service-ports --no-deps zoomy`
  docker-compose.yml            # TWO services: `comfy` (ports 8188/7860/9000, volumes ./Comfy:/comfy ./input:/input comfy:/root/.cache, GPU reservations) + `zoomy` (build ./Zoomy, port 7861:7861, volume ./Comfy/output:/comfy/output, depends_on comfy, ZOOMY_* env)
  Dockerfile                    # FROM pytorch/pytorch:2.9.0-cuda12.8-cudnn9-devel, bakes comfy-cli/mcp + custom nodes + requirements.txt
  entry.sh                      # dual-process: ComfyUI :8188 + comfy-mcp bridged :9000 (Streamable HTTP /mcp), handles set-default, req installs, patches
  AGENTS.md                     # THIS FILE — wrapper facts + update discipline
  Zoomy/                        # zoomy app (see §10, full design in Zoomy/DESIGN.md)
    pyproject.toml              # ruff ALL (line-length 100) + mypy strict + pytest config
    requirements.txt            # runtime: gradio==6.26.0, httpx==0.28.1 (dev: ruff/mypy/pytest)
    Dockerfile                  # FROM python:3.12-slim, /application, CMD ["python","-m","zoomy"]
    DESIGN.md                   # app design doc — update with every behavior change
    zoomy/                      # package: settings/errors/family_catalog/graph/frame_repository/comfy_connection/frame_workflow/finalize_workflow/rendering/interface/main
    tests/                      # one test module per source module (49 tests, all green)
  Comfy/                        # upstream checkout (mounted as /comfy inside container), NOT rebuilt from scratch
    main.py                     # `python /comfy/main.py --listen 0.0.0.0 --port 8188` (invoked by entry.sh)
    models/
      diffusion_models/         # ernieImageTurboQT_fp8V2.safetensors (7.7G, Civitai 3028150 fp8 inference requested, symlink ernie-image-turbo.safetensors) + Juggernaut-Z pair (ZImageBase, Civitai 2600510, safetensors-full 12.3G each, sha-verified): juggernautZ_v10FastBy (v3011968 Fast by RunDiffusion) + juggernautZ_v10ByRundiffusion (v2921151) + put_*here placeholders
      text_encoders/            # ministral-3-3b.safetensors 7.2G, ernie-image-prompt-enhancer.safetensors 6.5G (currently NOT wired — see GPU note) + qwen_3_4b_fp8_mixed.safetensors 5.6G (Z-Image TE, HF Comfy-Org/z_image; full 8.04G rejected for 16GB VRAM)
      vae/                      # flux2-vae.safetensors 321M (HF Comfy-Org/ERNIE-Image) + ae.safetensors 335M (Z-Image VAE, HF Comfy-Org/z_image)
      loras/                    # c64style_ernie.safetensors 188M (Ernie C64, Civitai 304097 v2882202, trigger `c64style`, diffusion-only weights — no text-encoder keys, so strength_clip is a no-op) + Chalkboard01-1_CE_ZIMG_AIT4k.safetensors 170M (Civitai 648912 v2660695, no trainedWords) + ClayArt01a_CE_ZIMG_AIT3k.safetensors 170M (Civitai 1492159 v3021033) — both ZImageBase, diffusion-only (strength_clip no-op)
      frame_interpolation/      # film_net_fp16.safetensors 66M (FILM, default) + rife_v4.25/4.26/4.26_heavy 22M each (HF Comfy-Org/frame_interpolation)
    input/  -> ../input host mount (see docker-compose)
      ernie_zoom_seed.png       # cold-start seed frame for loop workflow ImageReceiver (rendered once via throwaway txt2img run, 1376×768 HD C64-style)
      z_zoom_seed.png           # cold-start seed for Z loop (rendered via Juggernaut quality model 22 steps, 1376×768)
    output/                     # ALL generations confined here — `rm -rf Comfy/output/*` cleans everything
      Ernie_Zoom_Frames/frame_*.png  # current loop: one frame per Queue via SaveImage (the sequence; video loader reads this dir)
      Ernie_Zoom_*.mp4          # current loop video via VHS VideoCombine (only when video group unbypassed; save_output toggle)
      Z_Zoom_Frames/frame_*.png # Z loop sequence (same mechanics, Juggernaut-Z frames)
      Z_Zoom_*.mp4              # Z loop video + `-audio.mp4` twin (only when 99 unbypassed)
      Zoomy/<sequence>/frame_*.png  # zoomy app sequences (`ernie_turbo`, `z_image`) — one frame per Render click
      Zoomy_<sequence>_*.mp4    # zoomy finalized videos: silent main + `-audio.mp4` twin WITH the soundtrack (the twin is the real artifact) + preview PNG
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
      batching_utils            # local 20-line pack — BatchPadToMin (pad image batch to ≥N frames for MMAudio; passthrough when long); host-written, torch only, no deps
      cg-use-everywhere         # chrisgoringe 8.0 (50ae9f8) — Anything Everywhere broadcast; ⚠️ BROKEN with frontend 1.51.9 — do not add UE nodes to workflows (see §7)
      comfyui-workflow-prettier # deepme987/comfyui-workflow-prettier — auto-positioning (Sugiyama Layered DAG layout; canvas-menu JS only, no CLI — hand-position new nodes carefully, see §6)
    user/default/
      workflows/                # browser-visible workflows (auto-formatted, see below)
        ernie_turbo_qt.json               # adapted official Ernie Turbo template (PreviewAny fixed, enhancer removed)
        ernie_turbo_qt_simple.json        # minimal variant
        ernie_turbo_qt_enhanced.json      # backup with enhancer chain (OOM on 16GB — don't use without >24GB)
        ernie_infinite_zoom.json          # unrolled 12-frame fixed batch, N=12 768px, CreateVideo fps8 (legacy, per-queue rebuilds all frames)
        ernie_infinite_zoom_loop.json     # ✅ incremental IN-zoom: one new frame per Queue, C64-style LoRA (120, strength 1.0) on Ernie DiT, HD 1376×768 (official preset), crop 1356×748@10,10 → rescale bicubic (1.0147×/frame), img2img denoise 0.60, ImageReceiver/ImageSender (Impact Pack) cross-Queue feedback, frames → output/Ernie_Zoom_Frames/, video group (VHS loader + 4× FILM + VideoCombine, only 99 bypassed) + audio group (ACE-Step 1.5 music + MMAudio SFX via pad-to-17 node 123, mixed → VideoCombine audio; duration auto via 115/116 `max(a/b,0.7)`; shared primitives 117 fps / 118 link_id / 119+121 frame-size / 80+122 crop-size) — see §6.2
        z_infinite_zoom_loop.json         # ✅ Z-Image twin of the ernie loop (same Send/Receive + frames-dir + bypassed-VHS-video + duplicated-audio mechanics): Juggernaut-Z Fast (10) / Quality (200) DiTs toggled by ONE boolean (208 QualityMode) via lazy MODEL+LATENT switches (201/214, only selected branch executes), LoRA 3-way int selector (207: 0 off / 1 chalkboard 203 / 2 clay-art 204) via cascaded lazy switches (209-212) after a shared shift-3 node (202); fast KSampler 53 (DDIM 6 steps cfg 1.0) vs quality 213 (res_multistep/beta 22 steps cfg 4); SEMANTIC prompts required (Z-Image); link_id is 2 (not 1 — avoids crosstalk with the ernie tab); frames → output/Z_Zoom_Frames/, video `Z_Zoom_*.mp4` — see §6.3
      comfy.settings.json       # frontend settings — *must* contain `"Comfy.LinkRenderMode":"Straight"` for square right-angle edges
```

## Non-Obvious Facts (read before touching anything)

### 1. Host vs Container Boundary
- **opencode runs on host only, comfy MUST stay in container** via `serve.sh` (`docker compose run --build --rm --detach ...`). Never `comfy launch` on host; host has no comfy binaries by design (`pip uninstall comfy-cli/comfy-mcp`, `~/.config/comfy-cli` removed, opencode `mcp: { comfy-mcp: { type: remote, url: http://localhost:9000/mcp, oauth:false }}`).
- `serve.sh` uses **`run` not `up`** intentionally — keep it that way. `run --service-ports` publishes 8188/7860/9000 under the fixed container name `comfy` (`--name comfy`, stable and resolvable on the compose network). `docker ps` shows `comfy Up (healthy)`. A second `serve.sh` while one runs fails fast on the name — same outcome as the old port-bind failure, clearer message.
- zoomy lifecycle is driven by `zoomy.sh`: it spawns comfy via `serve.sh` when no container named `comfy` is running, then starts zoomy itself (`run --build --rm --detach --service-ports --no-deps`, ephemeral `comfy-zoomy-run-<hash>`, UI at `http://localhost:7861`). The dependency is also declared as `depends_on` in compose, but `--no-deps` deliberately skips compose-managed startup — otherwise compose would auto-start a second, port-conflicting `comfy` instance next to serve.sh's. The comfy container carries a fixed `--name comfy` (see `serve.sh`), so zoomy reaches ComfyUI bridge-locally at `http://comfy:8188` — no host-network access, no `extra_hosts`. Never restart comfy or edit `serve.sh` beyond the name for zoomy networking.
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
- Resolution is **1376×768** (official Baidu ERNIE-Image-Turbo landscape preset; presets: 1024², 848×1264, 1264×848, 768×1376, 896×1200, 1376×768, 1200×896; turbo = 8 steps cfg 1.0). Zoom mechanics: crop 1356×748 centered (10 px shaved/side) → bicubic rescale back to 1376×768 (= 1.0147× dive per frame, ~1.5%), full-frame img2img denoise 0.60, steps 8 cfg1 euler/simple. ~20s/frame on 4060 Ti (was few sec at 512²).
- Models symlinked: `ernieImageTurboQT_fp8V2.safetensors` ↔ `ernie-image-turbo.safetensors` for CLIPLoader compatibility.

### 6. Workflows & Validation
- Browser location is `Comfy/user/default/workflows/` (not `Comfy/workflows/`). All 5 custom workflows are listed below; 3 are subgraph-based (Ernie Turbo QT), 2 are flat DAGs (infinite zoom).
- All workflows are auto-formatted via **comfyui-workflow-prettier** (Sugiyama Layered DAG: topoSort → assignLayers → 6-iter barycenter minimizeCrossings → median assignCoordinates, Left-to-Right, `hGap 100 vGap 100`). Never hand-tweak `pos` — re-run prettier (canvas-menu JS only, no CLI; the `/tmp/prettify.py` host port is ephemeral and may not exist — hand-position new nodes carefully and verify zero overlaps) after manual edits. Current bbox: `ernie_infinite_zoom_loop.json` 44 nodes 5870×2444 (43 active + 1 bypassed: only 99; group wrappers removed, bypass managed per-node); `z_infinite_zoom_loop.json` 60 nodes 4290×5362 (59 active + 1 bypassed: only 99); `ernie_infinite_zoom.json` 67 nodes 63 layers 21690×840 (linear chain inherently wide, legacy).
- Validate via `docker exec <cid> comfy workflow validate --workflow /comfy/user/default/workflows/<name>.json` (or comfy-mcp `validate_workflow`). Must be `valid: true, 0 errors` before queue. Current: `ernie_infinite_zoom_loop.json` 44 nodes valid (20 expected warnings: tail nodes unreachable while 99 bypassed — server prunes them so normal Queues stay cheap); `z_infinite_zoom_loop.json` 60 nodes valid (21 expected warnings, same class; 58 converted); `ernie_infinite_zoom.json` 67 nodes valid; `ernie_turbo_qt*.json` 3 top nodes + subgraph (9 or 14 inner nodes) valid.

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

**`ernie_infinite_zoom_loop.json` (✅ incremental IN-zoom, 44 nodes, 51 links, 0 groups, bbox 5870×2444):**

| id | type | widgets / inputs | role |
|----|------|------------------|------|
| 10 | `UNETLoader` | `["ernie-image-turbo.safetensors","default"]` | MODEL → 120 |
| 11 | `CLIPLoader` | `["ministral-3-3b.safetensors","flux2","default"]` | CLIP → 120 |
| 12 | `VAELoader` | `["flux2-vae.safetensors"]` | VAE → 85 (137), 54 (111) |
| 20 | `PrimitiveStringMultiline` | C64 psychedelic prompt (c64style trigger + fractal cathedral, eyeball mushrooms, jellyfish band, chrome skeletons, …) | STRING → 21 |
| 21 | `CLIPTextEncode` | `[""]` text 100 + clip ←120 (188) | COND → 22,53 |
| 22 | `ConditioningZeroOut` | — | COND → 53 (negative) |
| 120 | `LoraLoader` | `["c64style_ernie.safetensors",1.0,1.0]` model ←10 (185), clip ←11 (186) | MODEL → 53 (187), CLIP → 21 (188) — C64 pixel-art style DOMINATES Ernie DiT output |
| 95 | `ImageReceiver` | `["ernie_zoom_seed.png",1,False,"",False]` link_id ←118 | last frame from previous Queue (cold start: seed PNG from `/comfy/input`) → 83 |
| 80 | `PrimitiveInt` | `[1356]` "Crop Width" | width → 83 (link 131) |
| 122 | `PrimitiveInt` | `[748]` "Crop Height" | height → 83 (link 132) |
| 81 | `PrimitiveInt` | `[10]` "Crop offset X" | x → 83 (link 133) |
| 82 | `PrimitiveInt` | `[10]` "Crop offset Y" | y → 83 (link 134) |
| 117 | `PrimitiveFloat` | `[32]` "Frame Rate (fps)" | → 99.frame_rate (179) + 116.b (180) — single source for fps |
| 118 | `PrimitiveInt` | `[1]` "Send/Receive Link ID" | → 95.link_id (181) + 96.link_id (182) — pair must match |
| 119 | `PrimitiveInt` | `[1376]` "Frame Width" | → 84.width (183) |
| 121 | `PrimitiveInt` | `[768]` "Frame Height" | → 84.height (184) — split per-axis with 119 (was single 512 "Frame Size") |
| 83 | `ImageCrop` | `[1356,748,10,10]`, all 4 crop widgets wired to 80/122/81/82 | center-crop 1356×748 @ (10,10) — title "Zoom In: Crop" |
| 84 | `ImageScale` | `["bicubic",1376,768,"disabled"]` width ←119, height ←121 | rescale crop back to 1376×768 bicubic — title "Zoom In: Rescale" |
| 85 | `VAEEncode` | pixels←136, vae←137 | encode zoomed frame — title "Zoom In: Encode" |
| 53 | `KSampler` | `[<seed>,"randomize",8,1.0,"euler","simple",0.60]` | img2img restore detail, denoise 0.60 |
| 54 | `VAEDecode` | — | IMAGE → 96,97 (links 151/152) |
| 96 | `ImageSender` | `["Ernie_Zoom_Send",1]` link_id ←118 | holds frame for next Queue's Receiver (img-send event; Sender's own temp preview replaces deleted PreviewImage 60) |
| 97 | `SaveImage` | `["Ernie_Zoom_Frames/frame"]` | persists frame → `output/Ernie_Zoom_Frames/frame_*.png` (the sequence) |
| 89 | `FrameInterpolationModelLoader` | `["film_net_fp16.safetensors"]` | INTERP_MODEL → 90 — FILM 66M, alt rife_v4.26/heavy 22M |
| 90 | `FrameInterpolate` | `[4]` multiplier 4 | images ←98, interp ←89 — title "4x Interpolation (FILM)"; → 99 (154, video path) + → 123 (189, SFX conditioning path) |
| 98 | `VHS_LoadImagesPath` | `["/comfy/output/Ernie_Zoom_Frames",0,0,1]` | loads all saved frames → 90 |
| 99 | `VHS_VideoCombine` | dict widgets, frame_rate ←117 (shared 32), mode 4 (bypassed — ONLY bypassed node; server prunes whole tail on normal Queues) | 32fps mp4 + preview; save_output toggle; AUDIO ←114 (link 173, muxes `-audio.mp4` twin) |
| 87 | `MarkdownNote` | full future-agent doc: flow, shared-primitives table, HD math, video/audio groups, LoRA, ops/reset | reference card in-editor |

**Audio group (nodes 100-116 + 123; 103/112 deleted; group wrappers removed, only 99 bypassed — unbypass 99 to render):**

| id | type | widgets / inputs | role |
|----|------|------------------|------|
| 115 | `VHS_GetImageCount` | — title "Frame Count" | IMAGE ←90 (174) → INT count → 116 (175) |
| 116 | `ComfyMathExpression` | `["max(a / b, 0.7)"]` title "Audio Seconds (frames/fps)" | FLOAT = max(interp frames / shared fps, 0.7s floor) → 104 (176), 105 (177), 111 (178) — floor keeps MMAudio ≥16 sync frames on tiny renders; VHS `-shortest` trims any excess |
| 100 | `UNETLoader` | `["acestep_v1.5_turbo.safetensors","default"]` | ACE-Step 1.5 2B turbo DiT 4.79G → 106 (159) |
| 106 | `ModelSamplingAuraFlow` | `[3]` shift 3.0 | → 107 (160) |
| 101 | `DualCLIPLoader` | `["qwen_0.6b_ace15.safetensors","qwen_1.7b_ace15.safetensors","ace","default"]` | ACE text encoders (3.71G + small) → 105 (158) |
| 105 | `TextEncodeAceStepAudio1.5` | modern experimental electro-acoustic tags (granular dulcimer, feedback cello, spectral voices, acid bass pulses, circuit-bent toys, prepared piano, sub-bass swells), lyrics `""` (instrumental), seed 31 fixed, bpm 100, timesig 4, lang en; duration widget stale 5.0, live value ←116 via 177 | COND → 107 (161) |
| 104 | `EmptyAceStep1.5LatentAudio` | seconds widget stale 5.0, live ←116 via 176 | LATENT → 107 (163) |
| 107 | `KSampler` | `[31,"fixed",8,1,"euler","simple",1]` + negative ←22 (162, zeroed COND reused) | music latent → 108 (164) |
| 102 | `VAELoader` | `["ace_1.5_vae.safetensors"]` 337M | → 108 (165) |
| 108 | `VAEDecodeAudio` | — | AUDIO music → 114 (169, direct — 0dB adjust node 112 deleted as no-op) |
| 109 | `MMAudioModelLoader` | `["mmaudio_large_44k_v2_fp16.safetensors","fp16"]` 2.06G | MODEL → 111 (166) |
| 110 | `MMAudioFeatureUtilsLoader` | `[vae_44k_fp16 611M, synchformer_fp16 475M, DFN5B-CLIP_fp16 1.97G, "44k","fp16"]` (bigvgan auto-downloads at runtime — no VoCoderLoader needed) | FEATUREUTILS → 111 (167) |
| 111 | `MMAudioSampler` | `[5.0(stale,←116 via 178),25,4.5,7,"fixed",electro-acoustic SFX prompt,vocal-negative,True,True]` steps 25 cfg 4.5 seed 7 fixed, mask_away_clip True, force_offload True — disk carries `"fixed"` at index 4 because the frontend injects `control_after_generate` after seed (old-style INPUT_TYPES lacks it); without it every later value shifts one slot left (prompt←old text, negative←"True") | images ←123 padded interp frames (168) → synced SFX AUDIO → 113 (170) |
| 123 | `BatchPadToMin` | `[17]` (custom `batching_utils` pack) | images ←90 (189) → 111 (168): pads batch to ≥17 frames for MMAudio's 16-frame sync segments; passthrough when long — title "Pad To Min 17" |
| 113 | `AudioAdjustVolume` | `[-6]` dB | SFX level → 114 (172) |
| 114 | `AudioMerge` | `["add"]` | mix → 99.audio (173) |

**2026-09-08: C64 STYLE + prompt rework** (user request: highly psychedelic/experimental/weird/crazy/wild/vibrant). Downloaded the Ernie-native LoRA version (`Ernie C64`, modelVersionId 2882202, file `c64style_ernie.safetensors` 180M, trigger `c64style`) — other versions on the page target Krea 2 / ZImageTurbo / Illustrious / Flux / Pony / SDXL and would NOT load on the Ernie DiT. Wired as `LoraLoader` 120 (strength_model 0.85, later raised to 1.0 in the HD rebuild so C64 dominates) between loaders (10/11) and consumers (53/21); LoRA carries diffusion-only weights so strength_clip is a no-op. Reworked all three prompts (visual 20: C64 16-color palette + dithering + scanlines + dense specific objects; ACE 105: chiptune-psychedelia instruments, bpm 90→128; MMAudio 111: retro-arcade SFX, negative unchanged to keep music out of SFX). Verified: validate 0 errors, live Queue → frame_00051 success with 120 executed (warm-cache fast path, no errors). Cold-start `input/ernie_zoom_seed.png` still old-style — first img2img pass (denoise 0.60) conforms it; re-render seed later if a pure-C64 frame 1 matters.

**2026-09-08: ADDED full-mix audio (music + SFX) to the bypassed video group** (user request: local/free/SOTA/instrumental, wired into node 99). Music bed = ACE-Step 1.5 2B turbo (native nodes, instrumental via empty lyrics, quality-max that fits 16GB alongside MMAudio — XL 4B rejected); SFX = MMAudio large fp16 in-workflow (interp frames → video-synced rumble/whoosh, mask_away_clip True so CLIP text doesn't fight the video conditioning). Mix = music 0dB + SFX −6dB via `AudioMerge add` → `VHS_VideoCombine.audio` (slot 11 — node 99 has 11 inputs 0-10, so audio is 11 not 12; installed VHS exposes the optional slot, disk JSON needed the input entry appended). Duration was single knob 103 — **2026-09-08: REPLACED with automatic frame-count math** after user reported "short burst of sound in the first ~1s then nothing": 103 was stale at 0.657s while the video had grown to 185f/5.78s, so ACE+MMAudio generated 0.657s and VHS padded the rest with digital silence (measured: RMS>0.14 for 0-1s, exactly 0.0 after). Fix uses existing nodes only — 115 `VHS_GetImageCount` reads 90's interp batch → 116 `ComfyMathExpression` (`a / 32`) feeds 104/105/111; serialization of the autogrow `values.a` slot was captured from a live browser probe (slot name is `values.a`, type `FLOAT,INT,BOOLEAN`). Verified: 48-frame render → `Ernie_Zoom_00003.mp4` 189f @32fps = 5.906s with 5.888s audio, RMS 0.17-0.23 across every 0.5s window — full-length sound, no drift ever (fps is shared: 117 feeds both 99's frame_rate and 116's `b`, so `a / b` stays exact at any fps). Shared-primitive verification: executed prompt shows `99.frame_rate=['117',0]`, `95/96.link_id=['118',0]`, `84.w/h=['119',0]`, `116.expression='a / b'`; render → `Ernie_Zoom_00005.mp4` 193f @32fps + `-audio.mp4` 6.016s AAC, RMS ≥0.14 in every window. Deliberately NOT shared: sampler steps/cfg/seeds that merely coincide (53 vs 107) are independent tunables — coupling them would be wrong; stale 5.0 duration widgets on 104/105/111 are dead fallbacks overridden by the already-shared 116. Stale 5.0 widget fallbacks on 104/105/111 remain inert while linked. Verified end-to-end from scratch: full reset → 6 bypassed Queues (warm cache ≈5s each) → 1 unbypassed render (`Ernie_Zoom_00004.mp4` 21f + `-audio.mp4` with AAC, 0.657s) → 103 set 5.0→0.657 → final render 7 frames → `Ernie_Zoom_00005.mp4` 25f @32fps + `-audio.mp4` AAC, 0.782s, ≈10-20s render on 4060 Ti sequential (Ernie models stay loaded; audio only runs on video render). Stable Audio Open 1.0 kept as backup SFX source; TangoFlux rejected (no ComfyUI nodes); API audio (Seed/ElevenLabs) excluded — not signed in.

**2026-09-08: HD 1376×768 + optimization + electro-acoustic audio** (user request: HD at an officially supported Ernie resolution, C64 dominance, no redundant nodes, future-agent docs, modern experimental electro-acoustic audio that follows the images). Resolution = official Baidu preset 1376×768 (turbo = 8 steps cfg 1.0). Shared prims split per-axis: 119 Frame Width [1376] + new 121 Frame Height [768] → 84; 80 Crop Width [1356] + new 122 Crop Height [748] → 83 (offset rule `(1376−1356)/2 = 10` keeps the dive centered); zoom = 1376/1356 = 1.0147×/frame (~1.5%, gentler than the 512-era 4%). Removed 60 `PreviewImage` (dup of Sender's own temp preview) + 112 `AudioAdjustVolume` [0dB no-op] (108 wired direct to 114). C64 re-applied at 1.0 — the b24 wiring had been lost by a browser-tab re-save (disk back at 41 nodes, mtime 16:20); texts + node-120 template recovered from `/tmp/opencode/loop_c64_new.json` and re-applied onto the current disk file (never install stale copies — always re-dump from disk after edits). **Same clobber happened AGAIN later the same day** (disk back at 41 nodes/512px, mtime 20:40, user queued 24 512² frames from the stale tab); recovered from `/tmp/opencode/loop_test.json` (full 44-node HD file, reinstalled via docker cp, revalidated, frames reset, 2 test queues → 1376×768 C64-confirmed). HARD RULE: after any agent-side workflow change the user must reload the workflow in their browser tab BEFORE pressing save — a save from a stale tab silently overwrites the on-disk file with the old graph. Prompts restyled electro-acoustic (105 tags, bpm 100; 111 SFX; lyrics still `""` instrumental). Seed re-rendered at 1376×768 via throwaway txt2img (seed-branch nodes 32-34 don't exist since Send/Receive — clone structures from disk + hand-written EmptyFlux2LatentImage). Two audio bugs fixed: (a) MMAudio synchformer needs ≥16 sync frames — few-frame renders crashed `torch.stack([])` in `encode_video_with_sync` (5 interp frames → 3 sync frames → `range(-1)`); fix = 116 `max(a / b, 0.7)` floor (VHS `-shortest` trims excess, so the floor is safe) + new 123 `BatchPadToMin` [17] inline 90→123→111 (retargeted 168, no new link needed); (b) seed-widget off-by-one — kijai's old-style INPUT_TYPES lacks `control_after_generate`, frontend injects it after seed so prompt←old/negative←"True"; fix = disk 111 carries `"fixed"` at index 4. Verified: 4-frame render → 13f @32fps + `-audio.mp4` 0.405s full-length AAC (RMS 0.28/0.10, no silence), submitted 111.prompt = new text; in-zoom continuity holds at HD (27.05 vs 37.23).

**2026-09-02: node 70 (`Anything Everywhere`) REMOVED.** It was completely disconnected (input `link:null`, zero links — broadcast nothing) but its mere presence made every browser Queue fail with `res is undefined`: UE 8.0's `find_duplicate_broadcasted_types` crashes on `node.inputs === undefined` (frontend 1.51.9), the exception is swallowed by `call_function_with_modified_graph` which returns `undefined`, and VHS's outer `graphToPrompt` wrapper then dereferences `res.workflow` → error dialog. All MODEL/CLIP/VAE/COND wiring was already explicit links, so removal required no rewiring; verified end-to-end (queue OK, history 6→7, new mp4). Do not re-add UE nodes (see §7).

**2026-09-02: REDESIGNED from out-zoom (scale 0.85 + pad + inpaint) to true IN-zoom** (user request). Old nodes 50 `ImageScaleBy`/51 `ImagePadForOutpaint`/52 `VAEEncodeForInpaint` deleted; new 80-85 (crop→rescale→encode) + 87 docs; node 53 now full-frame img2img denoise 0.60; node 56 range flipped `"1-1000"`→`"1000-1"` (ascending range = newest→oldest batch order — WRONG for playback; descending = chronological). History was RESET (old 18 out-zoom frames purged, `rm -rf` must run INSIDE container — host lacks perms on root-owned files). Verified end-to-end: 4 sequential headless Queues → history 2→5, 4 mp4s; direction confirmed numerically (each frame closer to 1.04× magnified predecessor than to predecessor itself, all 4 pairs ~0.126 vs ~0.18 mean abs diff @128²). Later superseded by Send/Receive redesign below (loopback nodes 40/55/56/57 deleted).

**2026-09-02: REDESIGNED from loopback history to Send/Receive + saved frames** (user request: simplify). Deleted seed branch (30-34), all loopback nodes (40/55/56/57), old video tail (62/88/91-94) and the `ComfySwitchNode` toggle. Cross-Queue feedback is now Impact Pack `ImageSender` 96 / `ImageReceiver` 95 (link_id 1): `img-send` frontend event rewrites Receiver's image widget to the Sender's temp file (`Ernie_Zoom_Send_temp_*.png [temp]`), so each Queue's crop chain starts from the previous Queue's frame — memory-resident, lost on page reload → falls back to `/comfy/input/ernie_zoom_seed.png` (rendered once via throwaway txt2img run, re-rendered at 1376×768 HD in the 2026-09-08 rebuild). Frames persist via `SaveImage` 97 to `output/Ernie_Zoom_Frames/frame_*.png` (subfolder keeps Turbo PNGs / AnimateDiff leftovers out of the sequence). Video group (89/98/90/99, frontend group id 1) is **bypassed by default**: unbypass to render — `VHS_LoadImagesPath` 98 loads all frames → `FrameInterpolate` 90 4× FILM (`(n-1)*4+1` frames, `<2` frames pass through) → `VHS_VideoCombine` 99 @32fps with preview + optional save (`Ernie_Zoom_*.mp4`, save_output toggle). Zoom math unchanged: `512/492 = 1.0407` per frame (≈4% dive); offset rule `(512−size)/2 = 10` keeps it centered. Verified end-to-end: cold Q1 12.1s → Receiver rewritten to temp PNG; Q2 6.0s; zoom continuity mean|f2−zoom(f1)|=0.085 vs mean|f2−f1|=0.133; 4 dir frames → `Ernie_Zoom_00001.mp4` 13 frames @32fps = (4−1)*4+1, no pingpong. History under loopback is gone — `rm -rf` of `Ernie_Zoom_Frames/` + videos (INSIDE container) resets.

Flow per Queue (current HD design): `95(ImageReceiver: last frame or seed) —148→83(crop 1356×748@10,10, params from 80/122/81/82)→84(rescale 1376×768 bicubic, size from 119/121)→85(encode)→53(img2img 0.60, LoRA 120 on MODEL/CLIP)→54→96 ImageSender (151, feeds next Queue) + 97 SaveImage (152, persists sequence)`. Zoom math: `1376/1356 = 1.0147` per frame (~1.5% dive); offset rule `(1376−1356)/2 = 10` keeps it centered. Bypassed video group (unbypass 99 to render): `98 VHS_LoadImagesPath(dir) —153→90 FrameInterpolate 4× FILM (interp_model from 89) —154→99 VHS_VideoCombine @32fps (frame_rate ←117)` with preview + optional save toggle; `(n-1)*4+1` frames (4 dir frames → 13 @32fps verified, no pingpong). Bypassed audio group renders with it: 115 counts 90's frames → 116 `max(a/b,0.7)` → 104/105/111; 90's frames also condition MMAudio via 123 pad-to-17 → 111; mix 114 → 99.audio. **GOTCHA — VHS_VideoCombine `widgets_values` must be the 6 declared widgets only** `[32,0,"Ernie_Zoom","video/h264-mp4",False,True]` (format sub-widgets default: crf19/yuv420p/meta-true/trim-false); a 10-element array with sub-widget values misaligns positionally → `pingpong` got truthy `"yuv420p"` → pingponged videos (2 frames→8, 3→16). VHS counter quirk: its metadata PNG matches the `Ernie_Zoom_(\d+)` counter regex → numbering can skip (harmless).

#### 6.3 Z-Image Loop — `z_infinite_zoom_loop.json` (✅ Z twin, 60 nodes, 80 links, 0 groups, bbox 4290×5362)

Adapted duplicate of the ernie loop (same crop→rescale→encode→img2img→Sender/SaveImage per-Queue flow, same frames-dir + bypassed-VHS-video + duplicated-audio mechanics, same 1376×768 which is div-16 ✓ for Z-Image). Only the generation core + prompts differ; zoom/crop prims (80/122/81/82), size prims (119/121), fps prim (117) are per-workflow copies (NOT shared across files).

| id | type | widgets / inputs | role |
|----|------|------------------|------|
| 10 | `UNETLoader` | `["juggernautZ_v10FastBy.safetensors","default"]` 12.3G | fast MODEL → 201 (300) |
| 200 | `UNETLoader` | `["juggernautZ_v10ByRundiffusion.safetensors","default"]` 12.3G | quality MODEL → 201 (301) |
| 208 | `PrimitiveBoolean` | `[False]` "QualityMode" | THE one-button switch → 201 (303) + 214 (357) |
| 201 | `ComfySwitchNode` | switch ←208, on_false ←10, on_true ←200 (lazy) | selected MODEL → 202 (302) — unselected UNETLoader never executes |
| 11 | `CLIPLoader` | `["qwen_3_4b_fp8_mixed.safetensors","lumina2","default"]` 5.6G | CLIP → 203/204 (parallel, lazy) |
| 202 | `ModelSamplingAuraFlow` | `[3]` shift 3.0 (gallery-canonical for Z) | MODEL ←201 → 203/204/209-off |
| 207 | `PrimitiveInt` | `[0]` "Style" (0 off / 1 chalkboard / 2 clay-art) | → 205.a (348) + 206.a (349) |
| 205/206 | `ComfyMathExpression` | `["a == 1"]` / `["a == 2"]` (BOOL out slot 2) | → 209.switch (350) / 210.switch (352) |
| 203 | `LoraLoader` | `["Chalkboard01-1_CE_ZIMG_AIT4k.safetensors",0.85,1.0]` model ←202 (344), clip ←11 (345) | chalk MODEL → 209 (351); no trainedWords — style comes from prose, not a trigger |
| 204 | `LoraLoader` | `["ClayArt01a_CE_ZIMG_AIT3k.safetensors",0.85,1.0]` model ←202 (346), clip ←11 (347) | clay MODEL → 210 (353) |
| 209/210 | `ComfySwitchNode` | cascaded MODEL select (209: chalk vs 202-base; 210: clay vs 209-out) | MODEL → 53/213 (354/355) |
| 211/212 | `ComfySwitchNode` | same cascade for CLIP (211: chalk vs 11; 212: clay vs 211-out) | CLIP → 21 (356) |
| 21 | `CLIPTextEncode` | SEMANTIC prose prompt (215) + clip ←212 | COND → 53, 213 |
| 216 | `CLIPTextEncode` | trimmed official Juggernaut negative (illustrated/drawing/comic/CGI tags dropped — they fight the LoRAs) | COND → 53, 213 (negative) |
| 53 | `KSampler` | `[<seed>,"randomize",6,1.0,"ddim","normal",0.60]` (page-faithful fast: steps 4-8, CFG 1-1.5) | fast img2img → 214 (358) |
| 213 | `KSampler` | `[777,"randomize",22,4.0,"res_multistep","beta",0.60]` (official recipe: res_2s/beta/22/cfg 4; single pass — 3-step refiner omitted, gallery-canonical) | quality img2img → 214 (359) |
| 214 | `ComfySwitchNode` | switch ←208, on_false ←53, on_true ←213 (lazy) | LATENT → 54 (360) — only the selected sampler runs |
| 217 | `ImpactInt` | `[2]` "Send/Receive Link ID" | → 95 (335) + 96 (336); id **2** (not 1) avoids crosstalk with the ernie tab |
| 95/96 | `ImageReceiver`/`ImageSender` | widget link_id **must equal 2** (see GOTCHA below) | seed `z_zoom_seed.png` (rendered via quality model, 22 steps, 66s) → 83; frames → `Z_Zoom_Frames/` |
| 220 | `ComfyMathExpression` | `["max(a / b, 0.7)"]` (dup of 116) | audio seconds → new audio group |
| 225 | `BatchPadToMin` | `[17]` (dup of 123) | 90 → new MMAudio 240 |
| 230-243 | audio-group duplicate | ACE music (230-237) + MMAudio SFX (238-241) + mix 242 → 99.audio; 243 `ConditioningZeroOut` replaces shared-with-ernie node 22 as 236's negative | `Z_Zoom` video + `-audio.mp4` twin |

Flow per Queue: identical to ernie (95→83→84→85→branch→54→96/97), except the img2img stage is `85 → 53/213 (model ←210, pos ←21, neg ←216) → 214 → 54`. Video/audio groups are copies retargeted at `Z_Zoom_*` paths. **Prompts MUST be semantic sentences** (Z-Image requirement — no SDXL tag soup, no C64 trigger).

**2026-09-08: Z-Image support + canonical mappings** (research before build): Z-Image runs on native nodes — `CLIPLoader` offers `lumina2` type (Qwen3-4B), `KSampler` offers `res_multistep` (= res_2s family) + `ddim`, schedulers `beta`/`normal`, `EmptySD3LatentImage` works (ZImage latent is Lumina2-derived) — no custom sampler pack needed (`ClownsharKSampler_Beta` in the official Juggernaut wf maps to native `KSampler`). Gallery templates `image_z_image` (25 steps/cfg 4/res_multistep/simple/shift 3, note "steps 30-50, cfg 3-5") + `image_z_image_turbo` (8 steps/cfg 1) and the official wf (cfg 4 via PrimitiveFloat link; pass1 res_2s/beta/22/1.0, pass2 res_2s/normal/3/0.15; recommended 960×1440 ≈1.38MP, div-16 ideal) all agree: **quality cfg = 4**. Fast = DDIM 6 steps cfg 1.0 is page-faithful (gallery turbo uses res_multistep, page says DDIM — page wins). TE/VAE from HF `Comfy-Org/z_image` (ungated: `qwen_3_4b_fp8_mixed` + `ae.safetensors`); official `Tongyi-MAI/Z-Image` files are diffuser-format (not directly loadable). CivitAI downloads need the version-scoped URL form (`/api/download/models/{versionId}?fileId={fileId}` — the bare file form 404s); on-disk HF token is INVALID (breaks public HF calls if sent — use no Authorization header).

**GOTCHA — img-send matches the frontend widget VALUE, not the link** (Impact Pack `js/impact-pack.js`, frontend 1.51.9): the linked-widget branch requires `widgets[1].type == 'converted-widget'`, which this frontend never sets (probed live: `{name link_id, type 'number'}` with the link present) — so matching falls through to `widgets[1].value` vs the sent link_id. Consequence: 95/96 `widgets_values[1]` MUST equal the link_id primitive's value (2 for Z, 1 for ernie) or the receiver silently never updates; the `ImpactInt`-vs-`PrimitiveInt` type is irrelevant to the handler (swap was harmless but the value alignment was the real fix). The ernie loop carries the same latent trap (118=1 matches today) — DO NOT touch the ernie file for this (clobber risk); if its link_id ever changes, align its 95/96 widget values too.

Verified end-to-end: validate 0 errors (21 expected bypass warnings, 58 converted); fast branch ~10-15s/frame, quality 65.3s/frame; lazy pruning PROVEN (QualityMode false + fresh seed → 15.1s vs 65.3s true — a ~50s delta is impossible if both branches executed); receiver rewrite works in both modes (q2 `95.image` = `Z_Zoom_Send_temp_*.png [temp]`); directional continuity in both modes (fast 29.24<30.29, quality 21.17<22.66 at the exact 1.0147× factor); LoRA 3-way ALL confirmed visually (0 = photorealistic cathedral, 1 = black chalkboard + chalk linework, 2 = plasticine claymation); A/V render `Z_Zoom_00001.mp4` 53f @32fps = 1.656s = (14−1)×4+1 with full-length AAC (RMS 0.075-0.144 in every window). Test rig `/tmp/opencode/uebug/queue_z.mjs [N] [--video]` (single page session — receiver rewrite is in-memory; fresh seed per run or the server cache-hits and appends nothing; filter `/history` for node `203` to isolate Z entries from ernie-tab interleaving; prompt JSON carries widget values as direct `inputs`, not `widgets_values`).

- Straight edges enforced globally via `comfy.settings.json` `Comfy.LinkRenderMode=Straight` and per-workflow `extra._linkRenderMode` doc hint (see §3).

### 7. Custom Nodes Audit (QoL / Safety)
- **comfy-loopback-buffer** (holo-q, 1.0.0 MIT, deps torch/pillow/numpy/aiohttp): replaces filesystem Save/Load reimplementation for incremental accumulation. Small, tested, maintained — **keep installed** but LEGACY: no workflow uses it since the Send/Receive redesign (§6.2).
- **ComfyUI-VideoHelperSuite** (Kosinkadink, 1.7.9, ~4k★): `VideoCombine` etc. Thousands of dependents, active, deps already baked (`opencv-python imageio-ffmpeg`). `VHS_LoadImagesPath` + `VHS_VideoCombine` drive the loop workflow's bypassed video group — **keep**, now load-bearing for video rendering.
- **comfyui-impact-pack** (ltdrdata, ~1k★, active): `ImageSender`/`ImageReceiver` cross-Queue feedback drives the loop workflow (link_id pair; `img-send` frontend event rewrites Receiver widget to Sender's temp file — memory-resident, falls back to seed PNG on reload). Host git clone (NOT baked in Dockerfile — code persists via `./Comfy:/comfy` mount, pip deps via entry.sh runtime loop); `requirements.txt` trimmed of `git+.../sam2` (SAM nodes unneeded → faster boot; `[Impact Pack] SAM2 functionality unavailable` warning + `impact-sam-editor.js` 404 are expected/harmless). **keep** (required for perpetual zoom). ⚠️ img-send matches the frontend widget VALUE, not the link (frontend 1.51.9 never marks linked widgets `converted-widget`, so the linked branch of `imgSendHandler` is dead code) — Receiver/Sender `widgets_values[1]` MUST equal the link_id primitive's value or the receiver silently never updates (see §6.3 GOTCHA).
- **comfyui-mmaudio** (kijai, 574★ MIT, pushed 2026-02): `MMAudioModelLoader`/`MMAudioFeatureUtilsLoader`/`MMAudioSampler` drive the loop workflow's bypassed audio group (video-synced SFX from interp frames). Host git clone (same pattern as impact-pack — NOT baked in Dockerfile); light deps (librosa/torchdiffeq/einops/timm/omegaconf/open_clip/accelerate/ftfy). Models (~4.6G: large fp16 2.06G + synchformer 475M + vae 611M + CLIP 1.97G) under `models/mmaudio/`. **keep** (required for SFX). ⚠️ Two gotchas: (1) synchformer needs **≥16 sync frames** — short renders crash `torch.stack([])` in `encode_video_with_sync` (fixed in-workflow via 116 `max()` floor + 123 pad node, see §6.2); (2) old-style INPUT_TYPES has no `control_after_generate`, frontend injects it after seed — disk `widgets_values` must carry the control value or the prompt slot shifts (see §6.2).
- **batching_utils** (local, host-written, torch only, no deps): single node `BatchPadToMin` (images + min_frames, default 17) — pads short batches by whole-batch tiling for MMAudio's 16-frame sync segments, passthrough when long. Exists only because no native pad-to-min node fits (RepeatImageBatch is fixed-count → OOM at scale). **keep** (required by loop audio group).
- **cg-use-everywhere** (chrisgoringe, checkout 50ae9f8 = upstream main HEAD 2026-07-20, "8.0", 1k★, frontend-only): `Anything Everywhere` broadcast reduces link spaghetti, no python deps. **⚠️ INCOMPATIBLE with frontend 1.51.9** — any workflow containing a UE node fails at browser Queue: UE 8.0 assumes `node.inputs`/`node.outputs` always defined (`find_duplicate_broadcasted_types` use_everywhere_utilities.js:365 `reading 'length'`; also `fix_unconnected_inputs` `reading 'filter'` in afterConfigureGraph on load), the throw is swallowed by `call_function_with_modified_graph` → returns `undefined` → VHS's outer wrapper dereferences `res.workflow` → `res is undefined` error dialog. No newer upstream fix exists as of 2026-09-02. **Keep installed** (workflows without UE nodes queue fine — wrappers pass through cleanly) but **do not add UE nodes to any workflow** until upstream/compatible-frontend resolves; wire explicitly instead.
- **comfyui-workflow-prettier** (deepme987, Sugi­yama Layered/Compact/Linear/SortByType, 1103-line JS, zero python deps): auto-positions DAGs properly (barycenter crossing minimization, median coordinates). Successor to hand-rolled `/tmp/format_*.py` scripts — **keep** (required tool for all position edits, avoids horizontal explosion).
- **Native FrameInterpolate** (`comfy_extras.nodes_frame_interpolation`, no pack): `FrameInterpolationModelLoader` + `FrameInterpolate` (multiplier 2-16, align-aware, OOM-batch halving, feature caching across pairs). Models under `models/frame_interpolation/` — `film_net_fp16.safetensors` 66M (FILM, default, best for large-motion/zoom) + `rife_v4.25/4.26/4.26_heavy` 22M (RIFE, faster). FILMNet has no pad_align, IFNet pad_align 64; both load via same loader. Greedy `prev_frame` + `feat_cache` reuse across pairs. No extra pip deps — **keep**; do not install Fannovel16/ComfyUI-Frame-Interpolation pack (duplicate).
- No removal recommended; all are maintained, trustworthy, popular. If a new dep is added, check stars, commit recency, license, and whether it replaces a reimplementation.

### 8. Confinement & Cleanup Discipline
- Everything generated goes under `Comfy/output/` — verify after each workflow edit: `ls -R Comfy/output` should contain only intended files. Saved frames (`Ernie_Zoom_Frames/`) count as generation.
- `Comfy/temp/` should stay empty of persistent history; entry.sh migration ensures legacy temp history moves to output.
- To reset the loop sequence: `rm -rf Comfy/output/Ernie_Zoom_Frames` (+ videos `rm Comfy/output/Ernie_Zoom*.mp4`) — **must run INSIDE the container** (host lacks perms on root-owned files). Next Queue cold-starts from `input/ernie_zoom_seed.png`. Z loop: same with `Z_Zoom_Frames` / `Z_Zoom*.mp4` / `input/z_zoom_seed.png`.
- zoomy sequences reset the same way from inside the zoomy container: `rm -rf /comfy/output/Zoomy /comfy/output/Zoomy_<sequence>*` (frames dir + videos + `-audio` twins + preview PNGs), or the UI's two-click **Clear frames**. Next Render cold-starts from the family's seed image.
- `Comfy/output/loopback_ernie_zoom/` is LEGACY — remove if it reappears.

### 9. Update Discipline & Task Alignment
- **After each significant amount of work, update this AGENTS.md** if new relevant info/facts became known (new model, new workflow, new custom node, new volume, new port, new patch, perf numbers, VRAM lessons). Keep this file as the single source of truth for the wrapper; don't let facts live only in chat.
- **NEVER commit without the user's explicit request for *each* commit.** The user may approve several commits at once, but approval is *always* required per commit — never `git commit` (nor amend, push, or PR) on your own initiative, even for small or "obvious" changes. Leave changes uncommitted for review unless asked.
- **When defining/planning a task, go through rounds of Q&A to further specify the solution and ensure proper alignment** before building: ask clarifying questions (node choices, naming, durability, toggle mechanics, scope), present a researched plan, and only proceed once the user confirms. Never jump from a vague request straight to implementation.

### 10. Zoomy App & Engineering Standards
- **zoomy** (`Zoomy/`, full design in `Zoomy/DESIGN.md`) is the default interface: Gradio control panel (family dropdown → LoRA multi-select + strength sliders → Render next frame → Finalize video) driving ComfyUI over REST. Families `ernie_turbo` / `z_fast` / `z_quality` (z pair shares sequence `z_image`); new family = one `FamilyDefinition` appended, nothing else changes. Existing workflow JSONs are untouched (additive).
- **All zoomy deps are containerized — never pollute the host**: runtime `gradio==6.26.0 httpx==0.28.1`, dev `ruff==0.16.6 mypy==2.3.1 pytest==9.1.1`, all installed in the image. Host has no comfy/gradio/pytest binaries by design.
- **Standards (binding for all zoomy work)**: type annotations everywhere; ruff `ALL` (line-length 100) + `ruff format` + mypy `strict` + pytest after each significant piece of work until clean; document non-obvious code (module docstrings explain the *why*); good names — no acronyms/abbreviations/contractions/single-letter variables (domain terms `lora/comfy/json/http` allowed; prefer `autoencoder/text_encoder/base_model` over `vae/clip/unet` in our names, node class_types + input keys stay literal); keep modules small (split before hitting complexity limits); maintain `Zoomy/DESIGN.md` with every behavior change.
- **Quality loop runs against host files via bind mount** (image files are throwaway copies): `docker compose run --rm --no-deps -v ./Zoomy:/workspace ... zoomy sh -c "cd /workspace && ruff check --fix zoomy tests && ruff format zoomy tests && ruff check zoomy tests && mypy zoomy tests && python -m pytest -p no:cacheprovider"` — then rebuild + launch via `zoomy.sh`.
- **Verified live behaviors**: `EmptyAceStep1.5LatentAudio.seconds` enforces min **1.0** → audio floor is `max(interp/32, 1.0)` (ACE text encoder allows ≥0, MMAudio no min); VHS mux always applies `-shortest` so floored audio trims to the video, and writes a silent main mp4 + `-audio.mp4` twin WITH the soundtrack (the twin is the preview artifact — `latest_video_path` prefers it). Gradio 6.26: `Dropdown` tuple choices are `(display, key)`; `Timer` takes `value=` seconds; `gr.State(value=...)` keyword form; `gr.render` is decorator-only; event methods (`tick/change/click`) exist at runtime but are stub-omitted (targeted `type: ignore`s, runtime-verified). E2E timings on 4060 Ti: z_fast frame ~10-15 s, finalize 2 frames ~20 s, ernie frame ~23 s.

## Commands Cheat Sheet

```bash
./serve.sh                                   # build + run (ephemeral) — the ONLY way to start
docker ps --format "{{.Names}} {{.Status}}"   # comfy-comfy-run-... Up (healthy)
docker logs <cid> --tail 100                 # ComfyUI logs
docker exec <cid> comfy workflow validate --workflow /comfy/user/default/workflows/ernie_infinite_zoom_loop.json
docker exec <cid> comfy workflow validate --workflow /comfy/user/default/workflows/z_infinite_zoom_loop.json
docker exec <cid> bash -c 'ls /comfy/output/Ernie_Zoom_Frames | wc -l'
docker exec <cid> bash -c 'ls /comfy/output/Z_Zoom_Frames | wc -l'
curl -s http://localhost:9000/status | jq    # comfy-mcp bridge health
curl -s http://localhost:8188 | head         # ComfyUI health
curl -s -o /dev/null -w "%{http_code}\n" http://localhost:7861/  # zoomy UI health
./zoomy.sh                                 # build + run zoomy (spawns comfy via serve.sh when missing)
# zoomy quality loop (fixes host files via bind mount, then verify clean):
docker compose run --rm --no-deps -v ./Zoomy:/workspace -e RUFF_CACHE_DIR=/tmp/ruff-cache -e MYPY_CACHE_DIR=/tmp/mypy-cache zoomy sh -c "cd /workspace && ruff check --fix zoomy tests && ruff format zoomy tests && ruff check zoomy tests && mypy zoomy tests && python -m pytest -p no:cacheprovider"
# zoomy e2e inside its container (ZOOMY_* env already set): docker exec <zoomy-cid> python -c "from zoomy. ... render_next_frame / finalize_video ..."
# Format workflows after manual edits (Sugiyama Layered, straight edges):
python3 /tmp/prettify.py --all               # or workflow_prettier UI: select Layered → Save
```

## Known Gotchas
- Host `comfy` binary absent by design — use `docker exec <cid> comfy ...` instead.
- Changing `cache_path` to an absolute under `/comfy/output` is the ONLY way to confine loopback; relative paths are jailed under temp.
- Workflow `id` changes on save can drift hash key — fixed key patch prevents history loss.
- Frames are **1376×768** (official Ernie preset; ~20s/frame on 4060 Ti). Don't change resolution without re-checking VRAM + re-rendering the seed PNG at the new size (cold-start seed must match frame size).
- `Comfy.LinkRenderMode` must be `Straight`; if edges look curved, re-check `comfy.settings.json`.
- zoomy finalize: the `-audio.mp4` twin (not the same-named mp4) carries the soundtrack — preview/ship the twin. Audio floor is `max(interp/32, 1.0)` because `EmptyAceStep1.5LatentAudio.seconds` rejects anything below 1.0 with a 400.
- zoomy e2e writes real artifacts under `Comfy/output/Zoomy/` + `Zoomy_*` — clean them from inside the zoomy container after verification; never delete the user's `Ernie_Zoom_*`/`Z_Zoom_*` files by pattern accident (`Zoomy_` ≠ `Z_Zoom_`/`Ernie_Zoom_`).
