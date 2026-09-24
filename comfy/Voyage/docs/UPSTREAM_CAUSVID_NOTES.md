# UPSTREAM_CAUSVID_NOTES — CausVid integration notes (Stream D prep)

Pins live in code in `voyage/model_registry.py` (`CAUSVID_*` / `WAN21_*`,
the single source of truth); this file mirrors them for humans. Every repo
is **ungated** — no token required. Revisit every note if a pin moves.

Status: prep scaffolding only. There is no worker yet —
`voyage/workers/video_causvid.py` is explicitly a deferred worker slice
(TASK §30.1) — and no `models download/verify causvid-*` CLI wiring
in `voyage/cli.py` (the registry `download_causvid_models` /
`verify_causvid_models` entry points exist so that slice has support ready).

## Upstream pins (probed 2026-09-24)

| Artifact | URL | Pinned revision |
|----------|-----|-----------------|
| Code | [tianweiy/CausVid](https://github.com/tianweiy/CausVid) | `adb6a5ecd07666b4d0290042915c8406e6d5ce22` (short `adb6a5e`; master HEAD at probe time, tip commit 2025-08-07 "Update README.md") |
| DMD checkpoint | [tianweiy/CausVid](https://huggingface.co/tianweiy/CausVid) | `b545eb2728fc9d1515023a270b847f7b24b3aa89` (main, 2025-05-17) |
| Base model | [Wan-AI/Wan2.1-T2V-1.3B](https://huggingface.co/Wan-AI/Wan2.1-T2V-1.3B) | `37ec512624d61f7aa208f7ea8140a131f93afc9a` |
| Paper | [causvid.github.io](https://causvid.github.io/) (arXiv [2412.07772](https://arxiv.org/abs/2412.07772), CVPR 2025) | — |

## What to download (registry-ready, CLI wiring deferred)

From `tianweiy/CausVid` (CC BY-NC-SA 4.0): only
`autoregressive_checkpoint/model.pt` — the causal DMD generator the
upstream long-video script strict-loads via
`torch.load(<folder>/model.pt)['generator']`. Skipped: both
`bidirectional_checkpoint*` dirs, `autoregressive_checkpoint_warp_4step_cfg2`,
the ODE checkpoints, and the mixkit LMDB datasets.

From `Wan-AI/Wan2.1-T2V-1.3B` (Apache 2.0) as a subset — sizes measured from
the HF API file listing at pin time (no weights downloaded during prep):

| File | Bytes at pin time | Registry floor |
|------|------------------:|----------------|
| `diffusion_pytorch_model.safetensors` | 5,676,070,424 | 5 GB |
| `Wan2.1_VAE.pth` | 507,609,880 | 400 MB |
| `models_t5_umt5-xxl-enc-bf16.pth` | 11,361,920,418 | 10 GB |
| `google/umt5-xxl/*` (tokenizer) + `config.json` | — | presence only |

The DMD `model.pt` byte count is unmeasured (prep rule: no downloads), so
its registry floor is a 1 GB presence sanity check — any real 1.3B bf16 DiT
checkpoint (~2.6 GB of params) clears it. Replace with a measured threshold
after the first real download (open question 1).

## License implications

- CausVid weights and the upstream repo `LICENSE.md` are **CC BY-NC-SA
  4.0** ([deed](https://creativecommons.org/licenses/by-nc-sa/4.0/deed.en)).
  Non-commercial use (Voyage's current case) is compatible; commercial
  redistribution is not. **ShareAlike**: redistributing an adapted model
  must carry the same license. Attribute Yin et al. Every run manifest must
  record repo/revision/license alongside the checkpoint hash (DESIGN §5.4
  already requires it).
- The Wan2.1-T2V-1.3B base is **Apache 2.0** (permissive) — the copyleft
  constraint comes from the CausVid checkpoint, not the base.
- Practical consequence: keep the CausVid stack clearly labeled in
  `models download` output and manifests so a future commercial deployment
  cannot silently inherit it.

## Native geometry / fps / latent-overlap (from the pinned sources)

- Env (upstream README): python 3.10, `torch>=2.4.0`, `flash_attn`,
  bf16 (`pipeline.to(device="cuda", dtype=torch.bfloat16)`).
- Config `configs/wan_causal_dmd.yaml` @ pin: `denoising_step_list:
  [1000, 757, 522, 0]` (3 DMD steps), `num_frame_per_block: 3`,
  `image_or_video_shape: [1, 21, 16, 60, 104]` — 21 latent frames, 16
  channels, 60×104 latent spatial = 480×832 px @ 8× VAE; `timestep_shift:
  8.0`. Matches DESIGN §5.4.
- Dataset prep reshapes to **480×832×81 @ 16 fps** — 81 pixel frames =
  (21−1)×4+1, i.e. temporal compression 4, consistent with the 21-latent
  chunk. Native profile: **832×480 @ 16 fps**.
- Long-video rollout (`minimal_inference/longvideo_autoregressive_inference.py`
  @ pin): per rollout, fresh noise `[1, 21, 16, 60, 104]` bf16 on CUDA →
  `InferencePipeline.inference(noise, text_prompts, return_latents=True,
  start_latents=...)`. Continuation state is
  `cat([re-encoded decoded tail slice, previous latents[:, -(overlap-1):]])`.
  `num_overlap_frames` defaults to 3 and **must be divisible by
  `num_frame_per_block`** (asserted) — overlap is counted in blocks.
  Committed per rollout: all but the last `4*(overlap-1)+1` decoded frames
  (9 tail frames dropped at overlap 3 → **72 novel frames/rollout**);
  output written with `export_to_video(..., fps=16)`.
- Voyage fps policy (DESIGN §5.4, unchanged): native 16 fps end-to-end or a
  separately validated resample stage — never relabel 16 fps media as 24.

## Open questions for the worker slice

1. Exact `model.pt` bytes → replace the 1 GB floor with a measured
   threshold after the first real download.
2. `models download/verify causvid-*` CLI wiring in `voyage/cli.py`
   (registry functions already exist).
3. Checkpoint choice: `autoregressive_checkpoint` vs
   `autoregressive_checkpoint_warp_4step_cfg2` vs `bidirectional_checkpoint2`
   — benchmark which the worker loads.
4. Overlap sweep 1, 2, 3 (+ larger, in block units) per the DESIGN §5.4
   benchmark list; confirm the 72-novel-frames accounting on pinned code.
5. VAE tail re-encode device/dtype placement on 16 GB (script uses bf16
   CUDA); `torch.compile` only after an eager correctness baseline exists.
6. Recovery tape shape: reconstructable `start_latents` + tail metadata vs
   stored latent blocks — prefer the smallest reconstructable artifact
   (DESIGN §5.4).
