# 048 — LongLive `generate_blocks` holds latents + pixels + uint8 + frame-list simultaneously

**Severity:** MEDIUM

**File:line:** `voyage/workers/video_longlive.py:892` (`torch.cat`), `921-951` (decode block), `963-972` (write block)

**Description:**
After denoise: `latents = torch.cat(block_latents)` (full-segment bf16 latents retained) → full-segment `decode_to_pixel_chunk(chunk_size=T)` (comment at 921-927 admits the VAE transient is ~10 GiB regardless of chunk size — then passes full T anyway) → `video = (255*rearrange(generated,...).cpu()).to(uint8)` (second full copy, now uint8) → `frames = [video[0,i].numpy() ...]` (third: list of N numpy arrays while `video` is still alive) → `get_writer` append loop. At 1280×704×93f that is latents + fp pixels transient + uint8 + list co-resident. `chunk_size=int(latents.shape[1])` is literally "no chunking" despite the parameter name; VAE tiling/slicing (`enable_vae_tiling/slicing`) is never enabled — the diffusers production fix for exactly this bottleneck.

**Rationale:**
The file already documents VAE-offload-for-generate and generator-offload-for-decode wins with measured GiB; the remaining 2–3× copy fan-out is unmeasured because `profile_stages` covers CUDA-event ms but not peak bytes per stage (see 051).

**Live evidence (current tree):**
```python
video_longlive.py:892: latents = torch.cat(block_latents, dim=1)
video_longlive.py:921-927: # Full causal decode (93f per 3-block segment): the VAE transient at
    # 1280x704 is ~10 GB regardless of chunk size ...
video_longlive.py:949: generated = pipe.vae.decode_to_pixel_chunk(
video_longlive.py:950:     latents, use_cache=False, chunk_size=int(latents.shape[1]))
video_longlive.py:963: video = (255.0 * rearrange(generated, "b t c h w -> b t h w c").cpu()).to(torch.uint8)
video_longlive.py:965: del latents, generated, block_latents  # only AFTER all copies exist
video_longlive.py:967: frames = [video[0, index].numpy() for index in range(video.shape[1])]
video_longlive.py:968-972: with imageio.get_writer(...) as writer: for frame in frames: writer.append_data(frame)
```
`del` at 965 runs after `video` exists; `frames` keeps N arrays alive while `video` is alive. Arithmetic: 93f × 1280×704×3 uint8 ≈ 240 MiB × 2 (video + frames payloads) + bf16 latents + fp pixels transient.

**Repro:** Code-reading + arithmetic above; `rg enable_vae_tiling voyage/` → no hits in workers.

**Fix candidates:**
- True chunked decode (`chunk_size=8/16`) with per-chunk `rearrange→uint8→numpy→writer.append` + `del` + `gc.collect()` per chunk; never `torch.cat` the full latents (decode per `block_latents` chunk and concatenate on disk).
- `enable_vae_tiling()` / slicing probe on the 16 GiB card with measured peaks per `stage_ms` + `max_memory_allocated` per stage.
- Stream `writer.append_data` inside the chunk loop instead of materializing `frames`.

**Refs:** diffusers advanced VRAM article ("VAE decode becomes the bottleneck… `enable_vae_tiling()` splits pixelspace"); `video_longlive.py:828-833,876-879` measured evict discipline.

**Overlaps with:** 043/044/045/049 (all-at-once RAM/VRAM cluster — same chunked-decode fix pattern; not duplicates).
