# 125 — LongLive decodes VAE output to uint8 with no clip: highlight overshoot wraps to black (the 065 bug its siblings fixed)

- **Severity:** Medium (render correctness — bright highlights can speckle near-black on the default backend path)
- **File:line:** `Voyage/voyage/workers/video_longlive.py:963` (`video = (255.0 * rearrange(generated, "b t c h w -> b t h w c").cpu()).to(torch.uint8)` — bare cast, no clip); contrast the fixed siblings `Voyage/voyage/workers/video_ltxv.py:740-750` (`_save_mp4` → `video_common.clip_array_to_uint8`, comment cites the wrap) and `Voyage/voyage/workers/video_causvid.py:659-661` (`np.clip(frames_any * 255.0, 0, 255)`).
- **Area:** workers-internals tail — longlive VAE-decode write path (below pass-1 coverage; 065 fixed ltxv+causvid only)

## Description

VAE decoders overshoot `[0, 1]` on highlights (values like 1.01 are routine). The `.cpu()` in line 963 means the `.to(torch.uint8)` cast executes with **CPU cast semantics, which wrap modulo 256** — 1.01 × 255 = 257.55 → wraps to ~1-2, i.e. near-black. A bright pixel becomes a black pixel: the exact failure the tree already diagnosed and fixed twice ("1.01 becomes near-black", `video_common.py:252-258`, `video_ltxv.py:745-746`, causvid `:653-661`). Longlive's decode path was written before (or without) that lesson and never adopted it — `grep -n "clip" voyage/workers/video_longlive.py` finds only the `clip` in `reencode_window_frames` naming, no value clipping on the write path.

Notably the in-tree rationale for the sibling fixes applies verbatim here: the tensor being converted is VAE output in nominal `[0, 1]` with real overshoot, and the conversion target is uint8 video frames. There is no intervening normalization between `decode_to_pixel_chunk` (`:948-951`) and the cast.

## Rationale

Upstream evidence for CPU wrap semantics: pytorch/pytorch#169058 documents that CPU float→int casts wrap/truncate (their example: float 128 → int8 -128 on CPU, vs saturating on MPS) — i.e. out-of-range float→int conversion on CPU is modulo behavior, exactly the numpy `.astype` wrap the siblings guarded against. Whether any given highlight overshoots far enough to be visible depends on content (the slice-4 solarization work shows this model's highlights do blow out), but the defect is structural: two of three video backends clip, the third casts bare, and nothing in the longlive path makes its VAE output more bounded than its siblings'.

## Evidence (verified live 2026-09-30)

- Tree reads: longlive `:963` bare `.to(torch.uint8)`; ltxv `:745-746` `clip_array_to_uint8` with the "wraps modulo 256" comment; causvid `:660` `np.clip(..., 0, 255)`; `video_common.py:252-258` documents the 1.01→near-black mechanism.
- Upstream semantics: pytorch/pytorch#169058 — "contradicting the wraparound/truncation behavior observed on CPU" (CPU wraps; the cast in `:963` runs on CPU because of the preceding `.cpu()`).
- In-tree precedent that VAE overshoot is real on this stack: the slice-4 fp8-solarization work (DESIGN §140) measured blown highlights on the same class of decoder output.

## Repro

1. Static: compare the three write paths (`:963` vs ltxv `:740-750` vs causvid `:651-661`).
2. Numeric (needs torch, i.e. the video image): `torch.tensor([1.01 * 255]).to(torch.uint8)` on CPU → wraps to a small value instead of 255; then render any high-key segment on longlive and inspect highlight pixels vs the same prompt on ltxv.
3. The one-line fix makes the repro moot: route `:963` through `video_common.clip_array_to_uint8` (numpy, already imported in the function via `frames = [... .numpy()]` at `:967` — or clip in torch with `.clamp(0, 255)` before the cast).

## Fix candidates

1. `video = video_common.clip_array_to_uint8((255.0 * rearrange(...).cpu().numpy()))` — one call, identical to the ltxv path (module already imports `video_common`).
2. Or torch-native: `(...).clamp_(0, 255).to(torch.uint8)` — no numpy round-trip; either is fine, pick one and note why.
3. Regression test (video-image or stubbed): a synthetic `[0, 1.05]` decoded tensor converts with 255s where the siblings put 255s — pin all three backends to the same conversion (the 065 clip test shape, extended to longlive).

## Refs

- `Voyage/voyage/workers/video_longlive.py:945-975`; `Voyage/voyage/workers/video_common.py:252-258`; `Voyage/voyage/workers/video_ltxv.py:734-750`; `Voyage/voyage/workers/video_causvid.py:651-661`.
- pytorch/pytorch#169058 (CPU float→int wraparound vs MPS saturation).
- Adjacent, not overlapping: 065 (the ltxv/causvid clip fix — this file is the longlive instance it missed); slice-4 solarization notes in DESIGN §140 (model-side blowout — this file is the *conversion* side).
