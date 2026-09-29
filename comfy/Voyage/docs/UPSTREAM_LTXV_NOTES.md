# UPSTREAM_LTXV_NOTES — LTXV integration notes (Stream A)

Pins live in code in `voyage/model_registry.py` (`LTXV_*`, the single
source of truth); this file mirrors them for humans. Revisit every note
if a pin moves.

Status: worker landed. `voyage/workers/video_ltxv.py` serves the `ltxv`
backend (Stream-A 121/25/96 accounting, `video_tail.mp4` + §5.3 JSON
tape); `generate --backend ltxv` is the default.

## Upstream pins

| Artifact | URL | Pinned revision |
|----------|-----|-----------------|
| Code | [Lightricks/LTX-Video](https://github.com/Lightricks/LTX-Video) | `4b2d053057623ddd4d0a1d3e9cd28890e9ef487f` (short `4b2d053`) |
| DiT + upscaler | [Lightricks/LTX-Video](https://huggingface.co/Lightricks/LTX-Video) | `8984fa25007f376c1a299016d0957a37a2f797bb` |
| Text encoder | [PixArt-alpha/PixArt-XL-2-1024-MS](https://huggingface.co/PixArt-alpha/PixArt-XL-2-1024-MS) | `b89adadeccd9ead2adcb9fa2825d3fabec48d404` |

Installed with `--no-deps` (`[inference]` extra) in
`worker/Dockerfile.video`; the T5 path used here is stable across 4.x.

## Stream-A accounting (DESIGN §5.3)

Every segment renders 121-frame clips; fresh blocks commit all 121,
conditioned blocks drop the 25-frame prefix and commit 96 novel
(`_LTXV_NOVEL_BLOCK_FRAMES * blocks` in `voyage/cli.py` plans the
steady-state 96 so `generate --duration` never runs short).
`video_tail.mp4` (last 25 committed frames) beside the segment video is
the crash-recovery anchor; `recovery.pt` carries the §5.3 JSON record
(clean break from old torch-pickle tapes). Native 768×512 (432 % 32
!= 0 pads, so 512 stands); draft 640×352 verified.

## License implications

Check the weight repo for the exact Lightricks community-license text
before redistributing weights or images (`model_registry.py` records
the pointer; the run manifest records the selection).
