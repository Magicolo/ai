# `requirements-gpu.txt` mixes bare, floor, and exact pins

- Severity: medium (dependency drift on the GPU stack).
- Status: verified open. `Zoomy/requirements-gpu.txt:19-44`.

## Evidence

`requirements.txt` (3 lines) and `requirements-dev.txt` (4 lines) are
fully exact — but the GPU file, the one that decides what actually
renders, carries:

- bare: `hf_transfer`, `protobuf`, `ccvfi`, `pyyaml`, `tqdm`
- floors: `huggingface_hub>=0.34`, `loguru>=0.7.3`, `soundfile>=0.13.1`,
  `scipy>=1.10.1`, `numba>=0.63.1`, `librosa>=0.10.1`,
  `torchdiffeq>=0.2.3`, `timm>=1.0.8`, `omegaconf>=2.3.0`,
  `open_clip_torch>=2.29.0`, `ftfy>=6.1.1`
- exact: only 8 (`diffusers==0.40.0`, `transformers==5.17.0`, …)

`ccvfi` bare is the sharpest edge: RIFE code drift breaks
`_interpolate_frames` with no version record of what worked. The file
header already claims "every pin below matches the spike-unify image" —
make that claim true and machine-checked.

## Fix

Exact-pin everything to the versions in the last green image
(`pip freeze` from the spike-unify/proven image), same discipline as the
other two files. When a floor is deliberate (e.g. CUDA-adjacent ranges),
say so in a comment per line — unexplained floors are the violation.

## Verification

- Fresh rebuild installs byte-identical stack (`pip freeze` diff empty vs
  record); frame + finalize e2e still green on the 16 GB card.
- Gates: `Zoomy/scripts/quality-gates.sh` green.
