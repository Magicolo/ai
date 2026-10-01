"""Worker entry points: JSONL over stdin/stdout (DESIGN §§45-46).

Each worker is a long-lived process started by the supervisor via
`SubprocessWorker`. Diagnostics go to stderr; stdout carries only RPC.

- `voyage.workers.video`: fake video backend (CPU).
- `voyage.workers.video_ltxv`: LTXV video backend (CUDA, default).
- `voyage.workers.video_causvid`: CausVid video backend (CUDA).
- `voyage.workers.video_ltx25`: LTX-2.5 GGUF video backend (CUDA).
- `voyage.workers.audio`: fake audio backend (CPU).
- `voyage.workers.audio_acestep`: ACE-Step music backend (CUDA).
- `voyage.workers.sfx` / `sfx_mmaudio`: fake / MMAudio SFX backends.
- `voyage.workers.director`: evolution decisions (deterministic + Qwen).
- `voyage.workers.augment_worker`: finalize-time augment pass (FILM/Real-ESRGAN).
"""

from __future__ import annotations
