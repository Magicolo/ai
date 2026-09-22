"""Worker entry points: JSONL over stdin/stdout (DESIGN §§45-46).

Each worker is a long-lived process started by the supervisor via
`SubprocessWorker`. Diagnostics go to stderr; stdout carries only RPC.

- `voyage.workers.video`: video generation (fake backend in Phase 0,
  LongLive adapter in Phase 1/2).
- `voyage.workers.audio`: audio generation (fake backend in Phase 0,
  ACE-Step adapter in Phase 4).
- `voyage.workers.director`: evolution decisions (deterministic
  fallback in Phase 0, Qwen3-8B in Phase 3).
"""

from __future__ import annotations
