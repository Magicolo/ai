# Local-model failures silently fall back to official weights

- Severity: medium (observability — the user is never told their file failed).
- Status: FIXED. Local transformer failures (Ernie + Z) now raise
  `EngineConfigurationError` naming the file and the caught error — no
  silent swap to official weights (the Z loader also gained the
  `EngineConfigurationError` pass-through the Ernie loader already had).
  Local VAE failures keep the documented official-VAE fallback but announce
  it with `warnings.warn` naming file, error, and fallback. Design note:
  progress-stream surfacing was deferred — `render_frame` returns a plain
  `Image` by protocol, so no progress stream exists in the frame path;
  fail-loud errors reach the UI via the existing error path and warnings
  land in the server log. Tests (sys.modules fakes, no heavy imports):
  `test_broken_local_transformer_fails_loud`,
  `test_broken_local_autoencoder_warns_and_uses_official`.

## Evidence

```python
try:
    transformer = ErnieImageTransformer2DModel.from_single_file(...)
except Exception:  # noqa: BLE001
    # Any load failure (missing keys, dtype mismatch) falls back
    # to the official weights below; the error resurfaces there.
    transformer = None
```

A corrupt/truncated local DiT, an OOM during load, or a genuinely
incompatible file all funnel into `None`, and the pipeline then downloads
and runs the *official* weights — including burning hub bandwidth the
local file was meant to avoid. The comment claims "the error resurfaces
there", but it does not: the official load succeeds, so the local failure
vanishes from logs and UI entirely. Same shape for the Z transformer and
the VAE channel-mismatch fallback (the VAE case is closer to legitimate —
a known 32ch-vs-8ch config gap documented in AGENTS.md §10 — but it is
still silent).

## Fix

Log the local failure at warning level (see `logging` gap — there is
currently zero logging repo-wide) AND surface it in the progress stream;
for the transformer cases (not the known VAE channel gap), consider
failing loud instead of falling back — a corrupt 12 GB DiT rendering with
different weights is worse than an error. At minimum: `warnings.warn` +
yielded message naming the file and the caught error. Distinguish the
known-benign VAE channel mismatch (keep fallback, but announce it) from
unexpected failures (raise).

## Verification

- New tests: corrupt local transformer → user-visible warning/error naming
  the file (no silent success); VAE channel mismatch → fallback WITH a
  visible notice.
- Gates: `Zoomy/scripts/quality-gates.sh` green.
