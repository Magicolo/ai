# Local-model failures silently fall back to official weights

- Severity: medium (observability — the user is never told their file failed).
- Status: verified open. `zoomy/local_engine.py:677-680` (Ernie
  transformer), `:727-729` (Z transformer), `:739-741` (autoencoder).

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
