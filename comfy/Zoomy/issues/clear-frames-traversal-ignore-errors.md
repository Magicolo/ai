# `clear_frames` / `frame_directory`: unsanitized key + silent delete failures

- Severity: medium (validation + misleading success message).
- Status: verified open. `zoomy/frame_repository.py:61-63,98-100`.

## Evidence

```python
def frame_directory(self, sequence_key: str) -> Path:
    return self.output_directory / sequence_key
...
def clear_frames(self, sequence_key: str) -> None:
    shutil.rmtree(self.frame_directory(sequence_key), ignore_errors=True)
```

Two gaps at one boundary:

1. `sequence_key` is path-joined unsanitized. Callers today pass catalog
   constants, but `sequence_key` arrives from the same trust domain as the
   dropdown key that `find_family` validates — and `clear_frames` is the
   one operation that *deletes*. A `../` key escapes `output_directory`.
2. `ignore_errors=True` means a real delete failure (permissions,
   root-owned bind-mount files — the exact situation this repo hits, see
   AGENTS.md §8) still reports "Cleared all frames…" in the UI
   (`interface.py:433`).

## Fix

Validate at the boundary: reject keys containing separators/`..`/empty
(abs-path check included) with `ZoomyError`/`ValueError` in
`frame_directory` (single choke point — every method routes through it).
Replace `ignore_errors=True` with `missing_ok`-style semantics: missing
directory is fine, but a failed delete raises (or returns a success flag
the handler reports honestly). Tests: `../` key raises; read-only
directory reports failure instead of "Cleared".

## Verification

- New tests: traversal key rejected; undeletable directory does not claim
  success.
- Gates: `Zoomy/scripts/quality-gates.sh` green.
