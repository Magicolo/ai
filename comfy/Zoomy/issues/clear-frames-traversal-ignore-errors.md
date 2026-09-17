# `clear_frames` / `frame_directory`: unsanitized key + silent delete failures

- Severity: medium (validation + misleading success message).
- Status: FIXED. `_check_sequence_key` allowlist (`[A-Za-z0-9_-]+`, the
  catalog's key shape) enforced in all seven path-producing methods
  (`frame_directory`, `assembly_directory`, `segment_twin_paths`,
  `segment_stem_paths`, `next_video_stem`, `remove_segment_files`,
  `latest_video_path` — every other keyed method routes through these, and
  the allowlist also neutralizes glob injection). `clear_frames` returns on
  missing dirs and raises `ZoomyError` on failed deletes; the handler
  catches `(OSError, ZoomyError)` into the `**Error:**` shape. Tests:
  7-param traversal rejection, missing-dir no-op, file-at-dir failure at
  both repo and handler level.

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
