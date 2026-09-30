# 096 — `validate_video` divides by zero on `avg_frame_rate "0/0"` (sibling catches it) and exempts `nb_frames == 0` from frame-count validation

- Severity: LOW
- Area: correctness / media / validation
- File: `voyage/media.py:101-126` (`validate_video` fps + frames; `def` at `:101`, parse at `:114-121`), `voyage/media.py:128-141` (`_probe_video_fps` sibling)

## Description

Live code (re-verified 2026-09-30):

```python
# voyage/media.py:101-121
rate = str(video.get("avg_frame_rate", "0/1"))
num, _, den = rate.partition("/")
actual_fps = float(num) / float(den or 1) if num else 0.0   # <-- ZeroDivisionError on "0/0"
if abs(actual_fps - fps) > 0.5:
    raise MediaError(...)

# voyage/media.py:119-121
frames = int(video.get("nb_frames", 0) or 0)
if frames < min_frames and frames != 0:                     # <-- 0 carve-out
    raise MediaError(f"too few frames in {path}: {frames}")
```

Two quirks:

1. **`"0/0"` crash.** `num="0"` is truthy (non-empty string), `den="0"` is truthy, so `den or 1` stays `"0"` and `float("0")/float("0")` raises `ZeroDivisionError` — not a `MediaError`. The sibling `_probe_video_fps` (`:128-141`) does the *same* parse inside `try: ... except (ValueError, ZeroDivisionError): return 0.0`. `validate_video` has no such guard, so a file ffprobe reports as `"0/0"` (corrupt header, zero-duration, some mkv) crashes the commit/finalize with an unclassified exception instead of `MediaError(fps mismatch ...)`. Falls into `run_segments`' generic `except Exception → FatalWorkerError` (misleading class) instead of the media path.
2. **`nb_frames == 0` exempt.** `frames != 0` carve-out means "ffprobe did not report a count" skips the `min_frames` gate entirely. For mp4 (which normally reports `nb_frames`), a 0 means corrupt/unknown — yet it passes. The carve-out exists for containers that legitimately omit the count (mkv/ffv1 paths report 0 while duration is valid), but applied unconditionally it lets a truncated mp4 with `nb_frames=0` + valid duration slip through.

## Rationale

- Low severity (needs a corrupt file to trigger), but the fix is a two-line alignment with the sibling that already handles it correctly. The taxonomy break (raw `ZeroDivisionError` escaping `MediaError`) is the same family as 018.
- The `0` carve-out weakens the commit gate (Phase 6 slice A "frame/duration checks") for exactly the files most likely to be truncated.

## Live evidence

Stdlib probe (host, pure string math — no ffmpeg needed):

```
$ PYTHONPATH=Voyage python3 - <<'PY'
num, _, den = "0/0".partition("/")
print(num, den, bool(num), den or 1)
try: float(num)/float(den or 1) if num else 0.0
except Exception as e: print(f"{type(e).__name__}: {e}")
PY
0 0 True 0
ZeroDivisionError: division by zero
```

Source contrast (live):

```
$ sed -n '114,142p' comfy/Voyage/voyage/media.py
    rate = str(video.get("avg_frame_rate", "0/1"))
    num, _, den = rate.partition("/")
    actual_fps = float(num) / float(den or 1) if num else 0.0
...
def _probe_video_fps(info: dict[str, Any]) -> float:
    ...
    try:
        return float(num) / float(den or 1) if num else 0.0
    except (ValueError, ZeroDivisionError):
        return 0.0
```

- Identical expression; one guarded, one not.

## Repro

1. Unit: call `validate_video` with stubbed `probe` returning `{"streams":[{"codec_type":"video","width":W,"height":H,"avg_frame_rate":"0/0","nb_frames":"29"}],"format":{"duration":1.2}}` → `ZeroDivisionError`, not `MediaError`.
2. Unit: same stub with `"nb_frames":"0"`, `min_frames=29` → passes (no `too few frames`), while a sibling mp4 with `"nb_frames":"5"` correctly fails.
3. Real file: craft a zero-duration mp4 (or truncate one) until ffprobe reports `avg_frame_rate=0/0` → `voyage finalize` tracebacks instead of `finalize failed:`.

## Fix candidates

1. (Preferred) Copy the sibling guard into `validate_video`:
   ```python
   try:
       actual_fps = float(num) / float(den or 1) if num else 0.0
   except (ValueError, ZeroDivisionError):
       raise MediaError(f"unparseable fps in {path}: {rate}") from None
   ```
2. Narrow the `0` carve-out: exempt only when the container legitimately omits counts (e.g. `codec_name`/format allow-list, or `duration`-based fallback with an explicit comment), not unconditionally. At minimum, require `duration`-derived frame estimate (`duration * fps >= min_frames - slack`) when `nb_frames == 0`.
3. Regression tests: `"0/0"` → `MediaError`; `"0/1"` → fps mismatch or 0.0-handling; `nb_frames="0"` mp4 with `min_frames>0` → `MediaError` (or duration-backed pass with justification).

## Refs

- Overlaps with 018 (error-taxonomy family — raw `ZeroDivisionError` escaping `MediaError`) — ownership stays here (fps/frames quirks).
- `voyage/media.py:101-126` (unguarded) vs `:128-141` (guarded sibling — the pattern to copy).
- Python truthiness — non-empty `"0"` is truthy, so `if num` does not catch `"0"`: https://docs.python.org/3/library/stdtypes.html#truth-value-testing
- `float("0")/float("0")` → `ZeroDivisionError`: https://docs.python.org/3/library/exceptions.html#ZeroDivisionError
