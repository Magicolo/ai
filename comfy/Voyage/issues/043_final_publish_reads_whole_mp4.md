# 043 — Final publish loads the entire final MP4 into RAM (`read_bytes`)

**Severity:** HIGH

**File:line:** `voyage/media.py:1044`, `voyage/media.py:1115`; `voyage/sfx_finalize.py:559`; `voyage/atomic.py:37-58`

**Description:**
Both `finalize_run` fast-path (stream-copy concat) and re-encode path publish via `atomic_write_bytes(output_path, staged.read_bytes())`. `atomic_write_bytes(destination: Path, data: bytes)` takes `bytes`, so the whole staged `final.mp4` is materialized in the Python heap before the atomic write. The SFX remux does the same (`remuxed.read_bytes()` at `sfx_finalize.py:559`). A long voyage (hundreds of segments at 1280×720@32 h264) stages hundreds of MB; this holds staged file + `bytes` object + temp copy inside `atomic_write_bytes` simultaneously — 2–3× peak host RAM for no reason.

**Rationale:**
Atomic-publish discipline is correct, but the API forces all-at-once. Chunked copy (`shutil.copyfile` to `.partial` + `flush`/`fsync`/`os.replace`/`fsync_dir`) is constant-memory and crash-identical. Never hold a whole artifact in RAM when a streaming copy exists.

**Live evidence (2026-09-30, current tree):**
```
voyage/media.py:1044:             atomic_write_bytes(output_path, staged.read_bytes())
voyage/media.py:1115:         atomic_write_bytes(output_path, staged.read_bytes())
voyage/sfx_finalize.py:559:         atomic_write_bytes(final_path, remuxed.read_bytes())
voyage/atomic.py:37: def atomic_write_bytes(destination: Path, data: bytes) -> None:
voyage/atomic.py:39-47: mkstemp(.partial) + handle.write(data) + flush + fsync + replace + fsync_dir
```
`rg read_bytes voyage/media.py voyage/sfx_finalize.py` → 3 hits, all on staged finals (config.py/tui_state.py hits are tiny TOML files; tests hits are assertions). No `shutil.copy` in `media.py`. `ffmpeg 8.0.1` present (host probe).

**Repro (CPU, no GPU):**
```python
from pathlib import Path
import voyage.media as m, inspect
src = inspect.getsource(m.finalize_run)
assert "staged.read_bytes()" in src
assert "shutil.copy" not in src
```

**Fix candidates:**
- Add `atomic_copy(src: Path, dst: Path)` (temp `.partial` + 1 MiB `shutil.copyfileobj` + `flush`/`fsync`/`replace`/`fsync_dir`) and call it from all three sites.
- Alternatively stream `staged.open('rb')` through the existing `mkstemp` fd in chunks.
- Add a test asserting `finalize_run` never calls `read_bytes` on files > 64 MiB (mock `Path.read_bytes` to fail).

**Refs:** `voyage/atomic.py:37-58`; `voyage/media.py:1006-1045` (fast path), `1046-1115` (re-encode path); ffmpeg concat guidance (stream-copy vs re-encode).

**Overlaps with:** 102 (finalize concat escape/RAM/tmp — sibling finalize-RAM defect, same fix `atomic_copy` helps both; not a duplicate).
