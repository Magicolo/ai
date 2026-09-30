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

## Progress log

- 2026-09-30: re-verified every premise against live code before touching anything — all hold as-read: `voyage/media.py:1044` + `:1115` both `atomic_write_bytes(output_path, staged.read_bytes())`, `voyage/sfx_finalize.py:563` `atomic_write_bytes(final_path, remuxed.read_bytes())`, `voyage/atomic.py:37-58` bytes-taking API, no streaming publish in `media.py` (`shutil` there only serves disk-usage + the take-slice memo copy).
- 2026-09-30 (TDD red): wrote `Voyage/tests/test_media_memory.py` first; `test_atomic_copy_is_byte_identical` failed with `ImportError: cannot import name 'atomic_copy'` in-container, as required.
- 2026-09-30 (implement): added `COPY_CHUNK_BYTES = 1 MiB` + `atomic_copy(source, dst, *, chunk_bytes)` to `voyage/atomic.py` (sibling `.partial` temp + `shutil.copyfileobj` + flush/fsync/replace/fsync_dir; returns `dst`; same BaseException-cleanup discipline as `atomic_write_bytes`, which is untouched). Switched both `media.py` publish sites to `atomic_copy(staged, output_path)`; `sfx_finalize.py:563` deliberately left (out of scope — follow-up for its owning pass, see Residuals).
- 2026-09-30 (TDD green): 21 passed + 1 torch-gated skip in-container. Two test-side fixes during green-up (narrowed the `read_bytes` mock-gate to `.mp4` only is inherent; fixed a monkeypatched-`subprocess.run` recursion and a wrong `match=` string — both in the new test file, no source impact).
- 2026-09-30 (gates): `ruff check` + `ruff format --check` + `mypy` (project strict config) all clean on the six touched files; related suites 119 passed (`test_finalize_fastpath`, `test_vision_metrics`, `test_issue_032_commit_fanout`, `test_sfx_contract`, `test_sfx_finalize`, `test_final_blend_scale`, all audio suites); full suite 1167 passed, 2 failed — both foreign and unrelated (TUI Pilot viewport `OutOfBounds` in `test_tui_app`, a known-flaky class in another group's files; `test_ltxv` serve-map passes in isolation while its owning group edits `video_ltxv.py`).

## Resolution

- Verdict: fixed in scope (both `media.py` sites stream now). Public signatures unchanged (`atomic_write_bytes` kept; `finalize_run` signature untouched for the supervisor track).
- Files changed: `Voyage/voyage/atomic.py` (+`COPY_CHUNK_BYTES`, +`atomic_copy`), `Voyage/voyage/media.py` (import + 2 publish sites), `Voyage/tests/test_media_memory.py` (new: byte-identical 5 MiB copy, empty-file round-trip, missing-source leaves no `.partial`, fast-path + re-encode finalize runs with `Path.read_bytes` mocked to raise on any `.mp4`).
- Test evidence: `test_finalize_fastpath_publishes_without_read_bytes` (768x432@24 native, duration ≈4.0 s) and `test_finalize_reencode_path_publishes_without_read_bytes` (default floors → 1280x720@32 re-encode) both pass under the mock-gate — the publish path provably never materializes staged MP4 bytes.
- DESIGN.md as-built proposal (not applied — DESIGN.md untouched per directive): in §31, after the atomic-write rule, add "`atomic_copy(src, dst)` publishes large artifacts (final MP4s) with identical crash semantics at constant memory (1 MiB chunks); `atomic_write_bytes` stays for small state files." In §56 step 7 (publish), change "atomically publish" to "atomically publish via `atomic_copy` (never `read_bytes` the staged final)".
- Residuals / follow-ups: `voyage/sfx_finalize.py:563` still publishes via `read_bytes` (out of scope — needs its owning pass to switch to `atomic_copy`); chunk size is a fixed 1 MiB (no tuning knob — raise only with measured evidence).
