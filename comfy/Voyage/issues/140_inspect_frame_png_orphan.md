# 140 — Visual-inspector frame PNG lands untracked inside the committed segment

- Severity: LOW
- Area: observability — visual inspector artifact hygiene
- Files (as-read 2026-09-30):
  - `voyage/supervisor.py:1343-1371` (`_inspect_frame_view` — ffmpeg-extracts `inspect_frame.png`)
  - `voyage/supervisor.py:1313-1341` (`_run_previous_inspect` — merges `visual` into `metrics.json`)
  - `voyage/cli.py:825` (`validate_segment` — unreadable-`metrics.json` check, no PNG awareness)

## Technical description

When the experimental visual inspector fires, `_inspect_frame_view`
(`voyage/supervisor.py:1343-1371`, as-read) extracts one middle frame into the *previous*
segment's own directory:

```python
# voyage/supervisor.py:1346-1362 (as-read, abridged)
info = probe(prev_video).get("format", {})
duration = float(info.get("duration", 0.0) or 0.0) ...
frame_path = prev_dir / "inspect_frame.png"
proc = run_capture(["ffmpeg", "-hide_banner", "-nostdin", "-y",
                    "-ss", f"{max(duration / 2.0, 0.0):.6f}",
                    "-i", str(prev_video), "-frames:v", "1", str(frame_path)])
if proc.returncode != 0:
    return ""
```

The PNG is never cleaned up, never added to `sha256.json`, never checked by `_verify_segment`
(which pins only `video.mp4`/`audio.wav` + `metrics.json` frames), never mentioned in
`validate_segment` (`voyage/cli.py:817-825` checks `metrics.json` readability / frame counts /
durations only), and is invisible to the finalize path. Every inspected segment therefore gains
one permanent untracked sibling (~0.5-2 MB at 1280×704) inside the committed, checksummed
directory. With the inspector on for an infinite run, that is unbounded growth of files no
reader owns — and any future strict orphan scan (see issue 098) would flag every inspected
segment, since the file is neither an artifact nor in an allow-list.

The `visual` merge itself (`:1334-1339`: read `metrics.json`, `{**existing, "visual": visual}`,
`atomic_write_json`) is correctly atomic; only the frame PNG is orphaned.

## Why this is an issue

- Breaks the "committed segment dir contains exactly the checksummed artifacts" invariant the
  state-integrity work (Phase 6 slice A) established.
- Disk growth scales with segments on exactly the runs that run longest (inspector-on voyages).
- A future orphan-scan allow-list written without knowledge of this file will either miss it
  (gap persists) or flag it (false-positive storm on inspector runs).

## Live evidence

Live re-verification 2026-09-30 (read-only; probes per task brief ran in `voyage:latest` CPU-only).
Track A draft command+output bundle (`ses_f0fbea412ffeh3V7K1SlQpXjsv`) was not recoverable from
this writer's context, so evidence below is the as-read code, not invented command output:

```
$ sed -n '1343,1379p' voyage/supervisor.py
  1343:    def _inspect_frame_view(self, prev_dir: Path, prev_video: Path) -> str:
  1348:            frame_path = prev_dir / "inspect_frame.png"
  1349:            proc = run_capture([... "-y", "-ss", ..., "-i", str(prev_video),
                                      "-frames:v", "1", str(frame_path)])
  1364:            if proc.returncode != 0:
  1365:                return ""
  # no unlink / tmp / sha registration anywhere in 1343-1379 (as-read)

$ sed -n '817,825p' voyage/cli.py
  817:    errors: list[str] = []
  818:    metrics_path = segment / "metrics.json"
  824:    except (ValueError, KeyError, AttributeError, TypeError):
  825:        return [f"{segment.name} has unreadable metrics.json"], 0
  # no inspect_frame.png handling (as-read)
```

## Minimal repro

1. Enable `[experimental] visual_inspector = true` with a fake or GPU run; commit ≥2 segments.
2. `ls segments/000000/` → contains `inspect_frame.png` alongside `video.mp4`/`audio.wav`/
   `metrics.json`/`sha256.json`/`DONE`.
3. `sha256.json` has no `inspect_frame.png` key; `validate --run` reports no error about it;
   deleting it changes nothing — pure orphan.
4. Long-run projection: N inspected segments → N orphan PNGs.

## Fix candidates

1. (Preferred) Extract to a non-segment location: `logs/inspect/<segment>.png` or a
   `TemporaryDirectory` passed into `_inspect_frame_view`; keep segment dirs checksum-clean.
   Optionally log the chosen path in the `segment_inspected` metric for debugging.
2. Delete-after-read: `try/finally: frame_path.unlink(missing_ok=True)` once the `inspect` RPC
   returns. One-line fix, but loses the view artifact for later debugging.
3. Register-or-allow-list: if the PNG must persist in-segment, add it to `sha256.json` at commit
   or to the orphan-scan allow-list (issue 098) with a size cap. Heaviest option; only if the
   frame views are a product requirement.
4. Regression tests: inspector-on fake run → no `inspect_frame.png` under `segments/` (options
   1/2) or checksummed + allow-listed (option 3); orphan scan green on an inspected run.

## References

- In-tree: `voyage/supervisor.py:1300-1379`; `voyage/media.py:324-360` (`_verify_segment`);
  `voyage/cli.py:810-850` (`validate_segment`); `voyage/paths.py` (`SEGMENTS_DIRNAME`).
- Neighbor issues: 055 (inspect metrics rotation-blind), 062 (scoreboard hidden partials),
  095 (metrics without checksum), 098 (orphan-scan gaps — the scan this file would trip).
- External:
  - https://ffmpeg.org/ffmpeg.html#Main-options (`-ss` seek + `-frames:v 1` single-frame extract)

## Investigation log

- 2026-09-30: filed by Track A sweep; live re-verified via Read (concurrent uncommitted edits
  noted in `voyage/cli.py`, `voyage/tui_state.py`, `tests/test_generate.py` — citations are
  as-read values above).
