# 027 — `scoreboard_rows` inverts robustness: crashes on malformed metrics, advertises nonexistent media, deltas skip gaps, headers cryptic

- Severity: MEDIUM (read-only view fails the table on one bad row)
- Group: observability/scoreboard — Rank: 3/5
- File:line: `voyage/scoreboard.py:47-74` (`_stages_by_segment`), `voyage/scoreboard.py:77-128` (`scoreboard_rows`)
- Overlaps: 062 (scoreboard) — same module/same root cause; recommend keep 062, fold 027 on fix. Not a duplicate of 028.

## Description

(a) `float(visual["metrics"][key])` at line 97 raises on a malformed value — a single hand-edited `metrics.json` kills the whole scoreboard with `ValueError` (zero rows instead of one error cell).

(b) `video_path`/`audio_path` are synthesized unconditionally (`str(segment / "video.mp4")`, `str(segment / "audio.wav")`) even when the files are absent — the CLI then prints `view: …` for missing artifacts.

(c) `previous` only advances when `current is not None`, so a segment without visuals inherits a stale baseline — the delta is attributed to the wrong predecessor (skipped gap, wrong diff).

(d) Header truncates metric names to 12 chars (`motion_energ`, `visual_compl`, … — in the CLI renderer) with no legend; `frames` can be `None` (missing `video.frames`) and prints as `None`.

The hardened `_stages_by_segment` (skips torn lines, rotation-aware) shows the intended pattern — the row builder doesn't follow it.

## Rationale

- A read-only observability view must degrade per-row, never fail the table; the current code lets one corrupt segment hide every healthy one — exactly when the user most needs the scoreboard.
- Advertising missing files as viewable sends the user to a dead path and erodes trust in the view layer.
- Stale-baseline deltas are worse than no deltas: a plausible-looking number attributed to the wrong predecessor.

## Live evidence (read, 2026-09-30)

`voyage/scoreboard.py:91-128`:

```python
        video = metrics.get("video")
        frames = video.get("frames") if isinstance(video, dict) else None
        visual = metrics.get("visual")
        current: dict[str, float] | None = None
        if isinstance(visual, dict) and isinstance(visual.get("metrics"), dict):
            current = {
                key: float(visual["metrics"][key])
                for key in METRIC_KEYS
                if key in visual["metrics"]
            }
        ...
            "video_path": str(segment / "video.mp4"),
            "audio_path": str(segment / "audio.wav"),
        }
        rows.append(row)
        if current is not None:
            previous = current
    return rows
```

Sweep probe (preserved Track B result, in-container): crafted segment with `{"frames": "many", "visual": {"metrics": {"motion_energy": "NaN-string"}}}` → `ValueError: could not convert string to float: 'NaN-string'` (full crash, zero rows).

Note: `float("nan")`/`float("inf")` parse without raising — non-finite machine values slip through as cells; the crash is specifically on non-numeric strings, i.e. the hand-edit case.

## Repro

```bash
docker run --rm -v "$PWD:/app" -w /app voyage:latest python3 -c "
import json, tempfile
from pathlib import Path
from voyage import scoreboard
tmp = Path(tempfile.mkdtemp())
seg = tmp / 'segments' / '000001'; seg.mkdir(parents=True)
(seg / 'DONE').write_text('ok')
(seg / 'metrics.json').write_text(json.dumps({'video': {'frames': 'many'}, 'visual': {'metrics': {'motion_energy': 'NaN-string'}}}))
print(scoreboard.scoreboard_rows(tmp))"
# Buggy: ValueError, zero rows
```

## Fix candidates

1. Per-row try/except with an `errors` cell; guard `float()` (non-finite → skip cell or mark).
2. Existence-check view paths (`−` when missing) before advertising them.
3. Advance `previous` per committed segment (or record the baseline segment id alongside the delta) so gaps can't misattribute.
4. Full header names or a `--wide` legend; render missing `frames` as `?`, not `None`.

## Refs

- In-tree hardened pattern to copy: `voyage/scoreboard.py:47-74` (`_stages_by_segment` — torn lines skipped, live + rotated siblings, later-wins).
- Rotation helper: `voyage/logrotate.py:iter_metric_files` (DESIGN §60).
