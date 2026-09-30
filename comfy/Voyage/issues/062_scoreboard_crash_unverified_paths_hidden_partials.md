# 062 — Scoreboard is fail-closed: stage parsing crashes, unverified view paths, hidden partials, stale baseline, cryptic headers

**Severity:** MEDIUM

**File:line:** `voyage/scoreboard.py:73` (`_stages_by_segment` float cast), `86-87` (DONE-gate skip), `96-100` (`scoreboard_rows` metric float cast), `123-124` (synthesized paths); contrast hardened sibling `voyage/cli.py:616-640` (`_slowest_stage` guards with `isinstance(value, (int,float))`, `_read_all_metric_events` skips torn lines)

**Overlaps with:** 027 (scoreboard rows robustness — same module, same root cause; recommend merging into one) / 028 (console contract — sibling observability hardening, not a duplicate)
- **Description:** Merged docs-sweep item 8 + structure-sweep item 8 (same module, same root cause — deduped here). Gaps in one small module: (a) `_stages_by_segment` does `{str(key): float(value) …}` unguarded — one `None`/`"unknown"`/string stage value aborts the whole scoreboard, while `_slowest_stage` right next to it guards; same for `float(visual["metrics"][key])` in `scoreboard_rows` — a single hand-edited `metrics.json` kills the table with zero rows; (b) `video_path`/`audio_path` are synthesized unconditionally (`str(segment/"video.mp4")`) even when the files are absent — the CLI then prints `view: …` for missing artifacts; (c) non-`DONE` segment dirs are silently skipped, so a stalled partial commit is invisible in the very table meant for iteration; (d) `previous` only advances when `current is not None`, so a segment without visuals inherits a stale baseline — the delta is attributed to the wrong predecessor; (e) header truncates metric names to 12 chars (`motion_energ`, `visual_compl`, …) with no legend; `frames` can be `None` (missing `video.frames`) and prints as `None`.
- **Rationale:** A read-only observability view must degrade per-row, never fail the table; advertising missing files as viewable sends the user to a dead path. The module's own docstring cites the rotation-outage class it fixes, then reintroduces a fail-closed parse one function down. `bench.summarize_gauges` already learned this lesson (issue 071: `"unknown"` strings broke aggregation → `None`).
- **Evidence (re-verified live 2026-09-30):**
```python
# voyage/scoreboard.py:73 (live, raises ValueError/TypeError on bad input)
stages[segment_id] = {str(key): float(value) for key, value in raw_stages.items()}
# voyage/scoreboard.py:96-100 (live)
current = {key: float(visual["metrics"][key]) for key in METRIC_KEYS if key in visual["metrics"]}
# voyage/scoreboard.py:123-124 (live, assumed, unchecked)
"video_path": str(segment / "video.mp4"),
"audio_path": str(segment / "audio.wav"),
```
Probe: `float("not-a-number")` → `ValueError`; `float(None)` → `TypeError`; crafted segment with `{"frames":"many","visual":{"metrics":{"motion_energy":"NaN-string"}}}` → full crash, zero rows. `cmd_inspect scoreboard` (`cli.py`) iterates rows trusting both path fields.
- **Repro:** Append one `segment_committed` event with `"stages": {"video": "unknown"}` to `logs/metrics.jsonl`, run `inspect scoreboard` → traceback instead of a table. `mkdir segments/000007 && touch segments/000007/video.mp4` without `DONE` → row absent with no notice.
- **Fix candidates:** Guard like `_slowest_stage` (skip non-finite, count `stages_dropped`); per-row try/except with an `errors` cell; mark paths with `exists: bool` or `"(missing)"`; add a trailing `partial: [...]` line listing non-DONE dirs; advance `previous` per committed segment or record the baseline id; full header names or a `--wide` legend.
- **Refs:** `voyage/bench.py:40-56` (issue-071 precedent); `docs/STATE_AND_RECOVERY.md:1-8` (DONE-gating invariant the view should make visible, not silent).

## Progress log

- 2026-09-30 re-verified live (container `voyage:latest`): `_stages_by_segment` with `{"video":"unknown"}` raises `ValueError`; `scoreboard_rows` with `{"motion_energy":"NaN-string"}` raises `ValueError` (zero rows); committed row without `video.mp4`/`audio.wav` still advertises both paths (`exists False` on disk); non-DONE dirs skipped silently; `previous` advances only on visuals (stale baseline unattributed); `frames "many"` passes through raw. All five premises CONFIRMED, no drift.
- TDD: `tests/test_observability_rank2.py` first run — all 7 scoreboard assertions failed as designed (stages crash, metrics crash, missing `video_exists`/`audio_exists`/`baseline_segment_id`/`errors`, missing `partial_segment_ids`, raw `frames`).
- Fix in `voyage/scoreboard.py` only (no `cli.py` touches): `_finite_float` guard on stages + metric cells (bool/strings/None/non-finite skipped, matching `_slowest_stage`); per-row `errors` cell; `video_exists`/`audio_exists` flags (paths kept for compat); new `partial_segment_ids()` helper (DONE-gating kept, stalls now listable); `baseline_segment_id` recorded per row (gaps explicitly attributed instead of silently stale); `frames` validated to non-negative int else `None`.
- Gates (container): `ruff check` + `ruff format --check` + `mypy` clean; 41 passed (15 new + existing scoreboard/benchmark/console suites).
- 027 FOLDED here on fix (same module, same root cause — see 027 file).

## Resolution

- Fixed in `voyage/scoreboard.py` (`_finite_float`, guarded `_stages_by_segment`, `partial_segment_ids`, hardened `scoreboard_rows` with `errors`/`video_exists`/`audio_exists`/`baseline_segment_id`/validated `frames`) with coverage in `tests/test_observability_rank2.py` (7 scoreboard tests).
- Residuals (CLI track owns `voyage/cli.py:1660-1677`, untouched): header still truncates to 12 chars (`key[:12]`, no legend); `view:` line still prints raw paths (should check `video_exists`/`audio_exists` and mark `"(missing)"`); `partial:` trailing line still to render from `partial_segment_ids()`; missing `frames` still prints `None` (should render `"?"`).
- DESIGN patch proposals (text only): §59 scoreboard contract gains per-row degradation (`errors` cell, `baseline_segment_id`, existence flags) + `partial_segment_ids` as the stall-visibility source; CLI renderer uses full metric names or a `--wide` legend and `"?"` for missing frames.
