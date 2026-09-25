# 046 — Legacy shims, unused helpers, cross-module feature envy (`loop.py` double-decode)

- Status: open
- Severity: low-medium (cleanup; one real per-RPC waste + one swallowed-error path)
- Area: structure — dead code / layering
- Rank rationale: small individually, but 7 unused helpers + a double Pydantic
  decode on every RPC + a silent-`continue` that causes 600 s hangs (see 007).

## Technical description

```python
# concepts.py:85,93,110
legacy_path: Path | None = None, ...
if legacy_path is not None and legacy_path.exists() and ...: self._migrate_legacy(legacy_path)
def _migrate_legacy(self, legacy_path: Path) -> None:  # Phase 0/2 token-set history
# media.py:429
... pass overlap_fraction=0 to keep the legacy hard splice.
# supervisor.py:1249 — beat math lives in audio/beat.py but is re-imported lazily mid-commit
from voyage.audio.beat import beats_for_segment
beats, grid_bpm = beats_for_segment(duration, config.audio.beats_per_segment)
# loop.py:23-28 — decodes every line twice
decode_request(line)        # try/except discarded
request = decode_request(line)
# def-vs-usage scan (uses≤1, excluding tests): non_empty, config_segment_seconds
# (backends.py), timing_stats (bench.py), load_jsonl (concepts.py),
# build_prompt_plan (prompts.py), director_seed (seeds.py),
# histogram_distance (vision/metrics.py), decoded_frames_for_latents (causvid.py:132)
```

The Phase 0→21 concept migration has served (callers at `supervisor.py:1033`,
`cli.py:1060` still thread `legacy_path` through); "legacy hard splice" keeps two
audio-join paths; supervisor reaches into `audio.beat` for progress-only numbers
while `media.build_final_audio` re-derives the same timeline; `loop.py`
double-decodes and silently `continue`s on malformed input (the 007 hang).

## Why this is an issue

Individually small, collectively a fog: seven unused helpers force every reader to prove deadness before ignoring them, the double Pydantic decode taxes every single RPC, and the silent `continue` on malformed lines converts protocol errors into 600-second hangs (see 007) — the worst failure signature for operators. A migration shim that has served its purpose but keeps threading `legacy_path` through live call sites invites new code to depend on a path slated for removal. Cleanup here buys back readability and one real per-RPC cost.

## Evidence

`rg -n "legacy_path|_migrate_legacy" Voyage/voyage -g '*.py'`;
`sed -n '16,29p' Voyage/voyage/workers/loop.py`; def-vs-usage scan (sweep
transcript — filter test-only names like `inject_worker_crash` and Textual
callbacks before deleting).

Verified live 2026-09-25:

```
concepts.py:85: legacy_path ... :93-94: _migrate_legacy call ... :110: def _migrate_legacy
supervisor.py:1033: legacy_path=self._run_dir / paths.CONCEPTS_FILENAME
cli.py:1060: store = ConceptStore(run_dir / "novelty", legacy_path=...)
loop.py:23-28: try: decode_request(line) / except Exception: continue /
  request = decode_request(line)   ← double-decode + silent continue confirmed
```

## Reproduction

Commands above.

## Source references

- `voyage/concepts.py:85,93-94,110-119`; `voyage/media.py:429`;
  `voyage/supervisor.py:1249`; `voyage/workers/loop.py:23-28`;
  `voyage/backends.py:226`; `voyage/bench.py`; `voyage/prompts.py`;
  `voyage/seeds.py`; `voyage/vision/metrics.py`.

## Resolution candidates

Time-box the migration (log + drop after N releases); collapse finalize to
blend-only (or explicit `FinalizeOptions.joint_style`, see 045); move progress
beat/BPM into `audio/beat.py:segment_progress_info()`; fix `loop.py` to decode
once and log malformed lines; delete or test-pin the 7 unused helpers.
(`__pycache__/` verified present-but-ignored via `git check-ignore` — no action.)

## Investigation / progress / resolution log

- 2026-09-25: found by structure sweep. Module-docstring discipline is otherwise
  good (AST scan: zero missing module docstrings; all majors carry DESIGN refs) —
  only thin shims (`workers/video.py`, `workers/audio.py`, `workers/loop.py`,
  `bench.py`, `fake_backends.py`) lack WHY/DESIGN pointers; add them in this pass.
- 2026-09-25: repair pass — added `## Why this is an issue`; legacy_path /
  loop double-decode refs re-verified live, current; pasted rg/sed output into
  Evidence.
- Open: cleanup batch.
