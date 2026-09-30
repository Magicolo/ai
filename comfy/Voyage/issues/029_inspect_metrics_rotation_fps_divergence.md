# 029 — Rotation-blind `inspect metrics` + divergent `fps=0` fallbacks (`status` shows 0s, `validate` assumes 24)

- Severity: MEDIUM (observability lies by omission; divergent fallbacks)
- Group: observability/status — Rank: 3/5
- File:line: `voyage/cli.py:1632-1644` (`inspect metrics`), `voyage/models.py:205-225` (`RunState.fps`), `voyage/cli.py:681` vs `:922` (divergent fallbacks)
- Overlaps: 055 (rotation-blind inspect) + 061 (status gaps) — sibling observability gaps; not duplicates, fix readers together.

## Description

(a) After a daily log rotation, `inspect metrics` reports only the live file ("N metric events", last-5 from the tail) and can print `no metrics yet` while the run has full rotated history. The sibling readers (`scoreboard`, `status` last-commit, soak/benchmark averages) all use `iter_metric_files`. One reader was left behind.

(b) `RunState` accepts `fps=0` (legacy tolerance). `status` then shows `Timeline: 0.00s` (division guarded to 0 — hides corruption), while `validate_run` silently substitutes 24 for the SFX-ledger timeline check. Same stored value, two meanings, neither flags the corruption. A `fps=0` state (impossible from any current writer — all presets are 16/24) passes `validate` as VALID while `status` shows a zero-length timeline.

## Rationale

- Observability readers must share one log-source helper; a single direct `open` reintroduces the rotation blindness the `logrotate` module was built to kill (issue 049 fixed every other reader).
- Divergent fallbacks for the same corrupt field turn a detectable error into two plausible-looking lies. Fail-loud (one error naming `fps`) beats both.
- The `RunState` comment documents the tolerance as deliberate for legacy states — but tolerance at read time plus silence at check time means the legacy state can never be noticed and repaired.

## Live evidence (read, 2026-09-30)

`voyage/cli.py:1588-1600`:

```python
    if args.inspect_target == "metrics":
        import json

        metrics_path = run_dir / paths.LOGS_DIRNAME / "metrics.jsonl"
        if not metrics_path.exists():
            print("no metrics yet")
            return 0
        lines = metrics_path.read_text(encoding="utf-8").splitlines()
        print(f"{len(lines)} metric events")
        for line in lines[-5:]:
            event = json.loads(line)
            print(f"  {event.get('event')}: {event.get('segment_id', event.get('worker', ''))}")
        return 0
```

Contrast `voyage/cli.py:577-584`:

```python
def _read_all_metric_events(run_dir: Path) -> list[dict[str, object]]:
    """All metric events across live + rotated siblings, oldest-first (049). ..."""
    events: list[dict[str, object]] = []
    for events_path in iter_metric_files(run_dir):
```

`voyage/models.py:205-218`:

```python
class RunState(BaseModel):
    ...
    # fps is deliberately unguarded: the CLI
    # tolerates legacy fps=0 states, ...
    next_segment_number: int = Field(default=0, ge=0)
    committed_segments: int = Field(default=0, ge=0)
    timeline_frames: int = Field(default=0, ge=0)
    fps: int = 24
```

Sweep probe: `RunState(run_id='x', fps=0, timeline_frames=48)` validates; `status` math yields `0`, `validate` math yields `48/24 = 2.0`.

## Repro

```bash
docker run --rm -v "$PWD:/app" -w /app voyage:latest python3 -c "
from voyage.models import RunState
s = RunState(run_id='x', fps=0, timeline_frames=48)
print('validates:', s.fps, '| status shows:', (s.timeline_frames / s.fps if s.fps else 0), '| validate uses:', s.timeline_frames / (s.fps if s.fps > 0 else 24))"
sed -n '1588,1600p' voyage/cli.py
```

## Fix candidates

1. Route `inspect metrics` through `_read_all_metric_events` (one-line fix; report the file count so rotation is visible).
2. Add a `validate_run` error for `fps <= 0` (fail-loud) while keeping the 24-fallback for the SFX math only after reporting — legacy states become visible instead of silently normalized.
3. Optionally a `validate --fix` suggestion (`fps = <preset for backend>`) since the correct value is recoverable from the run config.

## Refs

- In-tree rotation helper: `voyage/logrotate.py:iter_metric_files` + `voyage/cli.py:577-627` (DESIGN §60).
