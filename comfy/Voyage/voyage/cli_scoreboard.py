"""`inspect scoreboard` table rendering (DESIGN §59, issue 036 split).

Verb module of the issue-036 god-module split: the `cmd_inspect`
scoreboard branch moved verbatim out of `voyage.cli_observe` (which sat
above the §12 ~500-line signal), so the per-segment table — frames,
per-stage seconds, deterministic visual metrics with deltas, director
destination/phase, take ids, view paths, and the trailing `final.mp4`
line — lives with its reader (`voyage.scoreboard`) instead of the
benchmark/soak verbs. `cmd_inspect` keeps its name and seam (issue 080:
`cli.cmd_inspect is cli_observe.cmd_inspect`) and delegates here, so
patched-name tests keep intercepting the seam.
"""

from __future__ import annotations

from pathlib import Path


def render_scoreboard(run_dir: Path) -> int:
    """Print the per-segment scoreboard table for `run_dir` (DESIGN §59)."""
    # Seam dispatch (issue 080): imported at call time, like the original
    # branch, so patching `voyage.scoreboard.scoreboard_rows` (issue 142
    # fail-soft tests) keeps intercepting the render.
    from voyage.scoreboard import METRIC_KEYS, scoreboard_rows

    rows = scoreboard_rows(run_dir)
    if not rows:
        print("no committed segments")
        return 0
    header = ["seg", "frames", "stages", *[f"{key[:12]}" for key in METRIC_KEYS]]
    print("  ".join(header))
    for row in rows:
        stages = row["stages"]
        stage_cells = (
            ",".join(f"{name}={seconds:.1f}" for name, seconds in stages.items())
            if isinstance(stages, dict) and stages
            else "-"
        )
        metrics = row["metrics"]
        deltas = row["deltas"]
        if isinstance(metrics, dict) and isinstance(deltas, dict):
            try:
                cells = [f"{metrics[key]:.3f}({deltas[key]:+.3f})" for key in METRIC_KEYS]
            except (KeyError, TypeError, ValueError):
                cells = ["no-visual"] * len(METRIC_KEYS)
        else:
            cells = ["no-visual"] * len(METRIC_KEYS)
        print("  ".join([str(row["segment_id"]), str(row["frames"]), stage_cells, *cells]))
        print(f"  -> {row['destination']} [{row['phase']}] takes={row['take_ids']}")
        print(f"  view: {row['video_path']} + {row['audio_path']}")
    final = run_dir / "final.mp4"
    if final.exists():
        print(f"final: {final}")
    return 0
