"""`inspect metrics` event listing (DESIGN §59, issue 036 split).

Verb module of the issue-036 god-module split: the `cmd_inspect`
metrics branch moved verbatim out of `voyage.cli_observe` (which sat
above the §12 ~500-line signal), so the rotation-aware metric-event
view — live + rotated siblings via `iter_metric_files`, header with
the file span, trailing five rows — lives beside its reader instead
of the benchmark/soak verbs. `cmd_inspect` keeps its name and seam
(issue 080: `cli.cmd_inspect is cli_observe.cmd_inspect`) and
delegates here, so patched-name tests keep intercepting the seam.
"""

from __future__ import annotations

from pathlib import Path

from voyage.cli_status import _read_all_metric_events
from voyage.logrotate import iter_metric_files


def render_inspect_metrics(run_dir: Path) -> int:
    """Print the metric-event view for `run_dir` (DESIGN §59)."""
    events = _read_all_metric_events(run_dir)
    if not events:
        print("no metrics yet")
        return 0
    files = iter_metric_files(run_dir)
    print(f"{len(events)} metric events across {len(files)} files")
    for event in events[-5:]:
        print(f"  {event.get('event')}: {event.get('segment_id', event.get('worker', ''))}")
    return 0
