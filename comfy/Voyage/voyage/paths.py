"""Run-directory layout (DESIGN §21 persistent files, §29 segment dir).

run/
  manifest.json   (carries the effective config — CLI-is-config, no TOML)
  state.json
  concepts.jsonl
  segments/
  logs/
  tmp/            (generation scratch — worker session dirs, mux/assembly
                   staging; disposable, never checksummed or validated)
"""

from __future__ import annotations

import shutil
from pathlib import Path

from voyage.errors import MediaError

SCHEMA_VERSION = 1

#: Segment-number bounds (issue 091): the six-digit `%06d` id doubles as
#: the lexicographic/contiguous invariant validate_run relies on, so the
#: range is closed — negative inputs previously rendered as '-00001' and
#: huge ids grew past six digits into directory names and ordering checks.
MIN_SEGMENT_NUMBER = 0
MAX_SEGMENT_NUMBER = 999999
#: Width of the zero-padded segment id; kept as a constant (not a bare
#: `06` in the format spec) so the padding visibly tracks the max above.
SEGMENT_ID_WIDTH = 6

MANIFEST_FILENAME = "manifest.json"
SEGMENT_MANIFEST_FILENAME = "manifest.json"
STATE_FILENAME = "state.json"
CONCEPTS_FILENAME = "concepts.jsonl"
SEGMENTS_DIRNAME = "segments"
LOGS_DIRNAME = "logs"
#: Generation scratch dir (boba /tmp-quota incident): every temp file a run
#: produces — worker session work_roots, mux/assembly staging — lives under
#: `run_dir/tmp/`, never on the host /tmp tmpfs (31G with usrquota; a
#: 12-segment LTX25 run exhausted it mid-mux). Disposable: evict paths
#: remove their session dirs, and `ensure_scratch_dir` prunes stale
#: `voyage-*` session leftovers from killed runs. Invisible to
#: `validate_run`: the orphan scan only covers segments/novelty/audio/
#: augment plus run-root `voyage-final-*`/`*.partial`, so `tmp/` never
#: flags — and nothing under it is checksummed or referenced by stored
#: paths (resolve_stored_path never needs a `tmp` anchor).
SCRATCH_DIRNAME = "tmp"
DONE_MARKER = "DONE"

#: Session-dir prefixes pruned as stale by `ensure_scratch_dir` (crashed
#: runs strand them — evict-time `rmtree` only runs on clean shutdown).
#: Popup-bench prefixes (`voyage-bench-`, `voyage-sfx-bench-`, mux dirs)
#: are `TemporaryDirectory`-owned and self-cleaning, so only the
#: never-removed session roots are listed.
STALE_SCRATCH_PREFIXES = frozenset({"voyage-ltx25-", "voyage-ltx23-", "voyage-acestep-cwd-"})

# Layout dirnames usable as re-anchor points for legacy absolute entries
# (issue 016): a moved run's stale absolute path still names the layout
# position (`audio/take_0000.wav`), so consumers can find it again.
_LAYOUT_ANCHORS = frozenset({"segments", "audio", "novelty", "logs"})


def segment_dir(run_dir: Path, segment_id: str) -> Path:
    return run_dir / SEGMENTS_DIRNAME / segment_id


def scratch_dir(run_dir: Path) -> Path:
    """Generation scratch dir for a run (never on host /tmp)."""
    return run_dir / SCRATCH_DIRNAME


def ensure_scratch_dir(run_dir: Path) -> Path:
    """Create `run_dir/tmp/` and prune stale session leftovers (best-effort).

    Stale `voyage-*` session dirs only exist after a kill/crash (clean
    evict paths `rmtree` their work_root); pruning here is safe because
    no worker is running yet when the supervisor starts workers. A
    concurrent holder of the run would fail on the fcntl run lock
    anyway. Never raises for prune failures — leftover scratch only
    costs disk, never correctness.
    """
    scratch = scratch_dir(run_dir)
    scratch.mkdir(parents=True, exist_ok=True)
    try:
        for child in sorted(scratch.iterdir()):
            if child.is_dir() and any(
                child.name.startswith(prefix) for prefix in STALE_SCRATCH_PREFIXES
            ):
                shutil.rmtree(child, ignore_errors=True)
    except OSError:
        pass
    return scratch


def staging_parent(scratch_dir_value: str | None) -> Path | None:
    """Parent for worker staging dirs, or None for the TMPDIR default.

    Production init payloads always carry the run scratch
    (`run_dir/tmp/`); legacy and bench-only callers without one fall back
    to the process TMPDIR (which the supervisor points at the run
    scratch — the backstop covers them). The dir is created here so
    `TemporaryDirectory(dir=...)` never races a missing parent.
    """
    if scratch_dir_value is None:
        return None
    parent = Path(scratch_dir_value)
    parent.mkdir(parents=True, exist_ok=True)
    return parent


def format_segment_id(number: int) -> str:
    """Format a segment number as a zero-padded six-digit id.

    Why the range check: negative or >999999 inputs break the %06d
    lexicographic/contiguous invariant validate_run relies on (issue 091)
    — -1 previously rendered as '-00001' and huge ids grew past six
    digits into directory names and ordering checks.
    """
    if number < MIN_SEGMENT_NUMBER or number > MAX_SEGMENT_NUMBER:
        raise ValueError(
            f"segment number must be within"
            f" [{MIN_SEGMENT_NUMBER}, {MAX_SEGMENT_NUMBER}] (got {number})"
        )
    return f"{number:0{SEGMENT_ID_WIDTH}d}"


def _is_within_run(anchor: Path, path: Path) -> bool:
    """True when `path` resolves inside the resolved run dir (issue 015).

    Both sides resolve first: lexical `..` (`segments/../../evil.wav`)
    and symlinked parents must not pass a pure `relative_to` check, and
    the anchor resolves too so a symlinked run dir still matches its own
    entries. Missing paths are fine — `resolve()` is non-strict — only
    escapes fail.
    """
    try:
        path.resolve().relative_to(anchor)
    except ValueError:
        return False
    return True


def resolve_stored_path(run_dir: Path, stored: str | Path) -> Path:
    """Resolve a persisted run artifact path (issues 015, 016 consumer side).

    Stored form is run-relative POSIX (e.g. `audio/take_0000.wav`,
    `segments/000000/recovery.pt`); legacy entries are absolute. Relative
    entries resolve against `run_dir`, so `cp -r`/`mv` of a run keeps every
    consumer working. A missing absolute entry is re-anchored on the first
    known layout dirname (`segments/`, `audio/`, `novelty/`, `logs/`) when
    that target exists inside the run — this heals pre-fix runs after a
    move; the in-run copy wins even when the stale absolute still exists
    at the old location.

    Containment is enforced on the resolved path (issue 015): stored paths
    come from worker reports, ledger lines, and metrics — all untrusted
    (issue 006) — so a relative `..` escape or an absolute path outside
    the run raises `MediaError` (the `Supervisor._checked_tape_path`
    convention) instead of being read/written outside the run. A trusted
    in-run path is returned in its stored form, so the lexical
    `run_dir`-join contract for relatives is unchanged.
    """
    anchor = run_dir.resolve()
    candidate = Path(stored)
    if candidate.is_absolute():
        for index, part in enumerate(candidate.parts):
            if part in _LAYOUT_ANCHORS:
                reanchored = run_dir.joinpath(*candidate.parts[index:])
                if reanchored.exists() and _is_within_run(anchor, reanchored):
                    return reanchored
        if _is_within_run(anchor, candidate):
            return candidate
        raise MediaError(f"stored path escapes the run dir: {stored!r}")
    joined = run_dir / candidate
    if not _is_within_run(anchor, joined):
        raise MediaError(f"stored path escapes the run dir: {stored!r}")
    return joined
