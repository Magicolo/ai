# 098 — Orphan scan covers `segments/` + `novelty/` but not `audio/` nor run root (`state.json.*.partial`)

- Severity: LOW
- Area: correctness / CLI / durability
- File: `voyage/cli.py:855` (patterns), `:865-873` (collector), `:914-918` (roots in `validate_run` — verified live 2026-09-30; concurrent edits keep moving these, see drift note below)

> Note on line numbers: the task mapping cites `voyage/cli.py:866-870,807-825`. A concurrent agent has uncommitted edits in `voyage/cli.py` (verified via `git diff HEAD -- comfy/Voyage/voyage/cli.py`: `--name` alias + `_effective_run_id`). All refs below cite **live** numbers observed 2026-09-30 via `grep -n`. Review pass 2026-09-30: a second concurrent-edit wave moved them again (`_ORPHAN_PATTERNS` now `:855`, collector `:865`, roots `:914-918`) — excerpts below quote content, which is unchanged; only the anchors drifted.

## Description

Live code (re-verified 2026-09-30):

```python
# voyage/cli.py:816-823
_ORPHAN_PATTERNS = ("*.partial", "*.tmp.npy", "*.tmp*")
"""Transient-file globs ... `*.partial` covers atomic-write staging; ..."""

# voyage/cli.py:826-834
def _collect_transient_orphans(root: Path, base: Path) -> list[str]:
    found: set[str] = set()
    if root.exists():
        for pattern in _ORPHAN_PATTERNS:
            for candidate in root.rglob(pattern):
                if candidate.is_file():
                    found.add(str(candidate.relative_to(base)))
    return sorted(found)

# voyage/cli.py:875-879
orphans = _collect_transient_orphans(segments_root, segments_root)
orphans.extend(_collect_transient_orphans(run_dir / "novelty", run_dir))
orphans = sorted(set(orphans))
if orphans:
    errors.append(f"orphan transient files: {orphans}")
```

Scanned roots: `segments/` and `novelty/` only. Not scanned:

- **`audio/`** — `takes.jsonl` appends (`audio/planner.py:211-220`) and take renders write `audio/take_*.wav` (+ `audio/sfx/` stems + `sfx.jsonl` in the SFX pass). A crashed take render can leave `take_*.wav.partial`-style staging (any `atomic_write_*` in the audio path) or orphaned `slice_*.wav` / pid-suffixed temps. None is reported.
- **Run root** — `state.json`, `voyage.toml`, `run_manifest.json`, `concepts.jsonl` are all written via `atomic_write_*` (`*.partial` staging in the run dir itself). A crash between `mkstemp` and `os.replace` leaves `state.json.<rand>.partial` at the root. `validate` never looks there, so the canonical "is this run clean?" tool misses the most critical staging residue (the state file's own temp).
- (`logs/` rotation siblings are intentionally *not* orphans — dated `metrics-YYYY-MM-DD.jsonl` are legitimate; the current globs do not match them, so extending roots to `audio/` + root does not create false positives.)

## Rationale

- Low severity: leftover `.partial` files are harmless (ignored by readers, overwritten on next commit) — but the orphan scan exists precisely to surface crashed-commit residue (issue 058), and a scan that misses two of the four write locations gives false confidence ("VALID, no orphans" while `audio/` holds a half-written take).
- Run-root gap is the sharper one: `state.json.*.partial` means the last state advance may not have persisted (013 family) — exactly what an operator needs to know.

## Live evidence

```
$ grep -n "_ORPHAN_PATTERNS\|def _collect_transient_orphans\|orphans = _collect" comfy/Voyage/voyage/cli.py
816:_ORPHAN_PATTERNS = ("*.partial", "*.tmp.npy", "*.tmp*")
826:def _collect_transient_orphans(root: Path, base: Path) -> list[str]:
875:    orphans = _collect_transient_orphans(segments_root, segments_root)
```

Excerpt (`:875-879`):

```
orphans = _collect_transient_orphans(segments_root, segments_root)
orphans.extend(_collect_transient_orphans(run_dir / "novelty", run_dir))
```

No mention of `audio` or run root. Writers that can leave residue outside the scanned roots (verified by grep):

```
$ grep -rn "atomic_write_\|mkstemp\|\.partial" comfy/Voyage/voyage/audio/ comfy/Voyage/voyage/atomic.py | head
audio/planner.py: append_take → ledger.open("a") (no .partial, but take renders use atomic paths)
atomic.py:37: tempfile.mkstemp(dir=str(destination.parent), ... suffix=".partial")
```

`atomic_write_*` stages `*.partial` in *whatever* `destination.parent` is — including run root (`state.json`) and `audio/` — while the scan only looks under `segments/` + `novelty/`.

Host probe (stdlib):

```
$ PYTHONPATH=Voyage python3 - <<'PY'
from pathlib import Path
import tempfile
with tempfile.TemporaryDirectory() as td:
    run = Path(td); (run/"segments").mkdir(); (run/"novelty").mkdir(); (run/"audio").mkdir()
    (run/"state.json.abc.partial").write_text("x")
    (run/"audio"/"take_0000.wav.partial").write_text("x")
    # current scan roots miss both:
    print("root partial exists:", (run/"state.json.abc.partial").exists())
    print("audio partial exists:", (run/"audio"/"take_0000.wav.partial").exists())
    print("scanned roots: segments/, novelty/ only -> both MISSED")
PY
root partial exists: True
audio partial exists: True
scanned roots: segments/, novelty/ only -> both MISSED
```

## Repro

1. Init a run; `touch run/state.json.foo.partial run/audio/take_0000.wav.partial`.
2. `voyage validate --run <dir>` → `VALID` (or only unrelated errors), no `orphan transient files` entry.
3. Move the same files under `segments/000000/` → validate now reports them. Proves the gap is roots, not patterns.

## Fix candidates

1. (Preferred) Extend the scan:
   ```python
   orphans = _collect_transient_orphans(segments_root, segments_root)
   orphans.extend(_collect_transient_orphans(run_dir / "novelty", run_dir))
   orphans.extend(_collect_transient_orphans(run_dir / "audio", run_dir))
   orphans.extend(_collect_transient_orphans(run_dir, run_dir))  # root *.partial only
   ```
   Scope the root scan to top-level `*.partial` (non-recursive or filtered) to avoid double-counting `segments/` + `novelty/` + `audio/` hits when scanning `run_dir` recursively — or collect once over `run_dir` and allow-list `logs/` rotation siblings.
2. Keep `logs/` excluded (rotation siblings are legitimate) — assert in tests that `metrics-2026-01-01.jsonl` is never flagged.
3. Regression test: fixtures in `audio/` + root → `orphan transient files` reported; `logs/` dated sibling → not reported.

## Refs

- Not a duplicate of 095 (integrity coverage vs residue detection — complementary, both touch `validate_run`).
- `voyage/cli.py:855-873` (patterns + collector), `:914-918` (roots — the two missing lines).
- `voyage/atomic.py:37-58` (every `atomic_write_*` stages `*.partial` in `destination.parent` — including root and `audio/`).
- Issue 058 (orphan scan origin — segments + novelty only); Phase 6 slice A (validate recomputes + orphan-scans).
