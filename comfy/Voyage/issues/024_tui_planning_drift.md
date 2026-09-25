# 024 — TUI planning math is stale vs CLI truth; `causvid` omitted; plan derived twice

- Status: resolved (fixed 2026-09-25, TUI track)
- Severity: major (wrong segment estimates; backend rejected; paired lists
  already disagree)
- Area: UX correctness — `tui_state.plan_counts` / `plan_summary` / `gpu_warning`
- Rank rationale: TUI says 5 segments where CLI does 2 (ltxv); `causvid` users get
  a silent fallback to `ltxv`.

## Technical description

`plan_counts()`/`plan_summary()` hardcode (`voyage/tui_state.py:49-51,226,253`):

```python
_LTXV_FIRST_BLOCK_FRAMES = 25
_DEFAULT_SEGMENT_FRAMES = 48
_FPS = 24
if state.backend == "ltxv":
    frames_per_segment = 25 + (blocks-1)*24
else:
    frames_per_segment = 48
```

CLI truth (`voyage/cli.py:703,714,717-731`, orchestrator-verified live 2026-09-25):

| backend | CLI frames/seg (blocks=1) | fps | TUI predicts |
|---|---|---|---|
| ltxv | 96 | 24 | 25 |
| longlive2 | 29 | 24 | 48 |
| causvid | 72 | 16 | 48 (+ backend rejected) |
| fake | 48 | 24 | 48 (only correct case) |

```
$ python3 -c "..."
fake 48 / ltxv 96 / longlive2 29 / causvid 72
```

The comment "mirrors `cli._frames_per_segment`" is false — it mirrors the
pre-Stream-A 25/24 math. Additionally `BACKENDS = ("ltxv","longlive2","fake")`
(`tui_state.py:27-29`) omits `causvid` (live in CLI/generate/preset/models); a
saved `causvid` form loads back as `ltxv` via `_choice_field()` fallback; and
`gpu_warning()` (`tui_state.py:391-398`) returns `''` for causvid:

```
$ python3 -c "...gpu_warning('causvid')..."
causvid: ''
```

while `cli.py:734` treats it as CUDA. Finally `plan_counts` (226-250) and
`plan_summary` (253-282) each re-derive `frames_per_segment` — a fix to one can
miss the other — and `_read_form` (`tui.py:699-726`) hides form parsing in three
single-use closures.

## Why this is an issue

The launch screen predicts 5 segments where the CLI runs 2 (ltxv), 48 frames
where the CLI commits 29 (longlive2), and silently substitutes `ltxv` when a
saved `causvid` form reloads — so users budget time, cost, and expectations
from numbers that are wrong by factors of two to four. The comment claiming
the math mirrors `cli._frames_per_segment` is stale: it mirrors pre-Stream-A
constants. Because `plan_counts` and `plan_summary` each re-derive
frames-per-segment independently, fixing one can miss the other, as already
happened. Everyone who trusts the TUI plan before pressing Generate pays the
surprise.

## Evidence

Live probes above (all orchestrator-run 2026-09-25, idle GPU).

## Reproduction

`plan_counts(GenerateFormState(backend='ltxv',blocks='1',duration='5s',...))` →
`(5, 125, 5.2s)` vs CLI `ceil(5*24/96)=2` segments.

## Source references

- `voyage/tui_state.py:27-29,49-51,226,253,391-398`; `voyage/cli.py:697-731,734,
  1114,1186`; `voyage/workers/video_ltxv.py:52` (worker truth 121-25=96).

## Resolution candidates

Single source: `tui_state.plan_counts()` calls `cli._frames_per_segment()` +
preset fps (or one shared 3-branch table + fps map); `plan_summary()` formats the
struct (no recompute); `BACKENDS += ("causvid",)` + `gpu_warning` derives from the
shared CUDA set; lift `text/choice/flag` to module level; regression test all 4
backends × blocks 1..3.

## Investigation / progress / resolution log

- 2026-09-25: found by docs + structure sweeps (convergent); TUI-vs-CLI matrix
  and `gpu_warning` probes executed live by orchestrator.
- Open: implement single-source planning + tests.
- 2026-09-25 (repair pass): added `## Why this is an issue`; corrected
  `plan_counts` 242-250→226-250, `plan_summary` 253-282, `_read_form`
  656-663→699-726, constants 46-51→49-51.
- 2026-09-25 (fix, TUI track): relevance re-verified live — ltxv drift
  (25/49/73 vs CLI 96/192/288) and longlive2 drift (48 vs 29/61/93)
  confirmed on host; causvid/fake already correct, and `BACKENDS` +
  causvid plan branch + `gpu_warning` causvid were already landed by
  commit `1c3b46a` (kept, not re-done). Import-direction check FIRST:
  `voyage/cli.py` never imports `voyage.tui_state` at module level — its
  only TUI touch is the lazy `from voyage.tui import run_tui` inside
  `launch_tui` (`cli.py:90-98`) — so function-level imports from
  `tui_state` into `cli` cannot cycle (same pattern as the pre-existing
  `parse_duration` import; `cli.is_flat_folder_name`'s docstring
  documents this direction). No shared-table move into `config.py`
  needed (also out of scope). Implemented direct single source in
  `voyage/tui_state.py`: `_planning_frames_and_fps` (frames from
  `cli._frames_per_segment` on a planning-only `ProjectConfig`,
  fps from the config video preset) + `_plan_details` struct
  (`plan_counts` returns its first 3 fields, `plan_summary` formats all
  5 — no recompute); stale `_LTXV_FIRST_BLOCK_FRAMES` /
  `_DEFAULT_SEGMENT_FRAMES` / `_FPS` / `_BACKEND_FPS` constants deleted;
  `gpu_warning` derives from `cli._CUDA_BACKENDS`; `_read_form`
  closures lifted to module-level `_read_text_field` /
  `_read_choice_field` / `_read_flag_field` in `voyage/tui.py`.
  Live 12-case matrix (4 backends x blocks 1..3, host `PYTHONPATH=Voyage`):
  TUI == CLI exactly in all 12. Tests: `test_tui_state.py`
  `test_plan_counts_match_cli_truth_all_backends_and_blocks` (cross-test
  vs `cli._frames_per_segment` + preset fps),
  `test_plan_summary_uses_single_source_struct`,
  `test_gpu_warning_derives_from_shared_cuda_set`; `test_tui.py` stale
  ltxv expectations corrected (5seg/125f → 2seg/192f + 96f/segment;
  new longlive2 29f fragment test). Gates: `scripts/gates.sh` GREEN
  (ruff + format + mypy strict + 563 pytest).
