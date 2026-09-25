# 020 — God modules + god functions ("keep modules small" violated at the top)

- Status: open
- Severity: major (maintainability — 6 files hold ~60% of all lines)
- Area: structure — `cli.py`, `supervisor.py`, workers, `tui.py`, `media.py`
- Rank rationale: `commit_one_segment` spans six DESIGN sections in one function;
  `build_parser` + 13 `cmd_*` mix parsing, duration math, CUDA guards, console.

## Technical description

Sizes (sweep-measured, 12494 total py lines): `cli.py` 1336 (41 defs),
`supervisor.py` 1271 (32), `video_longlive.py` 1106 (41), `video_causvid.py` 1095
(50), `video_ltxv.py` 855 (30), `tui.py` 1012 (51), `model_registry.py` 663,
`media.py` 550, `config.py` 522. Long functions (AST, ≥80 lines):

```
261 supervisor.py:1011 commit_one_segment
216 cli.py:1098 build_parser
176 tui.py:500 compose
737 tui.py:276 class VoyageApp (class, not a function — listed for scale)
145 media.py:406 finalize_run
136 supervisor.py:581 _accept_director_decision
130 video_longlive.py:718 generate_blocks
...
```

`commit_one_segment` does director+prefetch+prompt-plan+video-payload+video-call+
audio-coverage+validate+checksums+DONE+state-advance+metrics+progress. Max indent
5 in supervisor/tui.

## Why this is an issue

Six files hold about 60% of all lines, and `commit_one_segment` alone spans
six DESIGN sections in a single 261-line function that cannot be unit-tested
in pieces; `cli.py` mixes parsing, duration math, CUDA guards, and console
output in the same module. Every feature therefore touches the same few files,
concentrating merge conflicts and cross-concern regressions exactly where
concurrent agents already collide. Indent depth of 5 and untestable middle
layers mean the riskiest code is also the hardest to verify. Every
contributor's next change pays in contention, review load, and regression
risk.

## Evidence

`wc -l Voyage/voyage/*.py Voyage/voyage/**/*.py | sort -n | tail`; AST length scan
(sweep transcript); `rg -c "def |class "` per file.

Re-verified 2026-09-25 (AST top functions per file):

```
$ python3 -c "import ast; ..."  # longest defs by (lines, lineno, name)
voyage/supervisor.py total-lines: 1271 top: [(114, 755, '_ensure_audio_coverage'), (136, 581, '_accept_director_decision'), (261, 1011, 'commit_one_segment')]
voyage/cli.py total-lines: 1336 top: [(90, 187, 'cmd_models'), (110, 783, 'cmd_generate'), (216, 1098, 'build_parser')]
voyage/tui.py total-lines: 1012 top: [(32, 921, '_generate_in_thread'), (57, 831, '_start_generation'), (176, 500, 'compose')]
voyage/media.py total-lines: 550 top: [(57, 152, 'assemble_segment_audio'), (101, 303, 'build_final_audio'), (145, 406, 'finalize_run')]
voyage/workers/video_longlive.py total-lines: 1106 top: [(88, 223, '_install_pos_only_caches'), (88, 608, '__init__'), (130, 718, 'generate_blocks')]
```

## Reproduction

Open `voyage/supervisor.py:1011` and `voyage/cli.py:1098` — read either end to end.

## Source references

- Files/lines in the table above.

## Resolution candidates

1. Split `commit_one_segment` → `_propose()/_render_video()/_cover_audio()/
   _commit()` helpers (each independently testable).
2. Split `cli.py` → `cli_parse.py` (parser only) + `cli_commands/{run,generate,
   models,inspect}.py`; move duration math out (see 024).
3. Split `VoyageApp` → `tui_form.py` + `tui_run.py`.

## Investigation / progress / resolution log

- 2026-09-25: found by structure sweep.
- Open: incremental extraction; no behavior change.
- 2026-09-25 (repair pass): added `## Why this is an issue`; corrected `tui.py`
  921→1012 lines, `compose` 460→500, `VoyageApp` 269→276, `build_parser`
  218→216, `generate_blocks` 136→130 + AST evidence.
