# 141 — Run manifest still advertises 768×432 while the finalizer ships ≥1280×720@32

- Severity: LOW
- Area: provenance — manifest timeline vs presentation floors
- Files (as-read 2026-09-30):
  - `voyage/persistence.py:23-24` (`FINAL_VIDEO_WIDTH = 768`, `FINAL_VIDEO_HEIGHT = 432`)
  - `voyage/persistence.py:53-56` (manifest `timeline: {fps, final_width, final_height}`)
  - `voyage/media.py:676-678` (`AUGMENT_DEFAULT_MIN_FPS = 32`, `MIN_WIDTH = 1280`, `MIN_HEIGHT = 720`)
  - `voyage/media.py:808-810` (`FinalizeOptions.min_fps/min_width/min_height` defaults 32/1280/720)

## Technical description

The run manifest hard-codes the Phase 0 geometry (`voyage/persistence.py:19-24`, as-read):

```python
#: Timeline geometry recorded in the run manifest. Informational only — ...
FINAL_VIDEO_WIDTH = 768
FINAL_VIDEO_HEIGHT = 432
```

and stamps it per run (`:53-56`):

```python
"timeline": {
    "fps": config.video.fps,
    "final_width": FINAL_VIDEO_WIDTH,
    "final_height": FINAL_VIDEO_HEIGHT,
},
```

Since the finalize-augmentation floor work, the shipped video is *never* 768×432 by default:
`finalize_run`/`FinalizeOptions` default to the presentation floors
(`voyage/media.py:676-678` constants, `:808-810` option defaults — 32 fps, 1280×720), and the
docstring (`:874-883`) states backends render native (CausVid 832×480@16, LTXV 768×512@24,
fake 768×432@24) but ship at ≥1280×720@32 unless explicitly disabled with `min_*=0`. The header
comment in `persistence.py` ("no reader scales media from it … timeline truth") is accurate
about *readers* but concedes the manifest value is decorative — while `status`/provenance
consumers and any future auditor will read `manifest.timeline` as what the run produced.

`fps` is already honest (it records `config.video.fps`, the source timeline); only the two
geometry constants are stale. `min_fps/min_width/min_height = 0` (floors disabled) is the one
configuration where 768×432 could still be literally true — and only for the fake backend.

## Why this is an issue

- Provenance lie, however low-stakes: manifest says 768×432, `final.mp4` probes 1280×720 (or
  larger). Debugging geometry reports against the manifest misleads.
- The constants' comment ("updates the manifest in exactly one place") centralizes the wrong
  value: a future geometry change edits the floors in `media.py`, not these constants, so the
  drift only grows.
- Not a duplicate of 042 (README geometry drift) or 096 (fps validation): this is the stored
  manifest contract, not docs or validation.

## Live evidence

Live re-verification 2026-09-30 (read-only; probes per task brief ran in `voyage:latest` CPU-only).
Track A draft command+output bundle (`ses_f0fbea412ffeh3V7K1SlQpXjsv`) was not recoverable from
this writer's context, so evidence below is the as-read code, not invented command output:

```
$ sed -n '19,24p;53,56p' voyage/persistence.py
  23: FINAL_VIDEO_WIDTH = 768
  24: FINAL_VIDEO_HEIGHT = 432
  53:        "timeline": {
  55:            "final_width": FINAL_VIDEO_WIDTH,
  56:            "final_height": FINAL_VIDEO_HEIGHT,

$ sed -n '676,678p;808,810p' voyage/media.py
  676: AUGMENT_DEFAULT_MIN_FPS = 32
  677: AUGMENT_DEFAULT_MIN_WIDTH = 1280
  678: AUGMENT_DEFAULT_MIN_HEIGHT = 720
  808:    min_fps: int = 32
  809:    min_width: int = 1280
  810:    min_height: int = 720
```

## Minimal repro

1. `init` any run; `cat <run>/manifest.json | python3 -c "import json,sys; print(json.load(sys.stdin)['timeline'])"`
   → `{'fps': <backend fps>, 'final_width': 768, 'final_height': 432}`.
2. `generate`/`run` + `finalize` with defaults; `ffprobe final.mp4` → ≥1280×720@32 (e.g. LTXV
   768×512@24 source lifted to the floors).
3. Compare: manifest claims 768×432; artifact is 1280×720+. No reader breaks (comment is right),
   but the provenance record is wrong.

## Fix candidates

1. (Preferred) Record the effective presentation contract at init: `timeline: {fps, min_fps,
   min_width, min_height, floors_enabled}` sourced from the same defaults `media.py` uses
   (import the `AUGMENT_DEFAULT_*` constants or a shared helper — single source, no duplicated
   numbers), plus a `final_geometry: null` slot filled by `finalize_run` with the probed output.
2. Minimal: reword the manifest keys to `native_hint_width/height` (or drop them) so nobody reads
   them as output geometry; keep `fps` as the source-timeline truth.
3. Docs: note in `ARCHITECTURE.md`/`OPERATIONS.md` that `manifest.timeline` is the source hint,
   not the shipped geometry, until option 1 lands.
4. Regression tests: `build_manifest` reflects the floor constants (no independent 768/432
   literals); `finalize` writes back probed geometry when wired.

## References

- In-tree: `voyage/persistence.py:19-59,63`; `voyage/media.py:670-830` (augment plan + floors),
  `:860-900` (`finalize_run` presentation contract); `DESIGN §§32-33` (manifest/state).
- Neighbor issues: 042 (README geometry drift), 061 (status omits fps divergence),
  096 (video fps validation), 133 (DESIGN handoff staleness incl. geometry notes).
- External: none (in-tree contract drift only).

## Investigation log

- 2026-09-30: filed by Track A sweep; live re-verified via Read (concurrent uncommitted edits
  noted in `voyage/cli.py`, `voyage/tui_state.py`, `tests/test_generate.py`,
  `config/persistence/rpc/supervisor` — citations are as-read values above).

## Progress log (Group C, 2026-09-30)

- Verdict: CONFIRMED live. `persistence.py:23-24` still
  `FINAL_VIDEO_WIDTH = 768` / `FINAL_VIDEO_HEIGHT = 432`, stamped at
  `:53-56`; `media.py:773-775` floors are 32/1280/720 and `config.py:438-440`
  `AugmentConfig` defaults match (line numbers drifted slightly from the
  filing — values identical).
- Fix candidates 1-2 are code (`persistence.py` — not owned), so the
  Group-C fix is candidate 3 (docs): added a "Provenance note" to
  `docs/STATE_AND_RECOVERY.md` stating `manifest.timeline` is the source
  hint, not the shipped presentation (read shipped geometry off the
  artifact / `docs/AUGMENT.md`).
- Files changed: `docs/STATE_AND_RECOVERY.md` (provenance note only).

## Resolution

- Docs: provenance note added; manifest constants untouched.
- Residual (code-owning track): record the effective presentation contract
  at init (`min_fps/min_width/min_height` + `final_geometry` slot filled by
  finalize) or reword the keys to `native_hint_*`, plus the
  `build_manifest`-reflects-floors regression test from the issue.
