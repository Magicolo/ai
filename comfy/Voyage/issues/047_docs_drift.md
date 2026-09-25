# 047 — Docs drift: README thin; BACKENDS/MODELS miss CausVid+LTXV; `UPSTREAM_LTXV_NOTES.md` missing

- Status: open
- Severity: medium (users can't discover verbs/backends; registry not "single
  source of truth" in docs)
- Area: documentation — README, `docs/`, `TASK.md`
- Rank rationale: README (100 lines) omits the TUI entirely plus 5+ verbs and all
  current flags; BACKENDS still describes the pre-Stream-A LTXV design.

## Technical description

- `Voyage/README.md` (105 lines) omits: bare-`voyage` TUI (zero mention —
  verified `rg -in "tui|textual|launcher" README.md` → no hits), `generate`
  flags (`--draft/--director/--blocks/--take-seconds/--quantization/
  --beats-per-segment/--drift-every-n/--final-video/--skip-bad/--verbose/
  --no-color`). `models download ltxv-2b|causvid` now present (lines 42-43 —
  fixed since filing); docs index omits nothing outstanding except the
  missing LTXV notes file below.
- `docs/BACKENDS.md` GPU table + §CausVid now present (lines 42, 67-79 —
  fixed since filing). Its "LTXV chaining" section (`:50-64`) still
  describes the pre-Stream-A tail-PNG design ("Each block writes a chain
  PNG", "The tail PNG doubles as the crash-recovery tape"), not the
  realigned 121-frame clips / 25-frame `video_tail.mp4` / 96-novel +
  §5.3 JSON tape (`voyage/workers/video_ltxv.py:15,59,561`).
- `docs/MODELS.md` LTXV-2B (`:19-27`, rev `8984fa25…`) and CausVid (`:30-38`,
  `adb6a5e…`/`b545eb27…`) pins now present (fixed since filing).
- `TASK.md:1711` deliverable `docs/UPSTREAM_LTXV_NOTES.md` does not exist
  (`ls docs/UPSTREAM*` → only CAUSVID + LONG_LIVE).

## Why this is an issue

Stale or thin docs are the first thing a new user (or a future agent) trusts,
so every gap here converts directly into wasted GPU hours and misfiled bug
reports: users cannot discover the TUI, the CausVid backend, or the correct
LTXV chaining model, and operators following BACKENDS.md will look for tail
PNGs that no longer exist. The blast radius is documentation-wide (README +
three docs pages + TASK deliverable), but the fix is low-risk text-only work
with no runtime behavior change.

## Evidence

Verified live 2026-09-25 (orchestrator) — the original "zero causvid hits"
claim is now STALE (fixed by Streams A–D, see log):

```
$ rg -n "causvid" README.md docs/BACKENDS.md docs/MODELS.md
README.md:43:./scripts/run.sh models download causvid        # ~28 GB CausVid ...
README.md:59:# --backend causvid renders 832x480 @ 16 fps ...
docs/MODELS.md:30:## Video — CausVid DMD + Wan2.1-1.3B base ...
docs/BACKENDS.md:42:| video | `causvid` | ... 832×480 @ 16 fps native, bf16 |
docs/BACKENDS.md:67:## CausVid chaining model (`causvid`, Stream D alternative)
$ rg -in "tui|textual|launcher" README.md
(no hits — TUI still undocumented)
$ ls docs/UPSTREAM*
docs/UPSTREAM_CAUSVID_NOTES.md
docs/UPSTREAM_LONG_LIVE_PATCHES.md
(no UPSTREAM_LTXV_NOTES.md — TASK.md:117,1711 deliverable still open)
```

Still-valid remainder: README omits the TUI entirely plus `generate` flags
(`--draft/--blocks/--take-seconds/--quantization/--beats-per-segment/
--drift-every-n/--final-video/--skip-bad/--verbose/--no-color`);
`docs/BACKENDS.md:50-64` "LTXV chaining" still describes the pre-Stream-A
tail-PNG design ("Each block writes a chain PNG", "The tail PNG doubles as
the crash-recovery tape") while `voyage/workers/video_ltxv.py:15,59,561`
uses `video_tail.mp4` (25 frames) + a §5.3 JSON tape.

## Reproduction

Commands above.

## Source references

- `Voyage/README.md`; `Voyage/docs/BACKENDS.md:42,50-64,67-79`,
  `Voyage/docs/MODELS.md:19-38`; `Voyage/TASK.md:117,1711`;
  `voyage/cli.py:1106-1311`; `voyage/workers/video_ltxv.py:15,59,561`.

## Resolution candidates

1. Extend README to ~140 lines: verb table + one `generate` example with
   `--draft`, TUI paragraph, inspect/benchmark/soak one-liners, full docs index.
2. Add CausVid rows (pins from `model_registry.py:CAUSVID_*/WAN21_*`); rewrite the
   LTXV chaining paragraph to Stream-A accounting; create or explicitly cancel the
   LTXV notes file.

## Investigation / progress / resolution log

- 2026-09-25: found by docs sweep.
- 2026-09-25 (repair): re-verified every reference live. PARTIALLY STALE —
  Streams A–D already fixed the CausVid/MODELS gaps (README:42-43,59;
  BACKENDS:42,67-79; MODELS:19-38). Remaining open: README TUI + generate
  flags, BACKENDS LTXV tail-PNG paragraph (:50-64 vs video_ltxv.py:15,59,561),
  missing UPSTREAM_LTXV_NOTES.md. Evidence replaced with real command output.
- Open: rewrite remaining docs batch.
