# 162 — AGENTS.md §11 voyage bullet is stale (768×432@24fps, 12 verbs, Phase-0 backends)

- **Severity:** LOW (wrapper-docs drift — code is correct; the §11 one-liner misdescribes the shipped product)
- **File:line:** `AGENTS.md:428` (stale bullet) + `AGENTS.md:433` (console-flags verb list predates `sfx`) + `AGENTS.md:454` (augment-floors bullet supersedes the 768×432 claim); contrast `Voyage/voyage/config.py:386-388` (32fps/1280×720 floors), `Voyage/voyage/media.py:670,736` (presentation/augment floors), `Voyage/voyage/cli.py:1773-1817` (`_add_sfx_args`) + `Voyage/voyage/cli.py:1992-2008` (`sfx` verb)
- **Area:** wrapper-docs drift (AGENTS §11 — orchestrator-owned; DO NOT EDIT here, fix belongs to the orchestrator)

## Description

`AGENTS.md:428` still reads (verified live 2026-09-30):

```
… transactional segments (`segments/NNNNNN/` + DONE marker), ffmpeg
finalize to 768×432@24fps, 12-command CLI. Phase 0 skeleton done
2026-09-21 (task groups A-D, G-partial, I, J, K-unit): fake backends
render real testsrc/sine media so commit/validate/finalize paths are
genuine; `LongLiveBackend`/`AceStepBackend` are NotImplementedError
stubs pointing at their phases. Next: Phase 1 group E (LongLive adapter).
```

At least four claims in that one bullet are stale against the live
tree:

1. **Geometry/fps.** Finalize no longer ships 768×432@24fps by
   default: `AugmentConfig` (`config.py:386-388`) floors every shipped
   video at `min_fps = 32`, `min_width = 1280`, `min_height = 720`,
   and `media.py:670` (`PRESENTATION_MIN_FPS = 24`) /
   `:673-677` (`AUGMENT_DEFAULT_*`) / `:736` (`out_fps =
   max(requested, floor, PRESENTATION_MIN_FPS)`) enforce it —
   documented in `AGENTS.md:454` itself (augment-floors bullet), which
   directly contradicts `:428`.
2. **Verb count.** The CLI is no longer 12 commands: `sfx`
   (`cli.py:1992-2008`) plus `generate`/`benchmark`/`soak`/`inspect`
   postdate the count.
3. **Backends.** `LongLive`/`AceStep` are not stubs — LongLive2,
   LTXV, CausVid video backends plus ACE-Step audio, Qwen director,
   and MMAudio SFX (`cli.py:1773-1817`, `sfx_finalize.py`) all ship.
4. **Console flags.** `AGENTS.md:433` lists `--verbose`/`--no-color`
   on `run/generate/status/validate/finalize/benchmark/soak/inspect`
   — a list that both wrongly includes non-flag verbs and predates
   the `sfx` verb (see 160: the live `_add_console_args` call sites
   are run/generate/finalize/sfx/soak).

## Rationale

AGENTS.md is the wrapper's single source of truth and the first thing
every agent reads; a stale §11 seeds wrong assumptions (output
geometry, verb inventory, backend maturity) into every downstream
plan. The fix is orchestrator-owned (AGENTS.md update discipline,
§9) — this file records the drift with live refs so the orchestrator
can correct it without re-probing. **Do not edit AGENTS.md in this
track.**

## Live evidence

- `AGENTS.md:428` — full stale bullet quoted above (768×432@24fps,
  12-command CLI, Phase-0 stubs, "Next: Phase 1 group E").
- `AGENTS.md:433` — console-flags verb list predating `sfx`.
- `AGENTS.md:454` — augment-floors bullet (≥32fps + ≥1280×720)
  contradicting `:428`'s 768×432@24fps in the same section.
- `voyage/config.py:386-388` — `min_fps: int = 32`,
  `min_width = 1280`, `min_height = 720`.
- `voyage/media.py:670` (`PRESENTATION_MIN_FPS = 24`) and `:736`
  (`out_fps = max(...)`) — the enforcement `:428` omits.
- `voyage/cli.py:1773-1817` (`_add_sfx_args`) and `:1992-2008`
  (`_add_sfx_parser`) — the SFX surface `:428`'s verb count misses.
- Overlap check: 042-adjacent README drift and 133 (DESIGN §140
  handoff staleness) cover other docs; no open file covers the
  AGENTS §11 bullet itself.

## Repro

Static: read `AGENTS.md:428` against `voyage/config.py:386-388` +
`voyage/media.py:670,736` (geometry) and `voyage --help` (verb list
now incl. `sfx`/`generate`). The contradictions are visible without
running anything.

## Fix candidates (orchestrator-owned — do not apply in this track)

1. Rewrite the `:428` bullet: finalize ships ≥32fps + ≥1280×720 by
   default (`0` disables); verb list incl. `sfx` (+ current count);
   backends enumerated as shipped (longlive2/ltxv/causvid +
   acestep/qwen/mmaudio); drop the Phase-0/Phase-1-group-E sentences
   ( Phases 1-7 + SFX + augment all landed per DESIGN §140).
2. Align `:433` with the live `_add_console_args` call sites
   (run/generate/finalize/sfx/soak) — or delete the verb list and
   point at OPERATIONS (after 160 lands).
3. Keep `:454` as is (it is the accurate statement the rewrite must
   agree with).

## Refs

- `AGENTS.md:428,433,454` (read-only here); `Voyage/voyage/config.py:386-388`;
  `Voyage/voyage/media.py:670-677,736`; `Voyage/voyage/cli.py:1773-1817,1992-2008`.
- Companion: 160 (OPERATIONS console-flags omission — the runbook half
  of the same drift).
