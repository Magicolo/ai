# 133 — DESIGN §140 handoff staleness batch: console flag list, command count, `(0 disables)`, fake pass-through (17 claims verified, 4 drifted)

- **Severity:** LOW (docs — §140 is the handoff record; stale claims misdirect the next agent, none affects runtime)
- **File:line:** `Voyage/DESIGN.md:6718-6734` (console entry flag list), `:5815-5856` (Phase-0 entry: "all 12 commands" at `:5831`, "finalize scale/pad is a pass-through on fake runs" at `:5848-5849`), `:7514-7538` (augment entry: "`--min-fps` / `--min-resolution` (`0` disables)" at `:7517-7518`); code: `Voyage/voyage/cli.py:1765-1789` (augment flags), `:1814`/`:1863`/`:1932` (run/gen/finalize wiring), `Voyage/voyage/media.py:670` (`PRESENTATION_MIN_FPS = 24`), `:736` (`out_fps = max(requested, floor, PRESENTATION_MIN_FPS)`), `:961`/`:991` in `cli.py` (`fps=config.video.fps` into finalize)
- **Area:** (c) DESIGN §140 vs current code — 11 concrete claims verified live, 4 drifted

## Description

Seventeen §140 claims were re-verified against the live tree (container `voyage:latest` + source read, 2026-09-30). Thirteen hold; four drifted:

**DRIFT 1 — console entry flag list (DESIGN `:6731-6732`).** The entry says "`--verbose` / `--no-color` parse on `run`, `generate`, `status`, `validate`, `finalize`, `benchmark`, `soak`, and `inspect`." Live `--help` greps for `verbose|no-color`: `run` 4 hits, `generate` 4, `finalize` 3, `soak` 3 — but `status`, `validate`, `inspect`, `benchmark` all **0 hits**. `tests/test_console.py:248-262` (`test_status_cli_rejects_console_flags`) pins the rejection for status/validate/inspect (exit 2, issue 050). The 050 change removed the flags from `status` without updating the entry; `benchmark` never has them. Any agent following the entry to add console output to `status`/`benchmark` starts from a false premise.

**DRIFT 2 — "all 12 commands" (Phase-0 entry, `:5831`).** Live verb set (`voyage --help`): `{init, doctor, models, run, generate, status, pause, resume, stop, validate, finalize, sfx, benchmark, soak, inspect}` — **15** verbs (`sfx`, `benchmark`, `soak` postdate the entry). Count-only staleness, but the entry is cited as the CLI inventory.

**DRIFT 3 — "`0` disables" overclaims on the fps axis (augment entry, `:7517-7518`).** `plan_augmentation` computes `out_fps = max(int(requested_fps), floor_fps, PRESENTATION_MIN_FPS)` (`media.py:736`) with `PRESENTATION_MIN_FPS = 24` (`:670`) — the code docstring itself admits "0 disables the new 32fps floor, not the shipped-video guarantee" (`:717-719`). Consequence: a CausVid run (`video.fps == 16`, `cli.py:961,991` passes `config.video.fps` as requested) finalized with `--no-augment` (floors to 0) still lifts 16→24 fps (`max(16, 0, 24) = 24`, re-encode + fps filter). "Disables" holds for the geometry axes (`max(target, 0) = target`) but never for fps below 24. The CLI help repeats the overclaim ("disable all finalize augmentation floors (fps + resolution floors to 0)", `cli.py:1786-1789`).

**DRIFT 4 — "finalize scale/pad is a pass-through on fake runs" (Phase-0 notes, `:5848-5849`).** True when written (fake 768×432@24 matched the old 768×432@24 target). Post-augment the default floors are 32 fps / 1280×720, so a fake run's finalize now re-encodes 768×432@24 → 1280×720@32 by default — the augment entry itself records the shift ("768x432@24 -> 1280x720@32", `:7533-7535`). The pass-through note survives verbatim two entries above it.

**VERIFIED HOLD (no action):** `--min-fps`/`--min-resolution`/`--no-augment` wired on all three of run/generate/finalize via one helper (`cli.py:1765-1789`, `:1814`, `:1863`, `:1932`); TUI `min_fps`/`min_resolution` fields + validation (`tui_state.py:83-84,112-113,226-242`); `AugmentConfig` defaults 32/1280/720 (`config.py:386-388`); `required_specs` fake-early-return `[]` + CUDA augment pair (`models_ensure.py:71-110`); `BackendRecord` sfx rows CUDA→mmaudio/cuda:0, fake→fake/cpu (`config.py:145-146,160-161,175-176,193-194`); `to_generate_namespace` sfx attrs (`tui_state.py:305-306`); `--sfx-caption` on the finalize family (`cli.py:1740`); director `INIT_STR_KEYS` + `models_dir` (`workers/director.py:43-52`, `supervisor.py:321`); `_add_generation_overrides` on run+generate + `effective_*` helpers (`cli.py:1670-1712,1813,1875`, `supervisor.py:251-260`); `run.sh` `nvidia-smi -L` CUDA default + `--entrypoint voyage` (`scripts/run.sh:77-78,113-124`); `SfxConfig` fake/cpu defaults (`config.py:369-370`); `commit_one_segment` without workers → `FatalWorkerError` (`supervisor.py:431`); TUI still has no music/video caption fields (the SFX-slice-4 "follow-up" is still open, entry accurate).

## Rationale (non-overlap)

- 050 (status rejects console flags) changed behavior + pinned it in `test_console.py` but never updated the §140 console entry — this file covers the doc side, 050 the code side.
- 092 (docs one-line batch + qual leg) and 065/091 (operator docs) touch other docs; none names the console flag list, the verb count, the 24 fps residual, or the fake pass-through note.
- The augment entry's own "fixed in place" paragraph (`:7533-7537`) records the *test* fallout of the default shift, not the two stale sentences flagged here.

## Live evidence (verified 2026-09-30)

```
$ for v in run generate status validate finalize benchmark soak inspect; do
    docker run --rm ... --entrypoint voyage voyage:latest $v --help | grep -c "verbose\|no-color"; done
run 4 / generate 4 / status 0 / validate 0 / finalize 3 / benchmark 0 / soak 3 / inspect 0
$ docker run --rm ... voyage:latest --help
{init,doctor,models,run,generate,status,pause,resume,stop,validate,finalize,sfx,benchmark,soak,inspect}  # 15
```

- `Voyage/DESIGN.md:6731-6732` vs the counts above (4 verbs mismatch).
- `Voyage/DESIGN.md:5831` ("all 12 commands") vs 15 live verbs.
- `Voyage/voyage/media.py:670,736` + `Voyage/voyage/cli.py:961,991`: `max(16, 0, 24) = 24` — pure-math proof the "0 disables" claim fails for CausVid sources.
- `Voyage/DESIGN.md:5848-5849` vs `:7533-7535` (the same file admits the shift three entries later).

## Repro

```bash
grep -n "verbose.*no-color.*run.*generate.*status" Voyage/DESIGN.md  # :6731-6732
docker run --rm -v $PWD/Voyage:/app -w /app --entrypoint voyage voyage:latest status --help  # no verbose flags
docker run --rm -v $PWD/Voyage:/app -w /app --entrypoint voyage voyage:latest --help  # count the verbs
sed -n '670p;736p' Voyage/voyage/media.py  # 24 floor + max() formula
```

## Fix candidates

1. Console entry: reduce the verb list to `run`, `generate`, `finalize`, `soak` (the four that parse the flags), with a pointer to 050 for why `status` rejects them.
2. Phase-0 entry: annotate "12 commands (15 as of 2026-09-30: +sfx/benchmark/soak)" and strike/annotate the fake pass-through note with the augment default shift.
3. Augment entry + CLI help: qualify "`0` disables" → "`0` disables the 32 fps / 1280×720 floors; sources below 24 fps are still lifted to the 24 fps presentation floor" (matches the code docstring's own wording).

## Refs

- `Voyage/DESIGN.md:5815-5856,6718-6734,7514-7538`; `Voyage/voyage/media.py:670-736`; `Voyage/voyage/cli.py:1765-1789,1814,1863,1932`; `Voyage/tests/test_console.py:248-262` (050 pin); `Voyage/issues/050_*` (behavior side), `092_*` (other docs).

## Progress log (Group C, 2026-09-30)

- Verdict: all 4 drifts CONFIRMED live (DESIGN.md not owned — verify-only,
  proposals below as quoted text).
- DRIFT 1 confirmed: `docker run --rm ... voyage:latest <verb> --help |
  grep -c "verbose|no-color"` → run 4 / generate 4 / finalize 3 / soak 3 /
  sfx 3, status 0 / validate 0 / benchmark 0 / inspect 0. DESIGN:7000-7001
  still lists all eight. (Note: `sfx` also parses the flags — see 160.)
- DRIFT 2 confirmed: `--help` verb set =
  {init,doctor,models,run,generate,status,pause,resume,stop,validate,finalize,sfx,benchmark,soak,inspect}
  (15); DESIGN:6102 still says "all 12 commands".
- DRIFT 3 confirmed: `media.py:833 out_fps = max(requested, floor,
  PRESENTATION_MIN_FPS)` with `:767 PRESENTATION_MIN_FPS = 24` and the
  code docstring `:809-815` admitting "0 disables the new 32fps floor, not
  the shipped-video guarantee". CausVid (fps 16) with floors 0 still lifts
  16→24. CLI help text for the flags lives in `cli*.py` (not owned).
- DRIFT 4 confirmed: fake preset is 768x432@24 (`config.py:167-179`) while
  default floors ship 1280x720@32 — the Phase-0 "pass-through" note is stale.
- Files changed: none (DESIGN.md is orchestrator-owned).

## Resolution

- No edit applied (out of scope). Proposed DESIGN text for the owner:
- 1. Console entry: "reduce the verb list to `run`, `generate`,
  `finalize`, `sfx`, `soak` (the five that parse the flags), with a
  pointer to 050 for why `status` rejects them."
- 2. Phase-0 entry: "annotate '12 commands (15 as of 2026-09-30:
  +sfx/benchmark/soak)' and strike/annotate the fake pass-through note
  with the augment default shift."
- 3. Augment entry + CLI help: "qualify '`0` disables' → '`0` disables
  the 32 fps / 1280x720 floors; sources below 24 fps are still lifted to
  the 24 fps presentation floor' (matches the code docstring's own
  wording)."
- Residual: CLI help overclaim (`cli*.py` augment-flag help) belongs to
  the CLI-owning track.
