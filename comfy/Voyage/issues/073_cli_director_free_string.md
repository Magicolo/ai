# 073 — CLI `--director` is a free string while TUI allowlists; bogus values fail late at the worker

- Status: resolved (2026-09-25, CLI track — choices done; model Literal deferred, see log)
- Severity: low-medium (late `RecoverableWorkerError` for a typo; TUI rejects
  inline)
- Area: CLI/config — `voyage/cli.py:1143-1146` (`run`), `:1217-1220`
  (`generate`), `voyage/config.py:143-153`
- Rank rationale: pass-2 finding; no prior director-validation item.

## Technical description

Argparse: `run.add_argument("--director", default=None, help="override…")` (no
`choices=`); `DirectorConfig.backend: str = "deterministic"` (no Literal/
validator). TUI `DIRECTORS = ("qwen", "deterministic")` rejects anything else
inline (`tui_state.py:147-150`).

## Why this is an issue

The CLI accepts any director string while the TUI allowlists, so typos
(`bogus`, `QWEN`, trailing spaces) sail through init and config and fail late
at the worker as a dead-stdout `RecoverableWorkerError` — after a full run
setup, with no hint the backend name was wrong. Two layers disagree on the
valid set, and the lenient one fails last.

## Evidence (live probes by pass-2 sweep)

```
DirectorConfig(backend='bogus') ACCEPTED; 'QWEN', 'qwen ' ACCEPTED
voyage run --director bogus → prints "effective settings: director=bogus …" + rule, then
RecoverableWorkerError: worker voyage.workers.video closed stdout
```

TUI with `director='bogus'` → inline `director must be one of qwen, deterministic`.

## Reproduction

`voyage run --run <dir> --segments 1 --director bogus` on a fake run.

## Source references

- Files/lines above; `voyage/tui_state.py:147-150` (good pattern, also feeds 022).

## Resolution candidates

`choices=("qwen","deterministic")` on both CLI flags + `Literal` on
`DirectorConfig.backend`.

## Investigation / progress / resolution log

- 2026-09-25: found by pass-2 CLI/TUI sweep with live probes.
- 2026-09-25: issue-file repair — added `## Why this is an issue`; re-verified
  cited lines live (`cli.py:1143-1146` run flag, `:1217-1220` generate flag,
  `config.py:143-153` `DirectorConfig`, `tui_state.py:147-150` allowlist —
  all match; no `choices=`/`Literal` on either CLI flag).
- 2026-09-25 (CLI track): FIXED the CLI half. `choices=("qwen",
  "deterministic")` on both `--director` flags (`voyage/cli.py:1320` run,
  `:1395` generate — same pair as the TUI `DIRECTORS` allowlist), so
  `bogus`/`QWEN`/trailing-space typos fail at parse (exit 2,
  `invalid choice`) instead of a late worker `RecoverableWorkerError`.
  `Literal` on `DirectorConfig.backend` NOT applied — `config.py` is
  outside this track's scope (programmatic construction still accepts
  anything; noted for the config-owning track). Tests:
  `tests/test_cli_hardening.py` (run rejects `bogus`; generate parser
  accepts `qwen`/default-None and rejects `QWEN`; deterministic director
  runs 1 fake segment). Gates green.
