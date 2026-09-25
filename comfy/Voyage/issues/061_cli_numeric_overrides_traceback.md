# 061 — CLI numeric overrides raise unhandled `ValidationError` traceback instead of a clean error

- Status: open
- Severity: medium (Python traceback + non-2 exit for plain user input errors)
- Area: CLI — `voyage/cli.py:283-304` (`cmd_run`), `:783-823` (`cmd_generate`),
  `voyage/config.py:173-178,114-119`
- Rank rationale: pass-2 CLI finding; every `--blocks 0`-class typo dumps a
  traceback.

## Technical description

```python
config = apply_draft_overrides(config, draft=args.draft, director=args.director,
    blocks=args.blocks, take_seconds=args.take_seconds, ...)
```

`main()` (`cli.py:1328-1332`) catches only `VoyageError`.
`apply_draft_overrides` rebuilds `VideoConfig/AudioConfig/VoyageConfig`, whose
`positive` validators raise pydantic `ValidationError` for `blocks=0/-2`,
`beats_per_segment=0`, `drift_every_n=0`, `take_seconds=-1`.

## Why this is an issue

A plain user typo (`--blocks 0`) dumps a Python traceback with a non-standard
exit instead of a clean usage error: the actual constraint ("must be positive")
is buried in pydantic internals rather than reported at the CLI layer. This
breaks script composability (callers expect exit 2 + stderr for input errors)
and makes every numeric override flag a small UX trap.

## Evidence (live probe by pass-2 sweep, real run dir)

```
{'blocks': 0} -> RAISED ValidationError: 1 validation error for VideoConfig
  blocks_per_segment Value error, must be positive
{'beats_per_segment': 0} -> RAISED ValidationError ... AudioConfig ...
{'drift_every_n': 0} -> RAISED ValidationError ... VoyageConfig ...
main(['run','--run',...,'--blocks','0']) -> main raised ValidationError
```

## Reproduction

`voyage init … && voyage run --run <dir> --segments 1 --blocks 0` → traceback,
no `exit 2`/stderr message.

## Source references

- Files/lines above.

## Resolution candidates

Catch `ValidationError` in `cmd_run`/`cmd_generate` (print `error: …` to stderr,
return 2), or add argparse-level `type=` validators for `--blocks/
--beats-per-segment/--drift-every-n/--take-seconds`.

## Investigation / progress / resolution log

- 2026-09-25: found by pass-2 CLI/TUI sweep with live probes.
- 2026-09-25: issue-file repair — added `## Why this is an issue`; re-verified
  cited lines live (`cli.py:283-304` `cmd_run`, `:783-823` `cmd_generate`,
  `main` catches only `VoyageError` at `:1328-1332`, `config.py` positive
  validators at `:114-119`/`:173-178` — all match).
- Open: implement + test (each flag, exit code 2, stderr message).
