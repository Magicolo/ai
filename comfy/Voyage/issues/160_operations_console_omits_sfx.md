# 160 — OPERATIONS.md console-flags list omits the `sfx` verb (which accepts them)

- **Severity:** LOW (docs-only — `sfx` honors `--verbose`/`--no-color`, the runbook says it does not)
- **File:line:** `Voyage/docs/OPERATIONS.md:39-41` vs `Voyage/voyage/cli.py:2007` (`_add_sfx_args(sfx, include_no_sfx=False)` + `_add_console_args(sfx)`) and `Voyage/voyage/cli.py:145-156` (`_add_console_args` def); contrast `Voyage/voyage/cli.py:1935-1953` (`status`/`pause`/`resume` parsers, correctly without console args)
- **Area:** CLI/docs drift (SFX verb grew console flags; the allow-list sentence never followed)

## Description

`docs/OPERATIONS.md:39-41` states:

```
`--verbose` / `--no-color` are honored on `run`, `generate`,
`finalize`, and `soak` only — `status`, `validate`, `benchmark`, and
`inspect` print plain text and do not accept those flags.
```

But the `sfx` verb wires the same console flags as the honored four:

```python
# cli.py:2006-2008
_add_sfx_args(sfx, include_no_sfx=False)
_add_console_args(sfx)
sfx.set_defaults(func=cmd_sfx)
```

where `_add_console_args` (`cli.py:145-156`) adds exactly `--verbose`
and `--no-color`. So `voyage sfx --verbose` / `--no-color` parse and
take effect (via `get_console(args)`), yet the doc's "only" sentence
tells the user they do not exist on `sfx`. The second half of the
sentence is accurate: `status` (`cli.py:1935-1939`), `pause`
(`cli.py:1942-1946`), and `resume` (`cli.py:1949-1953`) add only
`--run` and never call `_add_console_args` — they genuinely print
plain text. The omission is `sfx` alone: it belongs in the honored
list but appears in neither list.

## Rationale

A user scripting non-TTY finalization dubs (`sfx` onto an existing
video) will read the "only" sentence, conclude `--no-color` is
unsupported there, and either omit it (ANSI escapes in captured logs)
or avoid the verb's flags entirely. The fix is a one-sentence doc
edit; the code already does the right thing.

## Live evidence

- `sed -n '39,41p' docs/OPERATIONS.md` — honored list is
  `run`/`generate`/`finalize`/`soak` "only"; `sfx` appears nowhere in
  the paragraph (nor anywhere else in the console-flags context —
  `grep -n "sfx" docs/OPERATIONS.md` hits only the models-ensure `line
  193` and unrelated sections).
- `sed -n '1992,2008p' voyage/cli.py` — `_add_sfx_parser` calls
  `_add_console_args(sfx)` at `:2007`.
- `sed -n '145,156p' voyage/cli.py` — the helper adds `--verbose` and
  `--no-color` only.
- `sed -n '1935,1953p' voyage/cli.py` — `status`/`pause`/`resume`
  parsers add `--run` only, no console-args call: the doc is right
  about them, wrong only about `sfx`.
- Wiring completeness: `_add_console_args` call sites are `run`
  (`:1870`), `generate` (`:1931`), `finalize` (`:1988`), `sfx`
  (`:2007`), `soak` (`:2042`) — five verbs accept the flags, the doc
  names four.

## Repro

Static: `grep -n "_add_console_args" voyage/cli.py` (five hits incl.
`:2007`) vs the four-verb list at `docs/OPERATIONS.md:39-41`.
Dynamic: `voyage sfx --help` shows `--verbose`/`--no-color`;
`voyage status --help` does not.

## Fix candidates

1. Amend the sentence to "`run`, `generate`, `finalize`, `sfx`, and
   `soak` only — …" (one word + keep the plain-text list as is).
2. Alternatively enumerate the negative list with `sfx` removed from
   it — equivalent; prefer option 1 (smaller diff).
3. Test: optional docs-vs-parser assertion that every
   `_add_console_args` call site appears in the OPERATIONS allow-list
   (cheap grep test, guards the next verb the same way).

## Refs

- `Voyage/docs/OPERATIONS.md:39-41`; `Voyage/voyage/cli.py:145-156,1870,1931,1935-1953,1988,2007,2042`.
- Adjacent, not overlapping: 028 (console contract — output shape, not
  the verb allow-list); 063 (stream-split/timing — renderer internals).
