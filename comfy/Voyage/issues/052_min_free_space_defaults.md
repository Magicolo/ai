# 052 — `min_free_space_gib` triple default confuses spec readers; reserve invisible in `status`

- Status: open
- Severity: low-medium (small-dev-box validation aborts from an omitted key)
- Area: config/docs — `voyage/config.py:247,292`, DESIGN, `docs/`
- Rank rationale: documented in three places but the failure mode (implicit 20
  GiB on hand-written TOML) still bites, and `status` can't show which default
  applies.

## Technical description

Three truths: pydantic default `20.0`, generated TOML `5.0`, spec example `20`.
DESIGN §140 + TROUBLESHOOTING + STATE_AND_RECOVERY document "20 spec, 5 dev,
raise for production" — so not undocumented. The remaining papercut: a
hand-written TOML omitting the key silently gets 20 GiB (validation abort on
small dev boxes), while every `init`-generated run gets 5 GiB. `status` shows
`Free: X GiB` but never the configured reserve, so users can't tell which default
they're under.

## Why this is an issue

A small dev box with 12 GiB free will abort validation on a hand-written
TOML (implicit 20 GiB reserve) while an `init`-generated run on the same box
succeeds (5 GiB) — same machine, same workload, opposite outcomes, with no
visible indication of which default applied. Users debug "disk full" errors
that are really config-default confusion. Blast radius is limited to
hand-written configs on small disks, and the fix is display-only (surface
the reserve in `status`/`init`) plus a default-alignment decision.

## Evidence

```
$ sed -n '247p;292p' Voyage/voyage/config.py
    min_free_space_gib: float = 20.0      # pydantic model default
min_free_space_gib = 5.0                  # init-generated TOML default
```

`voyage/config.py:247,292`; `DESIGN.md:2784,5807`;
`docs/STATE_AND_RECOVERY.md:41`; `docs/TROUBLESHOOTING.md:44-46`.

## Reproduction

Hand-write a TOML without the key on a <20 GiB-free box → validation abort that
`init`-generated runs don't hit.

## Source references

- Files/lines above.

## Resolution candidates

Emit the reserve in `status` Storage section + in `cmd_init`'s confirmation line;
consider aligning the model default with the generator (or warn on implicit
default in `load_config`).

## Investigation / progress / resolution log

- 2026-09-25: found by docs sweep.
- 2026-09-25 (repair): re-verified refs live (`config.py:247` pydantic 20.0,
  `:292` generator 5.0 — output pasted in Evidence). Added
  `## Why this is an issue`. No staleness.
- Open: surface the reserve; decide on default alignment.
