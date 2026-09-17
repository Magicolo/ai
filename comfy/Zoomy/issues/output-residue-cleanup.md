# E2E residue accumulating under `Zoomy/output/`

- Severity: low (chore — gitignored, but hides real artifacts and bloats
  the bind mount).
- Status: FIXED. Sep-16 demo residue (81 `z_image/` frames + both
  `z_image_00001*.mp4`, ~50 MB) deleted from inside the container via the
  new `Zoomy/scripts/clean-output.sh` (explicit relative names/patterns
  only; refuses empty runs, absolute paths, `..`, and `.gitkeep`; globs
  expand in-container so host-side typos cannot reach `Comfy/output`).
  `/output` now holds only `.gitkeep`; `git status` clean.

## Evidence

`Zoomy/output/` currently contains post-verification renders
(`Zoomy_z_image_00001*.mp4` + frames). Gitignored per root `.gitignore`,
so no commit risk — but the AGENTS.md §8 cleanup
(`rm` from *inside* the container — host lacks perms on root-owned files)
was not done after the last e2e, and `Zoomy_` vs `Z_Zoom_`/`Ernie_Zoom_`
prefix discipline only helps if residue never piles up.

## Fix

Clean from inside the zoomy container (never `Zoomy_*` by sloppy glob
from the host — `Z_Zoom_*`/`Ernie_Zoom_*` are the user's Comfy files and
must not match):

```bash
docker compose run --rm --no-deps zoomy sh -c 'rm -f /output/Zoomy_z_image_* …'
```

Keep only artifacts the user explicitly wants. Consider a
`Zoomy/scripts/clean-output.sh` allowlist-based cleaner so cleanup is
one reviewed command, not a hand-typed glob.

## Verification

- `ls` inside the container shows only intended files; host `git status`
  clean.
