# 307 — OPERATIONS.md backend presets + image-selection omit ltx25/ltx23

Severity: MEDIUM (pass-2 DESIGN-docs drift sweep).

## Technical description

Presets paragraph lists only `ltxv` (768×512/cuda:0), `causvid` (832×480@16), `fake`
(CPU); image-selection sentence lists (`ltxv`, `causvid`, `acestep`). Default backend
`ltx25` (1216×704@24, `voyage-ltx` image) and `ltx23` are missing; `mmaudio`/`ltx25`/
`ltx23` CUDA triggers missing.

## Rationale

A default-config run's geometry/device/image choice is undocumented on the runbook page;
users guess the ltxv row.

## Live evidence

- Docs: `docs/OPERATIONS.md:61-64` (presets), `:74-76` (image selection).
- Code: `voyage/config.py:204-232` (`ltx25` 1216×704@24 + `ltx23` rows, `_DEFAULT_ROW =
  BACKEND_REGISTRY["ltx25"]` at `:240`, `VideoConfig.backend = "ltx25"` at `:333`);
  `scripts/run.sh:89-111` (CUDA set includes `ltxv|causvid|acestep|mmaudio|ltx25|ltx23`,
  `ltx25|ltx23` → `voyage-ltx`).
- Command: `grep -n "ltx25\|ltx23" scripts/run.sh | head` vs
  `grep -n "ltxv.*causvid.*acestep" docs/OPERATIONS.md`.

Repro: `configure vdemo` (no `--backend`) → manifest video backend `ltx25`, geometry
1216×704 — neither appears in OPERATIONS presets.

## Source refs

As above.

## Online sources

- None (code + run.sh are the authority).

## Fix candidates

- Add `ltx25` (1216×704@24/cuda:0, `voyage-ltx`) + `ltx23` rows to both paragraphs; mirror
  run.sh's full CUDA set.

## Log

- 2026-10-07: filed from read-only pass-2 docs-drift sweep; no code touched.

## Evaluation (2026-10-07)

Claim re-verified live: `docs/OPERATIONS.md:61-64` presets listed only
ltxv/causvid/fake and `:74-76` image selection listed ltxv/causvid/acestep, while
`voyage/config.py:117-240` carries 5 rows with `_DEFAULT_ROW =
BACKEND_REGISTRY["ltx25"]` (`:240`, `VideoConfig.backend = "ltx25"` at `:333`)
and `scripts/run.sh:144` CUDA set is `{ltxv,causvid,acestep,mmaudio,ltx25,ltx23}`
plus the `llama` director branch (`:195-200`, ltx→`voyage-ltx:latest`). File:line
refs fresh. No downgrade: full fix scope applied.

## Progress log

- 2026-10-07: `docs/OPERATIONS.md` presets paragraph now lists `ltx25` (1216×704 @
  24 fps on `cuda:0`, the default) + `ltx23` rows and pairs ACE-Step music on
  `ltx25`/`ltx23`/`ltxv`/`causvid`; image-selection sentence now mirrors run.sh's
  full CUDA set with the `voyage-ltx` vs `voyage-video` split (`ltx25`/`ltx23` →
  `voyage-ltx:latest`; rest + `llama` sidecar → `voyage-video:latest`).
  Grep-verified against `voyage/config.py` + `scripts/run.sh`.

## Resolution (2026-10-07)

RESOLVED. Default-config geometry/device/image choice is documented; nothing left open.
