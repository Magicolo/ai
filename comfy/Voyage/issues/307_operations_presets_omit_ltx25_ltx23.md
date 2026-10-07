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
