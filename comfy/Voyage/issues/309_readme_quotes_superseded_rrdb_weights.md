# 309 — Model docs quote stale pins/headers: README RRDB size, MODELS.md license note + LTXV "(default)"

Severity: MEDIUM (pass-2 DESIGN-docs drift sweep).

## Technical description

`README.md:46` lists `realesrgan-anime ~18 MB`. The pinned weight is the 2.5 MB SRVGG
`realesr-animevideov3.pth`; the 18 MB `RealESRGAN_x4plus_anime_6B` RRDB is documented as
superseded (~11-13× slower).

## Rationale

Users provision/verify the wrong file; size mismatch looks like a corrupt download.

## Live evidence

- Docs: `README.md:46` (~18 MB) vs `docs/MODELS.md:78-88` (~2.5 MB SRVGG, "old 18 MB …
  superseded").
- Code: `voyage/registry_realesrgan.py:39`
  (`REALESRGAN_ANIME_FILE = "realesr-animevideov3.pth"`), `:24-30` (2,504,012 bytes,
  0.055 vs 0.63 s/frame).
- Command: `grep -n "realesrgan" README.md docs/MODELS.md voyage/registry_realesrgan.py`.

Repro: compare the two doc lines + registry constant.

## Source refs

`README.md:46`; `docs/MODELS.md:78-88`; `voyage/registry_realesrgan.py:24-39`.

## Online sources

- None (in-tree SRVGG swap is the anchor).

## Fix candidates

- `~2.5 MB SRVGG (realesr-animevideov3.pth)` in README; keep the 18 MB RRDB mention only
  as superseded.

## Log

- 2026-10-07: filed from read-only pass-2 docs-drift sweep; no code touched.

## Consolidated from 311_models_doc_default_header_and_license_note_stale (2026-10-07)

Severity: LOW-MEDIUM (pass-2 DESIGN-docs drift sweep).

### Technical description

(a) `docs/MODELS.md:10` heads LTXV-2B as "(default)" while the config default is `ltx25`;
(b) `docs/MODELS.md:172-173` says "Voyage pins the Real-ESRGAN anime 6B mirror above"
when the artifact above is the SRVGG XS file.

### Rationale

Provisioning + license review reference the wrong pin.

### Live evidence

`docs/MODELS.md:10` vs `voyage/config.py:240`; `docs/MODELS.md:172-173` vs `:78-88` +
`voyage/registry_realesrgan.py:39`.

Repro: compare header vs config default; license bullet vs the pinned SRVGG file.

### Source refs

`docs/MODELS.md:10,78-88,172-173`; `voyage/config.py:240`;
`voyage/registry_realesrgan.py:39`.

### Online sources

- None.

### Fix candidates

- LTXV header → "(ltxv preset; config default is ltx25)"; license bullet → "pins the
  realesr-animevideov3 SRVGG mirror above."

### Log

- 2026-10-07: filed from read-only pass-2 docs-drift sweep; no code touched.
- 2026-10-07: consolidated into 309 (same stale-pin docs-drift class across README +
  MODELS.md).

## Evaluation (2026-10-07)

Both claims re-verified live. (a) README: `README.md:46` listed
`realesrgan-anime ~18 MB` while `voyage/registry_realesrgan.py:24-30,39` pins
`realesr-animevideov3.pth` (2,504,012 bytes SRVGGNetCompact XS, ~11.5x faster
than the RRDB) and `docs/MODELS.md:83-93` documents ~2.5 MB with the 18 MB
`RealESRGAN_x4plus_anime_6B` RRDB marked superseded. (b) Consolidated 311:
`docs/MODELS.md:15` headed LTXV-2B as "(default)" while `voyage/config.py:240`
defaults to `ltx25`, and `docs/MODELS.md:209-210` said "pins the Real-ESRGAN
anime 6B mirror above" while the artifact above (`:83-87`) is the SRVGG XS file.
All file:line refs fresh. No downgrade: fixed README + both MODELS.md spots
(311's consolidation into 309 covers them in this scope).

## Progress log

- 2026-10-07: `README.md:47` → `realesrgan-anime ~2.5 MB SRVGG (BSD-3-Clause)`;
  `docs/MODELS.md:15` header → "`ltxv` preset; config default is `ltx25`";
  `docs/MODELS.md:209-210` license bullet → "pins the realesr-animevideov3 SRVGG
  mirror above instead". Grep-verified against
  `voyage/registry_realesrgan.py:39` + `voyage/config.py:240`; no `~18 MB` /
  stale-default remnants.

## Resolution (2026-10-07)

RESOLVED. Provisioning/size/license references match the pinned SRVGG weight and
the ltx25 default; nothing left open.
