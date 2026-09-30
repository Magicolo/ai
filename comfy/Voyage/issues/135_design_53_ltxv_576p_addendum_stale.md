# 135 — DESIGN §5.3 addendum still claims the ltxv preset moved to 1024x576 (`ltxv-576p`); live preset is 768x512 (`ltxv-512p`), revert documented 1200 lines later

- **Severity:** LOW (docs — normative spec section contradicts the live default and the document's own later revert note)
- **File:line:**
  - Stale claim: `Voyage/DESIGN.md:545-550` ("Addendum (same day, after Stream A E2E): the `ltxv` preset in `config.py` was moved 768x512 → **1024x576** (`ltxv-576p`) … the worker is geometry-agnostic, so no Stream A code change was needed").
  - Revert note (same document, same day): `Voyage/DESIGN.md:6638-6646` ("Resolution: ltxv preset stays 768x512 … Native 1024x576 was tried and reverted the same day: the forward needs ~15.6 GB … beyond the 16 GB card").
  - Live code: `Voyage/voyage/config.py:165-170` (`"ltxv": BackendRecord(profile="ltxv-512p", width=768, height=512, …)`); `video_ltxv.py:811` (request defaults 768/512), `:858-859` (benchmark defaults 768/512).
- **Area:** DESIGN as-built/admonition drift — §5.3 normative backend spec (below both previous windows; 092 covers §5.1 + §§56-57 + §87, never this §5.3 addendum)

## Description

§5.3 is the normative LTXV backend section (granularity verdict + segment accounting + tail + tape). Its closing addendum states the preset *was moved* to 1024x576 as a completed fact. Twelve hundred lines later, the §140 rhythm-cut log records that the move was tried and **reverted the same day** (forward OOMs the 16 GB card at ~15.6 GB; the `torchao.quantization.quant` import break found during the attempt is also fixed), with the explicit resolution "ltxv preset stays 768x512". The config registry agrees with the revert (`ltxv-512p`, 768x512). The §5.3 addendum was never struck — so the spec section an implementer reads first prescribes a geometry the qualification run (6791-6792: "1024x576 LTXV probe: deterministic OOM 3/3 … the 768x512 preset stands") proved unshippable, while the correction lives only in a dated log entry the spec never points back to.

## Rationale

Stale geometry in a normative section causes real under-provisioning and wasted probes: a reader following §5.3 bottom-up configures (or benchmarks against) 1024x576, reproduces the 3/3 OOM, and only discovers the revert by reading to line 6638. The document's own convention supports the fix — later sections already carry "stays 768x512" resolutions (6638, 6791) — the §5.3 addendum just needs the one-line strike plus a pointer, not a rewrite.

## Evidence (verified live 2026-09-30)

- `sed -n '545,550p' Voyage/DESIGN.md` — the move claim verbatim (present tense, no revert pointer).
- `sed -n '6638,6646p' Voyage/DESIGN.md` — the revert verbatim ("stays 768x512", "~15.6 GB", import fix).
- `grep -n -A5 '"ltxv": BackendRecord' Voyage/voyage/config.py` → `profile="ltxv-512p", width=768, height=512` (lines 165-170).
- In-container probe (`voyage:latest`): `from voyage.config import BACKEND_REGISTRY; BACKEND_REGISTRY['ltxv']` → `768x512 ltxv-512p seg=96` (preset row, not an override).
- Overlap check: 092 names `DESIGN §5.1` (CausVid worker entry), `§§56-57` (768x432 finalize geometry), `§87` (doc list) — `grep -n "5\.3\|545\|1024x576\|576p" issues/092_*` → zero hits. 042/065 are README/INSTALL geometry and download lists, not this addendum. No existing file cites lines 545-550.

## Repro

```bash
sed -n '545,550p' Voyage/DESIGN.md   # claims 1024x576 move
sed -n '6638,6646p' Voyage/DESIGN.md # documents same-day revert
grep -n -A4 '"ltxv": BackendRecord' Voyage/voyage/config.py  # live: 768x512 ltxv-512p
```

## Fix candidates

1. Strike the §5.3 addendum in place: keep one line of history ("1024x576 tried 2026-09-24, reverted — forward needs ~15.6 GB, see §140 rhythm-cut note") and restore the normative sentence to 768x512 (`ltxv-512p`), matching config + the 6791 probe verdict.
2. Leave the §140 log untouched (history stays history) — the fix is a forward pointer from the spec, not a rewrite of the log.
3. Gate: `rg -n "576p|1024x576" DESIGN.md` should show only historical/log contexts after the fix, zero normative prescriptions.

## Refs

- `Voyage/DESIGN.md:500-550` (§5.3 verdict + accounting + addendum), `:6638-6646` + `:6791-6792` (revert + probe verdict), `:6539` (presets at 768x512); `Voyage/voyage/config.py:117-204`; `Voyage/voyage/workers/video_ltxv.py:56-58,811,858-859`.
- Adjacent, not overlapping: 092 (docs batch — §5.1/§§56-57/§87, never §5.3:545-550); 042/065 (README/INSTALL geometry-downloads, not DESIGN §5.3).

## Progress log (Group C, 2026-09-30)

- Verdict: CONFIRMED live (DESIGN.md not owned — verify-only, proposal
  below as quoted text). Live line numbers drifted from the filing:
  addendum now at `DESIGN.md:556-561` ("the `ltxv` preset in `config.py`
  was moved 768x512 → **1024x576** (`ltxv-576p`) ... All Stream A live
  evidence below is at 768x512; a 1024x576 render ... still needs its own
  VRAM/throughput probe"); revert now at `:6910-6919` ("Resolution: ltxv
  preset stays 768x512 ... Native 1024x576 was tried and reverted the
  same day: the forward needs ~15.6 GB ... beyond the 16 GB card").
- Live code agrees with the revert: `config.py:197-204` =
  `"ltxv": BackendRecord(profile="ltxv-512p", width=768, height=512, ...)`.
- Files changed: none (DESIGN.md is orchestrator-owned).

## Resolution

- No edit applied (out of scope). Proposed DESIGN text for the owner:
  "Strike the §5.3 addendum in place: keep one line of history
  ('1024x576 tried 2026-09-24, reverted — forward needs ~15.6 GB, see
  §140 rhythm-cut note') and restore the normative sentence to 768x512
  (`ltxv-512p`), matching config + the probe verdict. Leave the §140 log
  untouched. Gate: `rg -n '576p|1024x576' DESIGN.md` shows only
  historical/log contexts after the fix."
- Residual: none in this file's scope.
