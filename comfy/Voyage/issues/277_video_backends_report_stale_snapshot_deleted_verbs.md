# 277 — `reports/video-backends.md` is a 2026-09-24 snapshot referencing deleted verbs and a permanently "pending" leg

Severity: LOW (track E-13).

## Technical description

Environment table frozen 2026-09-24 (driver 595.84, image `6a55a3691940`); harness note
cites `run.sh init/run --segments 3` (both verbs deleted; `init` since 2026-10-04); "LTXV
leg (pending Stream A)" (`:51-56`) while Streams A–D, qual-driven fixes, and the
RIFE/SRVGG finalize stack all landed since; continuity gate prose duplicates
`boundary_metrics.BOUNDARY_RATIO_LIMIT` without referencing it.

## Rationale

§137A says "no static ranking overrides benchmark results; every GPU-gated field is
measured or PENDING". A report that is all-stale-but-unmarked invites citing fake-backend
testsrc numbers (144f VALID, 2.16 ratio) as GPU evidence.

## Live evidence

`reports/video-backends.md:30,44-56` vs `voyage/cli.py` (no `init`/`run`),
`DESIGN.md:7063-7133` (Stream B harness + longlive2 GPU leg landed after the report).

Repro: follow the report's `run.sh init/run` line → `invalid choice (exit 2)`.

## Source refs

`reports/video-backends.md:1-56`; `scripts/qualify.sh:1-20` (current driver the report
predates).

## Online sources

- None (in-tree §137A contract is the anchor).

## Fix candidates

- Header-banner the file as `ARCHIVED 2026-09-24 (fake-backend harness only; verbs
  init/run removed)` with a pointer to `qualify.sh` + `reports/qual-*.json`, or refresh
  the legs from current artifacts.

## Log

- 2026-10-07: filed from read-only Track E sweep; no code touched.

## Evaluation (2026-10-07)

Live-verified RELEVANT, no integrity deltas. `reports/video-backends.md:19-30`
(Environment frozen 2026-09-24, `run.sh init/run --segments 3` — both verbs
deleted, parser is `configure`+`generate` only) and `:51-56` ("LTXV leg (pending
Stream A)" while Streams A–D, qual-driven fixes, and the RIFE/SRVGG finalize
stack all landed since) reproduce exactly as reported. Per mandate the file
gets a header banner marking it ARCHIVED 2026-09-24 (fake-backend harness
only) with a pointer to `scripts/qualify.sh` + `reports/qual-*.json`;
historical numbers below the banner are left untouched.

## Progress log (2026-10-07)

- `reports/video-backends.md`: header banner added (ARCHIVED 2026-09-24,
  fake-backend harness only, `init`/`run` removed, numbers not GPU evidence;
  pointer to `scripts/qualify.sh` + `reports/qual-*.json`). All 56 historical
  lines below the banner byte-untouched.
- Verify: banner present; `run.sh init/run` now occurs only below the banner
  (historical snapshot, intentional).

## Resolution (2026-10-07)

Verdict: RESOLVED. Report is bannered as archived with a live-driver pointer;
nothing left open.
