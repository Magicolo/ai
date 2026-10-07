# 204 — `qualify.sh` run binding is broken: checks `<run-dir>/manifest.json` then generates `output/<basename>` — different runs — and `--segments N` is silently ignored (HIGH)

## Technical description
Header comment requires `<run-dir> MUST be absolute AND equal to
$PWD/output/<basename>`, but code enforces only absoluteness
(`scripts/qualify.sh:100-113`). `run.sh generate "$(basename "$run_dir")"`
(`:128`) resolves `output/<basename>` under the repo root. Verified live
2026-10-07:

```
if [ ! -f "$run_dir/manifest.json" ]; then ... exit 2; fi   # scripts/qualify.sh:120
./scripts/run.sh generate "$(basename "$run_dir")"          # scripts/qualify.sh:128
artifact="reports/qual-$(basename "$run_dir")-$(date +%F).json"  # :135
```

No `[[ "$run_dir" == "$PWD/output/$(basename ...)" ]]` check anywhere.

## Rationale
Qualification artifacts (`reports/qual-*.json`) then attest the wrong run.
The absolute-path gate (issue 064 leg b) was added precisely because relative
dirs "double up inside payload paths" — this reintroduces the same class one
layer up.

## Live evidence
See `sed -n '100,140p'` output above: absolute gate + disk preflight +
manifest gate, then basename-derived generate with no equality check.

## Repro
`./scripts/qualify.sh /tmp/qual-ltxv` with a valid
`/tmp/qual-ltxv/manifest.json` and no `output/qual-ltxv/manifest.json` →
manifest gate passes, `generate qual-ltxv` exits 2 missing-manifest (or
worse, generates an unrelated `output/qual-ltxv`).

## Source refs
- `Voyage/scripts/qualify.sh:7-11,100-128`

## Online sources
- Fail-closed shell gating practice (explicit exit 2/4/5 gates, as
  `qualify.sh` itself uses for path/disk/manifest).

## Fix candidates
1. Enforce `run_dir == $PWD/output/<basename>` (canonicalize + compare,
   exit 2 otherwise).
2. Or accept `--name` and derive `run_dir` instead of the reverse.
3. Add a shell test with `/tmp` vs `output/` divergence.

## Log
- Track B sweep, 2026-10-07. Verified live by orchestrator 2026-10-07.
  Read-only; nothing fixed.

## Consolidated from 217_qualify_segments_ignored (2026-10-07)

### Technical description (from 217)
`scripts/qualify.sh:33-34` defaults `segments=3`; `:46-53` parses
`--segments/--segments=`; `:79-84` validates positivity. The value is never
used: `:128` runs `./scripts/run.sh generate "$(basename "$run_dir")"` with
no segment forwarding, and the manifest gate (`:120-125`) demands a
pre-existing `manifest.json` (i.e. the stored plan wins).

```
$ grep -n "segments" scripts/qualify.sh
34:segments="3"
46:    --segments) … 50:    --segments=*) …
79:case "$segments" in … 128:./scripts/run.sh generate "$(basename "$run_dir")"
```
`$segments` appears on lines 4,6,34,46,50,55,69,79,81,122 — never on the
`generate` line.

### Rationale (from 217)
The §137A driver is the only documented path to "benchmarks, runs,
validates, tees JSON to `reports/`". A `--segments 9` invocation validates
`9`, prints nothing about ignoring it, and qualifies whatever the manifest
stored — results in `reports/qual-*.json` are mislabeled by construction.

### Live evidence (from 217)
See grep output above.

### Repro (from 217)
`mkdir -p output/q && echo '{"segments":2,…}' > output/q/manifest.json;
./scripts/qualify.sh --segments 9 $PWD/output/q` → generates 2 segments,
artifact named `qual-q-<date>.json` with no record of the requested 9.

### Source refs (from 217)
- `Voyage/scripts/qualify.sh:4-6,33-53,79-84,120-139`
- `Voyage/docs/BENCHMARKING.md:69-87`
- `Voyage/reports/video-backends.md:8-12`

### Online sources (from 217)
- CLI doctrine: parsed-and-validated-but-unused flags are silent
  mislabeling (cf. #203 — same failure class).

### Fix candidates (from 217)
1. Forward via the approved shorthand: `generate <name> --segments N`
   (extends stored plan) or `configure <name> --segments N` first; record
   requested vs rendered counts in the artifact.
2. Or delete the flag and require manifest pre-configuration (update usage +
   `docs/BENCHMARKING.md` together).

### Log (from 217)
- Track E sweep, 2026-10-07. Read-only; nothing fixed.
