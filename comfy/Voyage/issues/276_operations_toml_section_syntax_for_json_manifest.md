# 276 — Doc drift: TOML `[section]` syntax + deleted-verb pins across OPERATIONS.md and SFX.md

Severity: MEDIUM (merged scope: track E-12 + pass-2 TUI-remnant sweep).

## Technical description

OPERATIONS finalize section: "`--upscale`/`--interpolate`/`--presentation-fps` on
`configure`, stored in `[augment]`". `voyage/config.py:5` states "structured data inside
`manifest.json` — there is no TOML layer". `configure` writes flat `manifest.json`
(`voyage/cli_configure.py`, `paths.SEGMENT_MANIFEST_FILENAME`).

## Rationale

`[augment]` is TOML-section syntax from the deleted `voyage.toml` era. Operators go
looking for a `[augment]` table to hand-edit; none exists.

## Live evidence

`grep -n "\[augment\]" docs/OPERATIONS.md` → line 112; `sed -n '1,10p' voyage/config.py`
→ "there is no TOML layer"; `grep -rn "toml" voyage/cli_configure.py` → no TOML read
path.

Repro: `grep -rn "^\[augment\]" output/<run>/manifest.json` → no match (flat JSON keys).

## Source refs

`docs/OPERATIONS.md:105-116`; `voyage/config.py:5,736`; `docs/AUGMENT.md:14-45`
(correctly describes `configure` flags + manifest keys — use its wording).

## Online sources

- None (in-tree config doctrine is the anchor).

## Fix candidates

- Rephrase to "stored in the manifest's `augment` object"; sweep `[section]` TOML-isms
  from OPERATIONS (`[augment]`, `[director]`, `[experimental]` remnants) in one pass.

## Log

- 2026-10-07: filed from read-only Track E sweep; no code touched.

## Consolidated from 300_sfx_doc_toml_block_and_deleted_verbs (2026-10-07)

Severity: MEDIUM (pass-2 TUI-remnant sweep; same class as the OPERATIONS.md finding
above, different file).

### Technical description

TOML layer deleted (`voyage/config.py:5` "no TOML layer"; `voyage.toml` absent per `ls`).
`SFX.md` still shows a ```toml `[sfx]` block and pins behavior to deleted verbs. The
`OPERATIONS.md:112 [augment]` aspect is the canonical finding above.

### Rationale

Operators copy a config syntax that no reader parses, for verbs that exit 2.

### Live evidence

```
$ rg -n "TOML|toml|sfx verb|run.*generate|models download" docs/SFX.md
docs/SFX.md:76:Config (`voyage/config.py:361`, `[sfx]` TOML):
docs/SFX.md:78-85:```toml [sfx] backend/device/models_dir/model_size/dual_pan
docs/SFX.md:46,158: (`models download sfx-mmaudio` ...)
docs/SFX.md:118:--sfx-caption` (finalize + `sfx` verb)
docs/SFX.md:120:--music-caption`/`--video-caption` (`run`/`generate`)
docs/SFX.md:122-124:`--no-sfx` ... The standalone `sfx` verb has no `--no-sfx` — the verb IS the pass.
docs/SFX.md:143:`--sfx-device` override and `[sfx] device` TOML move it.
docs/SFX.md:116: Pins (all in-memory — ... no TOML):  ← contradicts :76-85 in same file
```

Repro: compare `SFX.md:76-85` vs `config.py:5` + `ls voyage.toml` (absent);
`voyage sfx` → `invalid choice`.

### Source refs

`docs/SFX.md:46,76-85,116-124,143,158`.

### Online sources

- None (in-tree CLI-is-config migration is the anchor).

### Fix candidates

- Replace ```toml block with manifest-JSON keys (copy `AUGMENT.md:14-45` wording cited as
  correct in 276); reword verb pins to `configure`/`generate`-finalize; fix `:149`
  `voyage/cli.py:1386` dead line ref (file is 417 lines).

### Log

- 2026-10-07: filed from read-only pass-2 TUI-remnant sweep; no code touched.
- 2026-10-07: consolidated into 276 (merged OPERATIONS.md + SFX.md doc-drift scope,
  severity LOW → MEDIUM).

## Evaluation (2026-10-07)

Live-verified RELEVANT, with two integrity deltas. Confirmed: `OPERATIONS.md:112`
(`stored in [augment]`), `SFX.md:76` (`[sfx]` TOML) + `:78-85` (toml block) +
`:118` (`sfx` verb pin) + `:120` (`run` verb pin) + `:122-124` (`[sfx]` +
standalone-verb sentence) + `:146` (`[sfx] device` TOML),
`config.py:5` ("there is no TOML layer", read-only check, not edited).
Deltas: (1) the mandate's reference `AUGMENT.md:14-45` (cited as correct
wording) is itself stale live — `AUGMENT.md:24,27,29-35` still use the
`[augment]` TOML block — so AUGMENT.md is fixed first to manifest-JSON keys
wording and OPERATIONS/SFX copy that wording; (2) the `:149 voyage/cli.py:1386`
dead-ref sub-finding does NOT reproduce — live `SFX.md` cites only
`voyage/cli_planning.py:183` (`SFX.md:155`), which exists; no edit needed for
that sub-item. `cli*.py`/`config.py` untouched throughout.

## Progress log (2026-10-07)

- `docs/AUGMENT.md:24-35` fixed first (`[augment]` section → `augment`
  object; toml block → manifest-JSON keys block matching `AugmentConfig`:
  `upscale`/`interpolate`/`interp_backend`, `presentation_fps` null-default
  noted).
- `docs/OPERATIONS.md:112` → "stored in the manifest's `augment` object".
- `docs/SFX.md`: `:76` → manifest `sfx` object; `:78-85` toml block →
  manifest-JSON keys block matching `SfxConfig` (`backend`/`device`/
  `models_dir`/`model_size`/`dual_pan`, CUDA pairing note kept); `:118-124`
  verb pins → `configure`/`generate`-finalize wording (verified live:
  `--no-sfx` exists on both `configure` stored and `generate`
  this-generate-only); `:146` → manifest `sfx` object `device` key.
  `SFX.md:116` ("no TOML") and `:155` (`cli_planning.py:183`) already
  correct, untouched.
- Verify: `toml` (lowercase) 0 hits across all 8 docs; `[augment]`/`[sfx]`
  bracket-isms 0 hits in SFX/OPERATIONS/AUGMENT; `cli*.py`/`config.py`
  untouched.

## Resolution (2026-10-07)

Verdict: RESOLVED (with one NOT-REPRODUCIBLE sub-item: the `cli.py:1386`
dead-ref does not exist live — `SFX.md` cites the valid
`cli_planning.py:183`). Left open (deliberate): `TROUBLESHOOTING.md:71`
"dev TOML" default-provenance phrase (not operator syntax, uncited by this
issue); `cli.py` `[sfx]`/`[augment]` help strings + `config.py` TOML comments
(forbidden files).
