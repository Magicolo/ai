# 205 — `run.sh` image sniff ignores the director backend and swallows malformed manifests (HIGH)

## Technical description
Sniff reads `video/audio/sfx.backend` from the flat manifest root and maps
`{ltxv,causvid,acestep,mmaudio,ltx25,ltx23}` → CUDA images
(`scripts/run.sh:83-103`):

```
bs = [cfg.get(s, {}).get("backend","") for s in ("video","audio","sfx")]  # run.sh:96
cuda = {"ltxv","causvid","acestep","mmaudio","ltx25","ltx23"}
```

Comment says `[director]` backends can never select the wrong image. But the
default director is `llama` (Qwen3.5 GGUF sidecar, `llama-server` binary baked
by `build-ltx.sh:28-34`), and `video=fake + director=llama` sniffs `fake` →
slim `voyage:latest`, which has no sidecar. The `fake`+`llama` combination is
the default CPU smoke shape.

## Rationale
Wrong-image selection is a late, confusing failure (`WORKER_ERROR` at decide
instead of fast image pick). The sniff's own comment overclaims.

## Live evidence
```
VOYAGE_DRY_RUN=1 bash scripts/run.sh generate testname  # no manifest → voyage-ltx:latest
VOYAGE_DRY_RUN=1 bash scripts/run.sh configure testname --backend fake --segments 1  # → voyage:latest
```
`build-ltx.sh` bakes `/opt/llama.cpp/bin/llama-server` +
`/usr/local/bin/llama-server`; slim `Dockerfile` carries no such bake.

## Repro
Configure `fake` video + default `llama` director → `run.sh generate NAME`
(dry-run shows `voyage:latest`) → director decide fails missing sidecar
binary.

## Source refs
- `Voyage/scripts/run.sh:83-112`
- `Voyage/scripts/build-ltx.sh:28-34`
- `Voyage/voyage/config.py:589-606` (`DirectorConfig.backend=llama` default)

## Online sources
- Container least-privilege: image selection is a trust boundary; select
  loudly, never by silent default.

## Fix candidates
1. Include `director.backend==llama` in the CUDA/ltx decision (or assert
   slim carries the sidecar).
2. Correct the comment.
3. Dry-run test matrix over `{fake+llama, fake+deterministic, ltx25+llama}`.

## Log
- Track B sweep, 2026-10-07. Read-only; nothing fixed.

## Consolidated from 240_run_sh_sniff_crash_swallowed_wrong_image (2026-10-07)

240 (MEDIUM, track E-05) is a concurrent duplicate-wave finding on the same `run.sh:83-103` sniff: non-dict manifest sections crash the comprehension and the crash is silenced into the same silent wrong-image selection.

### Technical description

`scripts/run.sh:93-98` runs `cfg.get(s,{}).get("backend","")` for `video/audio/sfx` with `2>/dev/null || true`. If any section is a non-dict (string, list, null — all reachable from a hand-edited or partially-migrated `manifest.json`), the snippet raises `AttributeError`, stderr is discarded, `requested_backend=""`, and selection falls to slim (or the `ltx25` fallback only for `generate`). CUDA run lands in `voyage:latest` → late `WORKER_ERROR: No module named …` instead of the intended `voyage-video/-ltx`.

### Rationale

Fail-closed parsing (cf. `qualify.sh`'s explicit exit 2/4/5 gates) is the repo standard. Here the failure is explicitly silenced one line above a comment claiming "never a launcher failure" — trading a loud config error for a late worker crash.

### Live evidence

```
$ python3 -c "
cfg={'video':'ltx25','audio':{'backend':'fake'}};
print([cfg.get(s,{}).get('backend','') for s in ('video','audio','sfx')])"
AttributeError: 'str' object has no attribute 'get'
```

The `run.sh` wrapper appends `2>/dev/null || true`, so this prints nothing and `requested_backend=""`.

### Repro

Write `output/x/manifest.json` with `"video": "ltx25"` (string) → `VOYAGE_DRY_RUN=1 ./scripts/run.sh generate x` prints `image=voyage-ltx:latest` only via the no-signal fallback, or `image=voyage:latest` for non-generate paths — never an error naming the malformed manifest.

### Source refs

`scripts/run.sh:83-103`.

### Online sources

None (in-tree `qualify.sh` fail-closed gates are the precedent).

### Fix candidates (240's, complementing 205's)

- Type-guard the comprehension (`(cfg.get(s) or {}).get(...) if isinstance(...) else ""`); on `json.load` failure or non-dict root, warn to stderr and (for `generate`) keep the `ltx25` fallback, otherwise exit 2 with the manifest path.

### Log

- 2026-10-07: filed from read-only Track E sweep; no code touched; consolidated into 205 the same day.
