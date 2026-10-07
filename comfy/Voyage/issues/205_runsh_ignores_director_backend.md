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

## Evaluation (2026-10-07)

Both claims still relevant; evidence re-verified fresh against the live tree before fixing:

- (a) CONFIRMED: staged `output/__205_fakellama__/manifest.json` with
  `{"video": {"backend": "fake"}, "director": {"backend": "llama"}}` →
  `VOYAGE_DRY_RUN=1 bash scripts/run.sh generate __205_fakellama__`
  printed `image=voyage:latest` (slim). Slim `Voyage/Dockerfile` bakes no
  `llama-server` (only `worker/Dockerfile.video:206-259` and
  `worker/Dockerfile.ltx:115-165` do), so the run would die late with
  `cannot spawn llama-server ... Errno 2` at sidecar start instead of at
  image selection. `DirectorConfig.backend` still defaults to `"llama"`
  (`voyage/config.py:595`), so fake+llama is the default CPU-smoke shape.
- (b) CONFIRMED: the old one-liner raises `AttributeError: 'str' object
  has no attribute 'get'` on a non-dict section (reproduced verbatim),
  and a truly unparseable manifest on the `run --run` path printed
  nothing on stderr and exited 0 with `image=voyage:latest` — silent
  wrong image, exactly as filed.

## Resolution (2026-10-07)

Design decision (picking fix candidate 1, first arm): include
`director.backend == "llama"` in the image decision, selecting
`voyage-video:latest` — NOT baking the sidecar into slim. Why slim is
left alone: the sidecar is GPU-only by design (`-ngl 99`, needs a CUDA
runtime), while slim is the CPU/offline image (`python:3.12-slim` +
ffmpeg, no CUDA, no llama.cpp toolchain); baking a CUDA server into it
would bloat the offline image for a workload that cannot run without a
GPU anyway. Why `voyage-video` over `voyage-ltx`: both CUDA images bake
the binary, but `-ltx` is reserved for the ComfyUI worker stack a
fake-video run never needs; media CUDA backends keep precedence, so
`ltx25+llama` still lands in `voyage-ltx` (which also carries the
sidecar). Explicit `--backend` still wins over every sniffed value
(unchanged guard). Only execution paths that resolve a `run_dir`
(`generate NAME`, `--run/--name`) can hit the sniff; `configure` never
sniffed and needs no change (it writes the manifest, never spawns the
sidecar).

Change (`Voyage/scripts/run.sh` only, no other files touched):

- Sniff now reads `director.backend` too; token `llama` maps to
  `needs_cuda=1` (→ `voyage-video:latest` + `--gpus all`).
- Type-guarded snippet: non-dict sections / non-string backends warn on
  stderr and are ignored (token still computed from the good sections);
  unreadable JSON / non-dict root warns and yields no signal. `2>/dev/null`
  removed — stderr stays loud.
- No-signal policy: `generate` keeps the historical `ltx25` fallback
  (warning already printed); other verbs exit 2 naming the manifest
  (fail-closed, qualify.sh precedent).
- Overclaiming comment corrected (director backends DO participate now).

Dry-run matrix (all `VOYAGE_DRY_RUN=1`, post-fix):

- `generate` fake+llama → `image=voyage-video:latest`, `gpus=--gpus all`
  (was `voyage:latest`/`none` — the bug).
- `generate` fake+deterministic → `image=voyage:latest`, `gpus=none`
  (unchanged, correct).
- `generate` ltx25+llama → `image=voyage-ltx:latest`, `gpus=--gpus all`
  (unchanged, correct — media CUDA wins, sidecar present there too).
- `generate` invalid-JSON manifest → stderr warning + `voyage-ltx`
  fallback, rc 0. `run --run` invalid-JSON / non-dict root → stderr
  warning + `run.sh: error: ... refusing to guess an image`, rc 2.
- `run --run` non-dict `video` section → stderr warning, token from
  remaining sections, rc 0. Explicit `--backend fake` still overrides a
  llama manifest (slim).

Verification: `bash -n scripts/run.sh` clean; matrix above on live
`run.sh`; `tests/test_run_sh.py` in-container
(`./scripts/test.sh tests/test_run_sh.py -m "not gpu" -q`):
18 passed. Probe manifests under `Voyage/output/__205_*__/` removed
afterwards; tree holds no stray files from this fix. No GPU workloads run.

Status: RESOLVED.
