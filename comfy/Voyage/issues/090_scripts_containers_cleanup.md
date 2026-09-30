# 090 — Scripts + containers: common.sh, gate matrix, run.sh audio gap, qualify generalize, dockerignore

- Severity: MEDIUM (toolchain / containers)
- Files: `scripts/run.sh` (126L), `qualify.sh` (36L), `gates.sh` (28L), `test.sh` (18L), `build.sh` (16L), `build-video.sh` (22L + inline smoke), `.dockerignore` (19L), `worker/Dockerfile.video:214` — `scripts/build-director.sh` (16L, cited by the sweep) deleted upstream
- Area: scripts / containers
- Decision: Q&A locked — full scripts cleanup (common lib + matrix + both fixes)

## Description

4× identical preamble (`build.sh/gates.sh/test.sh`: `set -euo pipefail; cd ..; docker build --build-arg UID/GID; docker run --rm --user=… -e PYTHONDONTWRITEBYTECODE -e RUFF/MYPY/HYPOTHESIS→/tmp -v $PWD:/app`); only axes are tag/mount/gate cmd. `build-video.sh` shares build half; `qualify.sh` tail is a copy minus cache env. (`scripts/build-director.sh` deleted upstream — one fewer preamble copy.) Scope drift: `gates.sh` myps voyage+conftest+3 property modules, `build-director.sh`/`build.sh` mypy voyage only (intentional 092/054 headers, but file list lives in 3 places). `build-video.sh` smoke (`inspect.getsource(main)` asserts `standard_serve_map` + 7 handlers ×3) duplicates `test_video_common`/`test_causvid_worker`/`test_ltxv` predicates at another layer. `qualify.sh` = `run.sh ×3` + `summarize_run`, hardcoded `longlive2` help while default is ltxv and helper is backend-agnostic; idle gate fragile (`held_mib` yields 0 when `nvidia-smi` absent → passes open; relative `<run-dir>` accepted despite doubling bug; JSON to stdout only, no artifact; no disk preflight — see 064). `run.sh` CUDA sniff covers `--backend` + `[video].backend` (+`generate→ltxv`) but `[audio].backend=acestep + video=fake` never selects video image → slim launch → late `cli._require_cuda_stack` fast-fail. `.dockerignore:1-4` header stale (claims tests/ COPY, unbaked since 054; omits `requirements.lock`); misses `.hypothesis/`, `.coverage`, `coverage.xml`. `Dockerfile.video:196` `chmod 777 /opt/longlive /opt/causvid/wan_models` broader than needed.

## Rationale

Preamble ×4 + scope ×3 + smoke ×2 = every gate change needs N edits; audio-gap misselects image; longlive2-only qual driver is stale on arrival; open-gate qual produces unreproducible numbers.

## Live evidence

- `cat scripts/*.sh` (line counts above); `grep -n "MYPY_CACHE\|RUFF_CACHE\|--user\|PYTHONDONTWRITE" scripts/*.sh`
- `scripts/qualify.sh:15-36`; `scripts/run.sh:67` backend case; `voyage/cli.py:_require_cuda_stack`
- `.dockerignore:1-19` full text; `ls -la .coverage .hypothesis`

Overlaps with 075 (`chmod 777` scope), 060 (thin benchmark env — the smoke/scope half), 064 (qualify fragility), 069 (dockerignore), 092 (snapshot-vs-tree); `tests/test_run_sh.py:142`, `tests/test_qualification.py:271`. Outside range: 154/163 cover the missing SFX benchmark targets — this issue owns scripts/containers only.

## Repro

```bash
diff <(sed -n '1,10p' scripts/build.sh) <(sed -n '1,10p' scripts/gates.sh)
PATH=/usr/bin:/bin ./scripts/qualify.sh ./output/q  # gate passes open without nvidia-smi
grep -n "backend" scripts/run.sh | head -n 20
```

## Fix candidates

1. `scripts/lib/common.sh` (`build_image`, `run_gate`, `CACHE_ENV`, `USER_ARGS`); callers become 5-10 lines; keep 092 snapshot-vs-tree headers at call sites. Single gate matrix `gates.sh --image voyage[-video|-director] [--no-mount] [--suite full|pytest|smoke]` replacing build/test/director-smoke files as flags. Do NOT merge `run.sh` (runtime selection vs build/gate lifecycle) — share only constants.
2. `run.sh`: extend TOML sniff to `[audio].backend` (any `backend=` under `[video]/[audio]`); add `test_run_sh.py` case.
3. `qualify.sh`: `--backend ltxv|causvid [--segments N]` default ltxv; `command -v nvidia-smi || exit 4`; absolute-path enforcement; `tee reports/qual-<date>-<backend>.json`; disk preflight.
4. `.dockerignore`: fix header (drop `tests/`, add `requirements.lock`), add `.hypothesis/ .coverage coverage.xml`; `chmod 777→chown`-only on next GPU rebuild.
5. Gate: `gates.sh` green + `test_run_sh` (7+DRAFT matrix) + `bash -n scripts/*.sh`.

## Refs

- Issues 054 (tests unbaked), 064 (qualify fragility), 069 (dockerignore), 092 (snapshot-vs-tree); `tests/test_run_sh.py:142`, `tests/test_qualification.py:271`
