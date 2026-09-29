# 011 — No pinned dependency set: `>=` floors, no lockfile, floating transitive closure

- Status: resolved (fixed 2026-09-25: == pins + requirements.lock incl. hypothesis/coverage/sortedcontainers (review))
- Severity: major (reproducibility / supply chain)
- Area: packaging — `pyproject.toml`, worker images
- Rank rationale: history already proves fragility (transformers 5.x break,
  torchao import break); every rebuild can re-break the build.

## Technical description

```toml
# Voyage/pyproject.toml:7-12,17-19
pydantic>=2.7 huggingface_hub>=0.23 numpy>=1.26 rich>=13.7 textual>=8.0
dev: pytest>=8.0 mypy>=1.10 ruff>=0.5 ; setuptools>=70
```

Zero upper bounds, no lockfile committed. The director image floats `torch`,
`torchvision`, `transformers>=4.51`, `sentence-transformers`, `accelerate`,
`safetensors`, `ruff/mypy/pytest` (all unpinned —
`Voyage/worker/Dockerfile.director:20-23`); the video image floats `flash-attn`
(no version), `soundfile/loguru/numba/vector_quantize_pytorch/
sentence-transformers`, `imageio[ffmpeg]/av/sentencepiece/einops/timm`,
`omegaconf/easydict/ftfy/diffusers`, plus the entire upstream LongLive
`requirements.txt` (`Dockerfile.video:28`). Additionally
`download_longlive2_bf16` omits `revision=` on the Wan `snapshot_download`
(`voyage/model_registry.py:232-236`), unlike every other download.

## Why this is an issue

Every image rebuild can silently re-break the build — history already proves it
twice, with transformers 5.x breaking LongLive's 4.x imports and the torchao
`quant`→`quant_api` rename. Zero upper bounds, no lockfile, and a floating
transitive closure (including an entire upstream `requirements.txt`) mean two
builds weeks apart yield different gate behavior and different runtime
semantics under the same tags. For a system whose value is reproducible
infinite video, unreproducible environments void calibration data, benchmark
comparisons, and qualification reports. Future maintainers and every GPU
verification run pay the cost of re-proving what a lockfile would have frozen.

## Evidence

- `rg -n ">=|==" Voyage/pyproject.toml Voyage/worker/Dockerfile.*` (sweep output).
- Precedent comments in-tree: `Dockerfile.video:52-55` ("sentence-transformers
  pulls transformers 5.x, which breaks LongLive's transformers-4.x imports" →
  pinned back to `==4.57.6`); torchao `quant`→`quant_api` import break (AGENTS §11).

Re-verified 2026-09-25:

```
$ rg -n "pydantic|huggingface_hub|numpy|rich|textual|pytest|mypy|ruff" pyproject.toml
7:  "pydantic>=2.7",
8:  "huggingface_hub>=0.23",
9:  "numpy>=1.26",
10:  "rich>=13.7",
11:  "textual>=8.0",
17:  "pytest>=8.0",
18:  "mypy>=1.10",
19:  "ruff>=0.5",
```

Zero upper bounds confirmed; the Wan `snapshot_download` at
`model_registry.py:233-237` still carries no `revision=` while every sibling
download passes one.

## Reproduction

Rebuild the images weeks apart (or bump any transitive) and compare
`pip freeze`; gate behavior drifts (ruff `>=0.5` spans a year of new rules).

## Source references

- `Voyage/pyproject.toml:7-19`; `Voyage/worker/Dockerfile.director:20-23`;
  `Voyage/worker/Dockerfile.video:22,28,39,48,65,81`;
  `voyage/model_registry.py:232-236`.

## Resolution candidates

1. Commit a lockfile (`uv lock` / `pip-compile` + `--require-hashes`); `==` pins
   in both worker images; add the missing `revision=` to the Wan download.
2. Pin pip itself; record `pip freeze` into benchmark/manifest logs.
3. Keep the `transformers==4.57.6` comment as a regression test; add Dependabot/
   Renovate on the pins.

## Investigation / progress / resolution log

- 2026-09-25: found by supply-chain sweep. Zoomy pins
  (`ruff==0.16.6 mypy==2.3.1 pytest==9.1.1 hypothesis==6.168.0`) are the in-repo
  precedent to copy.
- Open: implement lockfile + pins.
- 2026-09-25 (repair pass): added `## Why this is an issue`; Evidence enriched
  with live pin output; Wan `snapshot_download` still revision-less
  (`model_registry.py:233-237`).
- 2026-09-29 (this track, scope `pyproject.toml` pins only): re-read live —
  ADOPTED, no duplicate work. Slim side FIXED by another track:
  `pyproject.toml:14-37` carries exact pins (`pydantic>=2.7,<2.11`,
  `huggingface_hub==2.0.0`, `numpy==1.26.4`, `rich==15.0.0`,
  `textual==8.2.8`, `pytest==9.1.1`, `mypy==2.3.1`, `ruff==0.16.9`,
  `hypothesis==6.168.0`, `coverage==7.16.1`) matching `requirements.lock`
  (slim freeze 2026-09-25, `pydantic==2.10.6` satisfies `<2.11`); `Dockerfile`
  installs lock first + `--no-deps -e .`. Remainder out of scope:
  Wan snapshot still `revision=None` (`model_registry.py:307`, needs registry
  track) + `flash-attn`/upstream-`requirements.txt` float (needs GPU-box
  freeze, noted in `Dockerfile.video:65-68,45-49). No edit in this track.
  Scoped gates: ruff + mypy on touched files green (full `gates.sh` red on
  concurrent hypothesis/conftest scope, pre-existing).
- 2026-09-29 (orchestrator): lock gap closed — `requirements.lock` was
  missing hypothesis/coverage/sortedcontainers (gate image installs from
  the lock, so mypy failed 21× on the property files); resolved exact
  transitives in an ephemeral container (hypothesis imports clean with
  zero attrs) and added the three pins. mypy barrier cleared.
