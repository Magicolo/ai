# 011 — No pinned dependency set: `>=` floors, no lockfile, floating transitive closure

- Status: resolved in live tree (lockfile + `==` pins + bounded supervisor contract)
- Severity: MEDIUM (reproducibility / supply chain; resolved, record only; residuals → 067/068/078)
- Group: supply-chain/packaging — Rank: 3/5 (fixed; see 068 for live lock corruption)
- Overlaps: 068 owns the live `httpx2/httpcore2` lock rows (089 cites the same rows — fold into 068); this file is missing-pins, not corrupt-entries — not a duplicate.
- Area: packaging — `pyproject.toml`, worker images, `requirements.lock`
- Rank rationale: history already proved fragility (transformers 5.x break, torchao
  import break); every rebuild could re-break the build.

## Technical description

Pass-1 `pyproject.toml:7-19`:

```toml
pydantic>=2.7 huggingface_hub>=0.23 numpy>=1.26 rich>=13.7 textual>=8.0
dev: pytest>=8.0 mypy>=1.10 ruff>=0.5 ; setuptools>=70
```

Zero upper bounds, no lockfile committed. The director image floated `torch`,
`torchvision`, `transformers>=4.51`, `sentence-transformers`, `accelerate`,
`safetensors`, `ruff/mypy/pytest` (all unpinned —
`worker/Dockerfile.director:20-23` at pass 1); the video image floated
`flash-attn`, `soundfile/loguru/numba/vector_quantize_pytorch/
sentence-transformers`, `imageio[ffmpeg]/av/sentencepiece/einops/timm`,
`omegaconf/easydict/ftfy/diffusers`, plus the entire upstream LongLive
`requirements.txt`. Additionally `download_longlive2_bf16` omitted `revision=` on
the Wan `snapshot_download` (`voyage/model_registry.py:232-236` at pass 1), unlike
every other download.

Live state (re-verified 2026-09-30): `requirements.lock` committed (55 lines,
`==`-frozen, header dated 2026-09-25); `pyproject.toml:14`
`pydantic>=2.7,<2.11` (upper bound is a cross-image contract — director image
carries transformers 5.17.0); dev gates pinned `pytest==9.1.1 mypy==2.3.1
ruff==0.16.9` (`pyproject.toml:31-33`); all three Dockerfiles `==`-pin with
digest FROMs (see 012).

## Why this is an issue

Every image rebuild could silently re-break the build — history already proved it
twice, with transformers 5.x breaking LongLive's 4.x imports and the torchao
`quant`→`quant_api` rename. Zero upper bounds, no lockfile, and a floating
transitive closure (including an entire upstream `requirements.txt`) meant two
builds weeks apart yielded different gate behavior and different runtime semantics
under the same tags. For a system whose value is reproducible infinite video,
unreproducible environments void calibration data, benchmark comparisons, and
qualification reports.

## Evidence

Live verification 2026-09-30:

```
$ rg -n "pydantic|pytest|mypy|ruff" pyproject.toml | head
14:  "pydantic>=2.7,<2.11",
31:  "pytest==9.1.1",
32:  "mypy==2.3.1",
33:  "ruff==0.16.9",
$ ls -l requirements.lock && wc -l requirements.lock
-rw-rw-r-- 1 goulade goulade 1516 ... requirements.lock
55 requirements.lock
$ head -8 requirements.lock
annotated-types==0.8.0 / anyio==4.15.1 / ...
$ rg -n "pip==" Dockerfile worker/Dockerfile.*
Dockerfile:28:RUN pip install --no-cache-dir "pip==25.0.1" ...
worker/Dockerfile.video:30:RUN pip install "pip==25.0.1" ...
worker/Dockerfile.director:25:RUN pip install --no-cache-dir "pip==25.0.1" ...
```

Precedent comments in-tree: `Dockerfile.video` ("sentence-transformers pulls
transformers 5.x, which breaks LongLive's transformers-4.x imports" → pinned back
to `==4.57.6`); torchao `quant`→`quant_api` import break (AGENTS §11).

## Reproduction

Rebuild the images weeks apart (or bump any transitive) and compare `pip freeze`;
pre-fix gate behavior drifted (ruff `>=0.5` spans a year of new rules). Post-fix:
`requirements.lock` + `==` pins reproduce the set.

## Source references

- `pyproject.toml:9-33` (bounds + pins); `requirements.lock` (frozen set);
  `worker/Dockerfile.director:25-45`, `worker/Dockerfile.video:30-157`,
  `Dockerfile:28-29`; `voyage/model_registry.py` (revision-pinned downloads).

## Resolution candidates

1. (Landed) Commit a lockfile; `==` pins in worker images; `revision=` on
   downloads.
2. (Landed) Pin pip itself (`pip==25.0.1`); record `pip freeze` into
   benchmark/manifest logs.
3. Keep the `transformers==4.57.6` comment as a regression test; add
   Dependabot/Renovate on the pins.

## Online references

- pip requirements file format, `==` vs `>=` version specifiers:
  https://pip.pypa.io/en/stable/reference/requirements-file-format/
- PEP 440 version specifiers (why an upper bound is a compatibility contract):
  https://peps.python.org/pep-0440/
- torchao `quant` → `quant_api` migration (the in-tree precedent for float breakage):
  https://github.com/pytorch/ao

## Investigation / progress / resolution log

- 2026-09-25: found by supply-chain sweep. Zoomy pins
  (`ruff==0.16.6 mypy==2.3.1 pytest==9.1.1 hypothesis==6.168.0`) were the in-repo
  precedent.
- Resolution batch 1/5: lockfile + pins landed.
- 2026-09-30: re-verified live (lock + pins present); reconstructed from archived
  pass-1 text (commit `b5d7dda`). Status → resolved.
