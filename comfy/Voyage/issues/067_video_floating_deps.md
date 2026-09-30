# 067 — `worker/Dockerfile.video`: ~20 floating pip deps, no hashes — GPU image unreproducible by construction

**Severity:** HIGH

**File:line:** `Voyage/worker/Dockerfile.video:34`, `:50-54`, `:68`, `:80-82`, `:107`, `:111`, `:125`, `:144`, `:157`

**Area:** supply-chain / containers — video worker image

**Overlaps with:** 011 (unpinned deps/no lockfile — slim-image sibling; not a duplicate)

## Description

The video image (the one that runs all untrusted-weight code) installs most of its Python stack unpinned and hash-free: `huggingface_hub[cli]` + `pydantic>=2.7,<2.11` (`:34`), the entire upstream LongLive `requirements.txt` (`:54`), `flash-attn` (`:68`), `soundfile loguru numba vector_quantize_pytorch sentence-transformers` (`:80`), `open_clip_torch>=2.29.0 librosa>=0.10.1 torchdiffeq>=0.2.5 accelerate colorlog` (`:107`), `imageio[ffmpeg] av sentencepiece einops timm` (`:125`), `omegaconf easydict ftfy diffusers` (`:144`), `safetensors>=0.4.3 Pillow>=10.0.0` (`:157`). Floors (`>=`) are not pins — rebuilds silently absorb new releases. Only `torch==2.8.0`, `torchvision==0.23.0`, `torchao==0.13.0`, `torchaudio==2.8.0`, `transformers==4.57.6`, `rich==15.0.0/textual==8.2.8` are exact.

## Rationale

A floating tag/requirement is not a security boundary: upstream drift (or compromise) flows silently into rebuilds, and without `--require-hashes` an index/MITM substitution is undetectable. Docker's own supply-chain guidance is explicit: pin by digest + lock exact versions with hash verification, fail the build on mismatch.

## Evidence

Re-verified 2026-09-30 live:

```
Voyage/worker/Dockerfile.video:34:
RUN pip install torchao==0.13.0 "huggingface_hub[cli]" "pydantic>=2.7,<2.11"
Voyage/worker/Dockerfile.video:50-54:
RUN git clone ... /opt/longlive ... && pip install -r /opt/longlive/requirements.txt
Voyage/worker/Dockerfile.video:68:
RUN pip install flash-attn --no-build-isolation
Voyage/worker/Dockerfile.video:80-82:
    && pip install soundfile loguru numba vector_quantize_pytorch sentence-transformers ...
Voyage/worker/Dockerfile.video:107:
    && pip install "open_clip_torch>=2.29.0" "librosa>=0.10.1" "torchdiffeq>=0.2.5" accelerate colorlog
Voyage/worker/Dockerfile.video:125:
    && pip install "imageio[ffmpeg]" av sentencepiece einops timm
Voyage/worker/Dockerfile.video:144:
    && pip install omegaconf easydict ftfy diffusers
Voyage/worker/Dockerfile.video:157:
RUN pip install "safetensors>=0.4.3" "Pillow>=10.0.0"
```

The file confesses twice: `:46-49` ("The upstream requirements.txt itself floats… hashing it needs a GPU-box freeze first") and `:65-67` ("flash-attn floats with no version… can only be frozen on a GPU box"). `rg "pip install" worker/Dockerfile.video` returns 9 RUN lines; only the torch/torchao/transformers/rich/textual rows are exact.

## Repro

On a GPU box: `scripts/build-video.sh && docker run --rm voyage-video:latest pip freeze > /tmp/video-freeze.txt`; diff against any earlier freeze — LongLive/FA/leaf versions move with no tree change. `docker build --no-cache` twice, weeks apart, yields different `pip freeze` with identical source.

## Fix candidates

1. Freeze on a good GPU build (`pip freeze` → exact pins in the Dockerfile, same as the 2026-09-29 director freeze at `worker/Dockerfile.director:20-40`).
2. Generate with `pip-compile --generate-hashes` and install with `pip install --require-hashes`.
3. Never `pip install -r` upstream's floating file — vendor a hashed copy (the CausVid comment at `:129-131` already states the rule; apply it to LongLive/MMAudio too).

## Refs

- Docker "Pin dependencies and verify integrity… Pin language-level dependencies to exact versions with lock files, and verify the integrity of those lock files in CI" — https://www.docker.com/blog/software-supply-chain-security-best-practices/
- Reproducible-builds pattern `pip install --require-hashes -r requirements.txt` with `--hash=sha256:…` entries — https://www.systemshardening.com/articles/cicd/reproducible-builds/
- Docker digest-policy (`isCanonical`) — https://docs.docker.com/build/policies/validate-images

## Progress log

- 2026-09-30: premise re-verified against live `Voyage/worker/Dockerfile.video`
  (as-read lines match the issue evidence verbatim: `:34` hub/pydantic floors,
  `:50-54` LongLive `-r`, `:68` bare flash-attn, `:80` ACE leaf set,
  `:107` open_clip/librosa/torchdiffeq/accelerate/colorlog, `:125`
  imageio/av/sentencepiece/einops/timm, `:144` omegaconf/easydict/ftfy/diffusers,
  `:157` safetensors/Pillow floors, plus the venv floats at `:196-197`).
- 2026-09-30: freeze source resolved CPU-only — `docker run --rm
  voyage-video:latest pip freeze` (image 63b8c37d, 203 rows) plus the director
  venv freeze (`/opt/venvs/director/bin/pip freeze`). No GPU build needed to
  read the freeze; a rebuild is needed to *apply* it (residual below).
- 2026-09-30: TDD — `tests/test_registry_pins.py::test_video_dockerfile_has_no_floating_pins`
  failed pre-fix (all ~24 floating specs flagged), green post-fix. Scanner
  rewrite mid-pass: the first regex missed end-of-RUN rows (no trailing `&&`),
  hiding the `-r` row — now line-based, comment-aware, verified to catch the
  `-r` residual.
- 2026-09-30: gates on touched files — `ruff check` + `ruff format --check` +
  `mypy` (strict) clean; 14/14 new tests green in `voyage:latest`; full suite
  1181 passed with only the two documented 071 behavior-change failures
  (foreign file, one-line updates specified in 071's log).

## Resolution

- Fix candidate 1 applied: every direct pip row in `worker/Dockerfile.video` is
  now `==`-pinned to the 2026-09-30 live-image freeze — `huggingface_hub[cli]==0.36.2`,
  `pydantic==2.10.6`, `flash-attn==2.8.3.post1`, `soundfile==0.14.0`,
  `loguru==0.7.3`, `numba==0.67.0`, `vector-quantize-pytorch==1.31.6`,
  `sentence-transformers==6.1.0`, `open_clip_torch==3.3.0`, `librosa==0.11.0`,
  `torchdiffeq==0.2.5`, `accelerate==1.15.0`, `colorlog==6.12.0`,
  `imageio[ffmpeg]==2.38.0` + explicit `imageio-ffmpeg==0.6.0`, `av==13.1.0`,
  `sentencepiece==0.2.2`, `einops==0.8.2`, `timm==1.0.30`, `omegaconf==2.3.1`,
  `easydict==1.13`, `ftfy==6.3.1`, `diffusers==0.31.0`, `safetensors==0.8.0`,
  `Pillow==12.3.0`, and the venv rows (`sentence-transformers==6.1.0`,
  `safetensors==0.8.0`, `huggingface_hub[cli]==1.33.0`, `pydantic==2.10.6`).
  Freeze source + re-freeze procedure documented in the Dockerfile header.
- Regression gate: the Dockerfile scanner test fails on any new floating spec;
  the single `-r /opt/longlive/requirements.txt` row is allowlisted as the
  documented residual.
- Residuals (GPU box required): (a) the pins take effect on the next
  `scripts/build-video.sh` rebuild — verify with `docker run --rm
  voyage-video:latest pip freeze | diff /tmp/video-freeze.txt -` (expect only
  the `-r`-row drift, if any); (b) candidates 2+3 still open — full
  `--require-hashes` lockfile and vendoring a hashed LongLive requirements
  copy need the same GPU-box freeze (transitive deps still ride the direct
  rows); (c) install-order coupling stays: `sentence-transformers` installs
  before the `transformers==4.57.6` re-pin, so the re-pin line must never be
  deleted.
