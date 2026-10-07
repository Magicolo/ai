# 232 — No `--require-hashes` anywhere; worker images add `--index-url`, `--no-build-isolation`, an unverified PEP-508 git URL, and four bare unpinned SFX-venv packages

Severity: MEDIUM (track F-06).

## Technical description

Slim lock is `==`-pinned without hashes; worker Dockerfiles pin `==` without hashes and
explicitly note "a full `--require-hashes` lockfile is the follow-up"
(`Dockerfile.video:44-45`). Additional wideners: `--index-url
https://download.pytorch.org/whl/cu128|cu130` on 4 install lines (second index consulted
for every package on those lines), `flash-attn==2.8.3.post1 --no-build-isolation`
(build runs unisolated against ambient site-packages), and `ltx-video[inference] @
git+https://github.com/Lightricks/LTX-Video@4b2d053…` with no hash check (Dockerfile
itself flags it: "no --depth/hash check is expressible in a PEP 508 URL",
`Dockerfile.video:124-127`).

## Rationale

PyPI / index compromise → arbitrary code at image build, running as root.

## Live evidence

```
Dockerfile:28-29: RUN pip install ... "pip==25.0.1" && pip install ... -r requirements.lock   # no hashes
worker/Dockerfile.video:35-37,46,72,85-86,128-129,148,161,179,186,200-204
worker/Dockerfile.ltx:48-50,55,89,100-104,187-189,212-215
worker/Dockerfile.video:72: RUN pip install flash-attn==2.8.3.post1 --no-build-isolation
worker/Dockerfile.video:128: RUN pip install --no-deps "ltx-video[inference] @ git+https://...@4b2d053..."
worker/Dockerfile.ltx:215: ... Pillow tqdm huggingface_hub numpy "omegaconf==2.3.1" ...  # 4 bare names
```

Repro: `grep -rn "require-hashes\|only-binary" Dockerfile worker/Dockerfile.*` → empty.

## Source refs

As above; self-acknowledged at `worker/Dockerfile.video:39-45`.

## Online sources

- pip `secure-installs` ("By default, pip does not perform any checks … Enable
  Hash-checking Mode `--require-hashes`, Disallow sdists `--only-binary :all:`; hashes
  required for all requirements and all dependencies").
- Docker build best practices (pin + digest, reproducible installs).

## Fix candidates

- `pip-compile --generate-hashes` lockfiles per image + `pip install --require-hashes
  --only-binary :all:`; vendor `ltx-video@4b2d053` as a sha256-gated tarball (same pattern
  as the llama.cpp source gate); scope `--index-url` via `--extra-index-url` +
  `--trusted-host` review or a local wheel mirror; drop `--no-build-isolation` or pin the
  build closure.

## Log

- 2026-10-07: filed from read-only Track F sweep; no code touched.

## Consolidated from 236_sfx_venv_four_bare_unpinned_packages (2026-10-07)

### Technical description (from 236)
Every other pip row in both worker Dockerfiles is `==`-pinned.
`Dockerfile.ltx:215` ends the SFX venv with `Pillow tqdm huggingface_hub numpy`
unpinned — the next rebuild floats all four (notably `huggingface_hub`, whose v1→v2
migration renames `httpx→httpx2` and changes offline/cache behavior).

### Rationale (from 236)
Rebuild drift + unpinned supply chain in the audio path.

### Live evidence (from 236)
`worker/Dockerfile.ltx:215`; contrast `worker/Dockerfile.video:111`
(`"open_clip_torch==3.3.0" … "colorlog==6.12.0"`) and `requirements-ltx.txt` (all 100+
rows `==`).

Repro: `grep -n "Pillow tqdm" worker/Dockerfile.ltx` → line 215.

### Source refs (from 236)
`worker/Dockerfile.ltx:215`.

### Online sources (from 236)
- Project's own pinning contract (`requirements-ltx.txt:1-12`,
  `Dockerfile.video:39-45` re-freeze procedure).
- Docker reproducibility practices.

### Fix candidates (from 236)
- Pin all four `==` to the current freeze (`pip freeze` in a good build), note in
  `docs/INSTALL.md` per the file's own procedure.

### Log (from 236)
- 2026-10-07: filed from read-only Track F sweep; no code touched.
