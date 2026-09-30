# 078 — No hash-verified installs or SBOM anywhere

- Severity: LOW (defense-in-depth; closes the loop on issues 067-068)
- File: `Voyage/Dockerfile:28-30`, `Voyage/worker/Dockerfile.video` (all `pip install` lines), `Voyage/requirements.lock:10-12` (regen = `pip freeze`, versions only) — the former `Voyage/worker/Dockerfile.director:25-45` exact pins cited by the sweep no longer exist (director image file deleted upstream)
- Area: supply-chain / containers — hash verification + SBOM

## Description

Even the pinned installs verify nothing: no `--require-hashes`, no hashed requirements file, no SBOM/attestation generation. The lock regen procedure (`pip freeze | grep -v '^-e '`) records versions, not digests — a compromised index serving the "right" version with different bytes is undetectable at build time.

## Rationale

Version pins answer "which release"; only hashes answer "which bytes". Current best practice stacks both, plus signed SBOM/provenance so downstream consumers can verify the built image.

## Live evidence

Re-verified 2026-09-30 live:

```
$ rg -n "require-hashes|hash=sha256|sbom|attest|provenance" \
    Dockerfile worker/Dockerfile.video \
    requirements.lock scripts/build*.sh
worker/Dockerfile.video:47:# itself floats (issue 011 remainder): hashing it (`--require-hashes`)
→ zero hits outside that comment (verified during sweep)

Voyage/Dockerfile:28-30:
RUN pip install --no-cache-dir "pip==25.0.1" \
    && pip install --no-cache-dir -r requirements.lock \
Voyage/requirements.lock:10-12:
# Regenerate after any deliberate upgrade:
#   docker build -t voyage:latest .
#   docker run --rm voyage:latest pip freeze | grep -v '^-e ' > requirements.lock
```

Absence-of-control: the audit grep above is the repro. (The sweep's `Dockerfile.director:37-40` exact-pins excerpt is stale — file deleted upstream.)

## Repro

n/a (absence-of-control); audit is the repro: the grep above.

## Fix candidates

1. `pip-compile --generate-hashes` → `requirements.lock` with `--hash` entries, installs with `--require-hashes`; extend to the video image once frozen (issue 067).
2. Emit SBOM + build provenance at image build (`docker build --sbom --provenance`, or `syft`/Docker Scout in CI) and store alongside release tags.

## Refs

- "Pin language-level dependencies… verify the integrity of those lock files in CI… generate SBOMs at every build… signed SBOMs, and OpenVEX exploitability data" — https://www.docker.com/blog/software-supply-chain-security-best-practices/
- Hashed-requirements pattern — https://www.systemshardening.com/articles/cicd/reproducible-builds/

## Progress log (2026-09-30, tests-only pass — EVALUATE)

- Premise re-verified live (read-only): `rg -n
  "require-hashes|hash=sha256|sbom|attest|provenance" Dockerfile
  worker/Dockerfile.video requirements.lock scripts/build*.sh` → only hit
  is the `worker/Dockerfile.video:46,61` comments naming `--require-hashes`
  as future work (zero enforcement lines); `Dockerfile:28-30` still bare
  `pip install -r requirements.lock`; `requirements.lock:10-12` still
  `pip freeze` versions-only regen. The absence-of-control still holds
  verbatim; the stale `Dockerfile.director` cite remains stale (file still
  absent — unified `voyage-video` image).
- Tests-scope triage: NOTHING closable from `tests/` alone. Both fix
  candidates require network + container/build ownership + the GPU-box
  video-image freeze (issue 067): (1) needs `pip-compile --generate-hashes`
  against the live index + lockfile ownership + install-line edits in both
  Dockerfiles (frozen container scope); (2) needs build/CI ownership
  (`docker build --sbom --provenance` or syft/Scout) + artifact storage.
  A tests-only "shim" (e.g. a test asserting hashes exist) would fail by
  design and prove nothing — defense-in-depth without the enforcement
  half is theater. No files changed.
- Gate evidence: n/a.

## Resolution (2026-09-30, tests-only pass)

- Verdict: accepted (formal accept-residual with rationale — not pretended
  done). Files changed: none. DESIGN proposals: none.
- Rationale: hash-verified installs + SBOM/provenance need the networked
  build box + video-image dependency freeze + CI artifact storage — none
  available to a CPU-only tests-scoped pass, and no honest tests-only
  subset closes the control gap.
- Residuals (exact handoff, containers/lock owner): (1)
  `pip-compile --generate-hashes` → hashed `requirements.lock` +
  `--require-hashes` installs in `Dockerfile:28-30` + video image once
  067 freezes, with the `tomli`-conditional + `httpcore2/httpx2` typo
  (068/089) resolved in the same lock pass; (2) SBOM + provenance at
  every image build stored alongside release tags. Re-audit with the
  issue's grep; non-zero enforcement hits close this issue.
