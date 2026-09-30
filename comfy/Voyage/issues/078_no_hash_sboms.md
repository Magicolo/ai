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
