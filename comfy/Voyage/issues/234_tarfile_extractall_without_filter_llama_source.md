# 234 — `tarfile.extractall` without `filter=` on Python 3.10/3.11 for the llama.cpp source (path traversal)

Severity: MEDIUM (track F-08).

## Technical description

Both images download `llama.cpp-b11146.tar.gz`, verify sha256 (good), then
`tarfile.open(...).extractall("/tmp/llama-source")` with no `filter=`
(`Dockerfile.video:237-238`, `Dockerfile.ltx:143-144`). Build Pythons are 3.10 (jammy)
and 3.11 (noble+deadsnakes) — `filter="data"` (PEP 706) is 3.12+ (warning backport
only), so nothing constrains absolute paths / `..` / symlinks / devices inside the
tarball.

## Rationale

Mitigated by the sha256 gate, but a single gate bypass = arbitrary file write as root at
build.

## Live evidence

```
worker/Dockerfile.video:220-238 / worker/Dockerfile.ltx:126-144 (heredoc python):
  urllib.request.urlretrieve(url, archive_path)   # TLS via system store, no cert pin (acceptable)
  ...sha256 gate...                                # good, single layer of defense
  with tarfile.open(archive_path) as archive:
      archive.extractall("/tmp/llama-source")     # no filter=, no member sanitization
```

Repro (static): `grep -n "extractall" worker/Dockerfile.*` → two hits, zero `filter=`.

## Source refs

`worker/Dockerfile.video:237-238`; `worker/Dockerfile.ltx:143-144`.

## Online sources

- Python `tarfile` docs / PEP 706 (`filter="data"` default in 3.14, available 3.12+;
  prior versions warn that `extractall` without filter is unsafe).
- Docker least-privilege layering.

## Fix candidates

- Add member validation before extract (reject absolute/`..`/symlink/hardlink/device
  members) or upgrade build Python ≥3.12 and pass `filter="data"`; keep the existing sha
  gate as defense-in-depth, not the sole control.

## Log

- 2026-10-07: filed from read-only Track F sweep; no code touched.
