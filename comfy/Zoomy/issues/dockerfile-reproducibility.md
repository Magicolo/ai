# Dockerfile reproducibility gaps

- Severity: medium (every rebuild can silently move the stack).
- Status: verified open. `Zoomy/Dockerfile`.

## Evidence

1. **ACE-Step unpinned** (`:41`): `git clone --depth 1 …ACE-Step-1.5.git`
   tracks a moving default branch, while MMAudio two lines below pins
   `@8eaeb72`. Asymmetric discipline — one rebuild can pull breaking
   upstream changes with no record of what was previously built. No
   `--branch`, no recorded `rev-parse`, no SBOM.
2. **Tautological smoke assert** (`:60`):
   `assert torch.cuda.is_available() is False or True` is `X or True` —
   always true, dead check. Either assert something real (e.g. CUDA
   *build* availability matching the base tag) or drop the line; a smoke
   test that cannot fail is worse than none because it pretends coverage.
3. **Base tag without digest** (`:1`): `2.10.0-cuda12.8-cudnn9-devel` is
   mutable. AGENTS.md §10 still says `2.9.0` — doc drift proving the tag
   moves without anyone noticing.
4. **Apt packages unpinned** (`:34-36`): `git curl libgl1 libglib2.0-0
   libxcb1` float (good `--no-install-recommends` + layer hygiene, still
   drifty).

## Fix

Pin ACE-Step to a commit hash like MMAudio; replace the tautology with a
real assertion or delete it; record base digest (`FROM …@sha256:…`) and fix
the AGENTS.md version string; optionally snapshot `apt list --installed`
into the build log. Keep the build-time heavy-import smoke tests — they
are the good part.

## Verification

- Rebuild from scratch succeeds; `git -C /opt/ACE-Step-1.5 rev-parse HEAD`
  equals the pinned hash; AGENTS.md version matches the Dockerfile tag.
- Gates: `Zoomy/scripts/quality-gates.sh` green.
