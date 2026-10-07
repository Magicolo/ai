#!/usr/bin/env bash
# Full gates: ruff lint + format check + mypy strict + pytest, all in-container.
# Scope contract (issue 092): this script gates the LIVE TREE (bind-mounted
# over /app) — what you just edited. scripts/build.sh gates the BAKED
# SNAPSHOT instead (no bind mount — what ships). Both rebuild the image
# first, so each verdict is self-consistent; when they disagree (green here,
# red there), check Dockerfile COPY coverage first — the file is likely
# missing from the image. scripts/test.sh is pytest-only by design (fast
# iteration); the video image gets its smoke gate in build-video.sh.
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR/.."
# DESIGN-ref ratchet (issue 037): every top-level voyage/*.py docstring
# carries its DESIGN section (host-side check — no container needed, fail
# fast before the build). Subpackages (workers/, audio/) belong to issue
# 151's pass, so only voyage/*.py is checked here. AST-based (issue 275):
# the module docstring itself must mention DESIGN — the old `head -n 15`
# grep both missed long headers and passed stray comments. (`DESIGN`, not
# `DESIGN §`: config/director/ltxv legitimately cite "DESIGN task
# group/Phase" without a section sign.)
missing_refs="$(python3 - <<'EOF'
import ast
from pathlib import Path
missing = []
for source_file in sorted(Path("voyage").glob("*.py")):
    docstring = ast.get_docstring(ast.parse(source_file.read_text(encoding="utf-8"))) or ""
    if "DESIGN" not in docstring:
        missing.append(str(source_file))
print("\n".join(missing))
EOF
)"
if [ -n "$missing_refs" ]; then
  echo "missing DESIGN refs in: $missing_refs" >&2
  exit 1
fi
docker build -q --build-arg UID="$(id -u)" --build-arg GID="$(id -g)" -t voyage:latest . > /dev/null
# Cache dirs stay out of the bind-mounted tree (issue 042): without these,
# container runs leave root-owned .ruff_cache/.mypy_cache/.hypothesis
# residue on the host. PYTHONDONTWRITEBYTECODE already suppresses __pycache__.
# GPU-marked tests never run in gates (issue 041) — they need model workers;
# run them explicitly via test.sh on an idle GPU.
# mypy scope (issues 033 + 275): `voyage` plus every tracked test module
# minus scripts/gates-mypy-excludes.txt (pre-existing errors that belong
# to the owning passes; delete a line there as each file is annotated —
# never edit the generated list below). `git ls-files` keeps
# untracked/in-flight test files out until they are committed, while any
# newly tracked file is typechecked by default (the old explicit 150-file
# list silently skipped new files — the exact rot 275 filed).
# `ruff check .` above still lints everything.
mapfile -t _mypy_test_files < <(LC_ALL=C comm -23 \
  <(git ls-files 'tests/test_*.py' | LC_ALL=C sort) \
  <(grep -v -E '^[[:space:]]*(#|$)' scripts/gates-mypy-excludes.txt | LC_ALL=C sort))
if [ "${#_mypy_test_files[@]}" -eq 0 ]; then
  echo "gates: no mypy test files resolved (git ls-files empty?)" >&2
  exit 1
fi
docker run --rm --user="$(id -u):$(id -g)" \
  -e PYTHONDONTWRITEBYTECODE=1 \
  -e RUFF_CACHE_DIR=/tmp/voyage-ruff-cache \
  -e MYPY_CACHE_DIR=/tmp/voyage-mypy-cache \
  -e HYPOTHESIS_STORAGE_DIRECTORY=/tmp/voyage-hypothesis \
  -v "$PWD:/app" voyage:latest bash -c \
  "ruff check . && ruff format --check . && ruff check --select PLR2004 voyage/config.py voyage/doctor.py voyage/media.py voyage/sfx_finalize.py && mypy voyage \
    tests/conftest.py ${_mypy_test_files[*]} \
    && python -m pytest -q -p no:cacheprovider -m 'not gpu' -n auto --cov=voyage --cov-report= && coverage report"
# Cache guard (issue 089): fail loud when gate caches leak into the
# bind-mounted tree. Sourced from lib/common.sh (owned) so the check and
# the VOYAGE_CACHE_ENV contract cannot drift apart.
# shellcheck disable=SC1091
source "$SCRIPT_DIR/lib/common.sh"
voyage_assert_no_cache_residue "$SCRIPT_DIR/.."
