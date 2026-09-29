#!/usr/bin/env bash
# Full gates: ruff lint + format check + mypy strict + pytest, all in-container.
set -euo pipefail
cd "$(dirname "$0")/.."
docker build -q --build-arg UID="$(id -u)" --build-arg GID="$(id -g)" -t voyage:latest . > /dev/null
# Cache dirs stay out of the bind-mounted tree (issue 042): without these,
# container runs leave root-owned .ruff_cache/.mypy_cache/.hypothesis
# residue on the host. PYTHONDONTWRITEBYTECODE already suppresses __pycache__.
# GPU-marked tests never run in gates (issue 041) — they need model workers;
# run them explicitly via test.sh on an idle GPU.
# mypy scope (issue 035): `voyage` plus the converted test modules. The rest
# of tests/ carries pre-existing errors (untyped helpers, stale ignores —
# see issues/035) that belong to the owning passes; append a module path
# here as each file is annotated. `ruff check .` above still lints everything.
docker run --rm --user="$(id -u):$(id -g)" \
  -e PYTHONDONTWRITEBYTECODE=1 \
  -e RUFF_CACHE_DIR=/tmp/voyage-ruff-cache \
  -e MYPY_CACHE_DIR=/tmp/voyage-mypy-cache \
  -e HYPOTHESIS_STORAGE_DIRECTORY=/tmp/voyage-hypothesis \
  -v "$PWD:/app" voyage:latest bash -c \
  "ruff check . && ruff format --check . && mypy voyage tests/conftest.py tests/test_seeds_properties.py tests/test_beat_properties.py tests/test_similarity_properties.py && coverage run -m pytest -q -m 'not gpu' && coverage report"
