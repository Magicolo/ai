# 042 — Residue/cache hygiene below Zoomy bar; `__pycache__` on disk; dev pins drift

- Status: open
- Severity: medium (dirty tree; uncomparable gates across images)
- Area: standards — scripts env, Dockerfile, dev pins
- Rank rationale: two distinct but adjacent hygiene gaps; the tree already shows
  35 `__pycache__` dirs and unbounded dev gates.

## Technical description

(a) Cache env: `Voyage/Dockerfile:1-17` sets no `ENV PYTHONDONTWRITEBYTECODE=1`,
no `RUFF_CACHE_DIR/MYPY_CACHE_DIR/HYPOTHESIS_STORAGE_DIRECTORY` (cf.
`Zoomy/Dockerfile:19`, `Zoomy/scripts/quality-gates.sh:14-16`); scripts set
`-e PYTHONDONTWRITEBYTECODE=1` at runtime only (`gates.sh:6`, `test.sh:6`,
`build.sh:6`, `run.sh:70-71`, `qualify.sh:31`). Observed:
`Voyage/tests/__pycache__/test_*.cpython-312-pytest-9.1.1.pyc` (35 files) +
`Voyage/voyage/__pycache__/*.pyc` present on host despite `.gitignore`
(`__pycache__/ *.pyc .mypy_cache/ .ruff_cache/ .pytest_cache/` — gitignored but
not suppressed). No `tests/conftest.py` hypothesis-profile/DB guard (Zoomy
`tests/conftest.py:10`).

(b) Dev pins: `pyproject.toml:16-20` (`pytest>=8.0, mypy>=1.10, ruff>=0.5` —
unbounded); `worker/Dockerfile.director:21` (`ruff/mypy/pytest` unpinned, `torch`
unpinned CPU, `transformers>=4.51` open); `Dockerfile:13`
(`pip install -e ".[dev]"` resolves latest each rebuild). `ruff>=0.5` spans a year
of new rules; `build-director.sh:7-8` runs the same unpinned gates in a different
image (CPU torch + transformers 5.x risk) — gate results aren't comparable across
images.

## Why this is an issue

Thirty-five stray `__pycache__` trees dirty the working tree and risk container/host bytecode confusion, punishing everyone who reads `git status` or debugs an import. Unbounded dev pins are slower poison: `ruff>=0.5` spans a year of new rules and the director image can resolve transformers 5.x, so gate results from two different months — or two different images in the same week — are not comparable, and a "green gates" claim stops meaning anything. Hygiene debt is paid by every contributor on every run.

## Evidence

`ls Voyage/tests/__pycache__ | head`; `rg -n "CACHE|HYPOTHESIS"
Voyage/scripts/*.sh Voyage/Dockerfile` → only `PYTHONDONTWRITEBYTECODE`;
`rg -n "ruff|mypy|pytest" Voyage/pyproject.toml Voyage/Dockerfile
Voyage/worker/Dockerfile.director`.

Verified live 2026-09-25: `ls tests/__pycache__` shows
`__init__.cpython-312.pyc test_audio_planner...test_backends_adapter...` (plus
`voyage/__pycache__/*.pyc` present); `CACHE|HYPOTHESIS` rg hits only
`PIP_NO_CACHE_DIR=1` (`worker/Dockerfile.director:10`) — no Ruff/Mypy/Hypothesis
cache env anywhere. Refs current.

## Reproduction

Run gates without the cache env and list `__pycache__`; rebuild a month apart and
diff resolved gate versions.

## Source references

- Files/lines above.

## Resolution candidates

Mirror Zoomy: `ENV PYTHONDONTWRITEBYTECODE=1` in `Dockerfile`,
`RUFF_CACHE_DIR=/tmp/... MYPY_CACHE_DIR=/tmp/... HYPOTHESIS_STORAGE_DIRECTORY=
/tmp/hypothesis` in all scripts, `addopts` already `no:cacheprovider` (keep), add
`conftest.py` disabling the example DB; pin `dev` (`==`) + `pip freeze` into
benchmark/manifest logs; pin director-image test deps to the same pins.

## Investigation / progress / resolution log

- 2026-09-25: found by standards sweep.
- 2026-09-25: repair pass — added `## Why this is an issue`; `__pycache__`
  presence + cache-env absence re-verified live on host; pasted output into
  Evidence.
- Open: implement env + pins.
