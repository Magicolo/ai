# 048 — `voyage doctor` implements ~3 of ~13 spec checks (+ build-time false gate)

- Status: resolved (fixed 2026-09-25: probe extension + gap docs + build gates)
- Severity: medium (over-promises; build implies a verified stack)
- Area: observability — `voyage/doctor.py:26-50`, `cli.py:122-132`, Dockerfiles
- Rank rationale: spec §64 requires 13 checks; `probe()` returns 6 fields with
  `torch_cuda: None` always; yet `Dockerfile:14` runs it at build as a gate and
  docs tell users to reach for it first.

## Technical description

Spec (`DESIGN.md:2952-2965`, §64): NVIDIA driver, CUDA runtime, GPUs, compute
capability, VRAM, PyTorch, FlashAttention/Triton, ffmpeg, model files, checkpoint
compat, fs permissions, free space, worker interpreters, ACE-Step availability.
Actual `probe()` → `{python, ffmpeg, ffprobe, nvidia_smi, gpus, torch_cuda: None}`
with comment "Phase 0: not required". `cmd_doctor` prints python/ffmpeg/gpu lines
only; exit code = ffmpeg only.

Live probe (orchestrator, host, 2026-09-25):

```
{'python': '3.12.3', 'ffmpeg': '/usr/bin/ffmpeg', 'ffprobe': '/usr/bin/ffprobe',
 'nvidia_smi': True, 'gpus': ['NVIDIA GeForce RTX 4060 Ti, 16380 MiB, ...',
 'NVIDIA GeForce RTX 2060, 6144 MiB, ...'], 'torch_cuda': None}
```

Both GPUs visible, but says nothing about torch/CUDA-visibility mismatch, disk,
models, permissions — yet `docs/TROUBLESHOOTING.md` says "`voyage doctor` first"
and `docs/INSTALL.md:54` claims "reports driver/GPU/ffmpeg facts".
`python3 -m voyage.cli doctor` → exit 0 even with empty `/models`, full disk,
missing torch. The video/director images don't run it at build at all (only
`CMD`).

## Why this is an issue

`doctor` is the documented first step in TROUBLESHOOTING.md and INSTALL.md,
so users treat a green report as "my box is fine" — but it currently passes
on boxes with no torch, empty model dirs, or full disks, sending users down
long GPU-debug rabbit holes for problems a disk/model check would have named
in seconds. Worse, the Dockerfile runs it at build time as a gate, which
implies the image stack was verified when only ffmpeg presence was checked.
Blast radius: every new install and every CI build; fix cost is small (wire
existing `verify_*` helpers + disk/torch checks).

## Evidence

Probe output above; `rg -n "doctor" Voyage/Dockerfile Voyage/worker/Dockerfile.*`.

## Reproduction

`python3 -m voyage.cli doctor` on a box with no torch / empty models / full disk
→ still exit 0 with a healthy-looking report.

## Source references

- `voyage/doctor.py:26-50`; `voyage/cli.py:122-132`; `DESIGN.md:2952-2965`;
  `docs/TROUBLESHOOTING.md:16`; `Voyage/Dockerfile:14`.

## Resolution candidates

At minimum wire existing helpers — `verify_*` summary,
`shutil.disk_usage('/')`, `torch.cuda.is_available()` when importable
(`find_spec` guard, §83-safe), `VOYAGE_MODELS` presence — and document the
remaining gaps in `--help`/docs. Either extend `doctor` (import smoke: `torch`,
`longlive`, `ltx_video`, `acestep`; `ffmpeg -version` record) and run it in all
three builds, or drop it from `Dockerfile:14` so the build doesn't imply a
verified stack.

## Investigation / progress / resolution log

- 2026-09-25: found by docs + supply sweeps (convergent); probe executed live.
- 2026-09-25 (repair): re-verified refs live — `voyage/doctor.py:26-50`
  (`probe()` + `check_ffmpeg`), `voyage/cli.py:122-132` (`cmd_doctor`),
  `Voyage/Dockerfile:14` (`voyage doctor` build gate),
  `docs/TROUBLESHOOTING.md:16`, `docs/INSTALL.md:54` all current. No changes
  needed to references; added `## Why this is an issue`.
- Open: extend doctor + wire into all builds.
- 2026-09-25 (fix, scope `voyage/doctor.py` only — `cli.py`/Dockerfiles
  belong to other tracks): re-verified live — `probe()` still returned
  the six old keys with `torch_cuda: None` always. Extended `probe()`
  (all new facts best-effort, never raise): `torch_cuda` via
  `find_spec`-guarded lazy `torch.cuda.is_available()` (§83-safe, None
  when torch absent/broken), `disk_free_gib` via `shutil.disk_usage`,
  `ffmpeg_version` first line, `models` presence summary
  (`dir`/`exists`/`manifest` + per-backend `verify_*` results via a lazy
  `model_registry` import) plus a `models_ok` flag that is False unless
  the dir exists and every check passes. `cmd_doctor` output text is
  unchanged (cli.py out of scope) — the facts are in `probe()` for the
  CLI track to print; exit code still ffmpeg-only. Remaining §64 gaps
  (compute capability, CUDA runtime, FlashAttention/Triton, checkpoint
  compat, permissions, worker interpreters, ACE-Step) documented in
  `docs/TROUBLESHOOTING.md` + `docs/INSTALL.md`; as-built coverage note
  appended under DESIGN §64 (`§64-doctor-coverage-2026-09-25`). Tests:
  new `tests/test_doctor.py` (8 tests: key matrix, torch
  absent/present/broken, disk failure, empty/missing models dir). Scoped
  gates green: ruff + format + mypy strict on `doctor.py`, 8/8 pytest.
  Full `gates.sh` is red on concurrent agents' files (`cli.py`
  `_last_commit_stages` F821 + E501, `tui_state.py` F401/F821) — none in
  this scope. The Dockerfile-build-gate half (run doctor in all three
  builds vs drop the false gate) is left for the Docker track.

## Resolution

FIXED (doctor half): `voyage/doctor.py` now reports torch-CUDA, disk,
and models facts; docs name the remaining gaps. Build-gate half
explicitly deferred (Dockerfiles out of scope).
- 2026-09-29 (this track, scope Dockerfile half only): re-read live —
  ADOPTED, no duplicate work. Build-gate half FIXED by another track:
  `RUN voyage doctor` in all three builds (`Dockerfile:33`,
  `worker/Dockerfile.video:151`, `worker/Dockerfile.director:54`) with
  issue-048 comments (ffmpeg-only exit, warns without failing on missing
  models). `docs/INSTALL.md:54-60` + `docs/TROUBLESHOOTING.md:16-22` name
  coverage + remaining gaps. No edit in this track. Note: full
  video/director image builds skipped per task (slim build via `gates.sh`
  only).
