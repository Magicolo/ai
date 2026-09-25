# 048 — `voyage doctor` implements ~3 of ~13 spec checks (+ build-time false gate)

- Status: open
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
