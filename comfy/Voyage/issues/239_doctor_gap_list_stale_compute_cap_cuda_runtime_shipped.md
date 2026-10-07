# 239 — `docs/TROUBLESHOOTING.md` + `docs/INSTALL.md` doctor-gap list is stale (claims `compute_cap` / CUDA runtime unchecked; both shipped in issue 066)

Severity: MEDIUM (track E-04).

## Technical description

Both files state the `voyage/doctor.py` probe "does NOT yet check compute capability,
CUDA runtime version, …" (`TROUBLESHOOTING.md:18-23`, `INSTALL.md:57-64`). Live
`doctor.probe()` returns `compute_cap`, `cuda_runtime`, per-GPU `driver/compute_cap/
temp_c`, `vram_total/free_gib`, `disk_by_mount`, `director_python`, split
`models_ok_required/all` (`voyage/doctor.py:397-447`, `_parse_gpu_details:181-213`,
`_cuda_runtime:216-229`).

## Rationale

Stale "not covered" lists are worse than missing docs: on-call follows the doc and
re-probes manually, or concludes doctor is useless and skips it.

## Live evidence

```
$ python3 -c "import sys; sys.path.insert(0,'.'); from voyage import doctor; print(sorted(doctor.probe().keys()))"
['compute_cap','cuda_runtime','director_python',…,'gpu_details',…,'models_ok_required',…]
$ grep -n "compute_cap\|cuda_runtime" voyage/doctor.py | head
199: compute_cap … 216: def _cuda_runtime …
```

vs `TROUBLESHOOTING.md:20-23`: "it does NOT yet check compute capability, CUDA runtime
version, FlashAttention/Triton, …".

Repro: diff the two doc paragraphs against `doctor.probe()` keys — `compute_cap` and
`cuda_runtime` are in code, listed as gaps in prose. Residual true gaps
(FlashAttention/Triton, checkpoint compat, fs perms, ACE-Step runtime) remain correct and
should stay.

## Source refs

`docs/TROUBLESHOOTING.md:14-23`; `docs/INSTALL.md:53-64`;
`voyage/doctor.py:1-24,181-229,397-447`; `DESIGN.md:3530-3543` (as-built already records
the fix; operator docs were never updated).

## Online sources

- None (in-tree as-built DESIGN note is the anchor).

## Fix candidates

- Edit both paragraphs to the issue-066 reality: covered = python/ffmpeg+ffprobe/version,
  nvidia-smi GPUs, per-GPU VRAM/driver/compute-cap/temp, CUDA runtime, torch-CUDA,
  per-mount disks, models presence split required/optional; remaining gaps =
  FlashAttention/Triton, checkpoint compat, fs perms, worker interpreters, ACE-Step
  runtime (keep the `models verify` pointer).

## Log

- 2026-10-07: filed from read-only Track E sweep; no code touched.

## Evaluation (2026-10-07)

Live-verified RELEVANT, no integrity deltas. `probe()` returns `compute_cap`
+ `cuda_runtime` (`voyage/doctor.py:439,464-465`, `_parse_gpu_details:192-224`,
`_cuda_runtime:227-240`, issue-066 as-built note at `doctor.py:14-19`), while
`TROUBLESHOOTING.md:20-23` and `INSTALL.md:60-64` still list both as unchecked
gaps. Residual true gaps named in both paragraphs (FlashAttention/Triton,
checkpoint compat, fs perms, worker interpreters, ACE-Step runtime) remain
correct and stay. Overlap noted: `doctor.py:12` also carries the deleted
`voyage models verify` string (issue-218 class) — fixed in the same edit
since this file is in scope for both.

## Progress log (2026-10-07)

- `docs/TROUBLESHOOTING.md:18-23` rewritten to the issue-066 reality
  (covered: python/ffmpeg+ffprobe/version, nvidia-smi GPUs, per-GPU
  VRAM/driver/compute-capability/temperature, CUDA runtime, torch-CUDA,
  per-mount disks, models presence required/optional split; remaining gaps:
  FlashAttention/Triton, checkpoint compat, fs perms, worker interpreters,
  ACE-Step runtime).
- `docs/INSTALL.md:53-64` rewritten to the same reality (same covered set,
  same remaining gaps, `configure --no-download` verify-only pointer kept).
- `voyage/doctor.py:12` dead-verb string fixed alongside (issue-218 overlap).
- Verify: `compute capability|does NOT yet check` 0 hits in both docs;
  `test.sh tests/test_causvid_worker.py` 32 passed (neighbor scope);
  ruff check + format clean on `doctor.py`.

## Resolution (2026-10-07)

Verdict: RESOLVED. Both doctor-gap paragraphs now match `probe()` keys;
nothing left open (residual gaps are true and intentionally kept).
