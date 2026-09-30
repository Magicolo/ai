# 171 — Recovery-tape path gates bound existence and containment but not size: a worker-reported multi-GB `.pt` sails through to `torch.load`

- Severity: LOW-MEDIUM (correctness / resource exhaustion on the failure path — a buggy or compromised worker turns resume into a worker OOM crash loop that burns the shared restart budget)
- Area: supervisor/worker trust boundary — tape *path-property* validation (below all prior windows; 006 fixed frame counts + tape containment, 016 the symlink/TOCTOU shape of the same gate, 122/129 the *write* durability, 123 the *content* trust — none bounds the *bytes* behind an accepted path)
- Files (as-read 2026-09-30):
  - `voyage/supervisor.py:403-421` (`_checked_tape_path`: lexical `relative_to` + `exists()`, no size/stat)
  - `voyage/workers/video_longlive.py:1173-1192` (`_load_recovery_tape`: suffix `.pt` + `is_file()`, then `torch.load(handle, weights_only=True)` — no size cap)
  - `voyage/workers/video_longlive.py:1195-1205,1218-1245` (`handle_resume`/`handle_rebuild`: profile-string check only, then `resume_from_tape`)
  - `voyage/supervisor.py:530-562` (`_latest_recovery_tape` + `_resume_video_worker`: no size/read check before the resume RPC — 139's discovery side)

## Technical description

Three gates stand between a worker-reported tape path and `torch.load`, and none looks at the file size:

1. Commit-time `_checked_tape_path` (`supervisor.py:403-421`): rejects lexical escapes and missing files. A 3 GiB `recovery.pt` inside the run passes both predicates.
2. Discovery-time `_latest_recovery_tape` (`:530-544`): `exists()` only (139's finding).
3. Worker-side `_load_recovery_tape` (`video_longlive.py:1183-1191`): `suffix == ".pt"` + `is_file()` (follows symlinks, like the supervisor gate), then straight into `torch.load(handle, map_location="cpu", weights_only=True)`.

`weights_only=True` (005) stops code execution, not allocation: torch still materializes every tensor in the archive. A multi-GB `.pt` — a runaway tape write, a worker bug pointing at a weights file renamed `.pt`, or a malicious worker exercising 006's threat model — OOMs the video worker process mid-`torch.load`. The OOM surfaces as a crash/EOF (`RecoverableWorkerError`), `_call_with_restart` spends a restart, the rebuild replays init, `resume` re-loads the same giant file, and the cycle burns the whole `max_worker_restarts` budget on an input that was knowably oversized before any RPC. Legitimate tapes are megabytes (tail latents bf16 `[1,8,C,H,W]` ≈ single-digit MB + embeds + RNG state — DESIGN §22 logs ~7 MB), so any gigabyte-scale tape is either corrupt or hostile; there is no legitimate case the cap would break.

Same gap, second facet (worker-side path properties beyond 123): `is_file()` follows symlinks and there is no run-confinement at the worker (the docstring at `:1180-1181` explicitly delegates containment supervisor-side — but the supervisor's resume path at `:556-561` never calls `_checked_tape_path` on the discovered tape, so on the resume leg *neither* side confines). Size is filed here as the primary defect; the resume-leg confinement hole is 139/016's to close — this file notes it only so the two fixes meet at the same call site.

## Why this is an issue

- The resume path already runs when things went wrong; converting one bad input into N wasted restarts (each replaying a multi-GB model init on 016-GB cards) is the budget-burn shape Phase 6 slice B was built to prevent. A byte cap fails the commit/resume *before* the first restart, with a `MediaError` the taxonomy (007) already handles.
- The check is nearly free (`stat().st_size`, one syscall) against a multi-minute GPU rebuild, and the legitimate size distribution is tight (MBs) — a cap at two orders of magnitude above legitimate (e.g. 1 GiB) has no false-positive surface.
- Threat-model continuity: 006 established worker reports as untrusted and landed count + containment bounds; size is the third dimension of the same boundary, still open.

## Live evidence

CPU-only probes 2026-09-30 (`voyage:latest` for the gate, host `rg` for the absence):

```
$ docker run --rm -v "$PWD:/app" -w /app voyage:latest python3 -c "
from pathlib import Path
from voyage.supervisor import Supervisor
import tempfile
with tempfile.TemporaryDirectory() as td:
    run = Path(td)/'run'; (run/'segments'/'000000').mkdir(parents=True)
    big = run/'segments'/'000000'/'recovery.pt'
    with open(big,'wb') as f: f.truncate(3*1024**3)  # 3 GiB sparse, instant
    print('size GiB:', big.stat().st_size/1024**3, 'is_file:', big.is_file())
    sup = Supervisor.__new__(Supervisor); sup._run_dir = run
    try:
        out = sup._checked_tape_path('segments/000000/recovery.pt', '000000')
        print('gate ACCEPTED:', out)
    except Exception as exc:
        print('gate rejected:', type(exc).__name__, str(exc)[:120])
"
size GiB: 3.0 is_file: True
gate ACCEPTED: /tmp/tmp4v1v11xn/run/segments/000000/recovery.pt
```

- A 3 GiB `.pt` — ~400× a legitimate tape — passes the commit-time gate. The worker side then applies only `suffix` + `is_file` (`:1184`) before `torch.load` (`:1189`).

```
$ rg -n "st_size|stat\(\)|getsize|MAX_TAPE|tape_size" voyage/supervisor.py voyage/workers/video_longlive.py voyage/workers/video_ltxv.py voyage/workers/video_causvid.py voyage/workers/video_common.py
(no size-gate hits; only unrelated symlink-anchor helpers in causvid:109-118 / longlive:203-208)
```

- No byte bound exists on any tape path: commit gate, discovery, worker load, or resume/rebuild handlers.

## Minimal repro

1. `truncate -s 3G segments/000000/recovery.pt` (sparse — instant, no disk use) with `DONE` present.
2. Kill the video worker mid-next-segment so `_resume_video_worker` runs (or report the path from a stub video worker at commit).
3. Observed: commit gate accepts; `resume` RPC `torch.load`s 3 GiB → worker OOM/EOF → `RecoverableWorkerError` → restart → same load → budget exhausted → FAILED. Expected: `MediaError` (implausible tape size) at the gate, zero restarts spent.

## Fix candidates

1. (Preferred) Cap bytes at both existing gates: `_checked_tape_path` rejects `st_size` above `MAX_RECOVERY_TAPE_BYTES` (e.g. 1 GiB — ~100× legitimate) with `MediaError`; `_load_recovery_tape` enforces the same cap before `torch.load` (defense in depth — the worker is a separate image and must not rely on supervisor checks). Use `stat(follow_symlinks=False)` + `is_file()` so a symlink's own size, not the target's, is compared — or resolve-and-confine first per 016 and stat the resolved file.
2. Emit a `video_resume_skipped`/`implausible_tape` metric on rejection so the skip is counted (pairs with 139's skip-to-older-tape candidate).
3. Test: sparse multi-GB `.pt` → `MediaError` at the gate and at `_load_recovery_tape`, zero worker restarts; legitimate ~7 MB tape → accepted; symlink-to-huge-outside-file → rejected (ties 016's resolve-and-confine to the size check).
4. Long-term: single shared `verify_tape_for_resume` in `video_common` (123's candidate-3 home) so the three workers plus both supervisor gates share one size + shape + hash contract.

## References

- In-tree: `voyage/supervisor.py:396-421,530-562,1631-1648` (commit re-gating); `voyage/workers/video_longlive.py:907-920` (legitimate tape contents — tail latents + embeds + RNG state, MBs); `voyage/workers/video_common.py:239-249` (JSON tapes — small by construction, same cap applies cheaply); DESIGN §27/§27.1 (recovery tapes).
- Neighbor issues — not a duplicate of 006 (frame-count ceiling + path containment — no byte bound), 016 (symlink/TOCTOU *shape* of the same gate — no size dimension), 122/129 (tape *write* durability/atomicity), 123 (tape *content* checksum/shape — pre-load byte bound is a different, cheaper layer), 139 (discovery validation — this file is the *size* dimension across commit, discovery, and worker-load gates).
- External: CWE-400 uncontrolled resource consumption (same class 006 cited for `frames=10**9` — this is its byte-size sibling): https://cwe.mitre.org/data/definitions/400.html

## Progress log

- 2026-09-30 (Group B): re-verified live first (`voyage:latest`, CPU-only):
  a sparse 3 GiB `.pt` passes `_checked_tape_path` (ACCEPTED), and `rg`
  finds no `st_size`/`MAX_TAPE` gate on any tape path (commit, discovery,
  worker load, resume/rebuild). Premise confirmed.
- TDD: wrote `tests/test_tape_trust_123_171.py` first — collection error
  before the fix, 8 passed after (shared file with 123).
- Fix: `MAX_RECOVERY_TAPE_BYTES = 1 GiB` (~100× legitimate ~7 MB tapes,
  per DESIGN §22 — no false-positive surface) + `check_recovery_tape_size`
  in `voyage/workers/video_common.py` (one `stat`, `ValueError` naming
  re-render-from-seed); enforced at both supervisor gates (owned hunks):
  `_checked_tape_path` rejects with `MediaError` (zero restarts spent),
  `_latest_recovery_tape` skips oversized candidates to the next-newest
  tape with a `recovery_tape_skipped` metric (mirrors the 139 torn-write
  skip pattern). `stat(follow_symlinks)` on the resolved path — the gates
  resolve-and-confine first per 016, so the size read is on the confined
  file.
- Gates (in-container): shared with 123 — 81 passed; `ruff check` +
  `ruff format --check` + `mypy` on `voyage/supervisor.py` +
  `voyage/workers/video_common.py` clean.

## Resolution

- Verdict: FIXED at the commit + discovery gates. Files changed:
  `voyage/workers/video_common.py` (constant + helper),
  `voyage/supervisor.py` (commit-gate reject + discovery skip),
  `tests/test_tape_trust_123_171.py` (new: helper accept/reject,
  commit-gate `MediaError`, discovery skip + metric).
- DESIGN proposal (quoted text only, for the DESIGN owner): in §27
  (recovery tapes), add: "Tape paths carry a byte bound (`1 GiB`, ~100× a
  legitimate tape): the commit gate rejects oversized reports and discovery
  skips them to the next-newest tape, both metric-visible — a runaway or
  hostile tape fails before the first restart, never inside `torch.load`."
- Residuals (other files, not touched per scope): enforce the same cap in
  `video_longlive.py:1173-1192` (`_load_recovery_tape`, before
  `torch.load(handle, ...)` — defense in depth, the worker image must not
  rely on supervisor checks) via `check_recovery_tape_size`; long-term,
  fold size + shape + hash into one `verify_tape_for_resume` in
  `video_common` (123's candidate-3 home) so the three workers plus both
  supervisor gates share one contract.

## Investigation log

- 2026-09-30: filed by the 168-177 tails sweep; gate probe run live in `voyage:latest` CPU-only per task brief; code citations are as-read values (concurrent uncommitted edits noted in `voyage/supervisor.py` among others).
