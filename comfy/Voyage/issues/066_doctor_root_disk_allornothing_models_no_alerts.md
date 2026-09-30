# 066 — `doctor.probe()` measures disk at `/`, reports `models_ok` all-or-nothing, and alerts on nothing

**Severity:** MEDIUM

**File:line:** `voyage/doctor.py:135-165` (`disk_free_gib: _disk_free_gib(Path("/"))`, `models_ok` conjunction); consumers `voyage/cli.py:698-704` (`status`)
- **Description:** Disk-full is a first-class state (`PAUSED_DISK_FULL`, finalize preflight, `min_free_space_gib`), but doctor measures free space at filesystem root `/` — not the run dir, not `/models`, not `/tmp` (all three are distinct mounts in the standard `run.sh`: `$PWD:/app`, `/tmp:/tmp`, `$models:/models`). `models_ok` is `exists and all(checks ok)` — one missing *optional* stack (inspector VLM) flips the whole flag red. There are no thresholds anywhere: no VRAM pressure line, no compute-capability / CUDA-runtime / temp / throttle facts (all listed as "remaining gaps" in the docstring and TROUBLESHOOTING), no comparison against `min_free_space_gib`.
- **Rationale:** Health checks should answer "can I start / should I pause" per mount with thresholds, not report one root number. Monitoring guidance converges on: disk-by-mount with >85–90% alerts, VRAM used/pressure + temp, service reachability — doctor covers none as alerts, and `status`/`generate` consume its numbers without thresholds.
- **Evidence (re-verified live 2026-09-30):**
```python
# voyage/doctor.py:149-164 (live)
models_ok = (
    bool(model_facts["exists"]) and all(check["ok"] for check in model_facts["checks"].values())
    if model_facts["checks"]
    else False
)
"disk_free_gib": _disk_free_gib(Path("/")),
```
`grep -in "vram\|temperature\|throttle\|capability\|cuda_runtime" voyage/doctor.py` → no VRAM/temp/throttle/capability/runtime facts (words appear only in gap documentation). `status` prints `Free: X GiB` with no reserve comparison.
- **Repro:** Mount a small `/tmp`, fill `/models` past 90% while `/` stays roomy → `voyage doctor` still reports healthy `disk_free_gib`. Delete only the optional inspector snapshot → `models_ok: false` although every required stack verifies.
- **Fix candidates:** Report `disk_free_gib` per path (`run_dir`, `/models`, `/tmp`) + `meets_reserve: bool` vs `min_free_space_gib`; split `models_ok` into `models_ok_required` / `models_ok_all`; add VRAM free/total + compute capability + CUDA runtime + driver from `nvidia-smi` (cheap, already shelled); emit WARN lines (and non-zero exit tiers) at 85%/90% disk and VRAM-pressure thresholds.
- **Refs:** `docs/TROUBLESHOOTING.md` (gap list + disk-full policy); `docs/STATE_AND_RECOVERY.md:40-42` (reserve knob); local-AI monitoring guidance ("disk space by mount", "VRAM used", "alert >85% disk / VRAM pressure / temp ≥84 °C sustained").

### Track-E scope note (structure-sweep items 7–12 triage)

Structure-sweep items 8 (scoreboard), 9 (console), 10a (rotation-blind metrics) and 10b (fps divergence) are merged into 062/063/055/061 respectively. Item 7 (`voyage/prompts.py:27-35,64-80,158-199` — staged-prompt truncation/repetition + 7-substring style-override blocklist) and item 11 (`voyage/workers/loop.py:91-168` — fd-level stdout quarantine, `checked_request` extra-key blindness, all-`VoyageError`-fatal taxonomy) were reviewed and left out of Track E as out-of-scope: prompts/director belongs to the prompts track, worker RPC framing/taxonomy to the workers/RPC track (the stdout-quarantine half is cross-referenced in 056). No finding dropped silently — route those two to their owning tracks.
