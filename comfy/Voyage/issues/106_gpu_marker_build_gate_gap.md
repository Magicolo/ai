# 106 — GPU-marker handling inconsistent across gate scripts + stale `test.sh` comment

- **Severity:** Low (tests/toolchain — harmless today, latent trap for the next real GPU test)
- **File:line:** `Voyage/scripts/test.sh:9-10` (comment); `Voyage/scripts/gates.sh:28`; `Voyage/scripts/build.sh:16`; `Voyage/scripts/build-video.sh` (smoke only, no pytest — no deselect gap there); `Voyage/tests/test_acestep_contract.py:83`
- **Description:** (a) `test.sh` deselects `gpu` with the comment "no in-tree gpu tests exist yet — GPU legs live in qualify.sh/manual runs", but `tests/test_acestep_contract.py:83` carries `@pytest.mark.gpu` in-tree (it triple-skips without CUDA/weights plus an unconditional final `pytest.skip`, so it is collect-and-skip, not absent). The comment is factually stale. (b) `test.sh` and `gates.sh` both run pytest with `-m "not gpu"`, but `build.sh` runs bare `python -m pytest -q` with no deselect (`build-video.sh` is smoke-only, no pytest). Today the single gpu test skips unconditionally, so all scripts stay green; the first *real* GPU test (weights + CUDA, non-skipping) will silently start executing inside the build gate, which runs on whatever box builds the image — including GPU-less CI.
- **Rationale:** 089 records the marker registration (gpu + endurance, 1 test each) and the gates deselect, but does not note the `build*.sh` deselect gap or the stale "no in-tree gpu tests" comment. The failure mode is a slow-motion one: a future GPU test author sees `test.sh` deselecting, assumes gates are safe, and breaks `build.sh` on boxes without CUDA. One-line fixes now prevent a confusing red gate later.
- **Evidence (verified live 2026-09-30):**
  - `scripts/test.sh:9-13`: `if [ $# -eq 0 ]; then set -- -m "not gpu"; fi` with the "no in-tree gpu tests exist yet" comment.
  - `scripts/gates.sh`: `coverage run -m pytest -q -m 'not gpu'`.
- `scripts/build.sh:16`: `voyage:latest bash -c "ruff check . && ruff format --check . && mypy voyage && python -m pytest -q"` — no `-m`.
- `scripts/build-video.sh`: smoke only (`video_causvid/longlive/ltxv` import + serve-map assert) — no pytest, so no deselect gap there.
  - `tests/test_acestep_contract.py:83-99`: `@pytest.mark.gpu` + three conditional skips + unconditional `pytest.skip("GPU render needs provisioned ACE-Step checkpoints (see DESIGN §37)")`.
  - `pyproject.toml:186-187`: `gpu` marker registered, so collection + deselect machinery already exists.
- **Repro:**
  ```bash
  grep -n "not gpu" scripts/*.sh   # test.sh + gates.sh only; build.sh/build-director.sh absent
  grep -rn "pytest.mark.gpu" tests/ # test_acestep_contract.py:83 — contradicts the test.sh comment
  ```
- **Fix candidates:**
  1. Add `-m 'not gpu'` to the `pytest` invocation in `build.sh` (matching test.sh/gates.sh), or extract the shared deselect into one variable if the common.sh consolidation (090) lands first.
  2. Fix the `test.sh` comment to "in-tree gpu tests always skip without CUDA/weights; deselected so gates never pay for them".
  3. Gate: `grep -c "not gpu" scripts/build.sh` == 1, plus the existing `test_run_sh.py` suite stays green.
- **Overlaps with:** 090 (common.sh proposal — the durable home for this fix); 089 (markers registered — this file is the build-gate half it does not cover).
- **Refs:** `Voyage/issues/089_test_hygiene_lock_markers.md` (markers registered, slow tail unmarked — this file is the build-gate half it does not cover); `Voyage/issues/090_scripts_containers_cleanup.md` (common.sh proposal — the durable home for this fix); `Voyage/tests/test_acestep_contract.py:83-99`.
