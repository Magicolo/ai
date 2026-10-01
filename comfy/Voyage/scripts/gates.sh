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
cd "$(dirname "$0")/.."
# DESIGN-ref ratchet (issue 037): every top-level voyage/*.py docstring
# carries its DESIGN section (host-side check — no container needed, fail
# fast before the build). Subpackages (workers/, audio/) belong to issue
# 151's pass, so only voyage/*.py is checked here.
missing_refs="$(for f in voyage/*.py; do head -n 15 "$f" | grep -q DESIGN || echo "$f"; done)"
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
# mypy scope (issue 033): `voyage` plus the converted test modules below
# (83 files: the package + 82 test modules). The remaining tests/ files
# carry pre-existing errors (untyped helpers, stale ignores — see issues
# 033/034) that belong to the owning passes; append a module path here as
# each file is annotated. Untracked/in-flight test files stay out until
# they are both committed and clean. `ruff check .` above still lints
# everything.
docker run --rm --user="$(id -u):$(id -g)" \
  -e PYTHONDONTWRITEBYTECODE=1 \
  -e RUFF_CACHE_DIR=/tmp/voyage-ruff-cache \
  -e MYPY_CACHE_DIR=/tmp/voyage-mypy-cache \
  -e HYPOTHESIS_STORAGE_DIRECTORY=/tmp/voyage-hypothesis \
  -v "$PWD:/app" voyage:latest bash -c \
  "ruff check . && ruff format --check . && ruff check --select PLR2004 voyage/config.py voyage/doctor.py voyage/media.py voyage/sfx_finalize.py && mypy voyage \
    tests/conftest.py tests/test_seeds_properties.py tests/test_beat_properties.py tests/test_similarity_properties.py tests/test_acestep_contract.py \
    tests/test_adapter_contract.py tests/test_audio_accounting.py tests/test_audio_request_validation.py tests/test_audio_take_ahead_guard.py tests/test_audio_workers.py \
    tests/test_augment_config.py tests/test_augment_models.py tests/test_augment_plan.py tests/test_augment_runner.py tests/test_augment_weight_loading.py \
    tests/test_av_alignment_consumer.py tests/test_backend_registry.py tests/test_benchmark.py tests/test_benchmark_counts.py tests/test_causvid_prep.py \
    tests/test_checkpoint_safety.py tests/test_cli_hardening.py tests/test_cli_tui_split.py tests/test_cli_validate_handoff.py tests/test_concept_integrity.py \
    tests/test_config_resolution.py tests/test_console.py tests/test_containers_rank2.py tests/test_cuda_preflight_021.py tests/test_director_default.py \
    tests/test_director_device.py tests/test_director_request_validation.py tests/test_doctor.py tests/test_enter_repo_trees.py tests/test_fake_backends.py \
    tests/test_feedback.py tests/test_finalize_fastpath.py tests/test_generate_ensure.py \
    tests/test_generation_stack.py tests/test_hashing.py tests/test_inspect_metrics_fps_029.py tests/test_inspector.py tests/test_inspector_wiring.py tests/test_integration.py \
    tests/test_ltxv.py tests/test_ltxv_failure_hygiene.py tests/test_ltxv_oom_fallback.py \
    tests/test_media_robustness_rank2.py tests/test_observability_rank2.py tests/test_paths.py tests/test_phase3.py \
    tests/test_recovery.py tests/test_registry_pins.py tests/test_rhythm.py tests/test_rpc_deadline_finite.py \
    tests/test_rpc_paths_hardening.py tests/test_rpc_start.py tests/test_rpc_timeout.py tests/test_run_relative_consumer.py tests/test_run_sh.py \
    tests/test_scoreboard.py tests/test_sfx_contract.py tests/test_sfx_finalize.py \
    tests/test_state_integrity.py tests/test_supervisor_av_align.py tests/test_supervisor_lifecycle.py tests/test_surface_rank2.py tests/test_three_captions.py \
    tests/test_tui.py tests/test_tui_absent_defaults_023.py tests/test_unit.py tests/test_vision_metrics.py \
    tests/test_vocoder_allowlist.py tests/test_wire_contract.py tests/test_worker_perf_rank2.py \
    && coverage run -m pytest -q -m 'not gpu' && coverage report"
# Cache guard (issue 089): fail loud when gate caches leak into the
# bind-mounted tree. Sourced from lib/common.sh (owned) so the check and
# the VOYAGE_CACHE_ENV contract cannot drift apart.
# shellcheck disable=SC1091
source "$(dirname "$0")/lib/common.sh"
voyage_assert_no_cache_residue "$(dirname "$0")/.."
