"""Issue 084 TDD: shared worker validators + resident helpers (failing-first).

Pins the unification contract before the implementation lands:
- `voyage.workers._validators` is the single home for the triplicated
  worker validators (sample_rate/channels/output_path/geometry/fps/
  frame_count/energy/window_id/duration-base).
- `voyage.workers._resident` is the single home for `BYTES_PER_GIB`
  plus the `find_spec` torch guard.
- The fake SFX worker no longer reverse-imports the GPU SFX worker
  (slim import must not pull GPU modules).
"""

from __future__ import annotations

import sys


def test_shared_validators_module_exists() -> None:
    from voyage.workers import _validators as validators

    for name in (
        "validate_sample_rate",
        "validate_channels",
        "validate_output_path",
        "validate_geometry",
        "validate_fps",
        "validate_frame_count",
        "validate_energy",
        "validate_window_id",
        "validate_duration_seconds",
    ):
        assert callable(getattr(validators, name)), name


def test_shared_validators_reject_bad_inputs() -> None:
    import pytest

    from voyage.workers import _validators as validators

    with pytest.raises(ValueError, match="sample_rate"):
        validators.validate_sample_rate(0)
    with pytest.raises(ValueError, match="channels"):
        validators.validate_channels(3)
    with pytest.raises(ValueError, match="output_path"):
        validators.validate_output_path("   ")
    with pytest.raises(ValueError, match="width"):
        validators.validate_geometry(0, 64)
    with pytest.raises(ValueError, match="fps"):
        validators.validate_fps(0)
    with pytest.raises(ValueError, match="frames"):
        validators.validate_frame_count(0)
    with pytest.raises(ValueError, match="energy"):
        validators.validate_energy(2.0)
    with pytest.raises(ValueError, match="window_id"):
        validators.validate_window_id("  ")
    with pytest.raises(ValueError, match="duration_seconds"):
        validators.validate_duration_seconds(0.0)


def test_shared_validators_reject_non_finite_and_bool_221() -> None:
    """Issue 221: range-only validators reject NaN/inf/bool like validate_energy."""
    import pytest

    from voyage.workers import _validators as validators

    hostile_numbers = [float("nan"), float("inf"), float("-inf"), True, False]
    for hostile in hostile_numbers:
        with pytest.raises(ValueError, match="sample_rate"):
            validators.validate_sample_rate(hostile)  # type: ignore[arg-type]
        with pytest.raises(ValueError, match="fps"):
            validators.validate_fps(hostile)  # type: ignore[arg-type]
        with pytest.raises(ValueError, match="frames"):
            validators.validate_frame_count(hostile)  # type: ignore[arg-type]
        with pytest.raises(ValueError, match="width"):
            validators.validate_geometry(hostile, 64)  # type: ignore[arg-type]
        with pytest.raises(ValueError, match="height"):
            validators.validate_geometry(64, hostile)  # type: ignore[arg-type]
        with pytest.raises(ValueError, match="channels"):
            validators.validate_channels(hostile)  # type: ignore[arg-type]


def test_resident_module_exports_gib_and_torch_guard() -> None:
    from voyage.workers import _resident as resident

    assert resident.BYTES_PER_GIB == 1024**3
    assert callable(resident.require_torch)


def test_fake_sfx_worker_does_not_reverse_import_gpu_worker() -> None:
    for module in ("voyage.workers.sfx", "voyage.workers.sfx_mmaudio"):
        sys.modules.pop(module, None)
    import voyage.workers.sfx as fake_sfx

    assert "voyage.workers.sfx_mmaudio" not in sys.modules
    assert callable(fake_sfx.handle_generate_sfx)
