"""`generate <NAME> --segments/--duration`: extend the stored plan, then run.

Extension is additive: the new plan is `manifest segments + added`,
where `--duration` converts via the steady-state segment math (rounds
up, same as `configure`). The flags are mutually exclusive (configure
precedent); bad values leave the manifest untouched.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pytest


def _extend_namespace(
    name: str, segments: object = None, duration: object = None
) -> argparse.Namespace:
    return argparse.Namespace(
        name=name, verbose=False, no_color=True, segments=segments, duration=duration
    )


def _write_plan(run_dir: Path, segments: int) -> None:
    manifest_path = run_dir / "manifest.json"
    manifest_path.write_text(
        json.dumps(
            {
                **json.loads(manifest_path.read_text()),
                "segments": segments,
                "final_video": None,
                "skip_bad": False,
                "no_sfx": True,
            }
        ),
        encoding="utf-8",
    )


def _stub_generation(monkeypatch: pytest.MonkeyPatch, calls: dict[str, object]) -> None:
    import voyage.cli_generate as gen_ops

    class _FakeSupervisor:
        def __init__(self, *args: object, **kwargs: object) -> None:
            pass

        def run_segments(self, count: object) -> list[str]:
            calls["segments"] = count
            return ["000000"]

    monkeypatch.setattr(gen_ops, "Supervisor", _FakeSupervisor)
    monkeypatch.setattr("voyage.cli_finalize.cmd_finalize", lambda ns: 0)
    monkeypatch.setattr(gen_ops, "validate_run", lambda run_dir: [])


def _plan(run_dir: Path) -> dict:
    return json.loads((run_dir / "manifest.json").read_text(encoding="utf-8"))


def test_generate_extend_segments_adds_to_plan(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    import voyage.cli_generate as gen_ops
    from tests.conftest import initialize_run_directory

    monkeypatch.chdir(tmp_path)
    run_dir = tmp_path / "output" / "grow"
    initialize_run_directory(run_dir, run_id="grow")
    _write_plan(run_dir, 2)
    calls: dict[str, object] = {}
    _stub_generation(monkeypatch, calls)
    assert gen_ops.cmd_generate(_extend_namespace("grow", segments=2)) == 0
    assert _plan(run_dir)["segments"] == 4
    assert calls["segments"] == 4
    assert "extended plan" in capsys.readouterr().out


def test_generate_extend_duration_rounds_up(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Fake runs 2.0s per segment (24fps x 48f): 2.1s needs 2 segments."""
    import voyage.cli_generate as gen_ops
    from tests.conftest import initialize_run_directory

    monkeypatch.chdir(tmp_path)
    run_dir = tmp_path / "output" / "dur"
    initialize_run_directory(run_dir, run_id="dur")
    _write_plan(run_dir, 1)
    calls: dict[str, object] = {}
    _stub_generation(monkeypatch, calls)
    assert gen_ops.cmd_generate(_extend_namespace("dur", duration=2.1)) == 0
    assert _plan(run_dir)["segments"] == 3
    assert calls["segments"] == 3


def test_generate_extend_duration_exact_boundary(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """4.0s is exactly 2 fake segments — never rounds up to 3."""
    import voyage.cli_generate as gen_ops
    from tests.conftest import initialize_run_directory

    monkeypatch.chdir(tmp_path)
    run_dir = tmp_path / "output" / "edge"
    initialize_run_directory(run_dir, run_id="edge")
    _write_plan(run_dir, 1)
    calls: dict[str, object] = {}
    _stub_generation(monkeypatch, calls)
    assert gen_ops.cmd_generate(_extend_namespace("edge", duration=4.0)) == 0
    assert _plan(run_dir)["segments"] == 3
    assert calls["segments"] == 3


def test_generate_extend_both_flags_is_exit_2(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    import voyage.cli_generate as gen_ops
    from tests.conftest import initialize_run_directory

    monkeypatch.chdir(tmp_path)
    run_dir = tmp_path / "output" / "both"
    initialize_run_directory(run_dir, run_id="both")
    _write_plan(run_dir, 2)
    assert gen_ops.cmd_generate(_extend_namespace("both", segments=1, duration=5.0)) == 2
    assert _plan(run_dir)["segments"] == 2
    assert capsys.readouterr().err != ""


@pytest.mark.parametrize("bad", [0, -1])
def test_generate_extend_non_positive_segments_is_exit_2(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, bad: int
) -> None:
    import voyage.cli_generate as gen_ops
    from tests.conftest import initialize_run_directory

    monkeypatch.chdir(tmp_path)
    run_dir = tmp_path / "output" / f"bad{bad}"
    initialize_run_directory(run_dir, run_id=f"bad{bad}")
    _write_plan(run_dir, 2)
    assert gen_ops.cmd_generate(_extend_namespace(f"bad{bad}", segments=bad)) == 2
    assert _plan(run_dir)["segments"] == 2


def test_generate_extend_preserves_coverage_stamp(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Extension keeps final_coverage; the gate reads it stale (fewer
    segments than committed) and re-finalizes — no silent fresh stamp."""
    import voyage.cli_generate as gen_ops
    from tests.conftest import initialize_run_directory
    from voyage.persistence import record_final_coverage

    monkeypatch.chdir(tmp_path)
    run_dir = tmp_path / "output" / "stamp"
    initialize_run_directory(run_dir, run_id="stamp")
    _write_plan(run_dir, 2)
    record_final_coverage(run_dir, presented_frames=96, segments=2)
    calls: dict[str, object] = {}
    _stub_generation(monkeypatch, calls)
    assert gen_ops.cmd_generate(_extend_namespace("stamp", segments=1)) == 0
    assert _plan(run_dir)["final_coverage"] == {"segments": 2, "presented_frames": 96}


def test_generate_without_flags_leaves_plan_untouched(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import voyage.cli_generate as gen_ops
    from tests.conftest import initialize_run_directory

    monkeypatch.chdir(tmp_path)
    run_dir = tmp_path / "output" / "plain"
    initialize_run_directory(run_dir, run_id="plain")
    _write_plan(run_dir, 2)
    calls: dict[str, object] = {}
    _stub_generation(monkeypatch, calls)
    assert gen_ops.cmd_generate(_extend_namespace("plain")) == 0
    assert _plan(run_dir)["segments"] == 2
    assert calls["segments"] == 2
