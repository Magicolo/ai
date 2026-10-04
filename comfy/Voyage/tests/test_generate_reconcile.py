"""`generate <NAME>` reconciliation (two-verb CLI, step 3).

The manifest plans; `state.json` rules. Generate reconciles the
directory against both: manifest-only means a new run, extras beyond
the committed count are removed before resuming, and a validated run
whose `final.mp4` already covers the timeline does no work.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pytest


def _generate_namespace(name: str) -> argparse.Namespace:
    return argparse.Namespace(name=name, verbose=False, no_color=True)


def _manifest(run_dir: Path) -> dict:
    return json.loads((run_dir / "manifest.json").read_text(encoding="utf-8"))


def test_generate_new_run_renders_planned_segments(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import voyage.cli_generate as gen_ops
    from tests.conftest import initialize_run_directory

    monkeypatch.chdir(tmp_path)
    run_dir = tmp_path / "output" / "fresh"
    initialize_run_directory(run_dir, run_id="fresh")
    manifest_path = run_dir / "manifest.json"
    manifest_path.write_text(
        json.dumps(
            {
                **json.loads(manifest_path.read_text()),
                "segments": 2,
                "final_video": None,
                "skip_bad": False,
                "no_sfx": True,
            }
        ),
        encoding="utf-8",
    )
    calls: dict[str, object] = {}

    class _FakeSupervisor:
        def __init__(self, *args: object, **kwargs: object) -> None:
            pass

        def run_segments(self, count: object) -> list[str]:
            calls["segments"] = count
            return ["000000", "000001"]

    def _fake_finalize(ns: argparse.Namespace) -> int:
        calls["finalize"] = ns
        return 0

    monkeypatch.setattr(gen_ops, "Supervisor", _FakeSupervisor)
    monkeypatch.setattr("voyage.cli_finalize.cmd_finalize", _fake_finalize)
    monkeypatch.setattr(gen_ops, "validate_run", lambda run_dir: [])
    assert gen_ops.cmd_generate(_generate_namespace("fresh")) == 0
    assert calls["segments"] == 2
    assert isinstance(calls.get("finalize"), argparse.Namespace)


def test_generate_noop_when_complete(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    import voyage.cli_generate as gen_ops
    from tests.conftest import initialize_run_directory
    from voyage.persistence import read_state, write_state

    monkeypatch.chdir(tmp_path)
    run_dir = tmp_path / "output" / "done"
    initialize_run_directory(run_dir, run_id="done")
    manifest_path = run_dir / "manifest.json"
    manifest_path.write_text(
        json.dumps({**json.loads(manifest_path.read_text()), "segments": 1}),
        encoding="utf-8",
    )
    segment = run_dir / "segments" / "000000"
    segment.mkdir(parents=True, exist_ok=True)
    (segment / "DONE").write_text("done\n", encoding="utf-8")
    (segment / "video.mp4").write_bytes(b"fake-video")
    (segment / "audio.wav").write_bytes(b"fake-audio")
    (segment / "manifest.json").write_text(
        json.dumps(
            {
                "format": 1,
                "transition": {},
                "prompt_plan": {},
                "audio_state": {},
                "world_state": {},
                "metrics": {"frames": 48},
                "checksums": {
                    "video.mp4": "c9b936a163cb84ee9137fa239ae9050c5831a36156f4e61ab291e13454b3b9ce",
                    "audio.wav": "69538b86470d5575fc0181cf3b0d0e79ecacb05b6bc6f58c17e759154848e35f",
                },
            }
        ),
        encoding="utf-8",
    )
    state = read_state(run_dir)
    state.committed_segments = 1
    state.next_segment_number = 1
    state.timeline_frames = 48
    write_state(run_dir, state)
    from voyage.persistence import record_final_coverage

    record_final_coverage(run_dir, presented_frames=48, segments=1)
    from voyage.fake_backends import FakeVideoBackend

    FakeVideoBackend().generate_segment(
        run_dir / "final.mp4", prompt="noop", seed=7, width=64, height=64, fps=24, frames=48
    )

    def _explode(*args: object, **kwargs: object) -> object:
        raise AssertionError("complete run must do no work")

    monkeypatch.setattr(gen_ops, "Supervisor", _explode)
    monkeypatch.setattr("voyage.cli_finalize.cmd_finalize", _explode)
    assert gen_ops.cmd_generate(_generate_namespace("done")) == 0
    assert "nothing to do" in capsys.readouterr().out


def test_generate_removes_extra_segments_then_resumes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import voyage.cli_generate as gen_ops
    from tests.conftest import initialize_run_directory
    from voyage.persistence import read_state, write_state

    monkeypatch.chdir(tmp_path)
    run_dir = tmp_path / "output" / "messy"
    initialize_run_directory(run_dir, run_id="messy")
    manifest_path = run_dir / "manifest.json"
    manifest_path.write_text(
        json.dumps({**json.loads(manifest_path.read_text()), "segments": 3}),
        encoding="utf-8",
    )
    for index in ("000000", "000001", "000002"):
        segment = run_dir / "segments" / index
        segment.mkdir(parents=True, exist_ok=True)
        (segment / "DONE").write_text("done\n", encoding="utf-8")
    state = read_state(run_dir)
    state.committed_segments = 2
    state.next_segment_number = 2
    state.timeline_frames = 96
    write_state(run_dir, state)
    calls: dict[str, object] = {}

    class _FakeSupervisor:
        def __init__(self, *args: object, **kwargs: object) -> None:
            pass

        def run_segments(self, count: object) -> list[str]:
            calls["segments"] = count
            return ["000002"]

    monkeypatch.setattr(gen_ops, "Supervisor", _FakeSupervisor)
    monkeypatch.setattr("voyage.cli_finalize.cmd_finalize", lambda ns: 0)
    monkeypatch.setattr(gen_ops, "validate_run", lambda run_dir: [])
    assert gen_ops.cmd_generate(_generate_namespace("messy")) == 0
    assert not (run_dir / "segments" / "000002").exists()
    assert calls["segments"] == 1


def test_generate_missing_manifest_is_exit_2(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    import voyage.cli_generate as gen_ops

    monkeypatch.chdir(tmp_path)
    assert gen_ops.cmd_generate(_generate_namespace("ghost")) == 2
    assert "configure" in capsys.readouterr().err


def test_generate_rejects_traversal_name(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """A crafted NAME cannot escape the output tree."""
    import voyage.cli_generate as gen_ops

    monkeypatch.chdir(tmp_path)
    assert gen_ops.cmd_generate(_generate_namespace("../evil")) == 2
    assert capsys.readouterr().err != ""
    assert not (tmp_path / "evil").exists()
