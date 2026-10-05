"""Finalize overlap: model pass + deferred ACE music fork-join on 2-GPU.

TDD contract for the user-approved overlap change (DESIGN §140):
`finalize_run` runs the Real-ESRGAN + FILM model pass (cuda:1) and the
deferred ACE-Step music takes (cuda:0) side by side when the pure gate
`model_music_parallel_armed` holds (tensor path taken, takes pending,
no model device is cuda:0 — so 1-GPU finalizes stay sequential).
"""

from __future__ import annotations

import threading

import pytest

from voyage.media import (
    model_music_parallel_armed,
    run_model_pass_and_music_parallel,
)


def test_parallel_armed_only_with_tensor_path_pending_and_other_card() -> None:
    """The gate is a conjunction: tensor path + pending takes + no cuda:0."""
    assert model_music_parallel_armed(
        tensor_path=True,
        deferred_pending=True,
        model_devices=("cuda:1",),
    )
    assert not model_music_parallel_armed(
        tensor_path=False,
        deferred_pending=True,
        model_devices=("cuda:1",),
    )
    assert not model_music_parallel_armed(
        tensor_path=True,
        deferred_pending=False,
        model_devices=("cuda:1",),
    )


def test_parallel_disarmed_on_single_gpu_collapse() -> None:
    """1-GPU boxes collapse the model pass onto cuda:0 — stay sequential."""
    assert not model_music_parallel_armed(
        tensor_path=True,
        deferred_pending=True,
        model_devices=("cuda:0",),
    )
    assert not model_music_parallel_armed(
        tensor_path=True,
        deferred_pending=True,
        model_devices=("cuda:0", "cuda:1"),
    )
    assert not model_music_parallel_armed(
        tensor_path=True,
        deferred_pending=True,
        model_devices=(),
    )


def test_parallel_branches_actually_overlap() -> None:
    """Rendezvous: each branch waits for the other to start (10 s budget).

    Sequential execution would deadlock here (the first branch would time
    out waiting for a branch that never started), so passing proves both
    branches run concurrently.
    """
    model_entered = threading.Event()
    music_entered = threading.Event()

    def model_work() -> None:
        model_entered.set()
        assert music_entered.wait(timeout=10.0)

    def music_work() -> None:
        music_entered.set()
        assert model_entered.wait(timeout=10.0)

    run_model_pass_and_music_parallel(model_work=model_work, music_work=music_work)
    assert model_entered.is_set()
    assert music_entered.is_set()


def test_parallel_model_error_propagates_after_music_joins() -> None:
    """A model failure never orphans the music branch: it still runs."""

    music_ran = threading.Event()

    def model_work() -> None:
        raise RuntimeError("model blew up")

    def music_work() -> None:
        music_ran.set()

    with pytest.raises(RuntimeError, match="model blew up"):
        run_model_pass_and_music_parallel(model_work=model_work, music_work=music_work)
    assert music_ran.is_set()


def test_parallel_music_error_propagates_after_model_joins() -> None:
    """A music failure never orphans the model branch: it still runs."""

    model_ran = threading.Event()

    def model_work() -> None:
        model_ran.set()

    def music_work() -> None:
        raise RuntimeError("music blew up")

    with pytest.raises(RuntimeError, match="music blew up"):
        run_model_pass_and_music_parallel(model_work=model_work, music_work=music_work)
    assert model_ran.is_set()


def test_parallel_model_error_takes_precedence() -> None:
    """Both branches fail: the model error wins (video decides the timeline)."""

    def model_work() -> None:
        raise RuntimeError("model blew up")

    def music_work() -> None:
        raise ValueError("music blew up")

    with pytest.raises(RuntimeError, match="model blew up"):
        run_model_pass_and_music_parallel(model_work=model_work, music_work=music_work)


@pytest.mark.parametrize(
    ("tensor_path", "presentation_fps", "stretch", "expected"),
    [
        (True, 32, 1.5, True),
        (True, 32, 1.0, False),
        (True, None, 1.5, False),
        (False, 32, 1.5, False),
        (False, None, 1.0, False),
    ],
)
def test_slowmo_gate_truth_table(
    tensor_path: bool, presentation_fps: int | None, stretch: float, expected: bool
) -> None:
    """Pin the slow-mo gate the pre-fork `audio_stretch` relies on.

    The fork precomputes `stretch if slowmo else 1.0` from `tensor_path`
    instead of the post-branch `tensor_intermediate is not None` — valid
    only because both model-pass entries return non-None exactly when
    the gate holds. If this truth table ever changes, the fork's
    timeline prediction must be re-proved.
    """
    from voyage.media import slowmo_video_active

    assert slowmo_video_active(tensor_path, presentation_fps, stretch) is expected
