"""GenerateBlocksRequest + BoundaryKind contract (issue 045).

All cross-process `generate_blocks` construction goes through
`from_payload`: multi/single forms, validated equal-length tuples, no
asserts (strippable under -O), no positional construction. CPU-only.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from voyage.workers.video_common import (
    BoundaryKind,
    GenerateBlocksRequest,
    boundary_from_scene_cut,
)


def _multi(**overrides: object) -> dict[str, object]:
    payload: dict[str, object] = {
        "segment_id": "000007",
        "output_path": "/tmp/seg.mp4",
        "fps": 24,
        "prompts": ["a", "b"],
        "seeds": [1, 2],
        "scene_cuts": [True, False],
    }
    payload.update(overrides)
    return payload


def test_multi_form_parses_with_geometry_defaults() -> None:
    request = GenerateBlocksRequest.from_payload(_multi(), width_default=768, height_default=512)
    assert request.prompts == ("a", "b")
    assert request.seeds == (1, 2)
    assert request.scene_cuts == (True, False)
    assert (request.width, request.height, request.fps) == (768, 512, 24)
    assert request.segment_id == "000007"
    assert request.boundaries == ("fresh", "continue")


def test_single_form_expands_to_one_block() -> None:
    request = GenerateBlocksRequest.from_payload(
        {
            "segment_id": "000000",
            "output_path": "/tmp/seg.mp4",
            "fps": 24,
            "prompt": "solo",
            "seed": 9,
            "scene_cut": True,
        },
        width_default=None,
        height_default=None,
    )
    assert request.prompts == ("solo",)
    assert request.seeds == (9,)
    assert request.boundaries == ("fresh",)
    assert request.width is None and request.height is None


def test_mismatched_lengths_rejected() -> None:
    with pytest.raises(ValueError, match="scene_cuts"):
        GenerateBlocksRequest.from_payload(
            _multi(scene_cuts=[True]), width_default=1, height_default=1
        )


def test_empty_prompts_rejected() -> None:
    with pytest.raises(ValueError, match="non-empty"):
        GenerateBlocksRequest.from_payload(
            _multi(prompts=[], seeds=[], scene_cuts=[]), width_default=1, height_default=1
        )


def test_missing_field_rejected() -> None:
    with pytest.raises(KeyError):
        GenerateBlocksRequest.from_payload(
            {"output_path": "/tmp/x.mp4", "fps": 24}, width_default=1, height_default=1
        )


def test_boundary_vocabulary() -> None:
    assert boundary_from_scene_cut(True) == "fresh"
    assert boundary_from_scene_cut(False) == "continue"
    kind: BoundaryKind = "fresh"
    assert kind == "fresh"


def test_output_path_and_digest_passthrough() -> None:
    request = GenerateBlocksRequest.from_payload(
        _multi(prompt_plan_hash="abc123", frames=96),
        width_default=1,
        height_default=1,
    )
    assert request.output_path == Path("/tmp/seg.mp4")
    assert request.prompt_plan_digest == "abc123"
    assert request.requested_frames == 96
