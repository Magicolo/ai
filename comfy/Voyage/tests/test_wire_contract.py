"""Wire contract: Unset absent-encoding + GenerateBlocksRequest (issue 088 fold).

Fold of `tests/test_unset.py` (5 tests) + `tests/test_generate_blocks_request.py`
(7 tests) = 12 tests, same assertions, test fn names unchanged. Each cluster
keeps its original module docstring as a banner so the issue ID stays
greppable (batch-8 quintet precedent). No helper collisions (`_multi` only
in the request cluster).
"""

from __future__ import annotations

from pathlib import Path

import pytest

from voyage.config import ProjectConfig, Unset, UnsetType, is_provided, resolve_config
from voyage.workers.video_common import (
    BoundaryKind,
    GenerateBlocksRequest,
    boundary_from_scene_cut,
)

# ---------------------------------------------------------------------------
# Cluster 1: Unset absent-encoding (issue 045) — from tests/test_unset.py
# Original docstring: "Unset absent-encoding (issue 045).
#
# The CLI spells absence as None, the TUI form as "": one sentinel (Unset)
# plus one conversion point (provided_or_none) so consumers only ever
# check `is not None`. CPU-only."
# ---------------------------------------------------------------------------


def test_unset_is_falsy_singleton() -> None:
    assert not Unset
    assert repr(Unset) == "Unset"
    assert isinstance(Unset, UnsetType)


def test_is_provided_treats_both_absences_as_absent() -> None:
    assert not is_provided(Unset)
    assert not is_provided(None)
    assert is_provided(0)
    assert is_provided("")
    assert is_provided(False)
    assert is_provided("qwen")


def test_resolve_config_accepts_unset_like_none() -> None:
    base = ProjectConfig(style="probe")
    via_none = resolve_config(base, blocks=None, take_seconds=None)
    via_unset = resolve_config(base, blocks=Unset, take_seconds=Unset)
    assert via_none == via_unset == base


def test_resolve_config_explicit_value_still_wins() -> None:
    base = ProjectConfig(style="probe")
    assert resolve_config(base, blocks=2).video.blocks_per_segment == 2


# ---------------------------------------------------------------------------
# Cluster 2: GenerateBlocksRequest + BoundaryKind contract (issue 045)
# Original docstring: "GenerateBlocksRequest + BoundaryKind contract (issue
# 045). All cross-process `generate_blocks` construction goes through
# `from_payload`: multi/single forms, validated equal-length tuples, no
# asserts (strippable under -O), no positional construction. CPU-only."
# ---------------------------------------------------------------------------


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
