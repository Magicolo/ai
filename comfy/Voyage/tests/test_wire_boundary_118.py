"""Issue 118: `GenerateBlocksRequest.from_payload` must reject mistyped wire
values instead of coercing them (`None -> "None"`, `"false" -> True`,
`1.9 -> 1`, `True -> 1px`). Fail-loud at the boundary, before GPU work.
CPU-only, no workers needed.
"""

from __future__ import annotations

import pytest

from voyage.workers.video_common import GenerateBlocksRequest


def _base(**overrides: object) -> dict[str, object]:
    payload: dict[str, object] = {
        "segment_id": "s",
        "output_path": "/tmp/x.mp4",
        "fps": 24,
    }
    payload.update(overrides)
    return payload


def test_none_prompt_rejected() -> None:
    with pytest.raises(TypeError):
        GenerateBlocksRequest.from_payload(
            _base(prompts=[None], seeds=[1]), width_default=8, height_default=8
        )


def test_int_prompt_rejected() -> None:
    with pytest.raises(TypeError):
        GenerateBlocksRequest.from_payload(
            _base(prompts=[123], seeds=[1]), width_default=8, height_default=8
        )


def test_string_seed_rejected() -> None:
    with pytest.raises(TypeError):
        GenerateBlocksRequest.from_payload(
            _base(prompts=["a"], seeds=["5"]), width_default=8, height_default=8
        )


def test_float_seed_rejected_not_truncated() -> None:
    with pytest.raises(TypeError):
        GenerateBlocksRequest.from_payload(
            _base(prompts=["a"], seeds=[1.9]), width_default=8, height_default=8
        )


def test_bool_seed_rejected() -> None:
    with pytest.raises(TypeError):
        GenerateBlocksRequest.from_payload(
            _base(prompts=["a"], seeds=[True]), width_default=8, height_default=8
        )


def test_string_scene_cut_rejected() -> None:
    with pytest.raises(TypeError):
        GenerateBlocksRequest.from_payload(
            _base(prompts=["a"], seeds=[1], scene_cuts=["false"]),
            width_default=8,
            height_default=8,
        )


def test_int_scene_cut_rejected() -> None:
    with pytest.raises(TypeError):
        GenerateBlocksRequest.from_payload(
            _base(prompts=["a"], seeds=[1], scene_cuts=[0]),
            width_default=8,
            height_default=8,
        )


def test_string_profile_stages_rejected() -> None:
    with pytest.raises(TypeError):
        GenerateBlocksRequest.from_payload(
            _base(prompt="hi", seed=1, profile_stages="false"),
            width_default=8,
            height_default=8,
        )


def test_bool_geometry_rejected() -> None:
    with pytest.raises(TypeError):
        GenerateBlocksRequest.from_payload(
            _base(prompt="hi", seed=1, width=True), width_default=8, height_default=8
        )


def test_string_geometry_rejected() -> None:
    with pytest.raises(TypeError):
        GenerateBlocksRequest.from_payload(
            _base(prompt="hi", seed=1, width="512"), width_default=8, height_default=8
        )


def test_float_geometry_rejected_not_truncated() -> None:
    with pytest.raises(TypeError):
        GenerateBlocksRequest.from_payload(
            _base(prompt="hi", seed=1, width=10.9), width_default=8, height_default=8
        )


def test_single_form_scene_cut_coercion_rejected() -> None:
    with pytest.raises(TypeError):
        GenerateBlocksRequest.from_payload(
            _base(prompt="hi", seed=1, scene_cut="false"),
            width_default=8,
            height_default=8,
        )


def test_valid_payloads_still_parse() -> None:
    multi = GenerateBlocksRequest.from_payload(
        _base(prompts=["a", "b"], seeds=[1, 2], scene_cuts=[True, False]),
        width_default=768,
        height_default=512,
    )
    assert multi.prompts == ("a", "b")
    assert multi.seeds == (1, 2)
    assert multi.scene_cuts == (True, False)
    assert (multi.width, multi.height) == (768, 512)
    single = GenerateBlocksRequest.from_payload(
        _base(prompt="solo", seed=9, scene_cut=True),
        width_default=None,
        height_default=None,
    )
    assert single.prompts == ("solo",)
    assert single.scene_cuts == (True,)
    assert single.width is None and single.height is None
    assert single.profile_stages is False
