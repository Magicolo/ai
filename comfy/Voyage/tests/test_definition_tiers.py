"""Definition tiers: `configure --low-definition / --medium-definition / --high-definition`.

Lowest / middle / highest native reasonable resolution per backend, resolved
against the effective backend (--backend first, then tier). Creates
default to high; updates without flags keep stored geometry; --from
inherits the source geometry unless --backend or a tier flag overrides
it. Any two tier flags together are exit 2. The ltx25/ltx23 workers accept
exactly the three tier geometries (high 1216x704 + medium 1024x576 + low
768x448; ltx25 runs 257f/25-carry windows, ltx23 121f/25-carry — the
graph-builder stage-1 override is
pure and tested here; the GPU commit path is covered by the worker suites.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pytest

from voyage import cli_configure, paths
from voyage.config import (
    BACKEND_REGISTRY,
    DEFINITION_TIERS,
    VideoBackendName,
    _definition_preset,
    preset_config,
    resolve_config,
)
from voyage.workers import video_ltx23, video_ltx25


@pytest.fixture(autouse=True)
def _never_touch_models(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("voyage.models_ensure.ensure_models", lambda *a, **k: 0)


# Tier map pins -------------------------------------------------------------


def test_tier_geometries_match_qa_mapping() -> None:
    """Low/high geometry per backend is the locked Q&A mapping."""
    expected: dict[VideoBackendName, tuple[tuple[int, int], tuple[int, int]]] = {
        "fake": ((512, 288), (768, 432)),
        "ltxv": ((512, 320), (768, 512)),
        "causvid": ((832, 480), (832, 480)),
        "ltx25": ((768, 448), (1216, 704)),
        "ltx23": ((768, 448), (1216, 704)),
    }
    for backend, (low, high) in expected.items():
        tier = DEFINITION_TIERS[backend]
        assert (tier["low"].width, tier["low"].height) == low
        assert (
            DEFINITION_TIERS[backend]["high"].width,
            DEFINITION_TIERS[backend]["high"].height,
        ) == high


def test_high_tiers_equal_registry_rows() -> None:
    """High is the backend's native row (geometry + profile + latent)."""
    for backend, row in BACKEND_REGISTRY.items():
        high = DEFINITION_TIERS[backend]["high"]
        assert (high.width, high.height) == (row.width, row.height)
        assert high.profile == row.profile
        assert list(high.latent_shape or row.latent_shape) == list(row.latent_shape)


def test_ltx_low_tier_latent_is_halved_stage1() -> None:
    """ltx low latent is [1,128,T,12,7] (384//32 x 224//32; T=33 ltx25, T=16 ltx23)."""
    assert DEFINITION_TIERS["ltx25"]["low"].latent_shape == (1, 128, 33, 12, 7)
    assert DEFINITION_TIERS["ltx23"]["low"].latent_shape == (1, 128, 16, 12, 7)


def test_tier_medium_geometries_match_spec() -> None:
    """Medium geometry per backend: 1024x576, causvid stays 832x480."""
    expected: dict[VideoBackendName, tuple[int, int]] = {
        "fake": (1024, 576),
        "ltxv": (1024, 576),
        "causvid": (832, 480),
        "ltx25": (1024, 576),
        "ltx23": (1024, 576),
    }
    for backend, medium in expected.items():
        tier = DEFINITION_TIERS[backend]["medium"]
        assert (tier.width, tier.height) == medium


def test_tier_medium_profile_names() -> None:
    """Medium profiles are the -576p rows; causvid keeps its fixed profile."""
    expected: dict[VideoBackendName, str] = {
        "fake": "fake-576p",
        "ltxv": "ltxv-576p",
        "causvid": "causvid-480p",
        "ltx25": "ltx25-576p",
        "ltx23": "ltx23-576p",
    }
    for backend, profile in expected.items():
        assert DEFINITION_TIERS[backend]["medium"].profile == profile


def test_ltx_medium_tier_latent_shape() -> None:
    """ltx medium latent is [1,128,T,16,9] (512//32 x 288//32; T=33 ltx25, T=16 ltx23)."""
    assert DEFINITION_TIERS["ltx25"]["medium"].latent_shape == (1, 128, 33, 16, 9)
    assert DEFINITION_TIERS["ltx23"]["medium"].latent_shape == (1, 128, 16, 16, 9)


def test_definition_preset_rejects_bad_tier_and_backend() -> None:
    """Unknown tiers and backends fail loud (no silent default)."""
    with pytest.raises(ValueError, match="unknown definition tier"):
        _definition_preset("ltx25", "ultra")
    with pytest.raises(ValueError, match="unknown backend"):
        _definition_preset("nope", "low")


# Resolver order: backend preset first, then tier ----------------------------


def test_resolve_config_definition_low_on_default_backend() -> None:
    """definition='low' on the default (ltx25) config yields 768x448."""
    effective = resolve_config(preset_config("r", "s", 1), definition="low")
    assert (effective.video.width, effective.video.height) == (768, 448)
    assert effective.video.profile == "ltx25-448p"
    assert effective.video.latent_shape == [1, 128, 33, 12, 7]


def test_resolve_config_backend_then_tier() -> None:
    """--backend ltxv --low-definition yields ltxv's low, not ltx25's."""
    effective = resolve_config(preset_config("r", "s", 1), backend="ltxv", definition="low")
    assert (effective.video.width, effective.video.height) == (512, 320)
    assert effective.video.profile == "ltxv-320p"


def test_resolve_config_without_definition_keeps_preset() -> None:
    """No tier flag leaves the backend preset geometry untouched."""
    effective = resolve_config(preset_config("r", "s", 1), backend="ltxv")
    assert (effective.video.width, effective.video.height) == (768, 512)
    assert effective.video.profile == "ltxv-512p"


def test_resolve_config_definition_medium_on_default_backend() -> None:
    """definition='medium' on the default (ltx25) config yields 1024x576."""
    effective = resolve_config(preset_config("r", "s", 1), definition="medium")
    assert (effective.video.width, effective.video.height) == (1024, 576)
    assert effective.video.profile == "ltx25-576p"
    assert effective.video.latent_shape == [1, 128, 33, 16, 9]


def test_resolve_config_backend_then_medium_tier() -> None:
    """--backend ltxv --medium-definition yields ltxv's medium, not ltx25's."""
    effective = resolve_config(preset_config("r", "s", 1), backend="ltxv", definition="medium")
    assert (effective.video.width, effective.video.height) == (1024, 576)
    assert effective.video.profile == "ltxv-576p"


def test_resolve_config_medium_tier_on_fixed_backend() -> None:
    """--backend causvid --medium-definition stays 832x480 (fixed geometry)."""
    effective = resolve_config(preset_config("r", "s", 1), backend="causvid", definition="medium")
    assert (effective.video.width, effective.video.height) == (832, 480)
    assert effective.video.profile == "causvid-480p"


# Worker acceptance (pure parts) ----------------------------------------------


def test_ltx_workers_accept_all_tier_sizes() -> None:
    """All three LTX commit sizes: high + medium + low."""
    expected_options = frozenset({(1216, 704), (1024, 576), (768, 448)})
    ltx25_options = video_ltx25.COMMIT_SIZE_OPTIONS
    ltx23_options = video_ltx23.COMMIT_SIZE_OPTIONS
    assert ltx25_options == expected_options
    assert ltx23_options == expected_options
    assert video_ltx25.STAGE1_FOR_COMMIT_SIZE[(1024, 576)] == (512, 288)
    assert video_ltx23.STAGE1_FOR_COMMIT_SIZE[(1024, 576)] == (512, 288)
    assert video_ltx25.STAGE1_FOR_COMMIT_SIZE[(768, 448)] == (384, 224)
    assert video_ltx23.STAGE1_FOR_COMMIT_SIZE[(768, 448)] == (384, 224)
    assert video_ltx25.STAGE1_FOR_COMMIT_SIZE[(1216, 704)] == (608, 352)
    assert video_ltx23.STAGE1_FOR_COMMIT_SIZE[(1216, 704)] == (608, 352)


def test_ltx_graph_builders_default_to_high_stage1() -> None:
    """No stage-1 override keeps the baked 608x352 node (backward compatible)."""
    for builder in (video_ltx25.build_mode_a_graph, video_ltx23.build_mode_a_graph):
        graph = builder(prompt="p", seed=1, save_prefix="s")
        assert graph["8"]["inputs"]["width"] == 608
        assert graph["8"]["inputs"]["height"] == 352


def test_ltx_graph_builders_accept_low_stage1() -> None:
    """stage1_size=(384,224) rewires node 8 for the low tier (both workers)."""
    for builder in (video_ltx25.build_mode_a_graph, video_ltx23.build_mode_a_graph):
        graph = builder(prompt="p", seed=1, save_prefix="s", stage1_size=(384, 224))
        assert graph["8"]["inputs"]["width"] == 384
        assert graph["8"]["inputs"]["height"] == 224


def test_ltx_validators_accept_low_tier_geometry() -> None:
    """768x448 clears the /64 validators both LTX workers enforce."""
    from voyage.workers import video_ltx23_validators, video_ltx25_validators

    video_ltx25_validators.validate_spatial_size(768, 448)
    video_ltx23_validators.validate_spatial_size(768, 448)


def test_ltx_graph_builders_accept_medium_stage1() -> None:
    """stage1_size=(512,288) rewires node 8 for the medium tier (both workers)."""
    for builder in (video_ltx25.build_mode_a_graph, video_ltx23.build_mode_a_graph):
        graph = builder(prompt="p", seed=1, save_prefix="s", stage1_size=(512, 288))
        assert graph["8"]["inputs"]["width"] == 512
        assert graph["8"]["inputs"]["height"] == 288


def test_ltx_validators_accept_medium_tier_geometry() -> None:
    """1024x576 clears the /64 validators both LTX workers enforce."""
    from voyage.workers import video_ltx23_validators, video_ltx25_validators

    video_ltx25_validators.validate_spatial_size(1024, 576)
    video_ltx23_validators.validate_spatial_size(1024, 576)


def test_ltx_tail_decode_missing_file_fails_loud(tmp_path: Path) -> None:
    """Tail decode fail-loud path is intact with the new signature."""
    with pytest.raises(RuntimeError, match="tail decode failed|needs an ffmpeg"):
        video_ltx25._decode_tail_frames(tmp_path / "nope.mp4", 25)
    with pytest.raises(RuntimeError, match="tail decode failed|needs an ffmpeg"):
        video_ltx23._decode_tail_frames(tmp_path / "nope.mp4", 25)


# Configure verb --------------------------------------------------------------


def _configure_namespace(name: str, **overrides: object) -> argparse.Namespace:
    base: dict[str, object] = {
        "name": name,
        "backend": None,
        "from_run": None,
        "duration": None,
        "segments": None,
        "style": None,
        "seed": None,
        "final_video": None,
        "skip_bad": False,
        "no_download": False,
        "force": False,
        "draft": False,
        "director": None,
        "director_device": None,
        "blocks": None,
        "take_seconds": None,
        "quantization": None,
        "beats_per_segment": None,
        "drift_every_n": None,
        "music_caption": None,
        "video_caption": None,
        "upscale": None,
        "interpolate": None,
        "presentation_fps": None,
        "no_sfx": False,
        "sfx_backend": None,
        "sfx_caption": None,
        "sfx_device": None,
        "sfx_model_size": None,
        "sfx_workers": 1,
        "verbose": False,
        "no_color": True,
        "low_definition": False,
        "medium_definition": False,
        "high_definition": False,
    }
    base.update(overrides)
    return argparse.Namespace(**base)


def _manifest_video(tmp_path: Path, name: str) -> dict[str, object]:
    manifest = json.loads((tmp_path / "output" / name / "manifest.json").read_text())
    video = manifest["video"]
    assert isinstance(video, dict)
    return video


def test_configure_create_defaults_to_high_definition(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """No resolution flag configures the highest native geometry."""
    monkeypatch.chdir(tmp_path)
    args = _configure_namespace("hi", style="dark harbors", segments=2, seed=7)
    assert cli_configure.cmd_configure(args) == 0
    video = _manifest_video(tmp_path, "hi")
    assert (video["width"], video["height"]) == (1216, 704)
    assert video["profile"] == "ltx25-704p"
    assert video["latent_shape"] == [1, 128, 33, 19, 11]


def test_configure_create_low_definition(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """--low-definition configures 768x448 + low profile + low latent."""
    monkeypatch.chdir(tmp_path)
    args = _configure_namespace("lo", style="dark harbors", segments=2, seed=7, low_definition=True)
    assert cli_configure.cmd_configure(args) == 0
    video = _manifest_video(tmp_path, "lo")
    assert (video["width"], video["height"]) == (768, 448)
    assert video["profile"] == "ltx25-448p"
    assert video["latent_shape"] == [1, 128, 33, 12, 7]


@pytest.mark.parametrize(
    "flags",
    [
        {"low_definition": True, "medium_definition": True},
        {"low_definition": True, "high_definition": True},
        {"medium_definition": True, "high_definition": True},
        {"low_definition": True, "medium_definition": True, "high_definition": True},
    ],
    ids=["low-medium", "low-high", "medium-high", "all-three"],
)
def test_configure_tier_mutual_exclusion_is_exit_2(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, flags: dict[str, bool]
) -> None:
    """Any two tier flags together (or all three) is an error."""
    monkeypatch.chdir(tmp_path)
    args = _configure_namespace("excl", style="dark harbors", segments=2, seed=7, **flags)
    assert cli_configure.cmd_configure(args) == 2


def test_configure_create_medium_definition(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """--medium-definition configures 1024x576 + medium profile + medium latent."""
    monkeypatch.chdir(tmp_path)
    args = _configure_namespace(
        "med", style="dark harbors", segments=2, seed=7, medium_definition=True
    )
    assert cli_configure.cmd_configure(args) == 0
    video = _manifest_video(tmp_path, "med")
    assert (video["width"], video["height"]) == (1024, 576)
    assert video["profile"] == "ltx25-576p"
    assert video["latent_shape"] == [1, 128, 33, 16, 9]


def test_configure_implicit_high_follows_selected_backend(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """No tier flag + --backend ltxv yields ltxv's high (768x512)."""
    monkeypatch.chdir(tmp_path)
    args = _configure_namespace("lv", style="dark harbors", segments=2, seed=7, backend="ltxv")
    assert cli_configure.cmd_configure(args) == 0
    video = _manifest_video(tmp_path, "lv")
    assert (video["width"], video["height"]) == (768, 512)


def test_configure_backend_then_low_tier(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """--backend ltxv --low-definition yields ltxv's low (512x320)."""
    monkeypatch.chdir(tmp_path)
    args = _configure_namespace(
        "lvlo",
        style="dark harbors",
        segments=2,
        seed=7,
        backend="ltxv",
        low_definition=True,
    )
    assert cli_configure.cmd_configure(args) == 0
    video = _manifest_video(tmp_path, "lvlo")
    assert (video["width"], video["height"]) == (512, 320)


def test_configure_update_without_flags_keeps_geometry(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Updating style only never touches the stored tier geometry."""
    monkeypatch.chdir(tmp_path)
    assert (
        cli_configure.cmd_configure(
            _configure_namespace("keep", style="first", segments=2, seed=7, low_definition=True)
        )
        == 0
    )
    assert cli_configure.cmd_configure(_configure_namespace("keep", style="second")) == 0
    video = _manifest_video(tmp_path, "keep")
    assert (video["width"], video["height"]) == (768, 448)
    assert video["profile"] == "ltx25-448p"


def test_configure_update_tier_on_fresh_run(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """--high-definition on an uncommitted low run switches geometry."""
    monkeypatch.chdir(tmp_path)
    assert (
        cli_configure.cmd_configure(
            _configure_namespace("up", style="first", segments=2, seed=7, low_definition=True)
        )
        == 0
    )
    assert cli_configure.cmd_configure(_configure_namespace("up", high_definition=True)) == 0
    video = _manifest_video(tmp_path, "up")
    assert (video["width"], video["height"]) == (1216, 704)
    assert video["profile"] == "ltx25-704p"


def test_configure_update_tier_on_committed_run_refused(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A tier change after commits is refused (geometries would mix)."""
    from voyage.persistence import read_state, write_state

    monkeypatch.chdir(tmp_path)
    assert (
        cli_configure.cmd_configure(
            _configure_namespace("mix", style="first", segments=3, seed=7, low_definition=True)
        )
        == 0
    )
    run_dir = tmp_path / "output" / "mix"
    state = read_state(run_dir)
    state.committed_segments = 1
    write_state(run_dir, state)
    assert cli_configure.cmd_configure(_configure_namespace("mix", high_definition=True)) == 2
    video = _manifest_video(tmp_path, "mix")
    assert (video["width"], video["height"]) == (768, 448)


def test_configure_from_inherits_source_geometry(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """--from without --backend keeps the source tier untouched."""
    monkeypatch.chdir(tmp_path)
    assert (
        cli_configure.cmd_configure(
            _configure_namespace("src", style="first", segments=2, seed=7, low_definition=True)
        )
        == 0
    )
    assert (
        cli_configure.cmd_configure(
            _configure_namespace("dst", from_run="src", style="second", segments=2)
        )
        == 0
    )
    video = _manifest_video(tmp_path, "dst")
    assert (video["width"], video["height"]) == (768, 448)
    assert video["profile"] == "ltx25-448p"


def test_configure_from_backend_then_tier_overrides_source(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """--from + --backend ltxv --low-definition yields ltxv's low geometry."""
    monkeypatch.chdir(tmp_path)
    assert (
        cli_configure.cmd_configure(_configure_namespace("srch", style="first", segments=2, seed=7))
        == 0
    )
    assert (
        cli_configure.cmd_configure(
            _configure_namespace(
                "dstl",
                from_run="srch",
                style="second",
                segments=2,
                backend="ltxv",
                low_definition=True,
            )
        )
        == 0
    )
    video = _manifest_video(tmp_path, "dstl")
    assert (video["width"], video["height"]) == (512, 320)


def test_configure_from_inherits_medium_geometry(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """--from a medium run without --backend keeps 1024x576 untouched."""
    monkeypatch.chdir(tmp_path)
    assert (
        cli_configure.cmd_configure(
            _configure_namespace("srcm", style="first", segments=2, seed=7, medium_definition=True)
        )
        == 0
    )
    assert (
        cli_configure.cmd_configure(
            _configure_namespace("dstm", from_run="srcm", style="second", segments=2)
        )
        == 0
    )
    video = _manifest_video(tmp_path, "dstm")
    assert (video["width"], video["height"]) == (1024, 576)
    assert video["profile"] == "ltx25-576p"


def test_configure_update_without_flags_keeps_medium_geometry(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Updating style only on a medium run never touches the stored geometry."""
    monkeypatch.chdir(tmp_path)
    assert (
        cli_configure.cmd_configure(
            _configure_namespace("keepm", style="first", segments=2, seed=7, medium_definition=True)
        )
        == 0
    )
    assert cli_configure.cmd_configure(_configure_namespace("keepm", style="second")) == 0
    video = _manifest_video(tmp_path, "keepm")
    assert (video["width"], video["height"]) == (1024, 576)
    assert video["profile"] == "ltx25-576p"


def test_manifest_filename_still_manifest_json() -> None:
    assert paths.MANIFEST_FILENAME == "manifest.json"
