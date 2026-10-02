"""Issue 141: the run manifest records the effective presentation contract.

`manifest.timeline` stays the source hint, but the shipped geometry (lifted
to the `[augment]` floors by default) is now recorded too: `presentation`
carries the run's floors at init, and `final_geometry` is filled by the
first finalize with the validated output box.
"""

from __future__ import annotations

import json
import tempfile
from pathlib import Path


def _read_manifest(run_dir: Path) -> dict[str, object]:
    return json.loads((run_dir / "run_manifest.json").read_text(encoding="utf-8"))  # type: ignore[no-any-return]


def test_manifest_records_presentation_floors(tmp_path: Path) -> None:
    """Init stamps the augment floors and a null final-geometry slot."""
    from tests.conftest import initialize_run_directory

    run_dir = tmp_path / "run"
    initialize_run_directory(run_dir)
    manifest = _read_manifest(run_dir)
    presentation = manifest["presentation"]
    assert isinstance(presentation, dict)
    assert presentation["min_fps"] == 24
    assert presentation["min_width"] == 1216
    assert presentation["min_height"] == 704
    assert manifest["final_geometry"] is None


def test_manifest_floors_match_media_defaults() -> None:
    """No independent literals: the manifest floors track the finalizer's."""
    from tests.conftest import DEFAULT_RUN_ID, DEFAULT_RUN_SEED, DEFAULT_STYLE
    from voyage.config import default_config_toml, load_config
    from voyage.media import (
        AUGMENT_DEFAULT_MIN_FPS,
        AUGMENT_DEFAULT_MIN_HEIGHT,
        AUGMENT_DEFAULT_MIN_WIDTH,
    )
    from voyage.persistence import build_manifest

    with tempfile.TemporaryDirectory() as tmp:
        config_path = Path(tmp) / "voyage.toml"
        config_path.write_text(
            default_config_toml(DEFAULT_RUN_ID, DEFAULT_STYLE, DEFAULT_RUN_SEED),
            encoding="utf-8",
        )
        config, digest = load_config(config_path)
    manifest = build_manifest(config, digest, {}, {})
    presentation = manifest["presentation"]
    assert isinstance(presentation, dict)
    assert presentation["min_fps"] == AUGMENT_DEFAULT_MIN_FPS == config.augment.min_fps
    assert presentation["min_width"] == AUGMENT_DEFAULT_MIN_WIDTH == config.augment.min_width
    assert presentation["min_height"] == AUGMENT_DEFAULT_MIN_HEIGHT == config.augment.min_height


def test_finalize_records_final_geometry(tmp_path: Path) -> None:
    """First finalize writes the validated output box back to the manifest."""
    from tests.conftest import initialize_run_directory
    from voyage import paths
    from voyage.config import load_config
    from voyage.media import finalize_run
    from voyage.supervisor import Supervisor

    run_dir = tmp_path / "run"
    initialize_run_directory(run_dir, run_id="geom", seed=7)
    config, _ = load_config(run_dir / paths.CONFIG_FILENAME)
    supervisor = Supervisor(run_dir, config)
    supervisor.start_workers()
    try:
        supervisor.commit_one_segment()
        supervisor.commit_one_segment()
    finally:
        supervisor.stop_workers()
    out = finalize_run(
        run_dir,
        run_dir / "final.mp4",
        min_fps=0,
        min_width=0,
        min_height=0,
    )
    assert out.exists()
    manifest = _read_manifest(run_dir)
    geometry = manifest["final_geometry"]
    assert isinstance(geometry, dict)
    assert geometry["width"] == 768
    assert geometry["height"] == 432
    assert geometry["fps"] == 24
    presentation = manifest["presentation"]
    assert isinstance(presentation, dict)
    assert presentation["min_fps"] == 0
    assert presentation["min_width"] == 0
    assert presentation["min_height"] == 0
