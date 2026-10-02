"""Issue 079 TDD (failing-first): longlive2 full delete with migration hint.

Silent remap to ltxv is FORBIDDEN (it would corrupt timelines: 1280x704/29f
-> 768x512/96f invalidates committed segment durations, and longlive `.pt`
tapes can never resume on the ltxv JSON-tape path). Stored
`backend = "longlive2"` runs must fail fast with a hint naming ltxv + tapes.
"""

from __future__ import annotations

from typing import get_args

import pytest


def test_no_longlive2_in_registry_or_literal() -> None:
    from voyage import config

    assert "longlive2" not in config.BACKEND_REGISTRY
    assert "longlive2" not in get_args(config.VideoBackendName)


def test_removed_preset_rejected_with_migration_hint() -> None:
    from voyage import config

    with pytest.raises(ValueError, match="longlive2"):
        config._video_preset("longlive2")
    try:
        config._video_preset("longlive2")
    except ValueError as exc:
        message = str(exc).lower()
        assert "ltxv" in message
        assert "tape" in message
    else:  # pragma: no cover
        raise AssertionError("expected ValueError")


def test_no_silent_remap_to_ltxv() -> None:
    from voyage import config
    from voyage.config import ProjectConfig

    base = ProjectConfig(style="test style")
    with pytest.raises(ValueError, match="longlive2"):
        config.with_video_backend(base, "longlive2")  # type: ignore[arg-type]


def test_video_worker_module_longlive2_hint() -> None:
    from voyage.errors import ConfigurationError
    from voyage.supervisor import video_worker_module

    with pytest.raises(ConfigurationError) as excinfo:
        video_worker_module("longlive2")
    message = str(excinfo.value).lower()
    assert "longlive2" in message
    assert "ltxv" in message
    assert "tape" in message


def test_stored_longlive2_toml_fails_with_migration_hint(tmp_path) -> None:  # type: ignore[no-untyped-def]
    from pathlib import Path

    from voyage import config
    from voyage.errors import ConfigurationError

    toml_path = Path(str(tmp_path)) / "voyage.toml"
    toml_path.write_text(
        config.default_config_toml("voyage", "test style", 0).replace(
            'backend = "ltx25"', 'backend = "longlive2"', 1
        ),
        encoding="utf-8",
    )
    with pytest.raises(ConfigurationError) as excinfo:
        config.load_config(toml_path)
    message = str(excinfo.value).lower()
    assert "longlive2" in message
    assert "ltxv" in message
    assert "tape" in message
