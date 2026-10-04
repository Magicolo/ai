"""Manifest carries no stale/legacy fields (flat-manifest hygiene).

Regression: the manifest once carried `director.model_id` (Qwen/Qwen3-8B,
a legacy bf16-CPU default ignored on every live path — the default decider
is the llama sidecar serving the Qwen3.5 GGUF, and the qwen+cuda path
substitutes the AWQ id anyway), `director.embedding_model_id` and
`director.inspector_model_id` (the supervisor never sends either in an
init/embed payload and never calls the inspect op — the worker falls back
to its hardcoded defaults), plus the in-memory-only caption pins
`video.video_caption` / `audio.music_caption` (documented never-persisted
but leaked via `model_dump`). All five are gone from the manifest; the
live director knobs stay.
"""

from __future__ import annotations

from pathlib import Path

from voyage.config import ProjectConfig, resolve_config
from voyage.paths import MANIFEST_FILENAME
from voyage.persistence import build_manifest, read_effective_config, write_manifest


def _pinned_config() -> ProjectConfig:
    return resolve_config(
        ProjectConfig(style="pastel neon line-art, peaceful"),
        music_caption="pinned music",
        video_caption="pinned video",
    )


def test_manifest_has_no_stale_director_model_fields() -> None:
    manifest = build_manifest(_pinned_config())
    director = manifest["director"]
    assert isinstance(director, dict)
    assert "model_id" not in director
    assert "embedding_model_id" not in director
    assert "inspector_model_id" not in director


def test_manifest_has_no_in_memory_caption_pins() -> None:
    manifest = build_manifest(_pinned_config())
    video = manifest["video"]
    audio = manifest["audio"]
    assert isinstance(video, dict)
    assert isinstance(audio, dict)
    assert "video_caption" not in video
    assert "music_caption" not in audio


def test_manifest_keeps_live_director_knobs() -> None:
    manifest = build_manifest(_pinned_config())
    director = manifest["director"]
    assert isinstance(director, dict)
    for key in (
        "backend",
        "device",
        "llama_endpoint",
        "temperature",
        "max_new_tokens",
        "enable_thinking",
    ):
        assert key in director, key


def test_manifest_round_trip_drops_pins(tmp_path: Path) -> None:
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    write_manifest(run_dir, build_manifest(_pinned_config()))
    assert (run_dir / MANIFEST_FILENAME).exists()
    revived = read_effective_config(run_dir)
    assert revived.audio.music_caption is None
    assert revived.video.video_caption is None
    assert revived.director.backend == "llama"
