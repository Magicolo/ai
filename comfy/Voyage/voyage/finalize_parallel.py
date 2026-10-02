"""Parallel finalize: model-pass video + SFX bed side by side (DESIGN §140).

GPU-defaults orchestration: on a 2-GPU box the Real-ESRGAN + FILM model
pass owns cuda:1 (the 2060, via `model_pass_devices`) while the MMAudio
SFX stack owns cuda:0 (the 4060) — the two stages share nothing, so
`run_parallel_finalize` runs them in one ThreadPoolExecutor(2) instead
of the legacy sequential finalize-then-SFX. Output is byte-identical to
the sequential path (same `finalize_run` call for the music-only video,
same demux/mix/remux argv for the dub), pinned by
`test_parallel_finalize_matches_sequential_bytes`.

Shape: Thread A runs the whole `finalize_run` to a tmp music-only final
(which also records the standard `finalize_completed` event — soak
trending sees one event per finalize, unchanged schema); Thread B
builds a cheap stream-copy reference concat of the same usable segments
and renders the SFX bed conditioned on it (same timeline math: segment
seconds are fps-lift invariant, so the reference and the shipped final
agree). Post: demux music from Thread A's final, mix under the bed,
remux video-copy + mixed AAC, atomically publish. A failed SFX stage
still publishes the music-only final (the legacy guarantee) and raises
`SfxBedError` so the CLI keeps the "(music-only final kept ...)"
message; a failed video stage raises as-is with nothing published.
"""

from __future__ import annotations

import tempfile
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from voyage.atomic import atomic_copy
from voyage.augment import augment_devices, model_pass_active, resolve_augment_weights
from voyage.errors import MediaError
from voyage.media import committed_usable_segments, finalize_run, write_concat_list
from voyage.media_audio import probe, run_capture
from voyage.sfx_finalize import (
    demux_music_audio,
    mix_music_and_sfx,
    remux_video_with_audio,
    render_sfx_bed,
    segment_sfx_bounds,
)


class SfxBedError(MediaError):
    """The SFX stage of a parallel finalize failed (music-only published)."""


def should_run_parallel(
    *,
    sfx_backend: str,
    use_model_pass: bool,
    models_dir: str | Path | None,
    num_workers: int,
    devices: tuple[str, ...] | None = None,
    weights_present: bool | None = None,
) -> bool:
    """True when the parallel finalize pays: SFX will run, the knob is on,
    the augment legs are provisioned, one SFX worker leaves cuda:1 free,
    and two GPUs are visible (a dual-worker SFX run already spans both
    GPUs, and a single GPU serializes the stages anyway).

    `devices`/`weights_present` are explicit-visibility seams so tests pin
    the branches without GPUs or weight files; `None` probes live state
    (`augment_devices` / `resolve_augment_weights` + `model_pass_active`).
    """
    if sfx_backend != "mmaudio" or not use_model_pass:
        return False
    if num_workers != 1 or models_dir is None:
        return False
    visible = augment_devices() if devices is None else devices
    if len(visible) < 2:
        return False
    if weights_present is not None:
        return weights_present
    return model_pass_active(True, resolve_augment_weights(models_dir))


def _reference_concat(usable: list[Path], tmpdir: Path) -> tuple[Path, float]:
    """Stream-copy concat of the usable segment videos (SFX conditioning).

    Cheap CPU-only reference carrying the exact shipped timeline; the SFX
    bed needs pixels + seconds, never the enhanced grade.
    """
    concat_list = write_concat_list(
        [segment / "video.mp4" for segment in usable], tmpdir / "parallel_ref.txt"
    )
    reference = tmpdir / "parallel_reference.mp4"
    proc = run_capture(
        [
            "ffmpeg",
            "-hide_banner",
            "-nostdin",
            "-y",
            "-f",
            "concat",
            "-safe",
            "0",
            "-i",
            str(concat_list),
            "-c",
            "copy",
            str(reference),
        ]
    )
    if proc.returncode != 0:
        raise MediaError(f"parallel finalize reference concat failed: {proc.stderr[-2000:]}")
    if not reference.exists() or reference.stat().st_size == 0:
        raise MediaError(f"parallel finalize reference concat produced empty {reference}")
    info = probe(reference)
    timeline = float(info.get("format", {}).get("duration", 0.0) or 0.0)
    if timeline <= 0.0:
        raise MediaError(f"parallel finalize: unprobable reference duration for {reference}")
    return (reference, timeline)


def run_parallel_finalize(
    run_dir: Path,
    output_path: Path,
    width: int = 768,
    height: int = 432,
    fps: int = 24,
    skip_bad: bool | None = None,
    min_free_space_gib: float = 0.0,
    sample_rate: int | None = None,
    channels: int | None = None,
    overlap_fraction: float | None = None,
    overlap_cap_seconds: float | None = None,
    min_fps: int | None = None,
    min_width: int | None = None,
    min_height: int | None = None,
    crf: int | None = None,
    preset: str | None = None,
    use_model_pass: bool | None = None,
    models_dir: Path | str | None = None,
    seed: int = 0,
    sfx_backend: str = "mmaudio",
    sfx_device: str = "cuda:0",
    sfx_model_size: str = "large_44k_v2",
    sfx_caption: str | None = None,
) -> Path:
    """Finalize with the model pass and the SFX dub running side by side.

    Same contract as `finalize_run` + `finalize_sfx_pass` composed: the
    scalar params mirror `finalize_run` (None = resolve to the stored
    defaults inside), the `sfx_*` params mirror `finalize_sfx_pass`.
    Returns the published output path.
    """
    if models_dir is None:
        raise MediaError("parallel finalize needs a models_dir (SFX + model pass)")
    skip = False if skip_bad is None else skip_bad
    sample_rate_value = 48000 if sample_rate is None else sample_rate
    channels_value = 2 if channels is None else channels
    usable = committed_usable_segments(run_dir, skip)
    with tempfile.TemporaryDirectory(prefix="voyage-parallel-", dir=run_dir) as tmp:
        tmpdir = Path(tmp)
        reference, timeline = _reference_concat(usable, tmpdir)
        bounds = segment_sfx_bounds(run_dir, usable, fps, sfx_caption)
        music_final = tmpdir / "parallel_music.mp4"
        bed_dir = tmpdir / "sfx_bed"
        bed_dir.mkdir(parents=True, exist_ok=True)

        def _run_video() -> Path:
            return finalize_run(
                run_dir,
                music_final,
                width=width,
                height=height,
                fps=fps,
                skip_bad=skip_bad,
                min_free_space_gib=min_free_space_gib,
                sample_rate=sample_rate,
                channels=channels,
                overlap_fraction=overlap_fraction,
                overlap_cap_seconds=overlap_cap_seconds,
                min_fps=min_fps,
                min_width=min_width,
                min_height=min_height,
                crf=crf,
                preset=preset,
                use_model_pass=use_model_pass,
                models_dir=models_dir,
            )

        def _run_bed() -> Path:
            try:
                return render_sfx_bed(
                    run_dir,
                    reference,
                    timeline,
                    bounds,
                    bed_dir,
                    sfx_backend,
                    str(models_dir),
                    sfx_device,
                    sfx_model_size,
                    seed,
                    sample_rate_value,
                    channels_value,
                    1,
                )
            except MediaError as exc:
                raise SfxBedError(str(exc)) from exc

        video_error: Exception | None = None
        bed_error: Exception | None = None
        music_done: Path | None = None
        bed: Path | None = None
        with ThreadPoolExecutor(max_workers=2) as pool:
            video_future = pool.submit(_run_video)
            bed_future = pool.submit(_run_bed)
            try:
                music_done = video_future.result()
            except Exception as exc:  # noqa: BLE001 — collected, re-raised below
                video_error = exc
            try:
                bed = bed_future.result()
            except Exception as exc:  # noqa: BLE001 — collected, re-raised below
                bed_error = exc
        if video_error is not None or music_done is None:
            raise (
                video_error
                if video_error is not None
                else MediaError("parallel finalize: video stage produced no output")
            )
        if bed_error is not None or bed is None:
            # Legacy guarantee: the music-only final still ships.
            output_path.parent.mkdir(parents=True, exist_ok=True)
            atomic_copy(music_done, output_path)
            raise (
                bed_error
                if bed_error is not None
                else MediaError("parallel finalize: sfx stage produced no bed")
            )
        music = tmpdir / "parallel_music.wav"
        demux_music_audio(music_done, music)
        mixed = tmpdir / "parallel_mixed.wav"
        mix_music_and_sfx(music, bed, mixed, sample_rate_value, channels_value)
        remuxed = tmpdir / "parallel_final.mp4"
        remux_video_with_audio(music_done, mixed, remuxed)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        atomic_copy(remuxed, output_path)
    return output_path
