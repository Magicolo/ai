"""Frame, video, and memory helpers shared by the engine stages.

This module owns the single-buffer frame path (one PNG open per source,
one RGB array reused by interpolation, the silent encode, and the effects
conditioning), the chunked silent-video encode, artifact verification, and
the system/video memory probes. The ffmpeg binary path is resolved once
and cached instead of re-importing ``imageio_ffmpeg`` on every window.
"""

from __future__ import annotations

import contextlib
import subprocess
from pathlib import Path
from typing import TYPE_CHECKING, Any, cast

from zoomy import final_assembly
from zoomy.engine_protocol import VIDEO_CONSTANT_RATE_FACTOR, VIDEO_FRAMES_PER_SECOND
from zoomy.errors import AssemblyError, EngineConfigurationError

if TYPE_CHECKING:
    from collections.abc import Callable

    from PIL.Image import Image

# Frames written per stdin write: one syscall per chunk instead of one per
# frame keeps the GIL-bound encode loop off the hot path.
_ENCODE_FRAMES_PER_WRITE = 16
# Grace period for a failed ffmpeg child to exit after terminate, before kill.
_ENCODE_TERMINATE_TIMEOUT_SECONDS = 5.0
# A /proc/meminfo line always holds name, value, and unit ("kB").
_MEMINFO_PART_COUNT = 3

_cached_ffmpeg_binary: str | None = None


def ffmpeg_binary_cached() -> str:
    """Return the ffmpeg binary path, resolving it only once."""
    global _cached_ffmpeg_binary  # noqa: PLW0603
    if _cached_ffmpeg_binary is None:
        _cached_ffmpeg_binary = final_assembly.ffmpeg_binary()
    return _cached_ffmpeg_binary


def converted_frame(source_path: Path) -> Image:
    """Open one PNG, copy its pixels to RGB, and close the file handle.

    ``Image.open`` is lazy — the handle stays open until the image is
    closed — so converting without closing leaks one descriptor per source
    frame on interpreters without refcounting (and depends on decoder
    internals everywhere else). ``convert`` copies the pixels, making it
    safe to close the original before returning.
    """
    from PIL import Image as PillowImage  # noqa: PLC0415

    with PillowImage.open(source_path) as source_image:
        return cast("Image", source_image.convert("RGB"))


def frame_arrays(source_paths: list[Path]) -> tuple[list[Image], list[Any]]:
    """Open each source once, returning PIL frames plus one numpy array each.

    The arrays are the single shared buffer: interpolation consumes them,
    the silent encode writes them, and the effects conditioning stacks
    them — no stage re-opens a PNG or re-converts a PIL frame it already
    holds.
    """
    import numpy as np  # noqa: PLC0415

    frames = [converted_frame(path) for path in source_paths]
    arrays = [np.asarray(frame) for frame in frames]
    return frames, arrays


def pad_frames_to_minimum(frames: list[Any], minimum_frames: int) -> list[Any]:
    """Tile a short frame batch up to ``minimum_frames`` (passthrough above).

    MMAudio's synchformer encodes video in 16-frame sync segments and crashes
    on an empty segment list, so short renders repeat the whole batch (the
    retired BatchPadToMin custom node did exactly this). Callers must pass a
    non-empty batch: there is nothing sensible to tile from zero frames, and
    extending an empty list would spin forever.
    """
    if not frames:
        message = "Cannot pad an empty frame batch to the sync minimum"
        raise EngineConfigurationError(message)
    padded = list(frames)
    while len(padded) < minimum_frames:
        padded.extend(frames)
    return padded


def verify_paths(window_index: int, artifacts: tuple[Path, ...]) -> None:
    """Reject missing or empty window outputs at the step that caused them."""
    for artifact in artifacts:
        if not artifact.is_file() or artifact.stat().st_size == 0:
            message = f"Segment {window_index} produced no output: {artifact}"
            raise AssemblyError(message)


def discard_partial_artifacts(artifacts: tuple[Path, ...]) -> None:
    """Best-effort delete of one window's partial outputs after a failure.

    A failed attempt must not orphan its silent video or landed stems: the
    retry renders the same window paths again, and anything left behind
    would survive until a fully successful finalize. Unlink failures are
    suppressed — cleanup runs on the failure path and must never mask the
    error that caused it.
    """
    for artifact in artifacts:
        with contextlib.suppress(OSError):
            artifact.unlink(missing_ok=True)


def reap_encode_process(process: subprocess.Popen[bytes]) -> None:
    """Best-effort cleanup of a failed ffmpeg child: close, terminate, wait.

    Escalates to kill when the child ignores terminate past the grace
    period, so a mid-stream encode failure never leaves a zombie plus an
    open pipe behind (the window retry would otherwise orphan one per
    attempt).
    """
    try:
        if process.stdin is not None:
            process.stdin.close()
    except (OSError, ValueError):
        pass
    process.terminate()
    try:
        process.wait(timeout=_ENCODE_TERMINATE_TIMEOUT_SECONDS)
    except subprocess.TimeoutExpired:
        process.kill()
        process.wait()


def write_silent_video(frames: list[Any], destination: Path) -> None:
    """Encode frames as h264 (crf 19, yuv420p, 32 fps) in chunked writes."""
    import numpy as np  # noqa: PLC0415

    ffmpeg = ffmpeg_binary_cached()
    height, width, _channels = np.asarray(frames[0]).shape
    command = (
        ffmpeg,
        "-hide_banner",
        "-loglevel",
        "error",
        "-y",
        "-f",
        "rawvideo",
        "-pix_fmt",
        "rgb24",
        "-s",
        f"{width}x{height}",
        "-r",
        str(VIDEO_FRAMES_PER_SECOND),
        "-i",
        "-",
        "-c:v",
        "libx264",
        "-pix_fmt",
        "yuv420p",
        "-crf",
        str(VIDEO_CONSTANT_RATE_FACTOR),
        str(destination),
    )
    try:
        # Tuple argv, no shell: every argument comes from our own builders.
        process = subprocess.Popen(command, stdin=subprocess.PIPE)  # noqa: S603
    except OSError as failure:
        message = f"Silent video encode failed for {destination}: {failure}"
        raise AssemblyError(message) from failure
    if process.stdin is None:
        reap_encode_process(process)
        message = f"Silent video encode failed for {destination}: no input pipe"
        raise AssemblyError(message)
    try:
        chunk: list[bytes] = []
        for frame in frames:
            chunk.append(np.asarray(frame).tobytes())
            if len(chunk) >= _ENCODE_FRAMES_PER_WRITE:
                process.stdin.write(b"".join(chunk))
                chunk = []
        if chunk:
            process.stdin.write(b"".join(chunk))
        process.stdin.close()
        process.wait()
    except (OSError, ValueError) as failure:
        reap_encode_process(process)
        message = f"Silent video encode failed for {destination}: {failure}"
        raise AssemblyError(message) from failure
    if process.returncode != 0:
        message = f"Silent video encode failed for {destination}"
        raise AssemblyError(message)
    if not destination.is_file() or destination.stat().st_size == 0:
        message = f"Silent video encode produced no output: {destination}"
        raise AssemblyError(message)


def slice_audio(
    source: Path,
    destination: Path,
    *,
    start_seconds: float,
    duration_seconds: float,
    run: Callable[[tuple[str, ...]], None] | None = None,
) -> None:
    """Slice one window stem out of a global music track with ffmpeg.

    Seeks to ``start_seconds`` and copies ``duration_seconds`` of audio as
    FLAC, so a single global ACE-Step render covers every window without
    per-window diffusion passes.
    """
    ffmpeg = ffmpeg_binary_cached()
    command = (
        ffmpeg,
        "-hide_banner",
        "-loglevel",
        "error",
        "-y",
        "-ss",
        f"{start_seconds:.3f}",
        "-i",
        str(source),
        "-t",
        f"{duration_seconds:.3f}",
        "-c:a",
        "flac",
        str(destination),
    )
    try:
        if run is not None:
            run(command)
        else:  # Tuple argv, no shell: every argument comes from our builders.
            subprocess.run(list(command), check=True)  # noqa: S603
    except (OSError, subprocess.CalledProcessError) as failure:
        message = f"Audio slice failed for {destination}: {failure}"
        raise AssemblyError(message) from failure
    if not destination.is_file() or destination.stat().st_size == 0:
        message = f"Audio slice produced no output: {destination}"
        raise AssemblyError(message)


def read_system_memory_bytes() -> tuple[int | None, int | None]:
    """Return free and total system RAM, or Nones when /proc is unreadable.

    A total /proc failure (non-Linux host, sandboxed container) reports
    unknown rather than a zero reading the stats line could mistake for
    real figures; per-line parse tolerance below is unchanged.
    """
    try:
        meminfo = Path("/proc/meminfo").read_text()
    except OSError:
        return (None, None)
    values: dict[str, int] = {}
    for line in meminfo.splitlines():
        parts = line.split()
        if len(parts) == _MEMINFO_PART_COUNT and parts[2] == "kB":
            try:
                values[parts[0].rstrip(":")] = int(parts[1]) * 1024
            except ValueError:
                continue
    return (values.get("MemAvailable"), values.get("MemTotal"))


def collect_free_video_memory() -> None:
    """Collect garbage and hand freed blocks back to the CUDA allocator."""
    import gc  # noqa: PLC0415

    import torch  # noqa: PLC0415

    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
