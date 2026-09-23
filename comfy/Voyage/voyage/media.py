"""ffmpeg/ffprobe wrappers + validation + finalizer (DESIGN §§54-57, I).

All invocations use argument lists — never shell strings. The finalizer
never mutates source segment files; it publishes the final path
atomically.
"""

from __future__ import annotations

import json
import subprocess
import tempfile
from pathlib import Path
from typing import Any

from voyage import paths
from voyage.atomic import atomic_write_bytes
from voyage.errors import MediaError


def run_capture(argv: list[str]) -> subprocess.CompletedProcess[str]:
    return subprocess.run(argv, capture_output=True, text=True, check=False)


def probe(path: Path) -> dict[str, Any]:
    proc = run_capture(
        [
            "ffprobe",
            "-hide_banner",
            "-v",
            "error",
            "-show_format",
            "-show_streams",
            "-of",
            "json",
            str(path),
        ]
    )
    if proc.returncode != 0:
        raise MediaError(f"ffprobe failed for {path}: {proc.stderr[-2000:]}")
    data: Any = json.loads(proc.stdout or "{}")
    if not isinstance(data, dict):
        raise MediaError(f"ffprobe returned non-object for {path}")
    return data


def validate_video(
    path: Path, width: int, height: int, fps: int, min_frames: int = 1
) -> dict[str, Any]:
    info = probe(path)
    streams = [s for s in info.get("streams", []) if isinstance(s, dict)]
    video = next((s for s in streams if s.get("codec_type") == "video"), None)
    if video is None:
        raise MediaError(f"no video stream in {path}")
    if int(video.get("width", -1)) != width or int(video.get("height", -1)) != height:
        raise MediaError(
            f"resolution mismatch in {path}: "
            f"{video.get('width')}x{video.get('height')} != {width}x{height}"
        )
    rate = str(video.get("avg_frame_rate", "0/1"))
    num, _, den = rate.partition("/")
    actual_fps = float(num) / float(den or 1) if num else 0.0
    if abs(actual_fps - fps) > 0.5:
        raise MediaError(f"fps mismatch in {path}: {actual_fps} != {fps}")
    frames = int(video.get("nb_frames", 0) or 0)
    if frames < min_frames and frames != 0:
        raise MediaError(f"too few frames in {path}: {frames}")
    duration = float(info.get("format", {}).get("duration", 0.0) or 0.0)
    if duration <= 0:
        raise MediaError(f"non-positive duration in {path}")
    return {"frames": frames, "duration": duration, "fps": actual_fps}


def validate_audio(path: Path, sample_rate: int, channels: int) -> dict[str, Any]:
    info = probe(path)
    streams = [s for s in info.get("streams", []) if isinstance(s, dict)]
    audio = next((s for s in streams if s.get("codec_type") == "audio"), None)
    if audio is None:
        raise MediaError(f"no audio stream in {path}")
    if int(audio.get("sample_rate", -1)) != sample_rate:
        raise MediaError(f"sample-rate mismatch in {path}")
    if int(audio.get("channels", -1)) != channels:
        raise MediaError(f"channel mismatch in {path}")
    duration = float(info.get("format", {}).get("duration", 0.0) or 0.0)
    if duration <= 0:
        raise MediaError(f"non-positive duration in {path}")
    return {"duration": duration}


def slice_take(
    take_path: Path,
    start_seconds: float,
    duration_seconds: float,
    dest: Path,
    sample_rate: int,
    channels: int,
) -> Path:
    """Cut one segment-sized slice out of a music take (§35).

    Output is canonical segment audio (WAV s16le at the run's sample
    rate/channels) so slices from different takes always share the format
    the assembly crossfade requires.
    """
    dest.parent.mkdir(parents=True, exist_ok=True)
    proc = run_capture(
        [
            "ffmpeg",
            "-hide_banner",
            "-nostdin",
            "-y",
            "-ss",
            f"{max(start_seconds, 0.0):.6f}",
            "-t",
            f"{max(duration_seconds, 0.1):.6f}",
            "-i",
            str(take_path),
            "-ar",
            str(sample_rate),
            "-ac",
            str(channels),
            "-c:a",
            "pcm_s16le",
            str(dest),
        ]
    )
    if proc.returncode != 0:
        raise MediaError(f"take slice failed for {take_path}: {proc.stderr[-2000:]}")
    return dest


def assemble_segment_audio(
    slices: list[Path],
    dest: Path,
    crossfade_seconds: float,
) -> Path:
    """Join take slices into one segment audio.wav (§35).

    Consecutive slices (a take boundary falls inside the segment) are
    joined with an acrossfade; a single slice is copied through. The fade
    length is clamped to half the shortest slice so short segments can
    never collapse the filter (zoomy lesson: manual fades, never
    acrossfade on short tails — here takes are 30-60s but slices may be
    ~2s, hence the clamp).
    """
    if not slices:
        raise MediaError("no slices to assemble")
    dest.parent.mkdir(parents=True, exist_ok=True)
    if len(slices) == 1:
        proc = run_capture(
            [
                "ffmpeg",
                "-hide_banner",
                "-nostdin",
                "-y",
                "-i",
                str(slices[0]),
                "-c:a",
                "pcm_s16le",
                str(dest),
            ]
        )
        if proc.returncode != 0:
            raise MediaError(f"slice copy failed: {proc.stderr[-2000:]}")
        return dest
    durations = [float(probe(s).get("format", {}).get("duration", 0.0) or 0.0) for s in slices]
    if any(d <= 0 for d in durations):
        raise MediaError("slice with non-positive duration")
    fade = min(crossfade_seconds, min(durations) / 2.0)
    argv: list[str] = ["ffmpeg", "-hide_banner", "-nostdin", "-y"]
    for s in slices:
        argv += ["-i", str(s)]
    if fade < 0.1:
        filter_graph = "".join(f"[{i}:a]" for i in range(len(slices)))
        filter_graph += f"concat=n={len(slices)}:v=0:a=1[aout]"
    else:
        filter_graph = ""
        current = "[0:a]"
        for i in range(1, len(slices)):
            out = f"[a{i:02d}]"
            filter_graph += f"{current}[{i}:a]acrossfade=d={fade:.3f}:c1=tri:c2=tri{out};"
            current = out
        filter_graph += f"{current}anull[aout]"
    argv += ["-filter_complex", filter_graph, "-map", "[aout]", "-c:a", "pcm_s16le", str(dest)]
    proc = run_capture(argv)
    if proc.returncode != 0:
        raise MediaError(f"segment audio assembly failed: {proc.stderr[-2000:]}")
    return dest


def finalize_run(
    run_dir: Path,
    output_path: Path,
    width: int = 768,
    height: int = 432,
    fps: int = 24,
) -> Path:
    """Concat committed segments → single normalized MP4 (DESIGN §56).

    Exactly one final encode: scale/pad to 768×432, mux audio, validate,
    atomically publish.
    """
    segments_root = run_dir / paths.SEGMENTS_DIRNAME
    segment_dirs = (
        sorted(p for p in segments_root.iterdir() if p.is_dir()) if segments_root.exists() else []
    )
    committed = [d for d in segment_dirs if (d / paths.DONE_MARKER).exists()]
    if not committed:
        raise MediaError(f"no committed segments in {run_dir}")
    for segment in committed:
        if not (segment / "video.mp4").exists():
            raise MediaError(f"segment {segment.name} missing video.mp4")
        if not (segment / "audio.wav").exists():
            raise MediaError(f"segment {segment.name} missing audio.wav")

    with tempfile.TemporaryDirectory() as tmp:
        tmpdir = Path(tmp)
        # Per-segment A/V mux, then concat the muxed parts.
        parts: list[Path] = []
        for segment in committed:
            part = tmpdir / f"{segment.name}.mp4"
            proc = run_capture(
                [
                    "ffmpeg",
                    "-hide_banner",
                    "-nostdin",
                    "-y",
                    "-i",
                    str(segment / "video.mp4"),
                    "-i",
                    str(segment / "audio.wav"),
                    "-c:v",
                    "libx264",
                    "-pix_fmt",
                    "yuv420p",
                    "-preset",
                    "veryfast",
                    "-c:a",
                    "aac",
                    "-shortest",
                    str(part),
                ]
            )
            if proc.returncode != 0:
                raise MediaError(f"segment mux failed for {segment.name}: {proc.stderr[-2000:]}")
            parts.append(part)
        concat_list = tmpdir / "concat.txt"
        concat_list.write_text("".join(f"file '{part}'\n" for part in parts), encoding="utf-8")
        staged = tmpdir / "final.mp4"
        vf = (
            f"scale={width}:{height}:force_original_aspect_ratio=decrease,"
            f"pad={width}:{height}:(ow-iw)/2:(oh-ih)/2,setsar=1,fps={fps}"
        )
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
                "-vf",
                vf,
                "-c:v",
                "libx264",
                "-pix_fmt",
                "yuv420p",
                "-preset",
                "veryfast",
                "-c:a",
                "aac",
                "-b:a",
                "256k",
                str(staged),
            ]
        )
        if proc.returncode != 0:
            raise MediaError(f"final encode failed: {proc.stderr[-2000:]}")
        validate_video(staged, width, height, fps)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        atomic_write_bytes(output_path, staged.read_bytes())
    return output_path
