"""ffmpeg/ffprobe wrappers + validation + finalizer (DESIGN §§54-57, I).

All invocations use argument lists — never shell strings. The finalizer
never mutates source segment files; it publishes the final path
atomically.
"""

from __future__ import annotations

import hashlib
import json
import shutil
import subprocess
import tempfile
from pathlib import Path
from typing import Any

from voyage import paths
from voyage.atomic import atomic_write_bytes
from voyage.errors import DiskSpaceError, MediaError

#: Max |video duration − audio duration| per segment, seconds (DESIGN §56
#: step 6). Same budget the commit path enforces, so anything committed
#: stays finalizable.
AV_ALIGNMENT_TOLERANCE_SECONDS = 0.6


def check_free_space(run_dir: Path, min_free_gib: float) -> float:
    """Free GiB under `run_dir`; raise DiskSpaceError below the reserve.

    Shared by the commit precheck and the finalize preflight (DESIGN §53):
    a final metadata write must never be the thing that discovers a full
    disk. Moved here from supervisor (import direction is supervisor →
    media, never the reverse).
    """
    free_gib = shutil.disk_usage(run_dir).free / (1024**3)
    if free_gib < min_free_gib:
        raise DiskSpaceError(f"free space {free_gib:.1f} GiB below reserve {min_free_gib:.1f} GiB")
    return free_gib


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


def _sha256_file(path: Path) -> str:
    """Chunked SHA-256 (constant memory — takes can be multi-GB)."""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(65536), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _verify_segment(segment: Path) -> tuple[int, float, float]:
    """DESIGN §56 steps 4-6: checksums, frame ranges, A/V alignment.

    Returns (frames, video_duration, audio_duration). Raises MediaError
    on any mismatch — fail loud, never finalize corrupt media silently.
    """
    name = segment.name
    checksums_path = segment / "sha256.json"
    if not checksums_path.exists():
        raise MediaError(f"segment {name} missing sha256.json")
    try:
        expected = json.loads(checksums_path.read_text(encoding="utf-8"))
    except ValueError as exc:
        raise MediaError(f"segment {name} has unreadable sha256.json: {exc}") from exc
    if not isinstance(expected, dict):
        raise MediaError(f"segment {name} has malformed sha256.json")
    for artifact in ("video.mp4", "audio.wav"):
        recorded = expected.get(artifact)
        if not isinstance(recorded, str) or not recorded:
            raise MediaError(f"segment {name} sha256.json missing {artifact}")
        actual = _sha256_file(segment / artifact)
        if actual != recorded:
            raise MediaError(f"segment {name} checksum mismatch for {artifact}")
    metrics_path = segment / "metrics.json"
    try:
        metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
        frames = int(metrics.get("frames", 0))
    except (ValueError, KeyError, AttributeError) as exc:
        raise MediaError(f"segment {name} has unreadable metrics.json: {exc}") from exc
    if frames <= 0:
        raise MediaError(f"segment {name} has non-positive frame count {frames}")
    video_duration = float(probe(segment / "video.mp4").get("format", {}).get("duration", 0.0))
    audio_duration = float(probe(segment / "audio.wav").get("format", {}).get("duration", 0.0))
    if video_duration <= 0 or audio_duration <= 0:
        raise MediaError(f"segment {name} has non-positive media duration")
    drift = abs(video_duration - audio_duration)
    if drift > AV_ALIGNMENT_TOLERANCE_SECONDS:
        raise MediaError(
            f"segment {name} A/V alignment drift {drift:.3f}s "
            f"exceeds {AV_ALIGNMENT_TOLERANCE_SECONDS:.1f}s"
        )
    return frames, video_duration, audio_duration


def _segment_timeline(usable: list[Path], fps: int) -> tuple[list[float], list[float], float]:
    """Per-segment [start, end) video-times from committed metrics + total.

    The timeline follows frame counts (the supervisor's truthful
    accounting), not container durations, so the blended audio matches
    the concatenated video sample-exactly.
    """
    starts: list[float] = []
    ends: list[float] = []
    cursor = 0.0
    for segment in usable:
        try:
            metrics = json.loads((segment / "metrics.json").read_text(encoding="utf-8"))
            frames = int(metrics.get("frames", 0))
        except (ValueError, KeyError, AttributeError) as exc:
            raise MediaError(f"segment {segment.name} has unreadable metrics.json: {exc}") from exc
        if frames <= 0:
            raise MediaError(f"segment {segment.name} has non-positive frame count {frames}")
        starts.append(cursor)
        cursor += frames / fps
        ends.append(cursor)
    return starts, ends, cursor


def _concat_fallback_audio(inputs: list[Path], dest: Path) -> Path:
    """Join segment audio.wav files with a plain concat (no blend)."""
    dest.parent.mkdir(parents=True, exist_ok=True)
    argv: list[str] = ["ffmpeg", "-hide_banner", "-nostdin", "-y"]
    for source in inputs:
        argv += ["-i", str(source)]
    filter_graph = "".join(f"[{i}:a]" for i in range(len(inputs)))
    filter_graph += f"concat=n={len(inputs)}:v=0:a=1[aout]"
    argv += ["-filter_complex", filter_graph, "-map", "[aout]", "-c:a", "pcm_s16le", str(dest)]
    proc = run_capture(argv)
    if proc.returncode != 0:
        raise MediaError(f"final audio concat failed: {proc.stderr[-2000:]}")
    return dest


def build_final_audio(
    run_dir: Path,
    usable: list[Path],
    tmpdir: Path,
    fps: int,
    sample_rate: int,
    channels: int,
    overlap_fraction: float = 0.10,
    overlap_cap_seconds: float = 0.5,
) -> Path:
    """Blend committed segments into one timeline-exact final mix (§56).

    Each segment boundary gets an overlap crossfade: segment windows are
    extended by half the overlap on each side and re-sliced from the
    takes ledger (takes are continuous, so the extension is real musical
    content — not time-stretched), then chained with acrossfade. Total
    length stays exactly the video timeline, so no A/V drift.

    Falls back to a plain concat of the per-segment audio.wav previews
    when the takes ledger is unavailable (pre-take runs) — the old
    hard-splice behavior, clicks included. Per-segment previews are
    never rewritten: the blend exists only in the returned final mix.
    """
    from voyage.audio.planner import AudioPlanner, load_takes

    dest = tmpdir / "final_audio.wav"
    if len(usable) == 1:
        proc = run_capture(
            [
                "ffmpeg",
                "-hide_banner",
                "-nostdin",
                "-y",
                "-i",
                str(usable[0] / "audio.wav"),
                "-c:a",
                "pcm_s16le",
                str(dest),
            ]
        )
        if proc.returncode != 0:
            raise MediaError(f"final audio copy failed: {proc.stderr[-2000:]}")
        return dest
    starts, ends, timeline = _segment_timeline(usable, fps)
    durations = [end - start for start, end in zip(starts, ends, strict=True)]
    overlap = min(overlap_fraction * min(durations), overlap_cap_seconds)
    ledger = run_dir / "audio" / "takes.jsonl"
    takes: list[Any] = load_takes(ledger) if ledger.exists() else []
    planner = AudioPlanner(takes=takes)
    if not takes or overlap < 0.05:
        return _concat_fallback_audio([s / "audio.wav" for s in usable], dest)
    half = overlap / 2.0
    windows: list[Path] = []
    for index, segment in enumerate(usable):
        window_start = max(starts[index] - (half if index > 0 else 0.0), 0.0)
        window_end = min(ends[index] + (half if index < len(usable) - 1 else 0.0), timeline)
        # Slice the takes covering this (possibly extended) window; a take
        # joint inside the window yields two slices joined as usual.
        slices: list[Path] = []
        cursor = window_start
        piece = 0
        while cursor < window_end - 1e-6:
            serving = planner.take_for_time(cursor)
            if serving is None or not serving.path or not Path(serving.path).exists():
                return _concat_fallback_audio([s / "audio.wav" for s in usable], dest)
            piece_end = min(serving.covers_until(), window_end)
            if piece_end <= cursor:
                return _concat_fallback_audio([s / "audio.wav" for s in usable], dest)
            slice_path = tmpdir / f"{segment.name}_w{piece:02d}.wav"
            slice_take(
                Path(serving.path),
                cursor - serving.covers_from,
                piece_end - cursor,
                slice_path,
                sample_rate,
                channels,
            )
            slices.append(slice_path)
            cursor = piece_end
            piece += 1
        window_path = tmpdir / f"{segment.name}_window.wav"
        if len(slices) == 1:
            slices[0].replace(window_path)
        else:
            assemble_segment_audio(slices, window_path, overlap)
        windows.append(window_path)
    argv: list[str] = ["ffmpeg", "-hide_banner", "-nostdin", "-y"]
    for window in windows:
        argv += ["-i", str(window)]
    filter_graph = ""
    current = "[0:a]"
    for index in range(1, len(windows)):
        out = f"[a{index:02d}]"
        filter_graph += f"{current}[{index}:a]acrossfade=d={overlap:.3f}:c1=tri:c2=tri{out};"
        current = out
    filter_graph += f"{current}anull[aout]"
    argv += ["-filter_complex", filter_graph, "-map", "[aout]", "-c:a", "pcm_s16le", str(dest)]
    proc = run_capture(argv)
    if proc.returncode != 0:
        raise MediaError(f"final audio blend failed: {proc.stderr[-2000:]}")
    return dest


def finalize_run(
    run_dir: Path,
    output_path: Path,
    width: int = 768,
    height: int = 432,
    fps: int = 24,
    skip_bad: bool = False,
    min_free_space_gib: float = 0.0,
    sample_rate: int = 48000,
    channels: int = 2,
    overlap_fraction: float = 0.10,
    overlap_cap_seconds: float = 0.5,
) -> Path:
    """Concat committed segments → single normalized MP4 (DESIGN §56).

    Exactly one final encode: scale/pad to WxH, mux audio, validate,
    atomically publish. With skip_bad, corrupt segments are skipped with
    a warning instead of aborting the whole finalize. A positive
    `min_free_space_gib` runs the §53 preflight first so a full disk
    fails fast instead of mid-encode.

    Audio joints get a proportional overlap crossfade (re-sliced from
    the takes ledger — previews untouched); pass overlap_fraction=0 to
    keep the legacy hard splice.
    """
    if min_free_space_gib > 0:
        check_free_space(run_dir, min_free_space_gib)
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

    # §56 steps 4-6 per segment, before any encoding work.
    if not skip_bad:
        for position, segment in enumerate(committed):
            if segment.name != f"{position:06d}":
                raise MediaError(
                    f"segment numbering gap: expected {position:06d}, found {segment.name}"
                )
    usable: list[Path] = []
    for segment in committed:
        try:
            _verify_segment(segment)
        except MediaError as exc:
            if not skip_bad:
                raise
            print(f"finalize: skipping {segment.name} ({exc})")
            continue
        usable.append(segment)
    if not usable:
        raise MediaError(f"no usable segments in {run_dir}")

    with tempfile.TemporaryDirectory() as tmp:
        tmpdir = Path(tmp)
        # Per-segment video-only parts, then concat the parts.
        parts: list[Path] = []
        for segment in usable:
            part = tmpdir / f"{segment.name}.mp4"
            proc = run_capture(
                [
                    "ffmpeg",
                    "-hide_banner",
                    "-nostdin",
                    "-y",
                    "-i",
                    str(segment / "video.mp4"),
                    "-c:v",
                    "libx264",
                    "-pix_fmt",
                    "yuv420p",
                    "-preset",
                    "veryfast",
                    "-an",
                    str(part),
                ]
            )
            if proc.returncode != 0:
                raise MediaError(f"segment mux failed for {segment.name}: {proc.stderr[-2000:]}")
            parts.append(part)
        concat_list = tmpdir / "concat.txt"
        concat_list.write_text("".join(f"file '{part}'\n" for part in parts), encoding="utf-8")
        # Blended final mix (overlap re-sliced from takes; previews untouched).
        final_audio = build_final_audio(
            run_dir,
            usable,
            tmpdir,
            fps,
            sample_rate,
            channels,
            overlap_fraction,
            overlap_cap_seconds,
        )
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
                "-i",
                str(final_audio),
                "-vf",
                vf,
                "-map",
                "0:v:0",
                "-map",
                "1:a:0",
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
                "-shortest",
                str(staged),
            ]
        )
        if proc.returncode != 0:
            raise MediaError(f"final encode failed: {proc.stderr[-2000:]}")
        validate_video(staged, width, height, fps)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        atomic_write_bytes(output_path, staged.read_bytes())
    return output_path
