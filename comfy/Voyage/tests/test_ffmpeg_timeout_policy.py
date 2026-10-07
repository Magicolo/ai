"""FFmpeg timeout policy: unbounded for timeline-scaled ops, bounded otherwise.

Why this exists: `run_capture` bounds every spawn with a 600s default, but
finalize-time assembly ops (publishes, full-timeline mixes/converts,
demux/remux/concats) legitimately scale with total video length — jango's
~65k-frame publish died at exactly 600.0s. Those ops must pass
`timeout=None`; fixed-scope ops (probes, slices, windows, per-segment
assembly, single-stem copies) keep the 600s bound so hangs still fail fast.

Rule of thumb: if the op's legitimate runtime grows with the number of
committed segments/frames, it is unbounded. Everything else stays bounded.
"""

import ast
import subprocess
from pathlib import Path

from voyage import augment_drain, media

VOYAGE_DIR = Path(__file__).resolve().parent.parent / "voyage"

_SPAWN_CALLEES = ("run_capture", "run_publish_encode")


def _spawn_kinds(module: str) -> dict[str, list[str]]:
    """Map function name -> ordered timeout kinds of its direct ffmpeg spawns.

    Kinds: "default" (no timeout kwarg, keeps the 600s bound), "none"
    (explicit `timeout=None`, unbounded), "threaded" (timeout passed as a
    variable/constant — resolved by the caller-side tests below).
    """
    tree = ast.parse((VOYAGE_DIR / (module + ".py")).read_text())
    found: dict[str, list[str]] = {}

    def visit(node: ast.AST, func: str) -> None:
        for child in ast.iter_child_nodes(node):
            if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef)):
                visit(child, child.name)
            else:
                if isinstance(child, ast.Call):
                    callee = getattr(child.func, "id", "") or getattr(child.func, "attr", "")
                    if callee in _SPAWN_CALLEES:
                        kind = "default"
                        for keyword in child.keywords:
                            if keyword.arg == "timeout":
                                if (
                                    isinstance(keyword.value, ast.Constant)
                                    and keyword.value.value is None
                                ):
                                    kind = "none"
                                else:
                                    kind = "threaded"
                        found.setdefault(func, []).append(kind)
                visit(child, func)

    visit(tree, "<module>")
    return found


# Every direct spawn scales with total timeline length -> must be unbounded.
UNBOUNDED: set[tuple[str, str]] = {
    ("media_audio", "_blend_pair"),
    ("media_audio", "_join_audio_single_graph"),
    ("media_audio", "build_final_audio_with_metrics"),
    ("sfx_finalize", "mix_sfx_pair"),
    ("sfx_finalize", "mix_music_and_sfx"),
    ("sfx_finalize", "demux_music_audio"),
    ("sfx_finalize", "remux_video_with_audio"),
    ("sfx_finalize", "build_proxy_reference"),
    ("sfx_finalize", "stretch_and_dub_sfx_bed"),
    ("mastering", "_resample_wav"),
    ("mastering", "maybe_master_ship_audio"),
    ("augment_drain", "concat_chunk_mp4s"),
}

# Fixed-scope ops -> every direct spawn keeps the 600s bound.
BOUNDED: set[tuple[str, str]] = {
    ("media_audio", "probe"),
    ("media_audio", "slice_take"),
    ("media_audio", "assemble_segment_audio"),
    ("media_audio", "_convert_window_to_dest"),
    ("media", "presented_frames"),
    ("mastering", "_slice_wav"),
}


def test_fully_unbounded_functions_pass_none() -> None:
    for module, func in sorted(UNBOUNDED):
        kinds = _spawn_kinds(module).get(func, [])
        assert kinds, f"{module}.{func} has no ffmpeg spawns left to guard"
        assert all(kind == "none" for kind in kinds), (
            f"{module}.{func} spawns scale with total timeline length "
            f"but are not all unbounded (got {kinds})"
        )


def test_fixed_scope_functions_keep_the_bound() -> None:
    for module, func in sorted(BOUNDED):
        kinds = _spawn_kinds(module).get(func, [])
        assert kinds, f"{module}.{func} has no ffmpeg spawns left to guard"
        assert all(kind == "default" for kind in kinds), (
            f"{module}.{func} is fixed-scope but lost its 600s bound (got {kinds})"
        )


def test_finalize_run_spawn_order() -> None:
    # Full-timeline silent synth (unbounded), then the three publish encodes
    # via the shared helper (caller-threaded). Probes stay on the path via
    # the bounded `presented_frames` helper (pinned in BOUNDED above).
    kinds = _spawn_kinds("media").get("finalize_run", [])
    assert kinds == ["none", "threaded", "threaded", "threaded"]


def test_render_single_track_spawn_order() -> None:
    # Single-stem copy (bounded) then full joined bed convert (unbounded).
    from voyage import sfx_finalize  # noqa: F401  (import pins the module path)

    kinds = _spawn_kinds("sfx_finalize").get("_render_single_track", [])
    assert kinds == ["default", "none"]


def test_publish_helper_threads_caller_timeout() -> None:
    # `run_publish_encode` must forward its timeout argument, never hardcode
    # one — the three finalize publishes rely on it for the unbounded path.
    tree = ast.parse((VOYAGE_DIR / "media.py").read_text())
    forwarded = False
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) and node.name == "run_publish_encode":
            for child in ast.walk(node):
                if isinstance(child, ast.Call) and getattr(child.func, "id", "") == "run_capture":
                    for keyword in child.keywords:
                        if keyword.arg == "timeout" and isinstance(keyword.value, ast.Name):
                            forwarded = True
    assert forwarded, "run_publish_encode must forward its timeout argument"
    assert media.PUBLISH_TIMEOUT_SECONDS_DEFAULT is None, (
        "the shared publish default must stay unbounded (None)"
    )


def test_concat_chunk_mp4s_spawns_without_timeout_bound(
    monkeypatch: object, tmp_path: Path
) -> None:
    recorded_argvs: list[list[str]] = []
    recorded_timeouts: list[float | None] = []

    def _recording(argv: list[str], timeout: float | None = 600.0) -> object:
        recorded_argvs.append(argv)
        recorded_timeouts.append(timeout)
        return subprocess.CompletedProcess(argv, 0, "", "")

    monkeypatch.setattr(  # type: ignore[attr-defined]
        "voyage.media_audio.run_capture", _recording
    )
    chunks = [tmp_path / f"chunk_{index:02d}.mp4" for index in range(3)]
    for chunk in chunks:
        chunk.write_bytes(b"\x00")
    dest = tmp_path / "joined.mp4"
    assert augment_drain.concat_chunk_mp4s(chunks, dest) == dest
    assert recorded_timeouts == [None]
    assert "-c" in recorded_argvs[0]  # stream-copy shape preserved
