"""Run manifest + run state persistence (DESIGN §§32-33, task group C).

All writes are atomic (DESIGN §31). state.json is owned by the
supervisor only — the director proposes, the supervisor commits.
"""

from __future__ import annotations

from pathlib import Path

from voyage import paths
from voyage.atomic import atomic_write_json, read_json
from voyage.config import ProjectConfig
from voyage.errors import StateError
from voyage.models import RunState


def build_manifest(
    config: ProjectConfig,
    segments: int | None = None,
    final_video: str | None = None,
    skip_bad: bool = False,
    no_sfx: bool = False,
) -> dict[str, object]:
    """Build the flat run manifest: the effective config IS the root.

    The manifest root carries every ProjectConfig field plus the run
    plan (`segments`: planned segment count, the configure-time target
    the supervisor compares against `state.committed_segments`) and the
    finalize policy (`final_video`/`skip_bad`/`no_sfx`). No snapshots,
    no provenance, no duplication — readers validate the root straight
    into ProjectConfig (extra manifest keys are ignored).
    """
    manifest = config.model_dump(mode="json")
    # Caption pins are in-memory only (config.py): a stored pin would
    # freeze every future segment to one caption instead of letting the
    # director evolve them, so they never reach the manifest.
    for section_name, pin_key in (
        ("video", "video_caption"),
        ("audio", "music_caption"),
    ):
        section = manifest.get(section_name)
        if isinstance(section, dict):
            section.pop(pin_key, None)
    manifest["segments"] = segments
    manifest["final_video"] = final_video
    manifest["skip_bad"] = skip_bad
    manifest["no_sfx"] = no_sfx
    return manifest


def write_manifest(run_dir: Path, manifest: dict[str, object]) -> None:
    atomic_write_json(run_dir / paths.MANIFEST_FILENAME, manifest)


def record_final_coverage(
    run_dir: Path,
    presented_frames: int,
    segments: int,
    *,
    skip_key: str | None = None,
) -> None:
    """Stamp the shipped coverage into the manifest (redundant-finalize gate).

    `final_coverage` records what `final.mp4` actually presents (ffprobe
    presented frames + committed segment count at finalize success), so the
    generate 'nothing to do' gate compares presented-against-presented
    instead of presented-against-source-timeline (which never matched
    under interp + slow-mo). Extra manifest keys are ignored by
    `read_effective_config`, and `build_manifest` never emits this key —
    every `configure` wipes coverage, i.e. conservative invalidation of
    the freshness stamp on any plan/settings change for free.

    `skip_key` (optional, generate skip flags): the canonical behavior key
    for this finalize. Omitted keeps the legacy 2-key dict so old callers
    and tests stay green; provided stores it alongside for the freshness
    gate to compare against.
    """
    manifest = read_manifest(run_dir)
    coverage: dict[str, object] = {"segments": segments, "presented_frames": presented_frames}
    if skip_key is not None:
        coverage["skip_key"] = skip_key
    manifest["final_coverage"] = coverage
    write_manifest(run_dir, manifest)


def read_manifest(run_dir: Path) -> dict[str, object]:
    path = run_dir / paths.MANIFEST_FILENAME
    if not path.exists():
        raise StateError(f"missing {paths.MANIFEST_FILENAME} in {run_dir}")
    try:
        data = read_json(path)
    except (OSError, ValueError) as exc:
        # Torn manifest (SIGKILL mid-write) or hand-edit corruption must
        # read as StateError (issue 002) — never a bare JSONDecodeError
        # that escapes the commit boundary and strands the run at RUNNING.
        raise StateError(f"invalid {paths.MANIFEST_FILENAME} in {run_dir}: {exc}") from exc
    if not isinstance(data, dict):
        raise StateError(f"{paths.MANIFEST_FILENAME} is not a JSON object")
    return data


def read_effective_config(run_dir: Path) -> ProjectConfig:
    """Load the run's effective config from its flat manifest (clean break).

    The manifest root IS the config: validate it straight into
    ProjectConfig (extra manifest keys — segments, finalize policy —
    are ignored). Runs created before the flat-manifest migration carry
    a nested `effective_config` and fail loud with the re-generate hint.
    """
    manifest = read_manifest(run_dir)
    if isinstance(manifest.get("effective_config"), dict):
        raise StateError(
            f"{paths.MANIFEST_FILENAME} in {run_dir} uses the nested effective_config "
            "format (pre flat-manifest — re-generate; no legacy format is read)"
        )
    augment_section = manifest.get("augment")
    if isinstance(augment_section, dict):
        legacy_keys = {
            "min_fps",
            "min_width",
            "min_height",
            "min_resolution",
            "use_model_pass",
            "no_augment",
            "interp_multiplier",
        } & set(augment_section)
        if legacy_keys:
            raise StateError(
                f"{paths.MANIFEST_FILENAME} in {run_dir} carries pre-multiplier "
                f"augment keys ({sorted(legacy_keys)}) — re-configure the run "
                "with --upscale/--interpolate; no legacy format is read"
            )
    augment_section = manifest.get("augment")
    if isinstance(augment_section, dict):
        # Runs configured before the interp_backend knob existed rendered
        # with FILM (RIFE never existed) — backfill truthfully so those
        # ledgers keep hitting. Runs without an augment section never
        # rendered a model pass; the rife default stands.
        augment_section.setdefault("interp_backend", "film")
    try:
        return ProjectConfig.model_validate(manifest)
    except Exception as exc:
        raise StateError(
            f"invalid effective config in {paths.MANIFEST_FILENAME} ({run_dir}): {exc}"
        ) from exc


def write_state(run_dir: Path, state: RunState) -> None:
    atomic_write_json(run_dir / paths.STATE_FILENAME, state.model_dump())


def read_state(run_dir: Path) -> RunState:
    path = run_dir / paths.STATE_FILENAME
    if not path.exists():
        raise StateError(f"missing {paths.STATE_FILENAME} in {run_dir}")
    try:
        return RunState.model_validate(read_json(path))
    except (OSError, ValueError) as exc:
        # Same taxonomy as read_manifest above (issue 002): unreadable
        # files (OSError) and torn/hand-edited JSON or schema violations
        # (ValueError covers JSONDecodeError, UnicodeDecodeError, and
        # pydantic ValidationError) read as StateError. Anything else
        # (MemoryError and friends) propagates raw — masking resource
        # exhaustion as corrupt state would send recovery down the
        # wrong path.
        raise StateError(f"invalid state file {path}: {exc}") from exc


def initial_state(config: ProjectConfig) -> RunState:
    return RunState(
        name=config.name,
        status="CREATED",
        current_concept=config.style,
        destination_concept=config.style,
    )


def create_run_dir(
    run_dir: Path,
    config: ProjectConfig,
    segments: int | None = None,
    final_video: str | None = None,
    skip_bad: bool = False,
    no_sfx: bool = False,
) -> None:
    """Scaffold a fresh run directory (sole creator: `generate` + tests).

    mkdirs segments/logs, then writes the flat manifest (the effective
    config at the root plus plan + finalize policy) and the initial
    state. The caller owns all validation (non-empty guard, style,
    seed, backend) before calling — this function only writes.
    """
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / paths.SEGMENTS_DIRNAME).mkdir(exist_ok=True)
    (run_dir / paths.LOGS_DIRNAME).mkdir(exist_ok=True)
    write_manifest(
        run_dir,
        build_manifest(
            config,
            segments=segments,
            final_video=final_video,
            skip_bad=skip_bad,
            no_sfx=no_sfx,
        ),
    )
    write_state(run_dir, initial_state(config))
