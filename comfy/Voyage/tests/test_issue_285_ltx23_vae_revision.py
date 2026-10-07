"""Issue 285: ltx23 recovery tape carries a true VAE revision + enforces it.

`build_recovery_tape` aliased the DiT revision into `vae_revision`
while the registry had no VAE constant at all, and `parse_recovery_tape`
never compared revisions — a swapped VAE resumed silently. The registry
now owns `LTX23_VAE_REVISION` (same Hub repo as the DiT, so one pin
covers all three files) and `_validate_tape_trust` fails loud on a
mismatch. Legacy tapes grandfather in: the old aliased value equals the
new constant while the VAEs share the DiT repo.

All pure (slim gates image).
"""

from __future__ import annotations

from typing import Any

import pytest

from voyage.registry_ltx23 import LTX23_DIT_REVISION, LTX23_VAE_REVISION
from voyage.workers.video_ltx23 import (
    CONDITIONING_TAIL_FRAMES,
    SEGMENT_TARGET_FRAMES,
    build_recovery_tape,
    parse_recovery_tape,
)


def _fresh_tape() -> dict[str, Any]:
    """A valid tape straight from the builder (hashes self-consistent)."""
    return build_recovery_tape(
        source_segment_id="seg000001",
        conditioning_tail_path="/tmp/tail.mp4",
        prompts=["a calm ink landscape"],
        seeds=[11],
        width=1216,
        height=704,
        fps=24,
        segment_target_frames=SEGMENT_TARGET_FRAMES,
        conditioning_tail_frames=CONDITIONING_TAIL_FRAMES,
    )


def test_registry_vae_revision_pins_the_dit_repo() -> None:
    """One repo-level pin versions DiT + both VAEs (documented same-repo)."""
    assert isinstance(LTX23_VAE_REVISION, str) and LTX23_VAE_REVISION
    assert LTX23_VAE_REVISION == LTX23_DIT_REVISION


def test_tape_records_the_vae_revision() -> None:
    """The tape field now asserts VAE provenance, not the DiT string."""
    from voyage.workers.video_ltx23 import _validate_tape_trust

    tape = _fresh_tape()
    assert tape["vae_revision"] == LTX23_VAE_REVISION
    assert _validate_tape_trust(dict(tape)) == tape


def test_trust_gate_rejects_swapped_vae() -> None:
    """A foreign `vae_revision` fails loud instead of resuming silently."""
    from voyage.workers.video_ltx23 import _validate_tape_trust

    tape = _fresh_tape()
    tape["vae_revision"] = "deadbeef" * 5
    with pytest.raises(ValueError, match="vae_revision mismatch"):
        _validate_tape_trust(tape)


def test_trust_gate_rejects_missing_vae_revision() -> None:
    """A tape without the field cannot prove its VAE leg."""
    from voyage.workers.video_ltx23 import _validate_tape_trust

    tape = _fresh_tape()
    del tape["vae_revision"]
    with pytest.raises(ValueError, match="vae_revision mismatch"):
        _validate_tape_trust(tape)


def test_legacy_aliased_tape_grandfathers_in() -> None:
    """Pre-fix tapes (DiT rev in the VAE field) still validate today."""
    from voyage.workers.video_ltx23 import _validate_tape_trust

    tape = _fresh_tape()
    tape["vae_revision"] = LTX23_DIT_REVISION
    assert _validate_tape_trust(dict(tape))["vae_revision"] == LTX23_VAE_REVISION
    assert parse_recovery_tape(dict(tape))["vae_revision"] == LTX23_DIT_REVISION
