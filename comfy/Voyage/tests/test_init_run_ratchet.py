"""Ratchet for issue 040: legacy per-file scaffolds converge onto the helper.

Why this file exists: over twenty test modules each carry a private
scaffold that must stay byte-identical to `initialize_run_directory`
(DESIGN section 21, AGENTS section 12 pragmatic DRY). Every remaining
definition is a one-line delegation — no divergent directory layout,
no reseeded config — and the total reference count below can only
fall. Lower `MAXIMUM_SCAFFOLD_REFERENCES` each time a file converges;
never raise it (coordinate with concurrent tracks instead).

The needle is assembled by concatenation so this module's own source
never matches the text scan it performs (it would otherwise count
itself and defeat the ratchet).
"""

from __future__ import annotations

from pathlib import Path

TESTS_DIR = Path(__file__).resolve().parent

# As-left 2026-09-30 after folding the phase-2 remainder module into
# test_recovery and inlining the stage-timings, concept-integrity and
# av-alignment wrappers (153 references across 25 files as-read before
# the pass).
MAXIMUM_SCAFFOLD_REFERENCES = 144

# Tokens of a divergent hand-rolled scaffold: any definition that builds
# the run layout itself instead of delegating to the shared helper.
RAW_SCAFFOLD_TOKENS = (
    "write_manifest",
    "initial_state",
    "SEGMENTS_DIRNAME).mkdir",
)


def _needle() -> str:
    return "_init" + "_run"


def _scaffold_files() -> dict[str, str]:
    needle = _needle()
    found: dict[str, str] = {}
    for candidate in sorted(TESTS_DIR.glob("*.py")):
        if candidate.name == Path(__file__).name:
            continue
        text = candidate.read_text(encoding="utf-8")
        if "def " + needle in text:
            found[candidate.name] = text
    return found


def test_every_scaffold_definition_delegates_to_the_helper() -> None:
    """No private definition may build the run layout itself."""
    for name, text in _scaffold_files().items():
        assert "initialize_run_directory" in text, name
        for token in RAW_SCAFFOLD_TOKENS:
            assert token not in text, f"{name} carries raw scaffold {token!r}"


def test_scaffold_reference_count_only_falls() -> None:
    """Total references to the legacy scaffold never grow."""
    needle = _needle()
    total = 0
    for candidate in sorted(TESTS_DIR.glob("*.py")):
        if candidate.name == Path(__file__).name:
            continue
        text = candidate.read_text(encoding="utf-8")
        total += sum(1 for line in text.splitlines() if needle in line)
    assert total <= MAXIMUM_SCAFFOLD_REFERENCES, (
        f"{total} references exceed the ratchet {MAXIMUM_SCAFFOLD_REFERENCES}: "
        "converge a file onto initialize_run_directory and lower the cap"
    )
