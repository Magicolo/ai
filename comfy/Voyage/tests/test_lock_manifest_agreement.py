"""Lock-vs-manifest agreement gate (issues 011/068; overlaps 089).

`requirements.lock` is the trust anchor for the slim image: every row must
trace to a declared dependency in `pyproject.toml` through the installed
metadata, and every marker-applicable direct dep must appear in the lock.
The 2026-09-30 scare over `httpx2`/`httpcore2` turned out genuine
(`huggingface_hub==2.0.0` really requires `httpx2<3,>=2.0.0`, both
importable in voyage:latest) — which is why the agreement is now gated
instead of eyeballed: a future renamed or orphaned row fails here.

Reachability is extras-aware: `textual` requires `markdown-it-py[linkify]`,
which activates markdown-it-py's `extra == "linkify"` edge to
`linkify-it-py` — naive marker evaluation (no extras) false-flags it.
Same provenance check, all reachable: `ast_serialize` (mypy, PEP 503
separator equivalence), `librt` (mypy, CPython marker), `hf-xet`
(huggingface_hub, platform markers).

Rows installed and runtime-imported but declared nowhere in metadata live
in METADATA_ORPHAN_ALLOWLIST with justification; the gate keeps that list
minimal (an entry that becomes reachable fails until removed).
"""

from __future__ import annotations

import importlib.metadata
import re
from pathlib import Path

import pytest

try:
    import tomllib
except ImportError:  # Python 3.10 floor (same fallback idiom as voyage/config.py)
    import tomli as tomllib

from packaging.requirements import Requirement

REPO_ROOT = Path(__file__).resolve().parent.parent
LOCK_PATH = REPO_ROOT / "requirements.lock"
PYPROJECT_PATH = REPO_ROOT / "pyproject.toml"

METADATA_ORPHAN_ALLOWLIST = {
    "typing-inspection": (
        "Installed yet declared by nothing (Required-by empty): pydantic 2.10.6 "
        "imports typing_inspection at runtime but omits it from Requires-Dist "
        "(verified 2026-09-30 in voyage:latest) — upstream metadata gap, row "
        "is load-bearing so it stays until the metadata declares it."
    ),
}
"""Lock rows that are installed and needed but unreachable via metadata."""


def _canonical(name: str) -> str:
    """PEP 503 normalize (underscores/dots/dashes collapse, lowercase)."""
    return re.sub(r"[-_.]+", "-", name).lower()


def _lock_rows() -> dict[str, str]:
    """Parse `name==version` rows (skip comments/blanks)."""
    rows: dict[str, str] = {}
    for line in LOCK_PATH.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "==" not in line:
            continue
        name, _, version = line.partition("==")
        rows[_canonical(name.strip())] = version.strip()
    return rows


def _manifest_requirement_strings() -> list[str]:
    """Raw requirement strings: runtime deps + dev optional deps (both ship in the lock)."""
    with PYPROJECT_PATH.open("rb") as handle:
        pyproject = tomllib.load(handle)
    project = pyproject["project"]
    runtime = list(project["dependencies"])
    dev = list(project.get("optional-dependencies", {}).get("dev", []))
    return [*runtime, *dev]


def _applies(requirement: Requirement, active_extras: set[str]) -> bool:
    """True when a requirement's marker holds in an extras context.

    The bare evaluation covers unconditional and environment markers; each
    active extra (from `parent[extra]` edges) is tried for `extra == ...`
    gates such as markdown-it-py's `linkify`.
    """
    if requirement.marker is None:
        return True
    if requirement.marker.evaluate():
        return True
    return any(requirement.marker.evaluate({"extra": extra}) for extra in active_extras)


def _reachable_from(root_names: list[str], installed: dict[str, list[str]]) -> set[str]:
    """Transitive closure over installed Requires-Dist, following extras.

    `active` tracks which extras each distribution was requested with; a
    distribution is re-queued when new extras arrive so its
    extra-conditional edges get a second chance to apply.
    """
    active: dict[str, set[str]] = {root: set() for root in root_names}
    queue = list(root_names)
    reachable: set[str] = set()
    while queue:
        name = queue.pop()
        reachable.add(name)
        for req_string in installed.get(name, []):
            requirement = Requirement(req_string)
            if not _applies(requirement, active.get(name, set())):
                continue
            target = _canonical(requirement.name)
            known = active.setdefault(target, set())
            new_extras = set(requirement.extras) - known
            if target not in reachable or new_extras:
                known |= set(requirement.extras)
                queue.append(target)
    return reachable


def _unreachable(lock_names: set[str], reachable: set[str], allowlist: set[str]) -> set[str]:
    """Lock rows neither reachable from the manifest nor explicitly allowed."""
    return set(lock_names) - set(reachable) - set(allowlist)


def _installed_requirements() -> dict[str, list[str]]:
    """Canonical dist name -> Requires-Dist strings for the running image."""
    installed: dict[str, list[str]] = {}
    for distribution in importlib.metadata.distributions():
        name = (distribution.metadata["Name"] or "").strip()
        if name:
            installed[_canonical(name)] = list(distribution.requires or [])
    return installed


def _image_context() -> tuple[list[str], dict[str, list[str]]]:
    """Manifest roots + installed metadata, or skip outside the image.

    The gate is meaningful only where the lock is installed (the slim
    image); anywhere else (host) there is no metadata to check against.
    A partially-installed image is broken, not foreign — that fails loudly.
    """
    roots = [_canonical(Requirement(raw).name) for raw in _manifest_requirement_strings()]
    installed = _installed_requirements()
    if not any(root in installed for root in roots):
        pytest.skip("lock gate needs the installed image (no root dist metadata here)")
    return roots, installed


def test_direct_deps_are_pinned_in_lock() -> None:
    """Every marker-applicable direct dep has a lock row (the header contract)."""
    roots, _ = _image_context()
    lock = _lock_rows()
    inapplicable = set()
    missing = []
    for raw in _manifest_requirement_strings():
        requirement = Requirement(raw)
        name = _canonical(requirement.name)
        if requirement.marker is not None and not requirement.marker.evaluate():
            inapplicable.add(name)  # e.g. tomli on Python >= 3.11 — correctly absent
            continue
        if name not in lock:
            missing.append(raw)
    assert not missing, f"direct deps missing from requirements.lock: {sorted(missing)}"
    assert set(roots) - set(lock) <= inapplicable


def test_lock_rows_are_reachable_from_manifest() -> None:
    """Every lock row traces to the manifest (renames/orphans fail here)."""
    roots, installed = _image_context()
    reachable = _reachable_from(roots, installed)
    lock = _lock_rows()
    assert "httpx2" in reachable and "httpcore2" in reachable  # issue 068 rows are genuine
    offenders = _unreachable(set(lock), reachable, set(METADATA_ORPHAN_ALLOWLIST))
    assert not offenders, f"lock rows unreachable from pyproject.toml: {sorted(offenders)}"


def test_metadata_orphan_allowlist_stays_minimal() -> None:
    """Each allowlist entry is still installed yet still unreachable.

    If upstream metadata ever declares the edge, the entry becomes
    reachable and this test forces its removal from the allowlist.
    """
    roots, installed = _image_context()
    reachable = _reachable_from(roots, installed)
    for name in METADATA_ORPHAN_ALLOWLIST:
        assert name in installed, f"allowlist entry {name!r} no longer installed — drop it"
        assert name not in reachable, f"allowlist entry {name!r} now reachable — drop it"


def test_unreachable_names_are_flagged() -> None:
    """The gate discriminates: a fabricated orphan is reported, real rows are not."""
    installed = {"alpha": ["beta>=1"], "beta": []}
    reachable = _reachable_from(["alpha"], installed)
    assert reachable == {"alpha", "beta"}
    assert _unreachable({"alpha", "beta", "evil"}, reachable, set()) == {"evil"}
    assert _unreachable({"alpha", "beta", "evil"}, reachable, {"evil"}) == set()


def test_canonicalization_treats_separators_equal() -> None:
    """PEP 503 equivalence (the `ast_serialize` row matches `ast-serialize`)."""
    assert _canonical("ast_serialize") == _canonical("ast-serialize") == "ast-serialize"
    assert _canonical("linkify-it-py") == "linkify-it-py"
    assert _canonical("PyYAML") == "pyyaml"
