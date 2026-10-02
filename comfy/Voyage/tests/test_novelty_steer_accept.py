"""Novelty steer-and-accept (DESIGN §140, item 1).

The director is slow (~120s+/serve late-run), so novelty no longer
rejects: the prompt steers toward unvisited worlds (forbidden list +
explicit steering section), the supervisor scores every generation and
records it, and the first schema/style-valid generation always renders.
A revisit simply carries novelty_accepted=False. Speed comes from
single-serve accepts — no novelty retry loop burns extra LLM calls.
Supersedes test_novelty_leniency.py (cap-based override is gone).
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from tests.conftest import initialize_run_directory
from voyage import paths
from voyage.concepts import ConceptStore
from voyage.config import VoyageConfig, load_config
from voyage.director import DIRECTOR_SYSTEM_PROMPT, build_director_user_message
from voyage.models import DirectorDestination, DirectorVideoPlan, EvolutionDecision
from voyage.prompts import StyleSpec
from voyage.supervisor import Supervisor


def _valid_raw(index: int = 0) -> dict[str, Any]:
    decision = EvolutionDecision(
        decision_index=index,
        destination=DirectorDestination(canonical_name=f"steer valley {index}"),
        video=DirectorVideoPlan(stages=[f"a calm neon valley holds, stage {index}"]),
    )
    return dict(decision.model_dump())


def _metric_events(run_dir: Path, event: str) -> list[dict[str, Any]]:
    metrics_path = run_dir / paths.LOGS_DIRNAME / "metrics.jsonl"
    if not metrics_path.exists():
        return []
    lines = metrics_path.read_text(encoding="utf-8").splitlines()
    return [
        entry
        for entry in (json.loads(line) for line in lines if line.strip())
        if entry.get("event") == event
    ]


def _stubbed_supervisor(
    run_dir: Path,
    monkeypatch: pytest.MonkeyPatch,
    raws: list[dict[str, Any]],
) -> Supervisor:
    initialize_run_directory(run_dir, run_id="steer-accept")
    config, _ = load_config(run_dir / paths.CONFIG_FILENAME)
    supervisor = Supervisor(run_dir, config)
    queue = list(raws)

    def _queued_call(
        self: Supervisor,
        worker: Any,
        worker_name: str,
        segment_id: str,
        op: str,
        payload: dict[str, object],
        restart_hook: Any = None,
    ) -> dict[str, object]:
        assert queue, "director called more times than queued raws"
        return dict(queue.pop(0))

    monkeypatch.setattr(Supervisor, "_call_with_restart", _queued_call)
    monkeypatch.setattr(Supervisor, "_embed_texts", lambda self, texts: None)
    return supervisor


def _accept(supervisor: Supervisor, segment_id: str = "000000") -> EvolutionDecision:
    from voyage.persistence import read_state

    config = supervisor._config
    state = read_state(supervisor._run_dir)
    store = ConceptStore(supervisor._run_dir / "novelty")
    style = StyleSpec(prompt="pastel neon line-art, peaceful")
    decision, _tokens = supervisor._accept_director_decision(
        config, state, store, style, segment_id
    )
    return decision


def test_revisit_renders_without_rejection(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """An already-visited concept renders on the first serve — no rejection."""
    run_dir = tmp_path / "run"
    initialize_run_directory(run_dir, run_id="steer-accept")
    store = ConceptStore(run_dir / "novelty")
    store.append("steer valley 0", accepted=True, summary="seeded history", segment=0)
    supervisor = _stubbed_supervisor(run_dir, monkeypatch, [_valid_raw()])
    decision = _accept(supervisor)
    assert decision.destination_concept == "steer valley 0"
    assert decision.novelty_accepted is False
    assert _metric_events(run_dir, "director_rejection") == []
    assert _metric_events(run_dir, "novelty_overridden") == []
    scored = _metric_events(run_dir, "novelty_scored")
    assert len(scored) == 1
    assert scored[0]["score"] > 0.9
    assert scored[0]["accepted_novel"] is False


def test_novel_concept_accepted_novel(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """A fresh concept is accepted with novelty_accepted=True and scored."""
    run_dir = tmp_path / "run"
    supervisor = _stubbed_supervisor(run_dir, monkeypatch, [_valid_raw()])
    decision = _accept(supervisor)
    assert decision.novelty_accepted is True
    scored = _metric_events(run_dir, "novelty_scored")
    assert len(scored) == 1
    assert scored[0]["accepted_novel"] is True


def test_schema_style_retries_still_bounded(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Schema/style still reject and exhaust to the deterministic fallback."""
    run_dir = tmp_path / "run"
    supervisor = _stubbed_supervisor(
        run_dir, monkeypatch, [{"bogus": 1}, {"bogus": 2}, {"bogus": 3}]
    )
    decision = _accept(supervisor)
    assert decision.novelty_accepted is False
    assert len(_metric_events(run_dir, "director_rejection")) == 3
    assert len(_metric_events(run_dir, "director_fallback")) == 1


def test_steering_section_present_with_forbidden_list() -> None:
    """The prompt steers away from visited worlds when revisits are barred."""
    message = build_director_user_message(
        style_charter="pastel neon line-art",
        current_world="a reef",
        current_transition="none",
        recent_summary="reef",
        forbidden_summary="glass desert; basalt archive",
        audio_state="quiet",
        controller_metrics="ok",
    )
    assert "NOVELTY STEERING" in message
    assert "slightly different" in message


def test_no_steering_section_when_revisits_allowed() -> None:
    """No steering pressure when the run allows revisits."""
    message = build_director_user_message(
        style_charter="pastel neon line-art",
        current_world="a reef",
        current_transition="none",
        recent_summary="reef",
        forbidden_summary="(revisits allowed)",
        audio_state="quiet",
        controller_metrics="ok",
    )
    assert "NOVELTY STEERING" not in message


def test_system_prompt_steers_to_novelty() -> None:
    """The charter-level prompt tells the director to differ, not just avoid."""
    assert "slightly different" in DIRECTOR_SYSTEM_PROMPT


def test_legacy_rejections_cap_still_loads() -> None:
    """novelty_max_rejections is deprecated-unused but still parses (old TOMLs)."""
    assert VoyageConfig(novelty_max_rejections=5).novelty_max_rejections == 5
