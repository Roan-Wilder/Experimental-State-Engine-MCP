from __future__ import annotations

import uuid
from pathlib import Path

import pytest

from instrumentation.facade import InstrumentedStateFacade
from instrumentation.session import TrialSession
from instrumentation.store import ObservationStore
from state.experimental_provider import ExperimentalStateProvider
from state.snapshots import SnapshotStore


@pytest.fixture
def facade(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("AI_ECO_LAB_CONDITION", "C")
    provider = ExperimentalStateProvider(snapshots=SnapshotStore(tmp_path / "snapshots"))
    session = TrialSession(experiment_id="compare-test", source="test")
    session.trial_id = str(uuid.uuid4())
    store = ObservationStore(tmp_path / "observations.jsonl")
    return InstrumentedStateFacade(provider=provider, session=session, store=store)


def test_compare_observations_with_valid_uuids(facade: InstrumentedStateFacade):
    first = facade.capture_observation(stage="post_prediction", trial_id=facade.session.trial_id)
    facade.session.guard.advance_stage()
    second = facade.capture_observation(stage="post_contradiction", trial_id=facade.session.trial_id)
    result = facade.compare_observations(first["observation_id"], second["observation_id"])
    assert "channels" in result
    assert result["before_observation_id"] == first["observation_id"]


def test_compare_affective_states_rejects_keywords_by_default(facade: InstrumentedStateFacade, monkeypatch):
    monkeypatch.delenv("AI_ECO_LAB_ALLOW_KEYWORD_REFS", raising=False)
    result = facade.compare_affective_states("initial", "current")
    assert result["error"] == "invalid_reference"
