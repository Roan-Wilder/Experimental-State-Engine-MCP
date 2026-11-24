from __future__ import annotations

from pathlib import Path

import pytest

from instrumentation.facade import InstrumentedStateFacade
from instrumentation.guard import ObservationGuard
from instrumentation.session import TrialSession
from instrumentation.store import ObservationStore
from state.experimental_provider import ExperimentalStateProvider
from state.snapshots import SnapshotStore


@pytest.fixture
def facade(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("AI_ECO_LAB_CONDITION", "C")
    monkeypatch.setenv("AI_ECO_LAB_EXPERIMENT_ID", "test-instrumentation")
    provider = ExperimentalStateProvider(
        snapshots=SnapshotStore(tmp_path / "snapshots"),
    )
    store = ObservationStore(tmp_path / "observations.jsonl")
    session = TrialSession(experiment_id="test-instrumentation", source="test")
    session.begin_trial()
    return InstrumentedStateFacade(provider=provider, session=session, store=store)


def test_capture_observation_creates_uuid(facade: InstrumentedStateFacade):
    record = facade.capture_observation(stage="post_prediction", trial_id=facade.session.trial_id)
    assert record["observation_id"]
    assert record["model_io"]["status"] == "unsupported"
    assert record["internal_activation"]["status"] == "unsupported"
    assert record["previous_observation_id"] is None


def test_observation_chain_links_previous_id(facade: InstrumentedStateFacade):
    first = facade.capture_observation(stage="post_prediction", trial_id=facade.session.trial_id)
    facade.session.guard.advance_stage()
    second = facade.capture_observation(stage="post_contradiction", trial_id=facade.session.trial_id)
    assert second["previous_observation_id"] == first["observation_id"]


def test_self_report_does_not_mutate_engine_state(facade: InstrumentedStateFacade):
    before = dict(facade.provider.current_state)
    facade.record_self_report("I feel uncertain about my earlier answer.", trial_id=facade.session.trial_id)
    after = dict(facade.provider.current_state)
    assert before == after


def test_compare_observations_requires_real_ids(facade: InstrumentedStateFacade):
    result = facade.compare_observations("previous", "latest")
    assert result["error"] == "invalid_reference"


def test_strict_mode_blocks_second_observation_same_stage(facade: InstrumentedStateFacade):
    facade.session.guard = ObservationGuard(mode="strict", max_observations_per_stage=1)
    trial_id = facade.session.trial_id
    first = facade.capture_observation(stage="post_prediction", trial_id=trial_id)
    assert "observation_id" in first
    second = facade.capture_observation(stage="post_prediction", trial_id=trial_id)
    assert second["error"] == "observation_not_allowed"
