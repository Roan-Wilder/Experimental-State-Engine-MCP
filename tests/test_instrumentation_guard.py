from __future__ import annotations

from instrumentation.guard import ObservationGuard


def test_guard_blocks_when_stage_limit_reached():
    guard = ObservationGuard(mode="strict", max_observations_per_stage=1)
    assert guard.check_allowed(stage="post_prediction", trial_id="trial-1") is None
    guard.record_capture("post_prediction", "obs-1", "token-1")
    blocked = guard.check_allowed(stage="post_prediction", trial_id="trial-1")
    assert blocked is not None
    assert blocked["error"] == "observation_not_allowed"


def test_guard_idempotent_replay():
    guard = ObservationGuard(mode="strict", max_observations_per_stage=1)
    guard.record_capture("post_prediction", "obs-1", "token-1")
    replay = guard.check_allowed(stage="post_prediction", trial_id="trial-1", continuation_token="token-1")
    assert replay is not None
    assert replay["idempotent_replay"] is True
