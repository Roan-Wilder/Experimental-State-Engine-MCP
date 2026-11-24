from __future__ import annotations

from state.history import StateHistory


def test_history_is_bounded():
    history = StateHistory(limit=3)
    for index in range(5):
        history.append({"observation_id": f"obs-{index}"})
    entries = history.get()
    assert len(entries) == 3
    assert entries[0]["observation_id"] == "obs-2"


def test_history_respects_limit_param():
    history = StateHistory(limit=10)
    for index in range(4):
        history.append({"observation_id": f"obs-{index}"})
    entries = history.get(limit=2)
    assert len(entries) == 2
    assert entries[-1]["observation_id"] == "obs-3"


def test_history_summary_is_compact():
    history = StateHistory(limit=5)
    history.append({"observation_id": "a", "timestamp": "t", "event": "e", "state": {"activation": 0.5}, "delta": {"activation": 0.1}})
    summary = history.summary(limit=1)
    assert summary[0]["observation_id"] == "a"
