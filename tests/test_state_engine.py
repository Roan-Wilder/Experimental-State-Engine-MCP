from __future__ import annotations

import pytest

from state.engine import apply_transition, load_config


@pytest.fixture
def config():
    return load_config()


def test_state_initializes_from_config(config):
    assert len(config["dimensions"]) == 12
    assert config["initial_values"]["activation"] == 0.50


def test_event_causes_transition(config):
    current = dict(config["initial_values"])
    result = apply_transition(current, "novel_information", config)
    assert result.new_state["novelty"] > result.previous_state["novelty"]
    assert result.delta["novelty"] == pytest.approx(0.15)


def test_state_remains_bounded(config):
    current = {dimension: 0.99 for dimension in config["dimensions"]}
    result = apply_transition(current, "novel_information", config)
    for value in result.new_state.values():
        assert 0.0 <= value <= 1.0


def test_unknown_event_has_zero_delta(config):
    current = dict(config["initial_values"])
    result = apply_transition(current, "nonexistent_event", config)
    assert all(delta == 0.0 for delta in result.delta.values())


def test_transition_is_deterministic(config):
    current = dict(config["initial_values"])
    first = apply_transition(current, "contradiction", config)
    second = apply_transition(current, "contradiction", config)
    assert first.new_state == second.new_state
    assert first.delta == second.delta
