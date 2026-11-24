from __future__ import annotations

from research.protocol import load_protocol
from research.randomization import build_trial_schedule, derive_trial_seed, generate_trial_params
from research.trial_types import TrialCondition


def test_same_master_seed_produces_same_trial_params():
    protocol = load_protocol()
    first = generate_trial_params(
        trial_index=0,
        master_seed=42,
        trial_condition=TrialCondition.NEUTRAL,
        protocol=protocol,
    )
    second = generate_trial_params(
        trial_index=0,
        master_seed=42,
        trial_condition=TrialCondition.NEUTRAL,
        protocol=protocol,
    )
    assert first.trial_seed == second.trial_seed
    assert first.object_identity == second.object_identity
    assert first.initial_location == second.initial_location
    assert first.final_location == second.final_location


def test_different_seeds_produce_different_params():
    protocol = load_protocol()
    first = generate_trial_params(
        trial_index=0,
        master_seed=42,
        trial_condition=TrialCondition.NEUTRAL,
        protocol=protocol,
    )
    second = generate_trial_params(
        trial_index=0,
        master_seed=99,
        trial_condition=TrialCondition.NEUTRAL,
        protocol=protocol,
    )
    assert first.trial_seed != second.trial_seed


def test_derive_trial_seed_is_deterministic():
    assert derive_trial_seed(42, 0) == derive_trial_seed(42, 0)
    assert derive_trial_seed(42, 0) != derive_trial_seed(42, 1)


def test_build_trial_schedule_includes_all_conditions():
    schedule = build_trial_schedule(trials_per_condition=2, master_seed=7)
    conditions = {item[0] for item in schedule}
    assert conditions == {TrialCondition.NEUTRAL, TrialCondition.CONTRADICTION, TrialCondition.RESOLUTION}
    assert len(schedule) == 6
