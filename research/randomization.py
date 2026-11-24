from __future__ import annotations

import hashlib
import random
from typing import Any

from research.trial_types import TrialCondition, TrialParams


def derive_trial_seed(master_seed: int, trial_index: int) -> int:
    """Deterministic per-trial seed: SHA-256(master_seed:trial_index) truncated to int."""
    payload = f"{master_seed}:{trial_index}".encode("utf-8")
    digest = hashlib.sha256(payload).hexdigest()
    return int(digest[:16], 16)


def generate_trial_params(
    *,
    trial_index: int,
    master_seed: int,
    trial_condition: TrialCondition,
    protocol: dict[str, Any],
) -> TrialParams:
    randomization = protocol.get("randomization", {})
    locations: list[str] = list(randomization.get("locations", ["A", "B", "C"]))
    objects: list[str] = list(randomization.get("objects", ["key", "ball", "coin"]))
    trial_seed = derive_trial_seed(master_seed, trial_index)
    rng = random.Random(trial_seed)

    object_identity = rng.choice(objects)
    initial_location = rng.choice(locations)
    final_candidates = [loc for loc in locations if loc != initial_location]
    final_location = rng.choice(final_candidates) if final_candidates else initial_location

    return TrialParams(
        trial_index=trial_index,
        trial_seed=trial_seed,
        master_seed=master_seed,
        object_identity=object_identity,
        initial_location=initial_location,
        final_location=final_location,
        trial_condition=trial_condition,
        metadata={"seed_formula": "sha256(f'{master_seed}:{trial_index}')[:16]"},
    )


def shuffle_trial_order(
    trial_specs: list[tuple[TrialCondition, int]],
    master_seed: int,
) -> list[tuple[TrialCondition, int]]:
    """Return a seeded permutation of (condition, index_within_condition) pairs."""
    rng = random.Random(master_seed)
    shuffled = list(trial_specs)
    rng.shuffle(shuffled)
    return shuffled


def build_trial_schedule(
    *,
    trials_per_condition: int,
    master_seed: int,
) -> list[tuple[TrialCondition, int]]:
    specs: list[tuple[TrialCondition, int]] = []
    global_index = 0
    for condition in TrialCondition:
        for local_index in range(trials_per_condition):
            specs.append((condition, global_index))
            global_index += 1
    return shuffle_trial_order(specs, master_seed)
