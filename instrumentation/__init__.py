"""Multi-channel instrumentation layer for AI Eco Lab experiments."""

from instrumentation.sanitize import sanitize_metadata, validate_observation_reference

__all__ = [
    "InstrumentedStateFacade",
    "sanitize_metadata",
    "validate_observation_reference",
]


def __getattr__(name: str):
    if name == "InstrumentedStateFacade":
        from instrumentation.facade import InstrumentedStateFacade

        return InstrumentedStateFacade
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
