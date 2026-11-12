"""Optional same-process checkpoint recovery validation."""
from __future__ import annotations

from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass
from typing import Any, TypeVar


Record = TypeVar("Record")


@dataclass(frozen=True)
class RecoveryValidation:
    """Result of an optional same-process recovery experiment."""

    passed: bool | None
    metrics: dict[str, Any]

    @classmethod
    def not_run(cls, metric_names: Iterable[str]) -> "RecoveryValidation":
        return cls(None, {name: None for name in metric_names})


def validate_recovery_enabled(config: Mapping[str, Any]) -> bool:
    """Read the explicit recovery-validation switch without coercion."""
    enabled = config.get("validate_recovery", False)
    if type(enabled) is not bool:
        raise ValueError("validate_recovery must be a boolean")
    return enabled


def run_recovery_validation(
    *,
    enabled: bool,
    expected_step: int,
    metric_names: Iterable[str],
    update: Callable[[], Record],
    restore: Callable[[], int],
    fingerprint: Callable[[], str] | None,
    compare: Callable[[Record, Record, str | None, str | None], tuple[bool, dict[str, Any]]],
) -> RecoveryValidation:
    """Compare one continued update with a restored update when requested."""
    if type(enabled) is not bool:
        raise TypeError("recovery validation enabled must be a bool")
    metric_names = tuple(metric_names)
    if not enabled:
        return RecoveryValidation.not_run(metric_names)

    try:
        continuous = update()
        continuous_fingerprint = fingerprint() if fingerprint is not None else None
        restored_step = restore()
        resumed = update()
        resumed_fingerprint = fingerprint() if fingerprint is not None else None
        matched, metrics = compare(
            continuous, resumed, continuous_fingerprint, resumed_fingerprint
        )
        if set(metrics) != set(metric_names):
            raise ValueError("recovery validation metrics do not match the declared schema")
        return RecoveryValidation(restored_step == expected_step and matched, metrics)
    finally:
        # Leave the trainer at the final checkpoint, never at an experiment update.
        restore()
