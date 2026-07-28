"""ASHA defaults, imported lazily."""

from __future__ import annotations

from dataclasses import asdict, dataclass

from .optional import require_ray


@dataclass(frozen=True)
class ASHAConfig:
    metric: str = "objective"
    mode: str = "min"
    time_attr: str = "training_iteration"
    max_t: int = 100
    grace_period: int = 5
    reduction_factor: int = 3
    brackets: int = 1

    def __post_init__(self) -> None:
        if self.max_t < 1 or not 1 <= self.grace_period <= self.max_t:
            raise ValueError("ASHA requires 1 <= grace_period <= max_t")
        if self.reduction_factor <= 1 or self.brackets < 1:
            raise ValueError("Invalid ASHA reduction_factor or brackets")

    def to_dict(self) -> dict[str, object]:
        return asdict(self)

    def build(self):
        require_ray()
        from ray.tune.schedulers import ASHAScheduler
        return ASHAScheduler(**self.to_dict())
