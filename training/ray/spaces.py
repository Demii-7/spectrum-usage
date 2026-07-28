"""Ray-independent, bounded search-space declarations."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

from .optional import require_ray


@dataclass(frozen=True)
class Domain:
    kind: str
    lower: float | int | None = None
    upper: float | int | None = None
    choices: tuple[Any, ...] = ()

    def __post_init__(self) -> None:
        if self.kind not in {"choice", "uniform", "loguniform", "randint"}:
            raise ValueError(f"Unsupported domain kind: {self.kind}")
        if self.kind == "choice":
            if not self.choices:
                raise ValueError("choice domains cannot be empty")
        elif self.lower is None or self.upper is None or self.lower >= self.upper:
            raise ValueError(f"{self.kind} requires finite lower < upper bounds")

    def contains(self, value: Any) -> bool:
        if self.kind == "choice":
            return value in self.choices
        if not isinstance(value, (int, float)) or isinstance(value, bool):
            return False
        if self.kind == "randint":
            return isinstance(value, int) and self.lower <= value < self.upper
        return self.lower <= value <= self.upper

    def to_dict(self) -> dict[str, Any]:
        result = {"kind": self.kind}
        if self.kind == "choice":
            result["choices"] = list(self.choices)
        else:
            result.update(lower=self.lower, upper=self.upper)
        return result

    def to_ray(self):
        require_ray()
        from ray import tune

        if self.kind == "choice":
            return tune.choice(list(self.choices))
        return getattr(tune, self.kind)(self.lower, self.upper)


def choice(*values: Any) -> Domain:
    return Domain("choice", choices=tuple(values))


def uniform(lower: float, upper: float) -> Domain:
    return Domain("uniform", lower=lower, upper=upper)


def loguniform(lower: float, upper: float) -> Domain:
    return Domain("loguniform", lower=lower, upper=upper)


def randint(lower: int, upper: int) -> Domain:
    return Domain("randint", lower=lower, upper=upper)


def materialize_space(space: Mapping[str, Domain]) -> dict[str, Any]:
    return {name: domain.to_ray() for name, domain in space.items()}
