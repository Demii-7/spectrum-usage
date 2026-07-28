"""Three-seed model selection without a one-standard-error rule."""

from __future__ import annotations

from dataclasses import dataclass
from statistics import mean
from typing import Any, Iterable, Mapping


@dataclass(frozen=True)
class Selection:
    best: str
    best_simple: str
    reference: str
    mean_metrics: Mapping[str, float]


def select_candidates(results: Iterable[Mapping[str, Any]], *, seeds: tuple[int, ...] = (41, 42, 43),
                      metric: str = "objective", simplicity_tie: float = 0.01) -> Selection:
    rows = list(results)
    if len(set(seeds)) != 3:
        raise ValueError("Selection requires exactly three distinct seeds")
    grouped: dict[str, list[Mapping[str, Any]]] = {}
    for row in rows:
        grouped.setdefault(str(row["candidate_id"]), []).append(row)
    means: dict[str, float] = {}
    complexity: dict[str, float] = {}
    references: list[str] = []
    expected = set(seeds)
    for identifier, candidates in grouped.items():
        actual = {int(row["seed"]) for row in candidates}
        if actual != expected or len(candidates) != 3:
            raise ValueError(f"Candidate {identifier} does not have exactly seeds {sorted(expected)}")
        means[identifier] = mean(float(row[metric]) for row in candidates)
        complexity[identifier] = float(candidates[0].get("complexity", float("inf")))
        if candidates[0].get("capacity") == "reference":
            references.append(identifier)
    if not means:
        raise ValueError("No candidate results supplied")
    best = min(means, key=lambda key: (means[key], key))
    threshold = means[best] * (1.0 + simplicity_tie)
    eligible = [key for key, value in means.items() if value <= threshold]
    best_simple = min(eligible, key=lambda key: (complexity[key], means[key], key))
    if len(references) != 1:
        raise ValueError("Selection requires exactly one reference-capacity candidate")
    return Selection(best, best_simple, references[0], means)
