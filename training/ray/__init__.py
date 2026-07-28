"""Ray Tune integration for the shared training pipeline.

Importing this package does not require Ray. Callers only need Ray when they
materialize a search space, create a scheduler, probe a cluster, or run Tune.
"""

from .candidate import candidate_id
from .config import ConfigurationError, inject_parameters, validate_tuning_config
from .registry import MODEL_REGISTRY, ModelSpec, get_model_spec
from .selection import Selection, select_candidates

__all__ = [
    "ConfigurationError",
    "MODEL_REGISTRY",
    "ModelSpec",
    "Selection",
    "candidate_id",
    "get_model_spec",
    "inject_parameters",
    "select_candidates",
    "validate_tuning_config",
]
