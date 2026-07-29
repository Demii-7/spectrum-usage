"""Safe parameter injection into integrated YAML configurations."""

from __future__ import annotations

from copy import deepcopy
from typing import Any, Mapping

from .registry import ModelSpec, get_model_spec


class ConfigurationError(ValueError):
    pass


_DERIVED_KEYS = {
    "input_size", "input_channels", "grid_height", "grid_width", "map_height", "map_width",
    "enc_in", "dec_in", "c_out",
}


def _canonical(value: Any) -> Any:
    """Normalize Ray-serialized nested bundles for stable equality checks."""
    if isinstance(value, Mapping):
        return tuple(sorted((str(key), _canonical(item)) for key, item in value.items()))
    if isinstance(value, (list, tuple)):
        return tuple(_canonical(item) for item in value)
    return value


def _set_path(root: dict[str, Any], path: str, value: Any) -> None:
    parts = path.split(".")
    if not parts or any(not part for part in parts):
        raise ConfigurationError(f"Invalid parameter path: {path!r}")
    if parts[-1] in _DERIVED_KEYS:
        raise ConfigurationError(f"{path} is derived from data and cannot be tuned")
    current = root
    for part in parts[:-1]:
        child = current.setdefault(part, {})
        if not isinstance(child, dict):
            raise ConfigurationError(f"Cannot inject {path!r}; {part!r} is not a mapping")
        current = child
    current[parts[-1]] = deepcopy(value)


def validate_parameters(spec: ModelSpec, parameters: Mapping[str, Any]) -> None:
    unknown = set(parameters) - set(spec.space)
    if unknown:
        raise ConfigurationError(f"Parameters outside {spec.name} search space: {sorted(unknown)}")
    invalid = []
    for name, value in parameters.items():
        domain = spec.space[name]
        if name == "architecture" and isinstance(value, Mapping) and domain.kind == "choice":
            valid = any(_canonical(value) == _canonical(choice) for choice in domain.choices)
        else:
            valid = domain.contains(value)
        if not valid:
            invalid.append(name)
    if invalid:
        raise ConfigurationError(f"Values outside bounded search domains: {sorted(invalid)}")


def validate_tuning_config(config: Mapping[str, Any], model_name: str) -> None:
    spec = get_model_spec(model_name, require_executable=True)
    if model_name not in config:
        raise ConfigurationError(f"Config is missing model section {model_name!r}")
    ranges = ((config.get("data") or {}).get("split") or {}).get("ranges")
    if not isinstance(ranges, Mapping) or set(ranges) != {"train", "validation", "test"}:
        raise ConfigurationError(
            "Ray tuning requires explicit data.split.ranges with train, validation, and test; "
            "validation must not be inferred from the test partition."
        )
    training = config.get("training") or {}
    configured = training.get("models", [training.get("model_name")])
    if model_name not in [str(value).lower() for value in configured if value is not None]:
        raise ConfigurationError(f"training.models/model_name must select {model_name!r}")
    if spec.representation in {"1d", "2d", "4d"}:
        actual = str((config.get("data") or {}).get("representation", "")).lower()
        if actual != spec.representation:
            raise ConfigurationError(f"{model_name} requires {spec.representation} data, got {actual!r}")


def inject_parameters(config: Mapping[str, Any], model_name: str, parameters: Mapping[str, Any],
                      *, seed: int, validate: bool = True) -> dict[str, Any]:
    spec = get_model_spec(model_name, require_executable=True)
    validate_parameters(spec, parameters)
    result = deepcopy(dict(config))
    model_section = result.setdefault(model_name, {})
    for path, value in parameters.items():
        if path == "architecture":
            if not isinstance(value, Mapping):
                raise ConfigurationError("architecture must be a mapping of coupled parameter paths")
            for architecture_path, architecture_value in value.items():
                _set_path(model_section, str(architecture_path), architecture_value)
        else:
            _set_path(model_section, path, value)
    # Specialized models keep training keys at their root; regular models use train.seed.
    seed_path = "seed" if model_name in {"stsprednet", "tss_lcd"} else "train.seed"
    _set_path(model_section, seed_path, int(seed))
    if model_name not in {"stsprednet", "tss_lcd"}:
        _set_path(model_section, "train.selection_metric", spec.objective)
        _set_path(model_section, "train.early_stopping", False)
    result.setdefault("training", {})["models"] = [model_name]
    result["training"].pop("model_name", None)
    if validate:
        validate_tuning_config(result, model_name)
    return result
