"""Integrated model registry with safe architecture bundles and policies."""

from __future__ import annotations

from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Any, Mapping

from .spaces import Domain, choice, loguniform


PRIMARY_OBJECTIVE = "val_mean_horizon_mae_db"
FALLBACK_OBJECTIVE = "val_loss"


@dataclass(frozen=True)
class Resources:
    cpu: float
    gpu: float
    memory: int | None = None

    def as_ray(self) -> dict[str, float | int]:
        values: dict[str, float | int] = {"cpu": self.cpu, "gpu": self.gpu}
        if self.memory is not None:
            values["memory"] = self.memory
        return values


@dataclass(frozen=True)
class ModelSpec:
    name: str
    status: str
    policy: str
    representation: str
    anchors: Mapping[str, Mapping[str, Any]]
    space: Mapping[str, Domain]
    resources: Resources
    reason: str | None = None
    objective: str = PRIMARY_OBJECTIVE
    fallback_objective: str | None = FALLBACK_OBJECTIVE
    historical: Mapping[str, Mapping[str, Any]] = field(default_factory=dict)

    @property
    def executable(self) -> bool:
        """Whether integrated execution is allowed, including reference runs."""
        return self.status in {"tunable", "reference_only"}

    @property
    def hpo_executable(self) -> bool:
        return self.status == "tunable"

    def anchor(self, capacity: str) -> dict[str, Any]:
        try:
            return dict(self.anchors[capacity])
        except KeyError as exc:
            raise KeyError(f"{self.name} has no {capacity!r} anchor") from exc


def _freeze_anchors(tiny: Mapping[str, Any], small: Mapping[str, Any], reference: Mapping[str, Any]):
    return MappingProxyType({
        "tiny": MappingProxyType(dict(tiny)),
        "small": MappingProxyType(dict(small)),
        "reference": MappingProxyType(dict(reference)),
    })


def _bundle(**paths: Any) -> dict[str, Any]:
    return paths


_OPTIMIZER_2D = {
    "train.learning_rate": loguniform(3e-4, 3e-3),
    "train.weight_decay": choice(0.0, 1e-5, 1e-4),
    "train.batch_size": choice(32, 64, 128),
}

_OPTIMIZER_MAP = {
    "train.learning_rate": loguniform(3e-5, 5e-4),
    "train.weight_decay": choice(0.0, 1e-4, 1e-3, 4e-3),
    "train.batch_size": choice(1, 2, 4),
}


def _tunable(name: str, representation: str, bundles: tuple[dict[str, Any], ...],
              *, gpu: float, optimizer_space: Mapping[str, Domain],
              anchor_optimizer: Mapping[str, Any], cpu: float = 2.0,
              extra_space: Mapping[str, Domain] | None = None,
              historical: Mapping[str, Mapping[str, Any]] | None = None) -> ModelSpec:
    tiny, small, reference = (
        {"architecture": bundle, **dict(anchor_optimizer)} for bundle in bundles
    )
    historical = dict(historical or {})
    historical_bundles = tuple(
        dict(point["architecture"])
        for point in historical.values()
        if point.get("architecture") not in bundles
    )
    return ModelSpec(
        name=name,
        status="tunable",
        policy="asha",
        representation=representation,
        anchors=_freeze_anchors(tiny, small, reference),
        space=MappingProxyType({
            "architecture": choice(*bundles, *historical_bundles),
            **dict(optimizer_space),
            **dict(extra_space or {}),
        }),
        resources=Resources(cpu, gpu),
        historical=MappingProxyType(historical),
    )


def _reference(name: str, representation: str, *, reason: str) -> ModelSpec:
    return ModelSpec(
        name=name,
        status="reference_only",
        policy="single_reference",
        representation=representation,
        anchors=_freeze_anchors({}, {}, {}),
        space=MappingProxyType({}),
        resources=Resources(1.0, 0.0),
        reason=reason,
    )


def _blocked(name: str, representation: str, reason: str, *, gpu: float = 1.0) -> ModelSpec:
    return ModelSpec(
        name=name,
        status="not_publication_ready",
        policy="blocked_metric",
        representation=representation,
        anchors=_freeze_anchors({}, {}, {}),
        space=MappingProxyType({}),
        resources=Resources(4.0, gpu),
        reason=reason,
        fallback_objective=None,
    )


_specs = [
    _reference("arima", "2d", reason="Classical reference; integrated training does not optimize an HPO parameter."),
    _tunable("vanillalstm", "2d", (
        _bundle(**{"model.hidden_size": 8, "model.num_layers": 1, "model.dropout": 0.0}),
        _bundle(**{"model.hidden_size": 32, "model.num_layers": 1, "model.dropout": 0.0}),
        _bundle(**{"model.hidden_size": 128, "model.num_layers": 1, "model.dropout": 0.0}),
    ), gpu=0.5, optimizer_space=_OPTIMIZER_2D,
        anchor_optimizer={"train.learning_rate": 1e-3, "train.weight_decay": 1e-4, "train.batch_size": 128},
        historical={"summary70_residual_family": {
            "architecture": _bundle(**{"model.hidden_size": 128, "model.num_layers": 1, "model.dropout": 0.1}),
            "train.learning_rate": 1e-3, "train.weight_decay": 0.0, "train.batch_size": 32,
        }}),
    _tunable("residualvanillalstm", "2d", (
        _bundle(**{"model.hidden_size": 8, "model.num_layers": 1, "model.dropout": 0.0}),
        _bundle(**{"model.hidden_size": 16, "model.num_layers": 1, "model.dropout": 0.0}),
        _bundle(**{"model.hidden_size": 64, "model.num_layers": 1, "model.dropout": 0.0}),
    ), gpu=0.5, optimizer_space=_OPTIMIZER_2D,
        anchor_optimizer={"train.learning_rate": 1e-3, "train.weight_decay": 1e-4, "train.batch_size": 128},
        historical={"summary70_fast_2d": {
            "architecture": _bundle(**{"model.hidden_size": 32, "model.num_layers": 1, "model.dropout": 0.0}),
            "train.learning_rate": 1e-3, "train.weight_decay": 1e-4, "train.batch_size": 32,
        }}),
    _tunable("temporalconvnet", "2d", (
        _bundle(**{"model.hidden_channels": [8] * 6, "model.kernel_size": 2, "model.dropout": 0.0, "model.feature_mode": "joint"}),
        _bundle(**{"model.hidden_channels": [16] * 6, "model.kernel_size": 2, "model.dropout": 0.1, "model.feature_mode": "joint"}),
        _bundle(**{"model.hidden_channels": [32, 32, 32, 32, 32, 32], "model.kernel_size": 2, "model.dropout": 0.1, "model.feature_mode": "joint"}),
    ), gpu=0.5, optimizer_space=_OPTIMIZER_2D,
        anchor_optimizer={"train.learning_rate": 1e-3, "train.weight_decay": 0.0, "train.batch_size": 128},
        historical={"summary70_new_models": {
            "architecture": _bundle(**{"model.hidden_channels": [32] * 6, "model.kernel_size": 2, "model.dropout": 0.1, "model.feature_mode": "joint"}),
            "train.learning_rate": 1e-3, "train.weight_decay": 0.0, "train.batch_size": 32,
        }}),
    _tunable("lstmattn", "2d", (
        _bundle(**{"model.hidden_size": 16, "model.attention_size": 16, "model.num_layers": 1, "model.dropout": 0.0}),
        _bundle(**{"model.hidden_size": 64, "model.attention_size": 64, "model.num_layers": 1, "model.dropout": 0.1}),
        _bundle(**{"model.hidden_size": 128, "model.attention_size": 128, "model.num_layers": 1, "model.dropout": 0.1}),
    ), gpu=0.5, optimizer_space=_OPTIMIZER_2D,
        anchor_optimizer={"train.learning_rate": 1e-3, "train.weight_decay": 0.0, "train.batch_size": 128},
        historical={"summary70_new_models": {
            "architecture": _bundle(**{"model.hidden_size": 128, "model.attention_size": 128, "model.num_layers": 1, "model.dropout": 0.1}),
            "train.learning_rate": 1e-3, "train.weight_decay": 0.0, "train.batch_size": 32,
        }}),
    _tunable("autoformer_csa", "2d", (
        _bundle(**{"model.d_model": 32, "model.d_ff": 128, "model.encoder_layers": 1, "model.decoder_layers": 1, "model.n_heads": 1, "model.moving_avg": 5, "model.dropout": 0.05, "model.factor": 3, "model.csam_kernel_size": 3}),
        _bundle(**{"model.d_model": 128, "model.d_ff": 512, "model.encoder_layers": 2, "model.decoder_layers": 1, "model.n_heads": 4, "model.moving_avg": 13, "model.dropout": 0.05, "model.factor": 3, "model.csam_kernel_size": 5}),
        _bundle(**{"model.d_model": 64, "model.d_ff": 256, "model.encoder_layers": 2, "model.decoder_layers": 1, "model.n_heads": 8, "model.moving_avg": 25, "model.dropout": 0.05, "model.factor": 3, "model.csam_kernel_size": 7}),
    ), gpu=0.5,
        optimizer_space={
            "train.learning_rate": loguniform(3e-5, 3e-4),
            "train.weight_decay": choice(0.0, 1e-5, 1e-4),
            "train.batch_size": choice(8, 16, 32),
        },
        anchor_optimizer={"train.learning_rate": 1e-4, "train.weight_decay": 0.0, "train.batch_size": 8},
        historical={"summary70_residual_family": {
            "architecture": _bundle(**{"model.d_model": 64, "model.d_ff": 256, "model.encoder_layers": 2, "model.decoder_layers": 1, "model.n_heads": 8, "model.moving_avg": 25, "model.dropout": 0.05, "model.factor": 3, "model.csam_kernel_size": 7}),
            "train.learning_rate": 1e-4, "train.weight_decay": 0.0, "train.batch_size": 32,
        }}),
    _tunable("convlstm", "4d", (
        _bundle(**{"model.hidden_channels": [8], "model.num_encoder_layers": 1, "model.kernel_size": [[1, 3]], "model.decoder_hidden_channels": 8, "model.decoder_kernel_size": [1, 1], "model.decoder_lstm_hidden": 32, "model.dropout": 0.1}),
        _bundle(**{"model.hidden_channels": [16, 32], "model.num_encoder_layers": 2, "model.kernel_size": [[1, 3], [1, 1]], "model.decoder_hidden_channels": 16, "model.decoder_kernel_size": [1, 1], "model.decoder_lstm_hidden": 64, "model.dropout": 0.2}),
        _bundle(**{"model.hidden_channels": [32, 64], "model.num_encoder_layers": 2, "model.kernel_size": [[1, 3], [1, 1]], "model.decoder_hidden_channels": 32, "model.decoder_kernel_size": [1, 1], "model.decoder_lstm_hidden": 128, "model.dropout": 0.3}),
    ), gpu=0.5, optimizer_space=_OPTIMIZER_MAP,
        anchor_optimizer={"train.learning_rate": 2e-4, "train.weight_decay": 4e-3, "train.batch_size": 4},
        extra_space={"train.optimizer": choice("adam", "adamw")},
        historical={"summary70_c5_adamw": {
            "architecture": _bundle(**{"model.hidden_channels": [16, 32], "model.num_encoder_layers": 2, "model.kernel_size": [[1, 3], [1, 1]], "model.decoder_hidden_channels": 16, "model.decoder_kernel_size": [1, 1], "model.decoder_lstm_hidden": 64, "model.dropout": 0.2}),
            "train.learning_rate": 3e-4, "train.weight_decay": 1e-4, "train.batch_size": 4,
            "train.optimizer": "adamw",
        }}),
    _tunable("residualconvlstm", "4d", (
        _bundle(**{"model.hidden_channels": [4], "model.num_encoder_layers": 1, "model.kernel_size": [[1, 3]], "model.decoder_hidden_channels": 4, "model.decoder_kernel_size": [1, 1], "model.decoder_lstm_hidden": 16, "model.dropout": 0.0}),
        _bundle(**{"model.hidden_channels": [8], "model.num_encoder_layers": 1, "model.kernel_size": [[1, 3]], "model.decoder_hidden_channels": 8, "model.decoder_kernel_size": [1, 1], "model.decoder_lstm_hidden": 32, "model.dropout": 0.1}),
        _bundle(**{"model.hidden_channels": [16], "model.num_encoder_layers": 1, "model.kernel_size": [[1, 3]], "model.decoder_hidden_channels": 16, "model.decoder_kernel_size": [1, 1], "model.decoder_lstm_hidden": 64, "model.dropout": 0.1}),
    ), gpu=0.5, optimizer_space=_OPTIMIZER_MAP,
        anchor_optimizer={"train.learning_rate": 2e-4, "train.weight_decay": 4e-3, "train.batch_size": 4},
        historical={"summary70_better_residual": {
            "architecture": _bundle(**{"model.hidden_channels": [16, 32], "model.num_encoder_layers": 2, "model.kernel_size": [[1, 3], [1, 1]], "model.decoder_hidden_channels": 16, "model.decoder_kernel_size": [1, 1], "model.decoder_lstm_hidden": 64, "model.dropout": 0.1}),
            "train.learning_rate": 2e-4, "train.weight_decay": 4e-3, "train.batch_size": 4,
        }}),
    _tunable("convlstmfm", "4d", (
        _bundle(**{"model.hidden_channels": [8], "model.num_layers": 1, "model.kernel_size": [[3, 3]], "model.dropout": 0.0, "model.pretrain_epochs": 2, "model.pretrain_mask_ratio": 0.1}),
        _bundle(**{"model.hidden_channels": [16, 16], "model.num_layers": 2, "model.kernel_size": [[3, 3], [3, 3]], "model.dropout": 0.1, "model.pretrain_epochs": 10, "model.pretrain_mask_ratio": 0.2}),
        _bundle(**{"model.hidden_channels": [64, 64, 64, 64, 64], "model.num_layers": 5, "model.kernel_size": [[3, 3]] * 5, "model.dropout": 0.1, "model.pretrain_epochs": 20, "model.pretrain_mask_ratio": 0.2}),
    ), gpu=0.5, optimizer_space=_OPTIMIZER_MAP,
        anchor_optimizer={"train.learning_rate": 1e-4, "train.weight_decay": 4e-3, "train.batch_size": 2}),
    _tunable("dswinlstm_i", "4d", (
        _bundle(**{"model.embed_dim": 8, "model.hidden_dims": [8, 16], "model.encoder_units": 1, "model.decoder_units": 1, "model.swin_depths": [1, 1, 1, 1], "model.num_heads": [1, 2, 2, 1], "model.window_size": 2}),
        _bundle(**{"model.embed_dim": 16, "model.hidden_dims": [16, 32], "model.encoder_units": 1, "model.decoder_units": 1, "model.swin_depths": [1, 1, 1, 1], "model.num_heads": [1, 2, 2, 1], "model.window_size": 2}),
        _bundle(**{"model.embed_dim": 32, "model.hidden_dims": [32, 64], "model.encoder_units": 2, "model.decoder_units": 2, "model.swin_depths": [1, 1, 1, 1], "model.num_heads": [2, 4, 4, 2], "model.window_size": 4}),
    ), gpu=0.5, optimizer_space=_OPTIMIZER_MAP,
        anchor_optimizer={"train.learning_rate": 1e-4, "train.weight_decay": 0.0, "train.batch_size": 2}),
]

for family in ("lookbackmean", "linearar", "residuallinearar"):
    for dimension in ("1d", "2d", "4d"):
        reason = "Parameter-free reference." if family == "lookbackmean" else (
            "Reference only; no architecture parameter implemented for HPO."
        )
        _specs.append(_reference(f"{family}{dimension}", dimension, reason=reason))

_specs.extend([
    _blocked("stsprednet", "2d_or_4d", "Integrated callback lacks a comparable physical dB validation metric."),
    _blocked("tss_lcd", "2d_or_4d", "Multi-stage objective lacks a comparable physical forecasting dB metric."),
    ModelSpec(
        "deepspred", "not_publication_ready", "blocked_exact_minute", "spectrogram",
        _freeze_anchors({}, {}, {}), MappingProxyType({}), Resources(4.0, 1.0),
        "DeepSPred frame output cannot currently provide exact-minute horizon evaluation.",
        fallback_objective=None,
    ),
])

MODEL_REGISTRY: Mapping[str, ModelSpec] = MappingProxyType({spec.name: spec for spec in _specs})


def get_model_spec(name: str, *, require_executable: bool = False, require_hpo: bool = False) -> ModelSpec:
    normalized = name.lower()
    if normalized == "timeran":
        raise KeyError("timeran is intentionally excluded from Ray tuning")
    try:
        spec = MODEL_REGISTRY[normalized]
    except KeyError as exc:
        raise KeyError(f"Unknown integrated tuning model: {name!r}") from exc
    if require_executable and not spec.executable:
        raise RuntimeError(spec.reason or f"{normalized} is not executable")
    if require_hpo and not spec.hpo_executable:
        raise RuntimeError(spec.reason or f"{normalized} does not support HPO")
    return spec
