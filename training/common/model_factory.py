"""
Shared model construction and checkpoint-management utilities.

This module provides the model-agnostic interface used by the integrated
training and evaluation pipelines to create supported forecasting models,
resolve model-specific configuration, construct checkpoint paths, save trained
state, and restore model weights for later evaluation.

The factory inspects the selected model name and the shape of the loaded
training data to determine the correct input dimensions and architecture
arguments. It isolates model-specific setup from the higher-level training and
evaluation scripts so those scripts can operate through one common interface.

Primary responsibilities include:

- defining the set of model names supported by the integrated pipeline;
- validating model names and required configuration sections;
- deriving feature, frequency, and spatial dimensions from loaded data;
- instantiating sequence-based and map-based forecasting architectures;
- resolving standard or user-overridden checkpoint paths per frequency chunk;
- loading checkpoint dictionaries onto the configured runtime device;
- restoring model parameters and relevant training metadata;
- checking checkpoint compatibility with the current model and dataset;
- validating saved normalization and frequency metadata when available; and
- returning initialized models through a consistent model-agnostic API.

This module does not perform training, forecasting, evaluation, or data
preprocessing. It only creates models and manages their serialized state.
"""


from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn

from models.ARIMA import ARIMAForecaster
from models.ConvLSTM import ConvLSTMForecaster
from models.DSwinLSTM_I import DSwinLSTM_IForecaster
from models.LSTMAttn import LSTMAttnForecaster
from models.LookbackMean import LookbackMeanForecaster
from models.LinearAutoregressive import LinearAutoregressiveForecaster
from models.ResidualLinearAutoregressive import ResidualLinearAutoregressiveForecaster
from models.ResidualConvLSTM import ResidualConvLSTMForecaster
from models.ResidualVanillaLSTM import ResidualVanillaLSTMForecaster
from models.TimeRAN import TimeRANForecaster
from models.TemporalConvNet import TemporalConvNetForecaster
from models.VanillaLSTM import VanillaLSTMForecaster



SUPPORTED_MODELS = {
    "arima",
    "vanillalstm",
    "convlstm",
    "timeran",
    "dswinlstm_i",
    "lookbackmean1d",
    "lookbackmean2d",
    "lookbackmean4d",
    "linearar1d",
    "linearar2d",
    "linearar4d",
    "residuallinearar1d",
    "residuallinearar2d",
    "residuallinearar4d",
    "residualvanillalstm",
    "residualconvlstm",
    "temporalconvnet",
    "lstmattn",
}


#------ Model Building------------

def build_model( model_name: str, config: dict[str, Any], train_data: np.ndarray, ) -> nn.Module:
    """
    Build the selected model.

    Model-specific configuration can be added here as new models
    are integrated into the shared training script.
    """
    model_name = str(model_name).lower()
    model_cfg = config[model_name]["model"]
    print(f"[DEBUG] build_model: model_name={model_name}, train_data.shape={train_data.shape}")

    if model_name in ("temporalconvnet", "lstmattn", "arima"):
        if train_data.ndim != 2:
            raise ValueError(
                f"Error! {model_name} expects training data shaped "
                f"(time, features), got {train_data.shape}"
            )
        predictor_config = {
            "model": {
                **dict(model_cfg),
                "input_size": int(train_data.shape[-1]),
            }
        }
        model_types = {
            "temporalconvnet": TemporalConvNetForecaster,
            "lstmattn": LSTMAttnForecaster,
            "arima": ARIMAForecaster,
        }
        return model_types[model_name](predictor_config)

    if model_name == "vanillalstm":
        if train_data.ndim != 2:
            raise ValueError(
                "Error! VanillaLSTM expects training data shaped "
                f"(time, features), got {train_data.shape}"
            )
    
        model_cfg = config[model_name]["model"]
    
        predictor_config = {
            "model": {
                "input_sequence_length": int(model_cfg["input_sequence_length"]),
                "prediction_horizon": int(model_cfg["prediction_horizon"]),
                
                "input_size": int(train_data.shape[-1]),
                "hidden_size": int(model_cfg.get("hidden_size", 128)),
                "num_layers": int( model_cfg.get("num_layers", 1)),
                
                "dropout": float( model_cfg.get("dropout", 0.0)),
                "output_strategy": str(model_cfg.get("output_strategy", "final_hidden",)),
                "bidirectional": bool(model_cfg.get("bidirectional", False,)
                ),
            },
        }
        return VanillaLSTMForecaster(predictor_config)

    if model_name == "residualvanillalstm":
        if train_data.ndim != 2:
            raise ValueError(
                "Error! ResidualVanillaLSTM expects training data shaped "
                f"(time, features), got {train_data.shape}"
            )

        predictor_config = {
            "model": {
                **dict(model_cfg),
                "input_size": int(train_data.shape[-1]),
                "num_layers": 1,
                "bidirectional": False,
            },
        }
        return ResidualVanillaLSTMForecaster(predictor_config)

    if model_name == "timeran":
        if train_data.ndim != 2:
            raise ValueError(
                "Error! TimeRAN expects training data shaped "
                f"(time, features), got {train_data.shape}"
            )

        predictor_config = {
            "model": {
                **dict(model_cfg),

                "input_sequence_length": int(
                    model_cfg["input_sequence_length"]
                ),
                "prediction_horizon": int(
                    model_cfg["prediction_horizon"]
                ),

                # Derive the number of frequency features from
                # the actual loaded chunk.
                "input_size": int(train_data.shape[-1]),

                "checkpoint_size": str(
                    model_cfg.get("checkpoint_size", "base")
                ).lower(),
                "freeze_encoder": bool(
                    model_cfg.get("freeze_encoder", True)
                ),
                "freeze_embedder": bool(
                    model_cfg.get("freeze_embedder", True)
                ),
                "freeze_head": bool(
                    model_cfg.get("freeze_head", False)
                ),
            },
        }

        return TimeRANForecaster(predictor_config)

    if model_name == "convlstm":
        if train_data.ndim != 4:
            raise ValueError(
                "Error! ConvLSTM expects training map data shaped "
                f"(time, height, width, channels), got {train_data.shape}"
            )
        predictor_config = {
            "model": {
                **dict(model_cfg),
                
                "input_sequence_length": int( model_cfg["input_sequence_length"]),
                "prediction_horizon": int( model_cfg["prediction_horizon"]),
                
                # Derive dimensions from the actual loaded data.
                "input_channels": int(train_data.shape[3]), # Freq
                "grid_height": int(train_data.shape[1]),    # Lat
                "grid_width": int(train_data.shape[2]),     # Long
            },
        }
    
        return ConvLSTMForecaster(predictor_config)

    if model_name == "residualconvlstm":
        if train_data.ndim != 4:
            raise ValueError(
                "Error! ResidualConvLSTM expects training map data shaped "
                f"(time, height, width, channels), got {train_data.shape}"
            )
        predictor_config = {
            "model": {
                **dict(model_cfg),
                "input_sequence_length": int(model_cfg["input_sequence_length"]),
                "prediction_horizon": int(model_cfg["prediction_horizon"]),
                "input_channels": int(train_data.shape[3]),
                "grid_height": int(train_data.shape[1]),
                "grid_width": int(train_data.shape[2]),
            },
        }
        return ResidualConvLSTMForecaster(predictor_config)

    if model_name in ("lookbackmean1d", "lookbackmean2d", "lookbackmean4d"):
        if model_name == "lookbackmean4d" and train_data.ndim != 4:
            raise ValueError(
                "Error! LookbackMean4D expects map data shaped "
                f"(time, height, width, channels), got {train_data.shape}"
            )
        if model_name in ("lookbackmean1d", "lookbackmean2d") and train_data.ndim != 2:
            raise ValueError(
                "Error! LookbackMean1D/2D expects CSV data shaped "
                f"(time, features), got {train_data.shape}"
            )
        return LookbackMeanForecaster(config[model_name])

    if model_name in ("linearar1d", "linearar2d", "linearar4d"):
        if model_name == "linearar4d" and train_data.ndim != 4:
            raise ValueError(
                "Error! LinearAR4D expects map data shaped "
                f"(time, height, width, channels), got {train_data.shape}"
            )
        if model_name in ("linearar1d", "linearar2d") and train_data.ndim != 2:
            raise ValueError(
                "Error! LinearAR1D/2D expects CSV data shaped "
                f"(time, features), got {train_data.shape}"
            )
        predictor_config = {
            "model": {
                **dict(model_cfg),
                "input_size": int(np.prod(train_data.shape[1:])),
            }
        }
        return LinearAutoregressiveForecaster(predictor_config)

    if model_name in (
        "residuallinearar1d",
        "residuallinearar2d",
        "residuallinearar4d",
    ):
        if model_name == "residuallinearar4d" and train_data.ndim != 4:
            raise ValueError(
                "Error! ResidualLinearAR4D expects map data shaped "
                f"(time, height, width, channels), got {train_data.shape}"
            )
        if model_name in ("residuallinearar1d", "residuallinearar2d") and train_data.ndim != 2:
            raise ValueError(
                "Error! ResidualLinearAR1D/2D expects CSV data shaped "
                f"(time, features), got {train_data.shape}"
            )
        predictor_config = {
            "model": {
                **dict(model_cfg),
                "input_size": int(np.prod(train_data.shape[1:])),
            }
        }
        return ResidualLinearAutoregressiveForecaster(predictor_config)

    if model_name == "dswinlstm_i":
        if train_data.ndim != 4:
            raise ValueError(
                "Error! DSwinLSTM-I expects training map data shaped "
                f"(time, height, width, channels), got {train_data.shape}"
            )
        predictor_config = {
            "model": {
                **dict(model_cfg),
                "input_sequence_length": int(model_cfg["input_sequence_length"]),
                "prediction_horizon": int(model_cfg["prediction_horizon"]),
                "map_height": int(train_data.shape[1]),
                "map_width": int(train_data.shape[2]),
                "input_channels": int(train_data.shape[3]),
            },
        }
        print(f"[DEBUG] build_model: DSwinLSTM_I map_height={train_data.shape[1]} map_width={train_data.shape[2]} input_channels={train_data.shape[3]}")
        print(f"[DEBUG] build_model: creating DSwinLSTM_IForecaster ...")
        model = DSwinLSTM_IForecaster(predictor_config)
        print(f"[DEBUG] build_model: DSwinLSTM_IForecaster created, param_count={sum(p.numel() for p in model.parameters())}")
        return model

    raise ValueError(
        f"Unsupported model: {model_name}"
    )


    
# ===========================================================================
# Checkpoint handling
# ===========================================================================

def checkpoint_path_for_chunk(
    *,
    checkpoint_override: Path | None,
    default_checkpoint_directory: Path,
    chunk_id: str,
    model_name: str,
) -> Path:
    """
    Resolve a checkpoint path for one chunk.
    """

    if checkpoint_override is not None:
        return Path(
            str(checkpoint_override).replace(
                "{chunk_id}",
                chunk_id,
            ).replace("{model_name}", model_name)
        )

    return (
        default_checkpoint_directory
        / f"{chunk_id}_{model_name}.pt"
    )


def load_checkpoint_into_model(
    *,
    checkpoint_path: Path,
    model: nn.Module,
    model_name: str,
    data_normalization: dict[str, Any] | None,
    data_frequencies: list[float],
    device: torch.device,
) -> tuple[nn.Module, dict[str, Any]]:
    """
    Load a checkpoint created by the current integrated training script.

    Also validates that checkpoint metadata agrees with the freshly loaded
    evaluation data.
    """

    checkpoint = torch.load(
        checkpoint_path,
        map_location="cpu",
        weights_only=False,
    )

    if "model_state_dict" not in checkpoint:
        raise KeyError(
            f"Checkpoint {checkpoint_path} is missing "
            "'model_state_dict'."
        )

    checkpoint_model_name = str(
        checkpoint.get(
            "model_name",
            model_name,
        )
    ).lower()

    if checkpoint_model_name != model_name:
        raise ValueError(
            f"Checkpoint model {checkpoint_model_name!r} does not "
            f"match configured model {model_name!r}."
        )

    checkpoint_frequencies = checkpoint.get(
        "frequencies"
    )

    if checkpoint_frequencies is not None:
        checkpoint_frequencies_array = np.asarray(
            checkpoint_frequencies,
            dtype=np.float64,
        )

        data_frequencies_array = np.asarray(
            data_frequencies,
            dtype=np.float64,
        )

        if (
            checkpoint_frequencies_array.shape
            != data_frequencies_array.shape
            or not np.allclose(
                checkpoint_frequencies_array,
                data_frequencies_array,
                rtol=0.0,
                atol=1e-6,
            )
        ):
            raise ValueError(
                "Checkpoint frequencies do not match the "
                "currently loaded evaluation frequencies."
            )

    checkpoint_normalization = checkpoint.get(
        "normalization"
    )

    validate_normalization_consistency(
        checkpoint_normalization,
        data_normalization,
    )

    model.load_state_dict(
        checkpoint["model_state_dict"],
        strict=True,
    )

    model = model.to(device)
    model.eval()

    return model, checkpoint


def validate_normalization_consistency(
    checkpoint_normalization: dict[str, Any] | None,
    data_normalization: dict[str, Any] | None,
) -> None:
    """
    Confirm that evaluation preprocessing recreated the normalization used
    during training.
    """

    if (
        checkpoint_normalization is None
        and data_normalization is None
    ):
        return

    if (
        checkpoint_normalization is None
        or data_normalization is None
    ):
        raise ValueError(
            "Checkpoint normalization and evaluation-data "
            "normalization do not agree."
        )

    for key in (
        "mean_dbm",
        "std_dbm",
    ):
        checkpoint_value = np.asarray(
            checkpoint_normalization[key],
            dtype=np.float32,
        )

        data_value = np.asarray(
            data_normalization[key],
            dtype=np.float32,
        )

        if (
            checkpoint_value.shape
            != data_value.shape
            or not np.allclose(
                checkpoint_value,
                data_value,
                rtol=1e-5,
                atol=1e-5,
            )
        ):
            raise ValueError(
                f"Checkpoint normalization value {key!r} "
                "does not match evaluation preprocessing."
            )
