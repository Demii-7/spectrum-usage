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
SUPPORTED_MODELS = {
    "arima",
    "vanillalstm",
    "convlstm",
    "convlstmfm",
    "timeran",
    "dswinlstm_i",
    "lookbackmean1d",
    "lookbackmean2d",
    "lookbackmean4d",
    "vanillalstm1d",
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
    "autoformer_csa",
    "stsprednet",
    "tss_lcd",
    "deepspred",
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
        from models.ARIMA import ARIMAForecaster
        from models.LSTMAttn import LSTMAttnForecaster
        from models.TemporalConvNet import TemporalConvNetForecaster

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

    if model_name in ("vanillalstm", "vanillalstm1d"):
        from models.VanillaLSTM import VanillaLSTMForecaster

        if train_data.ndim != 2:
            raise ValueError(
                "Error! VanillaLSTM expects training data shaped "
                f"(time, features), got {train_data.shape}"
            )

        if model_name == "vanillalstm1d" and train_data.shape[1] != 1:
            raise ValueError(
                "Error! VanillaLSTM1D expects exactly one scalar feature, "
                f"got {train_data.shape[1]}"
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
        from models.ResidualVanillaLSTM import ResidualVanillaLSTMForecaster

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
        from models.TimeRAN import TimeRANForecaster

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
        from models.ConvLSTM import ConvLSTMForecaster

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

    if model_name == "convlstmfm":
        from models.ConvLSTM_FM import ConvLSTMFMForecaster

        if train_data.ndim != 4:
            raise ValueError(
                "Error! ConvLSTM-FM expects training map data shaped "
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
        pretrain_epochs = int(model_cfg.get("pretrain_epochs", 0))
        if pretrain_epochs > 0 and bool(model_cfg.get("freeze_backbone", False)):
            raise ValueError(
                "freeze_backbone conflicts with ConvLSTM-FM pretraining; use "
                "freeze_backbone_after_pretrain instead."
            )
        if bool(model_cfg.get("freeze_backbone_after_pretrain", False)) and pretrain_epochs <= 0:
            raise ValueError("freeze_backbone_after_pretrain requires pretrain_epochs > 0")
        return ConvLSTMFMForecaster(predictor_config)

    if model_name == "residualconvlstm":
        from models.ResidualConvLSTM import ResidualConvLSTMForecaster

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
        from models.LookbackMean import LookbackMeanForecaster

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
        from models.LinearAutoregressive import LinearAutoregressiveForecaster

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
        if model_name == "linearar1d" and train_data.shape[1] != 1:
            raise ValueError(
                "Error! LinearAR1D expects exactly one feature, "
                f"got {train_data.shape[1]}"
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
        from models.ResidualLinearAutoregressive import ResidualLinearAutoregressiveForecaster

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
        if model_name == "residuallinearar1d" and train_data.shape[1] != 1:
            raise ValueError(
                "Error! ResidualLinearAR1D expects exactly one feature, "
                f"got {train_data.shape[1]}"
            )
        predictor_config = {
            "model": {
                **dict(model_cfg),
                "input_size": int(np.prod(train_data.shape[1:])),
            }
        }
        return ResidualLinearAutoregressiveForecaster(predictor_config)

    if model_name == "dswinlstm_i":
        from models.DSwinLSTM_I import DSwinLSTM_IForecaster

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

    if model_name == "autoformer_csa":
        from models.AutoformerCSA import AutoformerCSAForecaster, DotConfig

        if train_data.ndim != 2:
            raise ValueError(
                "Error! Autoformer-CSA expects training data "
                "shaped (time, features), "
                f"got {train_data.shape}"
            )

        input_sequence_length = int(
            model_cfg["input_sequence_length"]
        )

        prediction_horizon = int(
            model_cfg["prediction_horizon"]
        )

        label_len = int(
            model_cfg["label_len"]
        )

        if label_len > input_sequence_length:
            raise ValueError(
                "Error! Autoformer-CSA label_len cannot exceed "
                "input_sequence_length. "
                f"Got label_len={label_len} and "
                f"input_sequence_length="
                f"{input_sequence_length}."
            )

        predictor_config = DotConfig(
            {
                # Shared-pipeline names translated to
                # Autoformer implementation names.
                "seq_len": input_sequence_length,
                "label_len": label_len,
                "pred_len": prediction_horizon,

                # Derived from the loaded frequency chunk.
                "enc_in": int(
                    train_data.shape[-1]
                ),
                "dec_in": int(
                    train_data.shape[-1]
                ),
                "c_out": int(
                    train_data.shape[-1]
                ),

                # Architecture configuration.
                "d_model": int(
                    model_cfg["d_model"]
                ),
                "d_ff": int(
                    model_cfg["d_ff"]
                ),
                "e_layers": int(
                    model_cfg["encoder_layers"]
                ),
                "d_layers": int(
                    model_cfg["decoder_layers"]
                ),
                "n_heads": int(
                    model_cfg["n_heads"]
                ),
                "moving_avg": int(
                    model_cfg["moving_avg"]
                ),
                "dropout": float(
                    model_cfg["dropout"]
                ),
                "factor": int(
                    model_cfg["factor"]
                ),
                "csam_kernel_size": int(
                    model_cfg["csam_kernel_size"]
                ),
                "output_attention": bool(
                    model_cfg.get(
                        "output_attention",
                        False,
                    )
                ),
            }
        )

        return AutoformerCSAForecaster(
            predictor_config
        )

    if model_name == "tss_lcd":
        from models.TSSLCD import (
            DiffusionModel,
            LatentSpaceDecoder,
            LatentSpaceEncoder,
            TSSConditionConstructor,
            TSSLCDForecaster,
        )

        if train_data.ndim != 2:
            raise ValueError(
                "Error! TSS-LCD expects training "
                "data shaped (time, frequencies), "
                f"got {train_data.shape}"
            )

        input_sequence_length = int(
            model_cfg["input_sequence_length"]
        )

        prediction_horizon = int(
            model_cfg["prediction_horizon"]
        )

        input_size = int(
            train_data.shape[-1]
        )

        spatial_locations = 1

        encoder = LatentSpaceEncoder(
            T_out=prediction_horizon,
            L=spatial_locations,
            F=input_size,
            latent_dim=int(
                model_cfg["latent_dim"]
            ),
            num_blocks=int(
                model_cfg.get(
                    "autoencoder_num_blocks", 3
                )
            ),
            init_channels=int(
                model_cfg.get(
                    "autoencoder_initial_channels", 32
                )
            ),
            kernel_size=int(
                model_cfg.get(
                    "autoencoder_kernel_size", 3
                )
            ),
            pool_kernel=int(
                model_cfg.get(
                    "autoencoder_pool_kernel", 2
                )
            ),
            pool_stride=int(
                model_cfg.get(
                    "autoencoder_pool_stride", 2
                )
            ),
            activation=str(
                model_cfg.get(
                    "autoencoder_activation", "relu"
                )
            ),
        )

        decoder = LatentSpaceDecoder(
            T_out=prediction_horizon,
            L=spatial_locations,
            F=input_size,
            latent_dim=int(
                model_cfg["latent_dim"]
            ),
            num_blocks=int(
                model_cfg.get(
                    "autoencoder_num_blocks", 3
                )
            ),
            init_channels=int(
                model_cfg.get(
                    "autoencoder_initial_channels", 32
                )
            ),
            kernel_size=int(
                model_cfg.get(
                    "autoencoder_kernel_size", 3
                )
            ),
            activation=str(
                model_cfg.get(
                    "autoencoder_activation", "relu"
                )
            ),
        )

        condition_constructor = TSSConditionConstructor(
            T_in=input_sequence_length,
            L=spatial_locations,
            F=input_size,
            hidden_dim=int(
                model_cfg.get("hidden_dim", 256)
            ),
            num_heads=int(
                model_cfg.get("attention_heads", 4)
            ),
            num_layers=int(
                model_cfg.get(
                    "num_attention_layers", 2
                )
            ),
            ffn_dim=int(
                model_cfg.get("ffn_dim", 1024)
            ),
            dropout=float(
                model_cfg.get("dropout", 0.1)
            ),
            latent_dim=int(
                model_cfg["latent_dim"]
            ),
            use_temporal=bool(
                model_cfg.get(
                    "use_temporal_branch", True
                )
            ),
            use_spectral=bool(
                model_cfg.get(
                    "use_spectral_branch", True
                )
            ),
            use_spatial=bool(
                model_cfg.get(
                    "use_spatial_branch", True
                )
            ),
        )

        diffusion = DiffusionModel(
            latent_dim=int(
                model_cfg["latent_dim"]
            ),
            n_timestep=int(
                model_cfg.get(
                    "diffusion_steps", 1000
                )
            ),
            device=torch.device("cpu"),
            noise_schedule=str(
                model_cfg.get(
                    "noise_schedule", "cosine"
                )
            ),
            nen_encoder_channels=list(
                model_cfg.get(
                    "nen_encoder_channels", [64, 128]
                )
            ),
            nen_bottleneck_channels=int(
                model_cfg.get(
                    "nen_bottleneck_channels", 256
                )
            ),
            nen_decoder_channels=list(
                model_cfg.get(
                    "nen_decoder_channels", [128, 64]
                )
            ),
            nen_kernel_size=int(
                model_cfg.get("nen_kernel_size", 3)
            ),
            time_embed_dim=int(
                model_cfg.get("time_embed_dim", 32)
            ),
            condition_proj_dim=model_cfg.get(
                "condition_proj_dim"
            ),
            condition_strategy=str(
                model_cfg.get(
                    "condition_strategy", "concat"
                )
            ),
            nen_activation=str(
                model_cfg.get(
                    "nen_activation", "relu"
                )
            ),
            nen_normalization=str(
                model_cfg.get(
                    "nen_normalization", "batchnorm"
                )
            ),
        )

        return TSSLCDForecaster(
            encoder=encoder,
            decoder=decoder,
            condition_constructor=condition_constructor,
            diffusion=diffusion,
            input_sequence_length=input_sequence_length,
            prediction_horizon=prediction_horizon,
            input_size=input_size,
        )

    if model_name == "stsprednet":
        from models.STSPredNet import (
            STSPredNetForecaster,
        )

        if train_data.ndim != 4:
            raise ValueError(
                "STS-PredNet expects map data shaped "
                "(time, height, width, frequencies), "
                f"got {train_data.shape}."
            )

        predictor_config = {
            "model": {
                **dict(model_cfg),
                "map_height": int(train_data.shape[1]),
                "map_width": int(train_data.shape[2]),
                "input_channels": int(train_data.shape[3]),
            },
            "branches": dict(config["stsprednet"]["branches"]),
        }

        return STSPredNetForecaster(predictor_config)

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
