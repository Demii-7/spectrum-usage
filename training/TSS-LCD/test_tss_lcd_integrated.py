from __future__ import annotations

import importlib.util
from pathlib import Path
import sys

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset


HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))


def _load(name: str):
    spec = importlib.util.spec_from_file_location(name, HERE / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


train = _load("train_integrated")
model = _load("model")


def test_nested_and_legacy_config_access() -> None:
    nested = {"tss_lcd": {"model": {"latent_dim": 7}, "train": {"batch_size": 3}}}
    legacy = {"tss_lcd": {"latent_dim": 7, "batch_size": 3}}
    assert train.config_sections(nested)[0]["latent_dim"] == 7
    assert train.config_sections(nested)[1]["batch_size"] == 3
    assert train.config_sections(legacy)[0]["latent_dim"] == 7
    assert train.config_sections(legacy)[1]["batch_size"] == 3


def test_integrated_dataset_preserves_layout_and_masks_deterministically() -> None:
    values = np.ones((12, 4), dtype=np.float32)
    starts = np.asarray([0, 2], dtype=np.int64)
    mask = {"missing_rate": 0.5, "masking_strategy": "random", "zero_pad_missing": True}
    first = train.TSSLCDWindowDataset(values, starts, 3, 2, mask, seed=9)
    second = train.TSSLCDWindowDataset(values, starts, 3, 2, mask, seed=9)
    assert first.X.shape == (2, 3, 1, 4)
    assert first.Y.shape == (2, 2, 1, 4)
    assert torch.equal(first.X, second.X)
    assert torch.equal(first.observation_mask, second.observation_mask)
    assert torch.all(first.X[~first.observation_mask] == 0)


def test_models_require_and_return_btlf() -> None:
    conditioner = model.TSSConditionConstructor(
        T_in=3, L=1, F=4, hidden_dim=4, num_heads=1, num_layers=1,
        ffn_dim=8, dropout=0, latent_dim=4,
    )
    encoder = model.LatentSpaceEncoder(2, 1, 4, 4, num_blocks=1, init_channels=2)
    decoder = model.LatentSpaceDecoder(2, 1, 4, 4, num_blocks=1, init_channels=2)
    x = torch.ones(2, 3, 1, 4)
    y = torch.ones(2, 2, 1, 4)
    assert conditioner(x).shape == (2, 4)
    assert decoder(encoder(y)).shape == y.shape


def test_tss_extractors_use_paper_token_axes() -> None:
    T, L, F = 2, 3, 4
    x = torch.arange(T * L * F, dtype=torch.float32).reshape(1, T, L, F)
    extractors = [
        (model.TemporalFE(T, L, F, 6, 1, 0, 8, 0), x.reshape(1, T, L * F)),
        (
            model.SpectralFE(T, L, F, 6, 1, 0, 8, 0),
            x.permute(0, 3, 2, 1).reshape(1, F, L * T),
        ),
        (
            model.SpatialFE(T, L, F, 6, 1, 0, 8, 0),
            x.permute(0, 2, 3, 1).reshape(1, L, F * T),
        ),
    ]

    for extractor, expected_tokens in extractors:
        captured = []
        handle = extractor.proj.register_forward_pre_hook(
            lambda _module, args, captured=captured: captured.append(args[0].detach().clone())
        )
        output = extractor(x)
        handle.remove()
        assert torch.equal(captured[0], expected_tokens)
        assert output.shape == expected_tokens.shape


def test_ffm_uses_temporal_q_spectral_k_and_reshaped_spatial_v() -> None:
    T, L, F, H = 2, 3, 4, 6
    ffm = model.FeatureFusionModule(T, L, F, H, num_heads=1, dropout=0)
    temporal = torch.arange(T * L * F, dtype=torch.float32).reshape(1, T, L * F)
    spectral = 100 + torch.arange(F * L * T, dtype=torch.float32).reshape(1, F, L * T)
    spatial = 200 + torch.arange(L * F * T, dtype=torch.float32).reshape(1, L, F * T)
    projected_inputs = {}

    for name in ("q_proj", "k_proj", "v_proj"):
        getattr(ffm, name).register_forward_pre_hook(
            lambda _module, args, name=name: projected_inputs.__setitem__(
                name, args[0].detach().clone()
            )
        )

    fused = ffm(temporal, spectral, spatial)
    expected_value = spatial.reshape(1, L, F, T).permute(0, 2, 3, 1).reshape(1, F, T * L)
    assert torch.equal(projected_inputs["q_proj"], temporal)
    assert torch.equal(projected_inputs["k_proj"], spectral)
    assert torch.equal(projected_inputs["v_proj"], expected_value)
    assert fused.shape == (1, T, H)


def test_condition_projection_retains_each_fused_token() -> None:
    projection = model.ConditionToLatentProjection(hidden_dim=2, latent_dim=1, num_tokens=3)
    with torch.no_grad():
        projection.fc.weight.copy_(torch.arange(1, 7, dtype=torch.float32).unsqueeze(0))
        projection.fc.bias.zero_()
    fused = torch.arange(6, dtype=torch.float32).reshape(1, 3, 2)
    assert projection(fused).item() == torch.dot(fused.flatten(), projection.fc.weight.flatten()).item()


def test_conditioner_preserves_branch_ablation_api() -> None:
    x = torch.ones(1, 2, 3, 4)
    for enabled in (
        (True, False, False),
        (False, True, False),
        (False, False, True),
        (True, True, False),
        (True, False, True),
        (False, True, True),
    ):
        conditioner = model.TSSConditionConstructor(
            T_in=2, L=3, F=4, hidden_dim=4, num_heads=1, num_layers=0,
            ffn_dim=8, dropout=0, latent_dim=5,
            use_temporal=enabled[0], use_spectral=enabled[1], use_spatial=enabled[2],
        )
        assert conditioner(x).shape == (1, 5)


def test_validation_forecast_metrics_are_physical_db() -> None:
    class Diffusion:
        def ddim_sample_loop(self, cond_z, steps, generator):
            return torch.zeros(2, 2, 1, 1)

    loader = DataLoader(
        TensorDataset(torch.zeros(2, 3, 1, 1), torch.ones(2, 2, 1, 1)),
        batch_size=2,
    )
    metrics = train.validation_forecast_metrics(
        torch.nn.Identity(), torch.nn.Identity(), Diffusion(), loader, [1, 2],
        {"mean_dbm": np.array([[-100.0]], dtype=np.float32),
         "std_dbm": np.array([[2.0]], dtype=np.float32)},
        torch.device("cpu"), sampler_steps=2, sampler_seed=7,
    )
    assert metrics["val_mae_db_t1"] == 2.0
    assert metrics["val_mae_db_t2"] == 2.0
    assert metrics["val_mean_horizon_mae_db"] == 2.0
