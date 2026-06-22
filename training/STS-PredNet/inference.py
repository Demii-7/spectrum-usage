import os
import sys
import argparse
import numpy as np
import torch
from torch.utils.data import DataLoader

from stsprednet import STSPredNet
from utils import get_device, load_checkpoint, denormalize
from dataset import load_csv, reshape_to_3d, compute_minmax_stats, minmax_neg1_pos1
from dataset import STSPredNetDataset, collate_branch_samples, generate_target_indices


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--input", required=True, help="Path to input CSV to predict on")
    parser.add_argument("--output", default="predictions.csv")
    args = parser.parse_args()

    device = get_device("auto")
    ckpt = load_checkpoint(args.checkpoint, device)
    config = ckpt["config"]
    stats = ckpt["norm_stats"]

    dcfg = config["data"]
    bcfg = config["branches"]
    n_nodes = dcfg["n_nodes"]
    bins_per_node = dcfg["bins_per_node"]
    prediction_offset = bcfg.get("prediction_offset", 1)

    use_c = bcfg["use_closeness"]
    use_p = bcfg["use_period"]
    use_t = bcfg["use_trend"]
    lc = bcfg["lc"]
    lp = bcfg["lp"]
    lq = bcfg["lq"]
    period_interval = bcfg["period_interval"]
    trend_interval = bcfg["trend_interval"]

    print(f"Loading input: {args.input}")
    raw = load_csv(args.input)
    data_3d = reshape_to_3d(raw, n_nodes, bins_per_node)

    method = stats.get("method", "minmax_neg1_pos1")
    if method == "minmax_neg1_pos1":
        dmin = stats["dmin"]
        dmax = stats["dmax"]
        data_norm = minmax_neg1_pos1(data_3d, dmin, dmax)
    elif method == "zscore":
        data_norm = ((data_3d - stats["mean"]) / (stats["std"] + 1e-8)).astype(np.float32)
    else:
        data_norm = data_3d.astype(np.float32)

    target_indices = generate_target_indices(
        len(data_3d), prediction_offset,
        use_c, use_p, use_t,
        lc, lp, lq, period_interval, trend_interval,
    )

    if len(target_indices) == 0:
        data_norm = np.tile(data_norm, (2, 1, 1))
        target_indices = generate_target_indices(
            len(data_norm), prediction_offset,
            use_c, use_p, use_t,
            lc, lp, lq, period_interval, trend_interval,
        )
        if len(target_indices) == 0:
            raise ValueError(f"Input too short ({len(data_3d)} steps). Cannot generate any samples.")

    ds = STSPredNetDataset(
        data_norm, target_indices,
        use_c, use_p, use_t,
        lc, lp, lq, period_interval, trend_interval,
        prediction_offset,
    )

    loader = DataLoader(ds, batch_size=32, shuffle=False, collate_fn=collate_branch_samples)

    model = STSPredNet(config).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    all_pred = []
    with torch.no_grad():
        for batch in loader:
            closeness = batch.get("closeness")
            period = batch.get("period")
            trend = batch.get("trend")

            if closeness is not None:
                closeness = closeness.to(device)
            if period is not None:
                period = period.to(device)
            if trend is not None:
                trend = trend.to(device)

            pred = model(closeness, period, trend)
            all_pred.append(pred.cpu())

    pred = torch.cat(all_pred, dim=0)
    B, C, H, W = pred.shape

    if method != "none":
        pred_dbm = denormalize(pred.numpy(), stats)
    else:
        pred_dbm = pred.numpy()

    pred_flat = pred_dbm.reshape(B, -1)
    np.savetxt(args.output, pred_flat, delimiter=",", fmt="%.2f")
    print(f"Predictions saved to {args.output} ({B} samples × {H * W} columns)")


if __name__ == "__main__":
    main()
