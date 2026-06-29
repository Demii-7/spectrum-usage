"""
Training script for Autoformer-CSA / AutoformerVanilla models.

Loads spectrum-usage CSV data, creates train/val/test windows, builds the
model, runs the training loop with early stopping, and evaluates the best
checkpoint on the test set.  Metrics, predictions, and diagnostic plots are
saved to the evaluation directory.
"""

import os, sys, argparse, json, copy
import numpy as np
import torch
import torch.nn as nn
import yaml

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from dataset import create_datasets
from utils import (
    set_seed, get_device, compute_metrics, compute_metrics_per_horizon,
    compute_metrics_per_node, save_checkpoint, load_checkpoint,
    denormalize, save_metrics_json, save_csv,
    plot_spectrogram_comparison, plot_error_analysis,
)


class DotDict(dict):
    """Dictionary subclass that allows attribute-style access (dot notation).

    Enables writing ``cfg.key`` instead of ``cfg["key"]``, which is
    convenient for passing configuration values to model constructors.
    """

    def __getattr__(self, k):
        try:
            return self[k]
        except KeyError:
            raise AttributeError(k)
    def __setattr__(self, k, v):
        self[k] = v
    def __delattr__(self, k):
        del self[k]


def build_model(config, device):
    """Construct model instance from a YAML configuration dict.

    Reads architecture variant (``autoformer_csa`` or ``autoformer``) and
    model hyper-parameters from *config*, builds a DotDict for the model
    constructor, and moves the model to *device*.

    Returns:
        Tuple of (model, model_cfg_dotdict).
    """
    arch = config.get("architecture", {})
    variant = arch.get("model_variant", "autoformer_csa")

    cfg = config["model"]
    windowing = config["windowing"]
    model_cfg = DotDict({
        "seq_len": windowing["seq_len"],
        "label_len": windowing["label_len"],
        "pred_len": windowing["pred_len"],
        "enc_in": cfg["enc_in"],
        "dec_in": cfg["dec_in"],
        "c_out": cfg["c_out"],
        "d_model": cfg["d_model"],
        "d_ff": cfg.get("d_ff", 4 * cfg["d_model"]),
        "e_layers": cfg["encoder_layers"],
        "d_layers": cfg["decoder_layers"],
        "n_heads": cfg["n_heads"],
        "moving_avg": cfg["moving_avg"],
        "dropout": cfg["dropout"],
        "factor": cfg["factor"],
        "activation": cfg["activation"],
        "output_attention": cfg.get("output_attention", False),
        "embed": cfg.get("embed", "timeF"),
        "freq": cfg.get("freq", "h"),
        "csam_kernel_size": cfg.get("csam_kernel_size", 7),
        "csam_reduction": cfg.get("csam_reduction", 16),
    })

    if variant == "autoformer":
        # Try local override first, else fall back to upstream Autoformer repo
        if "AUTOFORMER_REPO" in os.environ or os.path.isdir(
            os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "..", "extern", "Autoformer")
        ):
            from model import AutoformerVanilla
        else:
            _repo = os.environ.get(
                "AUTOFORMER_REPO",
                os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "..", "extern", "Autoformer"),
            )
            if os.path.isdir(_repo) and _repo not in sys.path:
                sys.path.insert(0, _repo)
            from models.Autoformer import Model as AutoformerVanilla
        model = AutoformerVanilla(model_cfg).to(device)
    else:
        from model import AutoformerCSA
        model = AutoformerCSA(model_cfg).to(device)

    return model, model_cfg


def main():
    """Entry point: parse args, load data, train, evaluate, and save results.

    Command-line arguments override corresponding fields in the YAML config.
    The best checkpoint (by validation loss) is used for test evaluation.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--checkpoints-dir", type=str, default=None)
    parser.add_argument("--evaluation-dir", type=str, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--seed", type=int, default=None)
    args = parser.parse_args()

    with open(args.config) as f:
        config = yaml.safe_load(f)

    # Allow CLI args to override paths in config
    if args.checkpoints_dir:
        config["paths"]["checkpoints_dir"] = args.checkpoints_dir
    if args.evaluation_dir:
        config["paths"]["evaluation_dir"] = args.evaluation_dir

    seed = args.seed or config.get("seed") or config["training"].get("seed")
    if seed is not None:
        set_seed(seed)

    device = get_device(config.get("device", {}).get("device", "auto"))
    print(f"Device: {device}")

    windowing = config["windowing"]
    data_cfg = config["data"]
    split_cfg = config["split"]
    preproc_cfg = config["preprocessing"]
    train_cfg = config["training"]
    eval_cfg = config["evaluation"]
    paths = config["paths"]
    arch = config.get("architecture", {})

    cc2_only = data_cfg.get("cc2_only_smoke_test", False)
    csv_path = data_cfg["dataset_path"]

    print(f"Loading data from {csv_path}")
    train_ds, val_ds, test_ds, stats = create_datasets(
        csv_path=csv_path,
        seq_len=windowing["seq_len"],
        label_len=windowing["label_len"],
        pred_len=windowing["pred_len"],
        train_stride=windowing["train_stride"],
        val_stride=windowing.get("val_stride"),
        test_stride=windowing.get("test_stride"),
        train_ratio=split_cfg["train_ratio"],
        val_ratio=split_cfg["val_ratio"],
        chronological=split_cfg.get("chronological_split", True),
        normalization=preproc_cfg.get("normalization", "zscore"),
        fit_on_train_only=preproc_cfg.get("fit_on_train_only", True),
    )

    batch_size = args.batch_size or train_cfg["batch_size"]
    train_loader = torch.utils.data.DataLoader(train_ds, batch_size=batch_size, shuffle=True, drop_last=True)
    val_loader = torch.utils.data.DataLoader(val_ds, batch_size=batch_size, shuffle=False) if val_ds else None
    test_loader = torch.utils.data.DataLoader(test_ds, batch_size=batch_size, shuffle=False) if test_ds else None

    print(f"Train samples: {len(train_ds)}, Val samples: {len(val_ds) if val_ds else 0}, "
          f"Test samples: {len(test_ds) if test_ds else 0}")

    model, model_cfg = build_model(config, device)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model: {arch.get('model_variant', 'autoformer_csa')}, Parameters: {n_params:,}")

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr or train_cfg["learning_rate"])

    epochs = args.epochs or train_cfg["epochs"]
    patience = train_cfg.get("patience", 6)
    grad_clip = train_cfg.get("gradient_clip")
    loss_fn = nn.MSELoss()

    checkpoints_dir = os.path.abspath(paths["checkpoints_dir"])
    evaluation_dir = os.path.abspath(paths["evaluation_dir"])
    os.makedirs(checkpoints_dir, exist_ok=True)
    os.makedirs(evaluation_dir, exist_ok=True)

    # Save a copy of the config alongside checkpoints for reproducibility
    with open(os.path.join(checkpoints_dir, "config.yaml"), "w") as f:
        yaml.dump(config, f)

    best_val_loss = float("inf")
    patience_counter = 0
    best_epoch = 0
    training_log = []

    for epoch in range(1, epochs + 1):
        model.train()
        train_loss = 0.0
        for seq_x, seq_y in train_loader:
            seq_x, seq_y = seq_x.to(device), seq_y.to(device)

            # Time features are unused (zeros) but required by the embedding layer
            x_mark_enc = torch.zeros(seq_x.shape[0], seq_x.shape[1], 4, device=device)
            x_mark_dec = torch.zeros(seq_y.shape[0], seq_y.shape[1], 4, device=device)

            # Decoder input: first label_len + pred_len steps of ground truth
            dec_input = seq_y[:, :model_cfg.label_len + model_cfg.pred_len, :]
            optimizer.zero_grad()
            output = model(seq_x, x_mark_enc, dec_input, x_mark_dec)
            loss = loss_fn(output, seq_y[:, -model_cfg.pred_len:, :])
            loss.backward()
            if grad_clip:
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            optimizer.step()
            train_loss += loss.item() * seq_x.size(0)

        train_loss /= len(train_loader.dataset)

        if val_loader:
            model.eval()
            val_loss = 0.0
            with torch.no_grad():
                for seq_x, seq_y in val_loader:
                    seq_x, seq_y = seq_x.to(device), seq_y.to(device)
                    x_mark_enc = torch.zeros(seq_x.shape[0], seq_x.shape[1], 4, device=device)
                    x_mark_dec = torch.zeros(seq_y.shape[0], seq_y.shape[1], 4, device=device)
                    dec_input = seq_y[:, :model_cfg.label_len + model_cfg.pred_len, :]
                    output = model(seq_x, x_mark_enc, dec_input, x_mark_dec)
                    loss = loss_fn(output, seq_y[:, -model_cfg.pred_len:, :])
                    val_loss += loss.item() * seq_x.size(0)
            val_loss /= len(val_loader.dataset)
        else:
            val_loss = train_loss

        training_log.append({"epoch": epoch, "train_loss": train_loss, "val_loss": val_loss})
        print(f"Epoch {epoch:3d}/{epochs}  Train Loss: {train_loss:.6f}  Val Loss: {val_loss:.6f}")

        # Save checkpoint if validation loss improved, else increment patience
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_epoch = epoch
            patience_counter = 0
            save_checkpoint(
                os.path.join(checkpoints_dir, "best_model.pt"),
                model, optimizer, epoch, stats, config,
                {"val_loss": val_loss, "train_loss": train_loss},
            )
        else:
            patience_counter += 1
            if train_cfg.get("early_stopping", True) and patience_counter >= patience:
                print(f"Early stopping at epoch {epoch}")
                break

    # Always save the final model (may be later used for resuming)
    save_checkpoint(
        os.path.join(checkpoints_dir, "last_model.pt"),
        model, optimizer, epoch, stats, config,
        {"val_loss": val_loss, "train_loss": train_loss},
    )

    with open(os.path.join(checkpoints_dir, "training_log.json"), "w") as f:
        json.dump(training_log, f, indent=2)

    print(f"\nBest epoch: {best_epoch} (val_loss={best_val_loss:.6f})")

    if test_loader:
        print("Running test evaluation...")
        # Reload the best checkpoint for test evaluation
        model.load_state_dict(
            torch.load(os.path.join(checkpoints_dir, "best_model.pt"), map_location=device, weights_only=False)["model_state_dict"],
        )
        model.eval()

        all_pred, all_true = [], []
        with torch.no_grad():
            for seq_x, seq_y in test_loader:
                seq_x, seq_y = seq_x.to(device), seq_y.to(device)
                x_mark_enc = torch.zeros(seq_x.shape[0], seq_x.shape[1], 4, device=device)
                x_mark_dec = torch.zeros(seq_y.shape[0], seq_y.shape[1], 4, device=device)
                dec_input = seq_y[:, :model_cfg.label_len + model_cfg.pred_len, :]
                output = model(seq_x, x_mark_enc, dec_input, x_mark_dec)
                all_pred.append(output.cpu())
                all_true.append(seq_y[:, -model_cfg.pred_len:, :].cpu())

        all_pred = torch.cat(all_pred, dim=0)
        all_true = torch.cat(all_true, dim=0)

        # Denormalise predictions back to original dBm scale
        mean_t = torch.from_numpy(stats["mean"]).float()
        std_t = torch.from_numpy(stats["std"]).float()
        pred_dbm = denormalize(all_pred, mean_t, std_t).numpy()
        true_dbm = denormalize(all_true, mean_t, std_t).numpy()

        # Compute multi-granularity metrics
        n_nodes = len(data_cfg.get("node_names", ["CC2"]))
        bins_per_node = data_cfg.get("bins_per_node", data_cfg["n_features"] // n_nodes)
        node_names = data_cfg.get("node_names", [f"Node{i}" for i in range(n_nodes)])

        overall = compute_metrics(all_pred, all_true)
        per_horizon = compute_metrics_per_horizon(all_pred, all_true)
        per_node = compute_metrics_per_node(all_pred, all_true, n_nodes, bins_per_node, node_names)

        metrics = {}
        metrics.update({f"overall_{k}": v for k, v in overall.items()})
        metrics.update(per_horizon)
        metrics.update(per_node)

        # Additional metrics at specific forecast horizons
        eval_horizons = eval_cfg.get("eval_horizons", [1, 3, 6, 12])
        for h in eval_horizons:
            if h <= all_pred.shape[1]:
                h_idx = h - 1
                m = compute_metrics(all_pred[:, h_idx], all_true[:, h_idx])
                for k, v in m.items():
                    metrics[f"h{h}_{k}"] = v

        save_metrics_json(metrics, os.path.join(evaluation_dir, "metrics.json"))

        # Save flat predictions and ground truth as CSV
        B, T_out, D = pred_dbm.shape
        pred_flat = pred_dbm.reshape(-1, D)
        true_flat = true_dbm.reshape(-1, D)
        save_csv(pred_flat, os.path.join(evaluation_dir, "predictions.csv"))
        save_csv(true_flat, os.path.join(evaluation_dir, "ground_truth.csv"))

        # Reshape to (time, node, bin) for per-node visualisation
        pred_3d = pred_dbm.reshape(B * T_out, n_nodes, bins_per_node)
        true_3d = true_dbm.reshape(B * T_out, n_nodes, bins_per_node)

        for n in range(n_nodes):
            node_name = node_names[n] if n < len(node_names) else f"Node{n}"
            plot_spectrogram_comparison(
                pred_3d[:, n, :], true_3d[:, n, :], node_name,
                os.path.join(evaluation_dir, f"spectrogram_{node_name}.png"),
            )

        # Aggregate across all nodes for overall error heatmap
        all_nodes_pred = pred_3d.reshape(-1, bins_per_node * n_nodes)
        all_nodes_true = true_3d.reshape(-1, bins_per_node * n_nodes)
        plot_error_analysis(all_nodes_pred, all_nodes_true, os.path.join(evaluation_dir, "error_analysis.png"))

        print(f"Test metrics saved to {evaluation_dir}")
        print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
