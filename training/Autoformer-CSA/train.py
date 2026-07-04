import os, sys, argparse, json, time
import torch
import torch.nn as nn
import yaml

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from dataset import create_datasets
from utils import set_seed, get_device, compute_metrics, save_checkpoint


class DotDict(dict):
    def __getattr__(self, k):
        try:
            return self[k]
        except KeyError:
            raise AttributeError(k)

    def __setattr__(self, k, v):
        self[k] = v


def build_model(config, device):
    arch = config.get("architecture", {})
    variant = arch.get("model_variant", "autoformer_csa")
    cfg = dict(config["model"])
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
        from model import AutoformerVanilla
        model = AutoformerVanilla(model_cfg).to(device)
    else:
        from model import AutoformerCSA
        model = AutoformerCSA(model_cfg).to(device)
    return model, model_cfg


def resolve_feature_dims(config, stats):
    feat_dim = int(stats["n_features"])
    config["model"]["enc_in"] = feat_dim
    config["model"]["dec_in"] = feat_dim
    config["model"]["c_out"] = feat_dim
    return feat_dim


def make_loss(name):
    loss_name = str(name).lower()
    if loss_name == "mse":
        return lambda pred, target: torch.mean((pred - target) ** 2)
    if loss_name == "rmse":
        return lambda pred, target: torch.sqrt(torch.mean((pred - target) ** 2) + 1e-12)
    if loss_name == "mae":
        return lambda pred, target: torch.mean(torch.abs(pred - target))
    raise ValueError(f"Unsupported loss: {name}")


def build_optimizer(name, params, lr):
    opt_name = str(name).lower()
    if opt_name == "adam":
        return torch.optim.Adam(params, lr=lr)
    if opt_name == "nadam":
        return torch.optim.NAdam(params, lr=lr)
    if opt_name == "adamw":
        return torch.optim.AdamW(params, lr=lr)
    raise ValueError(f"Unsupported optimizer: {name}")


def build_scheduler(train_cfg, optimizer):
    sched_name = str(train_cfg.get("lr_scheduler", "none")).lower()
    if sched_name == "reduce_on_plateau":
        return torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode="min",
            factor=float(train_cfg.get("lr_factor", 0.5)),
            patience=int(train_cfg.get("lr_patience", 5)),
        )
    if sched_name == "none":
        return None
    raise ValueError(f"Unsupported lr_scheduler: {train_cfg.get('lr_scheduler')}")


def run_model(model, model_cfg, seq_x, seq_y, device):
    x_mark_enc = torch.zeros(seq_x.shape[0], seq_x.shape[1], 4, device=device)
    x_mark_dec = torch.zeros(seq_y.shape[0], seq_y.shape[1], 4, device=device)
    dec_input = seq_y[:, : model_cfg.label_len + model_cfg.pred_len, :]
    return model(seq_x, x_mark_enc, dec_input, x_mark_dec)


def evaluate_loss(model, model_cfg, loader, device, loss_fn):
    model.eval()
    total_loss = 0.0
    all_pred, all_true = [], []
    with torch.no_grad():
        for seq_x, seq_y in loader:
            seq_x, seq_y = seq_x.to(device), seq_y.to(device)
            output = run_model(model, model_cfg, seq_x, seq_y, device)
            target = seq_y[:, -model_cfg.pred_len:, :]
            loss = loss_fn(output, target)
            total_loss += loss.item() * seq_x.size(0)
            all_pred.append(output.cpu())
            all_true.append(target.cpu())
    pred = torch.cat(all_pred, dim=0)
    true = torch.cat(all_true, dim=0)
    metrics = compute_metrics(pred, true)
    metrics["loss"] = total_loss / len(loader.dataset)
    return metrics


def main():
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
    paths = config["paths"]
    data_format = data_cfg.get("format", "csv")
    dataset_path = data_cfg["map_path"] if data_format == "interpolated_map" else data_cfg["dataset_path"]

    print(f"Loading data from {dataset_path}")
    train_ds, val_ds, test_ds, stats = create_datasets(
        data_format=data_format,
        dataset_path=dataset_path,
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
        data_cfg=data_cfg,
    )

    feat_dim = resolve_feature_dims(config, stats)
    batch_size = args.batch_size or train_cfg["batch_size"]
    train_loader = torch.utils.data.DataLoader(train_ds, batch_size=batch_size, shuffle=True, drop_last=True)
    val_loader = torch.utils.data.DataLoader(val_ds, batch_size=batch_size, shuffle=False) if val_ds else None

    print(f"Train samples: {len(train_ds)}, Val samples: {len(val_ds) if val_ds else 0}, Test samples: {len(test_ds) if test_ds else 0}")
    print(f"Feature dimension: {feat_dim}, Data format: {data_format}")

    model, model_cfg = build_model(config, device)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model: {config.get('architecture', {}).get('model_variant', 'autoformer_csa')}, Parameters: {n_params:,}")

    optimizer = build_optimizer(train_cfg.get("optimizer", "adam"), model.parameters(), args.lr or train_cfg["learning_rate"])
    scheduler = build_scheduler(train_cfg, optimizer)
    loss_fn = make_loss(train_cfg.get("loss", "rmse"))

    epochs = args.epochs or train_cfg["epochs"]
    patience = train_cfg.get("patience", 6)
    grad_clip = train_cfg.get("gradient_clip")
    checkpoints_dir = os.path.abspath(paths["checkpoints_dir"])
    os.makedirs(checkpoints_dir, exist_ok=True)
    with open(os.path.join(checkpoints_dir, "config.yaml"), "w") as f:
        yaml.dump(config, f)

    best_val_loss = float("inf")
    patience_counter = 0
    best_epoch = 0
    training_log = {"epochs": [], "summary": {}}
    training_start = time.perf_counter()

    for epoch in range(1, epochs + 1):
        epoch_start = time.perf_counter()
        model.train()
        train_loss_sum = 0.0
        for seq_x, seq_y in train_loader:
            seq_x, seq_y = seq_x.to(device), seq_y.to(device)
            optimizer.zero_grad()
            output = run_model(model, model_cfg, seq_x, seq_y, device)
            target = seq_y[:, -model_cfg.pred_len:, :]
            loss = loss_fn(output, target)
            loss.backward()
            if grad_clip:
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            optimizer.step()
            train_loss_sum += loss.item() * seq_x.size(0)

        train_loss = train_loss_sum / len(train_loader.dataset)
        if val_loader:
            val_metrics = evaluate_loss(model, model_cfg, val_loader, device, loss_fn)
            val_loss = val_metrics["loss"]
        else:
            val_metrics = {"loss": train_loss, "rmse": train_loss, "mae": train_loss, "r2": 0.0}
            val_loss = train_loss

        if scheduler is not None:
            scheduler.step(val_loss)

        epoch_time = time.perf_counter() - epoch_start
        lr = optimizer.param_groups[0]["lr"]
        training_log["epochs"].append({
            "epoch": epoch,
            "train_loss": train_loss,
            "val_loss": val_loss,
            "val_rmse": val_metrics.get("rmse"),
            "epoch_time_seconds": epoch_time,
            "learning_rate": lr,
        })
        print(
            f"Epoch {epoch:3d}/{epochs} | LR: {lr:.2e} | Train Loss: {train_loss:.6f} | "
            f"Val Loss: {val_loss:.6f} | Val RMSE: {val_metrics.get('rmse', 0):.4f} | Epoch Time: {epoch_time:.2f}s"
        )

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_epoch = epoch
            patience_counter = 0
            save_checkpoint(
                os.path.join(checkpoints_dir, "best_model.pt"),
                model,
                optimizer,
                epoch,
                stats,
                config,
                {"val_loss": val_loss, "train_loss": train_loss},
            )
        else:
            patience_counter += 1
            if train_cfg.get("early_stopping", True) and patience_counter >= patience:
                print(f"Early stopping at epoch {epoch}")
                break

    save_checkpoint(
        os.path.join(checkpoints_dir, "last_model.pt"),
        model,
        optimizer,
        epoch,
        stats,
        config,
        {"val_loss": val_loss, "train_loss": train_loss},
    )

    total_training_time = time.perf_counter() - training_start
    mean_epoch_time = sum(entry["epoch_time_seconds"] for entry in training_log["epochs"]) / len(training_log["epochs"])
    training_log["summary"] = {
        "epochs_completed": len(training_log["epochs"]),
        "best_epoch": best_epoch,
        "best_val_loss": best_val_loss,
        "elapsed_training_time_seconds": total_training_time,
        "mean_epoch_time_seconds": mean_epoch_time,
    }
    with open(os.path.join(checkpoints_dir, "training_log.json"), "w") as f:
        json.dump(training_log, f, indent=2)

    print(
        f"\nBest epoch: {best_epoch} (val_loss={best_val_loss:.6f}) | "
        f"Total training time: {total_training_time:.2f}s | Mean epoch time: {mean_epoch_time:.2f}s"
    )


if __name__ == "__main__":
    main()
