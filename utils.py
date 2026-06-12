"""
utils.py  –  Utility functions for typhoon track prediction
"""

import os
import random

import matplotlib.pyplot as plt
import numpy as np
import torch
from geopy.distance import great_circle


def set_seed(seed: int):
    """Set all relevant random seeds for reproducible experiments."""
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def compute_weighted_loss(preds, targets, pred_len, weights, criterion):
    """
    Compute the weighted sum of per-step reconstruction losses.

    Args:
        preds   (Tensor): Shape ``(B, T, 2)`` – model2 predictions.
        targets (Tensor): Shape ``(B, T, 2)`` – ground-truth positions.
        pred_len  (int): Number of prediction steps ``T``.
        weights (list[float]): Per-step loss weights (length ``T``).
        criterion: Loss module (MSE or L1).

    Returns:
        tuple:
            - weighted_loss (Tensor): Scalar weighted loss for back-prop.
            - step_losses (list[float]): Unweighted per-step loss values.
    """
    weighted_loss = 0.0
    step_losses = []

    for s in range(pred_len):
        loss_s = criterion(preds[:, s, :], targets[:, s, :])
        weighted_loss += loss_s * weights[s]
        step_losses.append(loss_s.item())

    return weighted_loss, step_losses


def calculate_metrics(preds, targets):
    """
    Compute MAE, RMSE, MSD, and mean great-circle distance (D) after
    de-normalising coordinates.

    Normalisation convention (applied during dataset construction):
        lat_norm = (lat - 30) / 5
        lon_norm = (lon - 130) / 5

    Args:
        preds   (Tensor): Shape ``(N, 2)`` – (lat_norm, lon_norm) predictions.
        targets (Tensor): Shape ``(N, 2)`` – (lat_norm, lon_norm) ground truth.

    Returns:
        tuple:
            - metrics_dict (dict): Scalar value for each metric.
            - metrics_txt  (str): Human-readable summary string.
    """
    m = len(preds)
    R = 6371

    # De-normalise
    preds = preds.clone().float()
    targets = targets.clone().float()
    preds[:, 0] = preds[:, 0] * 5.0 + 30.0
    preds[:, 1] = preds[:, 1] * 5.0 + 130.0
    targets[:, 0] = targets[:, 0] * 5.0 + 30.0
    targets[:, 1] = targets[:, 1] * 5.0 + 130.0

    pred_lat, pred_lon = preds[:, 0], preds[:, 1]
    target_lat, target_lon = targets[:, 0], targets[:, 1]

    if torch.isnan(pred_lat).any() or torch.isnan(target_lat).any():
        print("警告: 纬度数据包含NaN值")
    if torch.isnan(pred_lon).any() or torch.isnan(target_lon).any():
        print("警告: 经度数据包含NaN值")

    mae_lat = torch.mean(torch.abs(target_lat - pred_lat)).item()
    mae_lon = torch.mean(torch.abs(target_lon - pred_lon)).item()
    rmse_lat = torch.sqrt(torch.mean((target_lat - pred_lat) ** 2)).item()
    rmse_lon = torch.sqrt(torch.mean((target_lon - pred_lon) ** 2)).item()

    pred_lat_rad = torch.deg2rad(pred_lat)
    pred_lon_rad = torch.deg2rad(pred_lon)
    target_lat_rad = torch.deg2rad(target_lat)
    target_lon_rad = torch.deg2rad(target_lon)

    cos_term = (
        torch.sin(pred_lat_rad) * torch.sin(target_lat_rad)
        + torch.cos(target_lat_rad) * torch.cos(pred_lat_rad)
        * torch.cos(target_lon_rad - pred_lon_rad)
    )
    cos_term = torch.clamp(cos_term, -1.0, 1.0)
    if torch.any(cos_term < -1.0) or torch.any(cos_term > 1.0):
        print(f"警告: cos_term超出范围: min={cos_term.min()}, max={cos_term.max()}")

    msd = (R / m) * torch.sum(torch.acos(cos_term)).item()

    metrics_dict = {
        "MAE_lat":  mae_lat,
        "MAE_lon":  mae_lon,
        "RMSE_lat": rmse_lat,
        "RMSE_lon": rmse_lon,
        "MSD":      msd,
    }
    metrics_txt = (
        f"MAE_lat={mae_lat:.4f} | "
        f"MAE_lon={mae_lon:.4f} | "
        f"RMSE_lat={rmse_lat:.4f} | "
        f"RMSE_lon={rmse_lon:.4f} | "
        f"MSD={msd:.4f} | "
    )
    return metrics_dict, metrics_txt


def plot_training_loss(train_losses, val_losses, save_dir, figsize=(10, 6)):
    """
    Plot and save training / validation loss curves.

    Args:
        train_losses (list): Per-epoch training losses.
        val_losses   (list): Per-epoch validation losses.
        save_dir (str): Directory where the figure is saved.
        figsize (tuple): Matplotlib figure size.
    """
    def _to_float(v):
        return v.cpu().detach().item() if torch.is_tensor(v) else float(v)

    train_losses = [_to_float(v) for v in train_losses]
    val_losses = [_to_float(v) for v in val_losses]
    epochs = range(1, len(train_losses) + 1)

    fig, ax = plt.subplots(figsize=figsize)
    ax.plot(epochs, train_losses, "b-", label="Train loss")
    ax.plot(epochs, val_losses,   "r-", label="Val loss")

    best_idx = int(np.argmin(val_losses))
    ax.scatter(best_idx + 1, val_losses[best_idx], color="green", marker="*",
               s=200, label=f"Best val: {val_losses[best_idx]:.4f}")

    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.set_title("Training and Validation Loss")
    ax.legend()
    ax.grid(True, linestyle="--", alpha=0.7)
    fig.tight_layout()

    os.makedirs(save_dir, exist_ok=True)
    out_path = os.path.join(save_dir, "loss_curve.png")
    fig.savefig(out_path, dpi=300)
    plt.close(fig)
    print(f"Loss curve saved to: {out_path}")
