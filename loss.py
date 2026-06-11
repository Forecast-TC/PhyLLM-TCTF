"""
loss.py  –  Loss functions for typhoon track prediction
"""

import torch
import torch.nn as nn


def get_dynamic_weights(step_losses, epsilon=1e-6):
    """
    Compute normalised per-step weights inversely proportional to their losses.

    A step with a *larger* loss receives a *higher* weight so that the model2
    focuses more on the steps it finds hardest.

    Args:
        step_losses (list[float]): Mean loss value for each prediction step.
        epsilon (float): Small constant to avoid division by zero.

    Returns:
        list[float]: Normalised weights that sum to 1.
    """
    raw_weights = [1.0 / (loss + epsilon) for loss in step_losses]
    weight_sum = sum(raw_weights)
    return [w / weight_sum for w in raw_weights]


def trajectory_smooth_loss(traj):
    """
    Second-order trajectory smoothness regularisation.

    Penalises abrupt changes in direction / speed by computing the mean
    squared second-order finite difference along the time axis.

    Args:
        traj (Tensor): Shape ``(B, T, 2)`` – batch of (lon, lat) sequences.

    Returns:
        Tensor: Scalar smoothness loss.
    """
    assert traj.ndim == 3 and traj.shape[2] == 2, \
        "Expected trajectory shape (B, T, 2)"
    if traj.shape[1] < 3:
        return torch.tensor(0.0, device=traj.device)

    delta1 = traj[:, 1:, :] - traj[:, :-1, :]
    delta2 = delta1[:, 1:, :] - delta1[:, :-1, :]
    return torch.mean(torch.square(delta2))


def build_criterion(loss_type: str) -> nn.Module:
    """
    Return the base reconstruction loss module.

    Args:
        loss_type (str): ``'MSE'`` or ``'L1'``.

    Returns:
        nn.Module
    """
    if loss_type == "MSE":
        return nn.MSELoss()
    elif loss_type == "L1":
        return nn.L1Loss()
    else:
        raise ValueError(f"Unknown loss type '{loss_type}'. Choose 'MSE' or 'L1'.")
