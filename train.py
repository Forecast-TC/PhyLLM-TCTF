"""
main.py  –  Training entry point for PhysicCNN-LKASTA-LoRA-GPT2

Example
-------
python main.py \
    --data_root /data/401_81 \
    --llm_path  /models/gpt2-local \
    --output_dir ./runs \
    --train_years 2015 2016 2017 2018 2019 2020 2021 \
    --test_years  2022 2023 2024
"""

import os
from datetime import datetime

import torch
import torch.optim as optim
from torch.utils.data import DataLoader

from config import get_args
from loss import build_criterion, get_dynamic_weights, trajectory_smooth_loss
from sample import TyphoonDataConfig, TyphoonDataset, TyphoonDataLoader
from utils import calculate_metrics, plot_training_loss, set_seed, compute_weighted_loss
from model.PhyLLM_TCTF import TyphoonTrackPredictor


def train(model, loader, criterion, optimizer, current_weights, smooth_loss_weight, pred_len, device, epoch):
    model.train()

    total_loss = 0.0
    total_traj_loss = 0.0
    total_smooth_loss = 0.0
    step_losses_sum = [0.0] * pred_len

    for batch in loader:
        track = batch["track"].to(device)
        era5 = batch["era5"].to(device)
        himawari = batch["himawari"].to(device)
        target = batch["target"].to(device)

        pred = model(track, era5, himawari)

        traj_loss, step_losses = compute_weighted_loss(
            pred, target, pred_len, current_weights, criterion
        )

        track_total = torch.cat([track, pred], dim=1)
        smooth_loss = trajectory_smooth_loss(track_total) * smooth_loss_weight
        batch_loss = traj_loss + smooth_loss

        optimizer.zero_grad()
        batch_loss.backward()
        optimizer.step()

        total_traj_loss += traj_loss.item()
        total_smooth_loss += smooth_loss.item()
        total_loss += batch_loss.item()
        for s in range(pred_len):
            step_losses_sum[s] += step_losses[s]

    n = len(loader)
    avg_step_losses = [v / n for v in step_losses_sum]

    print(f"[Train] Epoch {epoch + 1}  "
          f"loss={total_loss / n:.4f}  "
          f"traj={total_traj_loss / n:.4f}  "
          f"smooth={total_smooth_loss / n:.4f}")
    for s, l in enumerate(avg_step_losses):
        print(f"  step {s + 1} ({(s + 1) * 6}h): {l:.4f}")

    return total_loss / n, avg_step_losses


def validate(model, loader, criterion, current_weights, pred_len, device, epoch):
    model.eval()

    total_weighted_loss = 0.0
    step_losses_sum = [0.0] * pred_len
    step_preds = [[] for _ in range(pred_len)]
    step_targets = [[] for _ in range(pred_len)]

    with torch.no_grad():
        for batch in loader:
            track = batch["track"].to(device)
            era5 = batch["era5"].to(device)
            himawari = batch["himawari"].to(device)
            target = batch["target"].to(device)

            pred = model(track, era5, himawari)

            val_loss, step_losses = compute_weighted_loss(
                pred, target, pred_len, current_weights, criterion
            )

            total_weighted_loss += val_loss.item()
            for s in range(pred_len):
                step_losses_sum[s] += step_losses[s]
                step_preds[s].append(pred[:, s, :].cpu())
                step_targets[s].append(target[:, s, :].cpu())

    n = len(loader)
    avg_step_losses = [v / n for v in step_losses_sum]

    print(f"[Val]   Epoch {epoch + 1}  weighted_loss={total_weighted_loss / n:.4f}")
    for s in range(pred_len):
        p = torch.cat(step_preds[s],   dim=0)
        t = torch.cat(step_targets[s], dim=0)
        _, metrics_txt = calculate_metrics(p, t)
        print(f"  step {s + 1} ({(s + 1) * 6}h): loss={avg_step_losses[s]:.4f}  {metrics_txt}")

    return total_weighted_loss / n, avg_step_losses


def main():
    args = get_args()
    set_seed(args.seed)

    os.makedirs(args.output_dir, exist_ok=True)
    weights_dir = os.path.join(args.output_dir, "model_weights")
    os.makedirs(weights_dir, exist_ok=True)

    g = torch.Generator()
    g.manual_seed(args.seed)

    data_cfg = TyphoonDataConfig(args)
    data_loader = TyphoonDataLoader(data_cfg)

    train_samples, val_samples, _ = data_loader.split_dataset(
        train_years=args.train_years,
        test_years=args.test_years,
    )

    train_set = TyphoonDataset(train_samples, data_cfg)
    val_set = TyphoonDataset(val_samples,   data_cfg)

    train_loader = DataLoader(
        train_set, batch_size=args.batch_size, shuffle=True,
        num_workers=args.num_workers, generator=g
    )
    val_loader = DataLoader(
        val_set, batch_size=args.batch_size, shuffle=False,
        num_workers=args.num_workers
    )

    sample = train_set[0]
    print(f"Input shapes – track: {sample['track'].shape}  "
          f"era5: {sample['era5'].shape}  "
          f"himawari: {sample['himawari'].shape}  "
          f"target: {sample['target'].shape}")

    model = TyphoonTrackPredictor(feat_dim=args.feat_dim, llm_ckp_dir=args.llm_path)
    model.to(args.device)
    print(model)

    criterion = build_criterion(args.loss)
    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    pred_len = args.pred_len

    best_val_loss = float("inf")
    no_improve = 0

    train_losses = []
    val_losses = []

    current_weights = [1.0 / pred_len] * pred_len
    print(f"\nInitial step weights: {[round(w, 3) for w in current_weights]}")
    print(f"Smooth loss weight: {args.smooth_loss_weight}")
    print(f"{'=' * 80}")

    for epoch in range(args.num_epochs):
        t0 = datetime.now()
        print(f"\nEpoch {epoch + 1}/{args.num_epochs}  "
              f"[{t0.strftime('%Y-%m-%d %H:%M:%S')}]  "
              f"weights={[round(w, 3) for w in current_weights]}")

        avg_train_loss, avg_train_step_losses = train(
            model, train_loader, criterion, optimizer,
            current_weights, args.smooth_loss_weight, pred_len,
            args.device, epoch
        )
        train_losses.append(avg_train_loss)

        avg_val_loss, avg_val_step_losses = validate(
            model, val_loader, criterion, current_weights,
            pred_len, args.device, epoch
        )
        val_losses.append(avg_val_loss)

        if epoch < args.num_epochs - 1:
            current_weights = get_dynamic_weights(avg_val_step_losses)
            print(f"  Next epoch weights: {[round(w, 3) for w in current_weights]}")

        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            no_improve = 0
            ckpt_path = os.path.join(weights_dir, f"checkpoint{epoch + 1}.pth")
            torch.save(model.state_dict(), ckpt_path)
            print(f"  -> Saved best model2 (epoch {epoch + 1}, val_loss={best_val_loss:.4f}): {ckpt_path}")
        else:
            no_improve += 1
            print(f"  No improvement for {no_improve} epoch(s)")

        elapsed = (datetime.now() - t0).total_seconds()
        print(f"  Epoch time: {elapsed:.1f}s")

        if no_improve >= args.patience:
            print(f"\nEarly stopping triggered at epoch {epoch + 1}.")
            break

    print(f"\n{'=' * 80}")
    print(f"Training complete. Best val loss: {best_val_loss:.4f}")
    plot_training_loss(train_losses, val_losses, args.output_dir)


if __name__ == "__main__":
    main()
