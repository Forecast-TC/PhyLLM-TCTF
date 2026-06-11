"""
test.py  –  Standalone evaluation script for PhyLLM-TCTF

Example
-------
python test.py \
    --data_root  ./data \
    --llm_path   ./LLM/gpt2-local \
    --checkpoint ./model_weights/checkpoint52.pth \
    --output_dir ./result \
    --test_years 2022 2023 2024
"""

import os

import torch
from torch.utils.data import DataLoader

from config import get_args
from loss import build_criterion
from sample import TyphoonDataConfig, TyphoonDataset, TyphoonDataLoader
from utils import calculate_metrics, set_seed, compute_weighted_loss
from model.PhyLLM_TCTF import TyphoonTrackPredictor


def test(model, loader, checkpoint_path, criterion, pred_len, output_dir, device):
    """
    Load a checkpoint, run inference on *loader*, print per-step metrics,
    and save per-step prediction / target tensors under *output_dir*/results/.

    Args:
        model: Uninitialised ``TyphoonTrackPredictor`` (weights are loaded here).
        loader: DataLoader for the test split.
        checkpoint_path (str): Path to the ``.pth`` checkpoint file.
        criterion: Loss module (MSE or L1).
        pred_len (int): Number of prediction steps.
        output_dir (str): Root output directory.
        device (str): ``'cuda'`` or ``'cpu'``.
    """
    print(f"\n{'=' * 80}")
    print(f"Loading checkpoint: {checkpoint_path}")
    model.load_state_dict(torch.load(checkpoint_path, map_location=device, weights_only=True))
    model.to(device)
    model.eval()

    test_weights = [1.0 / pred_len] * pred_len

    total_loss = 0.0
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

            loss, step_losses = compute_weighted_loss(
                pred, target, pred_len, test_weights, criterion
            )

            total_loss += loss.item()
            for s in range(pred_len):
                step_losses_sum[s] += step_losses[s]
                step_preds[s].append(pred[:, s, :].cpu())
                step_targets[s].append(target[:, s, :].cpu())

    n = len(loader)
    print(f"\nTest results  (checkpoint: {os.path.basename(checkpoint_path)})")
    print(f"  Weighted loss: {total_loss / n:.4f}")

    # ------------------------------------------------------ per-step metrics
    ckpt_name = os.path.splitext(os.path.basename(checkpoint_path))[0]
    results_dir = os.path.join(output_dir, "results", ckpt_name)
    os.makedirs(results_dir, exist_ok=True)

    for s in range(pred_len):
        p = torch.cat(step_preds[s],   dim=0)
        t = torch.cat(step_targets[s], dim=0)

        _, metrics_txt = calculate_metrics(p, t)
        avg_raw_loss = step_losses_sum[s] / n
        hours = (s + 1) * 6

        print(f"  step {s + 1} ({hours}h): loss={avg_raw_loss:.4f}  {metrics_txt}")

        results_path = os.path.join(results_dir, "results")
        os.makedirs(results_path, exist_ok=True)

        torch.save(p, os.path.join(results_path, f"preds_{hours}h.pt"))
        torch.save(t, os.path.join(results_path, f"targets_{hours}h.pt"))

    print(f"\nPer-step tensors saved to: {results_dir}")
    print(f"{'=' * 80}\n")


def main():
    args = get_args()
    set_seed(args.seed)

    if not os.path.exists(args.checkpoint):
        raise FileNotFoundError(f"Checkpoint not found: {args.checkpoint}")

    os.makedirs(args.output_dir, exist_ok=True)

    data_cfg = TyphoonDataConfig(args)
    data_loader = TyphoonDataLoader(data_cfg, seed=args.seed)

    _, _, test_samples = data_loader.split_dataset(
        train_years=args.train_years,
        test_years=args.test_years,
    )

    test_set = TyphoonDataset(test_samples, data_cfg)
    test_loader = DataLoader(
        test_set, batch_size=args.batch_size, shuffle=False,
        num_workers=args.num_workers
    )
    print(f"Test samples: {len(test_set)}")

    model = TyphoonTrackPredictor(feat_dim=args.feat_dim, llm_ckp_dir=args.llm_path)
    criterion = build_criterion(args.loss)

    test(
        model=model,
        loader=test_loader,
        checkpoint_path=args.checkpoint,
        criterion=criterion,
        pred_len=args.pred_len,
        output_dir=args.output_dir,
        device=args.device,
    )


if __name__ == "__main__":
    main()
