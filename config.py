"""
config.py  –  Centralised argument parser for PhyLLM-TCTF
              (typhoon track prediction)

Usage examples
--------------
Training:
    python main.py --data_root ./dataset --llm_path ./LLM/gpt2-local \
                   --train_years 2015 2016 2017 2018 2019 2020 2021 \
                   --test_years 2022 2023 2024

Testing:
    python test.py --checkpoint ./model_weights/checkpoint52.pth
"""

import argparse
import torch


def get_args():
    parser = argparse.ArgumentParser(
        description="PhyLLM-TCTF: Typhoon Track Prediction"
    )

    parser.add_argument(
        "--data_root", type=str, default="./dataset",
        help="Root directory of the dataset (expects CMA/, ERA5/, Himawari/ sub-dirs)"
    )
    parser.add_argument(
        "--llm_path", type=str, default="./LLM/gpt2-local",
        help="Local checkpoint directory of the GPT-2 LLM backbone"
    )
    parser.add_argument(
        "--output_dir", type=str,
        default="./",
        help="Root output directory"
    )

    parser.add_argument(
        "--checkpoint", type=str,
        default="./model_weights/checkpoint33.pth",
        help="[test.py only] Path to a trained model checkpoint (.pth)"
    )

    parser.add_argument(
        "--train_years", type=str, nargs="+",
        default=["2015", "2016", "2017", "2018", "2019", "2020", "2021"],
        help="Years used for training (default: 2015-2021)"
    )
    parser.add_argument(
        "--test_years", type=str, nargs="+",
        default=["2022", "2023", "2024"],
        help="Years used for testing (default: 2022-2024)"
    )

    parser.add_argument(
        "--seq_len", type=int, default=5,
        help="Input observation window length in time steps (default: 5)"
    )
    parser.add_argument(
        "--pred_len", type=int, default=4,
        help="Number of future time steps to predict (default: 4, i.e. 6h–24h)"
    )

    parser.add_argument(
        "--feat_dim", type=int, default=128,
        help="Feature / hidden dimension used inside the model2 (default: 128)"
    )
    parser.add_argument(
        "--k_neighbors", type=int, default=10,
        help="Number of nearest neighbours for RAG retrieval (default: 10)"
    )

    parser.add_argument(
        "--batch_size", type=int, default=8,
        help="Mini-batch size (default: 8)"
    )
    parser.add_argument(
        "--lr", type=float, default=1e-3,
        help="Learning rate for Adam optimiser (default: 1e-3)"
    )
    parser.add_argument(
        "--num_epochs", type=int, default=200,
        help="Maximum number of training epochs (default: 200)"
    )
    parser.add_argument(
        "--patience", type=int, default=20,
        help="Early-stopping patience in epochs (default: 20)"
    )
    parser.add_argument(
        "--smooth_loss_weight", type=float, default=0.3,
        help="Weight of the trajectory-smoothness regularisation term (default: 0.3)"
    )
    parser.add_argument(
        "--loss", type=str, default="MSE", choices=["MSE", "L1", "euclidean"],
        help="Base reconstruction loss: MSE, L1 or euclidean (default: MSE)"
    )
    parser.add_argument(
        "--lr_step_size", type=int, default=30,
        help="Number of epochs between StepLR reductions (default: 30)"
    )
    parser.add_argument(
        "--lr_gamma", type=float, default=0.5,
        help="Multiplicative StepLR decay factor (default: 0.5)"
    )
    parser.add_argument(
        "--min_early_stopping_patience", type=int, default=45,
        help="Minimum effective early-stopping patience (default: 45)"
    )
    parser.add_argument(
        "--seed", type=int, default=42,
        help="Global random seed (default: 42)"
    )
    parser.add_argument(
        "--num_workers", type=int, default=1,
        help="Number of DataLoader worker processes (default: 1)"
    )
    parser.add_argument(
        "--device", type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Device to use: 'cuda' or 'cpu' (auto-detected by default)"
    )

    parser.add_argument(
        "--train_ratio", type=float, default=0.9,
        help="Fraction of train-year typhoons used for training (rest → val, default: 0.9)"
    )
    parser.add_argument(
        "--val_ratio", type=float, default=0.1,
        help="Fraction of train-year typhoons used for validation (default: 0.1)"
    )
    parser.add_argument(
        "--himawari_channels", type=str, nargs="+",
        default=["tbb_08", "tbb_13", "tbb_15", "tbb_16"],
        help="Himawari channel names (default: tbb_08 tbb_13 tbb_15 tbb_16)"
    )
    parser.add_argument(
        "--era5_single", type=str, nargs="+",
        default=["sst", "msl", "v10", "u10"],
        help="ERA5 single-level variable names (default: sst msl v10 u10)"
    )
    parser.add_argument(
        "--era5_pres", type=str, nargs="+",
        default=["z", "u", "v"],
        help="ERA5 pressure-level variable names (default: z u v)"
    )

    return parser.parse_args()
