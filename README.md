# PhyLLM-TCTF
**Physics-enhanced Large Language Model for Tropical Cyclone Track Forecasting**

---

## 🧭 1. Project Overview

PhyLLM-TCTF is a deep learning model for tropical cyclone track forecast that integrates physical features with Large Language Model backbone. It fuses multi-source meteorological inputs — CMA best-track data, ERA5 reanalysis fields, and Himawari satellite imagery — through Physical Feature Extraction, LSTM-LoRA fine-tuned, and a Semantic-Guided Cross-modal Fusion (SGCF) module to produce 6h–24h track forecasts.

---

## ⚙️ 2. Environment Setup

```
python == 3.10
torch == 2.7.1+cu118
torchvision == 0.22.1+cu118
transformers == 4.57.3
numpy == 2.2.6
pandas == 2.3.3
xarray == 2025.6.1
netCDF4 == 1.7.4
geopy == 2.4.1
matplotlib == 3.10.8
```

---

## 🧩 3. Prepare Data

Place your data under the `dataset/` directory following the structure below:

```
dataset/
├── CMA/
│   ├── 2015/
│   │   ├── Atsani.csv
│   │   ├── Champi.csv
│   │   └── ...          # one CSV per typhoon
│   └── ...              # one sub-folder per year
├── ERA5/
│   ├── 2015/
│   │   ├── Atsani/
│   │   │   ├── 2015081400.nc
│   │   │   ├── 2015081406.nc
│   │   │   └── ...      # one .nc per 6-hour timestep
│   │   └── ...          # one sub-folder per typhoon
│   └── ...              # one sub-folder per year
└── Himawari/
    ├── 2015/
    │   ├── Atsani/
    │   │   ├── 2015081400.nc
    │   │   ├── 2015081406.nc
    │   │   └── ...
    │   └── ...
    └── ...
```

The default split uses **2015–2021 for training/validation** and **2022–2024 for testing**. This can be changed with `--train_years` and `--test_years`.

---

## 🤖 4. Download the Pre-trained LLM

The model uses **GPT-2** as its language model backbone. Download the pre-trained weights at https://huggingface.co/openai-community/gpt2 and place them under `LLM/gpt2-local/`.

---

## 🚀 5. Training and Evaluation

Pre-trained checkpoints are provided in `model_weights/`. To evaluate directly without training:

```bash
python test.py
```

---

You can also start training again with the following instructions and then proceed with testing.

```bash
# train
python train.py
# test
python test.py
```

Checkpoints are saved to `<output_dir>/model_weights/` as `checkpoint{epoch}.pth` whenever validation loss improves. A loss curve is saved as `<output_dir>/loss_curve.png`.

Prediction tensors for each lead time are saved to `<output_dir>/results/<checkpoint_name>/`.

Key configurable parameters:

```python
--seq_len            5       # Input observation window (time steps)
--pred_len           4       # Forecast steps (6h / 12h / 18h / 24h)
--feat_dim           128     # Hidden feature dimension
--batch_size         8       # Batch size
--lr                 1e-3    # Adam learning rate
--num_epochs         200     # Maximum training epochs
--patience           20      # Early-stopping patience
--smooth_loss_weight 0.3     # Track smoothness regularisation weight
--loss               MSE     # Base loss: MSE or L1
```
