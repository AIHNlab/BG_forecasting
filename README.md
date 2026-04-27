# Blood Glucose Forecasting

This repository accompanies the paper:

> **Multi-task Transformer with Unified Clinical Tokenizer for Effective Blood Glucose Prediction**
> Knut J. Strommen, Maria Panagiotou, Lorenzo Brigato, Stavroula Mougiakakou

It contains the official implementation of **MT-UCT**, a multi-task encoder-only Transformer for blood glucose (BG) forecasting that combines a **Unified Clinical Tokenizer (UCT)** with a multi-task training objective (masked reconstruction, deterministic forecast, probabilistic forecast with uncertainty, and hypo-/hyperglycaemia alarm tokens). The codebase additionally provides the LSTM and linear-regression baselines used in the paper, in. addition to the training and evaluation pipelines used across 12 publicly available clinical datasets.

## Highlights from the paper

- **MT-UCT architecture** — encoder-only Transformer with historical temporal tokens for CGM, carbohydrate intake, and bolus insulin; learnable demographic/clinical tokens (age, sex, BMI, diabetes type, treatment); and dedicated forecast / hypo / hyper task tokens. Implemented in [architectures/MTUCT.py](architectures/MTUCT.py) (the `Model` class, with the UCT backbone in the `UCT` class).
- **Multi-task training objective** — masked patch reconstruction (10% mask ratio), MSE forecast, Gaussian NLL uncertainty (with detached gradient at the encoder), and BCE alarm losses with soft sustained-event labels (1 h horizon, 3-step run rule). Implemented in [trainers/exp_long_term_forecasting.py](trainers/exp_long_term_forecasting.py).
- **Long context** — trained on sequences of 2,304 time steps (~8 days at 5-minute sampling), enabling MT-UCT to exploit multi-day history that LSTMs cannot.
- **Alarm system** — probabilistic early warnings for hypo-/hyperglycaemia with tunable thresholds, evaluated by event-level precision, recall, detection time, and daily false-alarm rate.
- **12-dataset benchmark** — OhioT1DM, Broll, Colas, Dubosson, Hall, Weinstock, Tidepool SAP, Tidepool HCL, T1DEXI, AI4FoodDB, ShanghaiT1DM, ShanghaiT2DM.

## Quick Start

```bash
# 1. Create and activate a virtual environment
python -m venv .venv
.venv\Scripts\activate   # Windows
# source .venv/bin/activate  # Linux/Mac

# 2. Install dependencies
pip install -r requirements.txt

# 3. Run a single experiment
# Edit the experiment_path in ml_pipeline.py, then:
python ml_pipeline.py

# 4. Run a hyperparameter sweep
python run_multiple.py    # LSTM sweep
python run_multiple2.py   # Transformer sweep
```

## Project Structure

```
BG_forecasting/
│
├── ml_pipeline.py              # Main entry point (shim → pipeline/)
├── run_multiple.py             # LSTM hyperparameter grid search
├── run_multiple2.py            # Transformer hyperparameter grid search
├── robustness.py               # Parameter sensitivity sweep runner
├── hp_summary.py               # Summarize HP search results
├── utils.py                    # Shared utilities (IOB/COB calculations, etc.)
│
├── pipeline/                   # Core pipeline (split from ml_pipeline.py)
│   ├── orchestrator.py         #   main(), init_experiment_directory()
│   ├── training.py             #   train_model(), create_temporal_split()
│   └── evaluation.py           #   evaluate_model(), plot_per_horizon()
│
├── data/                       # Data loading and preprocessing
│   ├── handler.py              #   DataHandler — dataset loading & caching
│   ├── prepper.py              #   DataPrepper — normalization, sequencing, masking
│   ├── dataset.py              #   SequenceDataset, EventBalancedSampler (PyTorch)
│   └── scaler.py               #   Scaler — feature normalization with persistence
│
├── evaluation/                 # Evaluation metrics and visualization
│   ├── alarm.py                #   Event-level alarm metrics, calibration curves
│   ├── forecast.py             #   RMSE/MAE by glycemic zone, CG-EGA, uncertainty
│   └── metrics.py              #   Evaluator (basic RMSE)
│
├── dataloaders/                # Dataset-specific loaders (one per dataset)
│   ├── dataloader.py           #   Abstract base class
│   ├── dataloader_ohio.py      #   OhioT1DM (2018, 2020)
│   ├── dataloader_tidepool_sap100.py
│   ├── dataloader_tidepool_hcl150.py
│   ├── dataloader_t1dexi.py
│   ├── dataloader_glucobench.py
│   ├── dataloader_shanghai.py
│   ├── dataloader_ai4food.py
│   ├── dataloader_geneva.py
│   └── ...
│
├── trainers/                   # Training loops
│   ├── exp_basic.py            #   Base class for Transformer trainers
│   ├── exp_long_term_forecasting.py  # Transformer training (cosine LR, calibration)
│   ├── trainer_basic.py        #   LSTM training loop with early stopping
│   └── trainer_linreg.py       #   Linear regression baselines (sklearn, Darts)
│
├── architectures/              # Model definitions
│   ├── lstms.py                #   TidepoolLSTM, MirshekarianLSTM
│   ├── iTransformer.py         #   Inverted Transformer
│   ├── iTransformerMasked.py   #   Masked iTransformer (periodicity-aware)
│   ├── MTUCT.py                #   MT-UCT: Multi-task Transformer w/ Unified Clinical Tokenizer
│   ├── BGiTransformer.py       #   Backwards-compat shim re-exporting MTUCT
│   └── layers/                 #   Embeddings, attention, encoder/decoder
│
├── experiments/                # Experiment configs and outputs
├── standardized_datasets/      # Cached preprocessed data (pickled DataFrames)
├── scripts/                    # Aggregation scripts (RMSE, CG-EGA, heatmaps)
├── other_scripts/              # Utilities, tests, visualization tools
├── CG-EGA/                     # Clarke/Consensus Error Grid Analysis (vendored)
└── OpenLTM/                    # Open Long-Term Models (vendored, experimental)
```

## How It Works

### Pipeline Flow

```
model_config.json
    │
    ▼
main(config)                          # pipeline/orchestrator.py
    ├── DataHandler.load_data()       # Load from dataset-specific loaders, cache as pickle
    ├── DataPrepper → SequenceDataset # Normalize, create sliding windows, mask missing data
    ├── train_model()                 # Temporal split → DataLoader → Trainer.train()
    └── evaluate_model()              # Per-participant inference → metrics → CSV/JSON/PNG
```

### Configuration

Each experiment is driven by a `model_config.json` with two sections:

- **`hp_config`** — Model hyperparameters: `architecture`, `learning_rate`, `batch_size`, `feature_window`, `forecast_steps`, `hidden_dim`, `lstm_layers`, `mask_ratio`, etc.
- **`run_config`** — Experiment settings: `experiment_path`, `dataset_names`, `dataloaders`, `features`, `targets`, `trainer`, `scaler`, `train`/`test` flags, `train_participants`, `test_participants`, etc.
- **`alarm_calibration`** (optional) — Post-hoc alarm recalibration. See [ALARM_RECALIBRATION_USAGE.md](ALARM_RECALIBRATION_USAGE.md).

See [.github/instructions/config.instructions.md](.github/instructions/config.instructions.md) for a complete field-by-field reference with types, defaults, and valid values.

See `experiments/` subdirectories for example configs.

### Supported Datasets

| Dataset | Loader | Notes |
|---------|--------|-------|
| OhioT1DM (2018, 2020) | `DataloaderOhio` | CGM + insulin + meals + heart rate |
| Tidepool SAP-100 | `DataloaderTidepoolSAP100` | 100 participants, sensor-augmented pump |
| Tidepool HCL-150 | `DataloaderTidepoolHCL150` | 150 participants, hybrid closed-loop |
| T1DEXI | `DataloaderT1DEXI` | Type 1 diabetes exercise study |
| GlucoBench (5 sites) | `DataloaderGlucobench` | Colas, Broll, Hall, Dubosson, Weinstock |
| Shanghai (T1DM, T2DM) | `DataloaderShanghai` | Chinese hospital cohort |
| AI4Food | `DataloaderAI4Food` | Diet study |
| Geneva | `DataloaderGeneva` | Feasibility study |

### Supported Architectures

| Architecture | Config name | Trainer |
|---|---|---|
| Multi-layer LSTM | `TidepoolLSTM` | `TrainerBasic` |
| Baseline LSTM | `MirshekarianLSTM` | `TrainerBasic` |
| Inverted Transformer | `iTransformer` | `Exp_Long_Term_Forecast` |
| Masked iTransformer | `iTransformerMasked` | `Exp_Long_Term_Forecast` |
| **MT-UCT** (paper model) | `MTUCT` | `Exp_Long_Term_Forecast` |
| Linear Regression | — | `SciKitLinearRegressionModel` |

`MTUCT` is the canonical config name for the paper's MT-UCT model. The legacy name `BGiTransformer` is still accepted as an alias so older `model_config.json` files keep working. The two LSTM and linear-regression entries correspond to the baselines reported in the experiments.

### Evaluation Outputs

After running with `"test": true`, the pipeline generates in `experiments/<name>/evaluation/`:

| File | Contents |
|------|----------|
| `rmse_summary.csv` | Per-participant RMSE/MAE by horizon and glycemic zone |
| `cg_ega_summary.csv` | CG-EGA accuracy (AP), bias (BE), error (EP) per horizon |
| `event_level_summary_multi_threshold.csv` | Alarm precision/recall/F1 at multiple thresholds |
| `calibration_summary.csv` | Uncertainty calibration (PICE, PICP) |
| `alarm_calibration_metrics_*.json` | Brier score, ECE for hyper/hypoglycemia |
| `aggregate_metrics.json` | Median MAE, MSE, RMSE across all samples |

### Hyperparameter Sweeps

Edit the `param_grid` dict in `run_multiple.py` or `run_multiple2.py`:

```python
param_grid = {
    "learning_rate": [1e-3, 1e-4, 1e-5],
    "batch_size": [32],
    "feature_window": [48, 72, 96],
    "hidden_dim": [512, 256, 128],
    "dataset_config": ["all"],  # or ["ohio", "tidepool_sap100", ...]
}
```

Each combination creates a subfolder like `run_lr-0.001_bs-32_fw-96_hd-512_...` with its own config, model, and results.

### Missing Data Handling

The pipeline uses sentinel values for missing/masked data:

| Value | Meaning |
|-------|---------|
| `-9` | Missing or disabled feature |
| `-8` | Padding marker |
| `-6` | Masked patch (data augmentation) |
| `-5` | Future forecast period |

### Alarm Recalibration

Post-hoc histogram binning recalibration is supported for alarm probabilities. See [ALARM_RECALIBRATION_USAGE.md](ALARM_RECALIBRATION_USAGE.md) for details.

## Backwards Compatibility

Root-level shim files (`datahandler.py`, `dataprepper.py`, `custom_dataset.py`, `scaler.py`, `evaluator.py`, `alarm_evaluation.py`, `forecast_evaluation.py`) re-export from their new package locations. Old imports like `from ml_pipeline import main` or `from datahandler import DataHandler` continue to work.

## Citation

If you use this code or the MT-UCT model in your research, please cite:

```bibtex
@article{strommen2026mtuct,
  title   = {Multi-task Transformer with Unified Clinical Tokenizer for Effective Blood Glucose Prediction},
  author  = {Strommen, Knut J. and Panagiotou, Maria and Brigato, Lorenzo and Mougiakakou, Stavroula},
  journal = {IEEE Journal of Biomedical and Health Informatics},
  year    = {2026}
}
```

## Acknowledgements

This work was supported by the Stiftung Sanitas (Sanitas Diabetes Technologie 2.0 project) and by the European Commission together with the Swiss Confederation–State Secretariat for Education, Research and Innovation (SERI) under project 101057730 MELISSA (Mobile Artificial Intelligence Solution for Diabetes Adaptive Care). Tidepool data were accessed via Vivli, Inc.; Vivli has not contributed to or approved, and is not responsible for, the contents of this repository.