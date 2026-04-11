---
description: "Use when creating, editing, or debugging model_config.json experiment configurations. Covers all hp_config and run_config fields, valid values, and which architecture/trainer combinations are supported."
---
# Configuration Reference (`model_config.json`)

Every experiment is driven by a `model_config.json` file with two required sections: `hp_config` (model hyperparameters) and `run_config` (experiment settings). An optional `alarm_calibration` section enables post-hoc alarm recalibration.

> **Note**: A `hp_search_config` section may exist in some configs but is unused by the pipeline.

---

## `hp_config` — Model Hyperparameters

### Training

| Field | Type | Example | Description |
|-------|------|---------|-------------|
| `batch_size` | int | `32` | Batch size for DataLoaders during training and evaluation. |
| `num_epochs` | int | `100` | Number of training epochs. |
| `learning_rate` | float | `1e-5` | Learning rate for the Adam optimizer. |
| `weight_decay` | float | `0.0001` | L2 regularization coefficient. Only used by `Exp_Long_Term_Forecast`. |
| `lradj` | string | `"cosine_annealing_warmup"` | Learning rate schedule. Only used by `Exp_Long_Term_Forecast`. Options: `"cosine_annealing_warmup"`, `"type1"`. |
| `lr_update_interval` | int | `2500` | Steps between LR scheduler restarts (for `cosine_annealing_warmup`). Only used by `Exp_Long_Term_Forecast`. |

### Architecture Selection

| Field | Type | Example | Description |
|-------|------|---------|-------------|
| `architecture` | string | `"iTransformerMasked"` | Model class to instantiate. Must match an imported class name. Valid values: `"TidepoolLSTM"`, `"MirshekarianLSTM"`, `"iTransformer"`, `"iTransformerMasked"`, `"BGiTransformer"`. |
| `model_tag` | string | `"Transformer"` | Descriptive label for experiment naming in hyperparameter sweeps. Not used by the pipeline itself. |

#### Architecture ↔ Trainer Mapping

| Architecture | Required Trainer |
|---|---|
| `TidepoolLSTM` | `TrainerBasic` |
| `MirshekarianLSTM` | `TrainerBasic` |
| `iTransformer` | `Exp_Long_Term_Forecast` |
| `iTransformerMasked` | `Exp_Long_Term_Forecast` |
| `BGiTransformer` | `Exp_Long_Term_Forecast` |

### LSTM-Specific Parameters

These fields are only used when `architecture` is `TidepoolLSTM` or `MirshekarianLSTM`:

| Field | Type | Example | Description |
|-------|------|---------|-------------|
| `hidden_dim` | int | `512` | LSTM hidden layer dimension. |
| `lstm_layers` | int | `5` | Number of stacked LSTM layers. |
| `first_dense_dim` | int | `256` | Size of the first fully-connected layer after the LSTM. Defaults to `hidden_dim // 2` if omitted. |

### Transformer-Specific Parameters

These fields are only used when `architecture` is `iTransformer`, `iTransformerMasked`, or `BGiTransformer`:

| Field | Type | Example | Description |
|-------|------|---------|-------------|
| `token_size` | int | `256` | Transformer embedding dimension (mapped to `d_model` internally). |
| `encoder_layers` | int | `6` | Number of transformer encoder layers. |
| `n_features` | int | `3` | Number of input feature channels. Used for periodicity reshaping in masked transformers. |
| `freeze_encoder` | bool | `false` | Freeze encoder parameters during training (for transfer learning). |
| `reconstruction` | bool | `true` | Enable masked autoencoder reconstruction task alongside forecasting. |

### Sequence & Windowing

| Field | Type | Example | Description |
|-------|------|---------|-------------|
| `feature_window` | int | `2304` | Input sequence length (number of timesteps of history). At 5-min resolution: 288 = 1 day, 2304 = 8 days. |
| `forecast_steps` | int | `24` | Number of future timesteps to predict. At 5-min resolution: 24 = 2 hours. |
| `patch_size` | int | `24` | Patch/token size for transformer architectures. Also used as the sequence length unit in `DataPrepper`. |
| `forecast_horizons` | list[int] | `[6, 12, 24]` | Horizons at which to compute per-step evaluation metrics (RMSE, MAE, CG-EGA). Each value is a step index within the `forecast_steps` window. |
| `history_of_days` | int | `0` | Number of past days to include as additional historical context beyond `feature_window`. 0 = no extra days. |
| `context_limit` | int or null | `null` | Maximum historical context length. `null` = no limit. Used in robustness/ablation studies. |

### Data Augmentation

| Field | Type | Example | Description |
|-------|------|---------|-------------|
| `mask_ratio` | float | `0.0` | Probability of masking patches during training (reconstruction pretext task). Range: 0.0–1.0. |
| `chance_feature_missing` | float | `0.0` | Probability of masking an entire feature channel in a training sample. Simulates sensor dropout. Range: 0.0–1.0. |
| `chance_of_smbg` | float | `0.0` | Probability of subsampling the CGM feature to simulate sparse self-monitored blood glucose (SMBG) readings. Range: 0.0–1.0. |
| `baseline` | bool | `false` | When `true`, disables all covariate features (only CGM remains). Used for baseline comparisons. |

### Clinical Thresholds

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `hypoglycemia_threshold` | int | `70` | Blood glucose level (mg/dL) below which hypoglycemia is classified. Used for alarm labels and zone-stratified RMSE. |
| `hyperglycemia_threshold` | int | `180` | Blood glucose level (mg/dL) above which hyperglycemia is classified. Used for alarm labels and zone-stratified RMSE. |

### Unused / Informational Fields

These fields may appear in configs but are **not read by the pipeline**:

| Field | Notes |
|-------|-------|
| `input_dim` | Not referenced in code. The number of input features is determined by the `features` list in `run_config`. |
| `output_dim` | Not referenced in code. Output dimension is always 1 (CBG). |
| `n_targets` | Not referenced in code. The number of targets is determined by the `targets` list. |
| `dropout` | Not read from config. Hardcoded to `0.1` in `Exp_Long_Term_Forecast`. |

---

## `run_config` — Experiment Settings

### Experiment Identity

| Field | Type | Example | Description |
|-------|------|---------|-------------|
| `experiment_path` | string | `"/experiments/MyExperiment"` | Path (relative to project root) where all experiment outputs are saved: model weights, scalers, evaluation CSVs, plots. |
| `parent_model_path` | string or null | `null` | Path to a pre-trained experiment folder. When set, copies `best_model.pth`, `scaler_input.pkl`, and `scaler_target.pkl` from this folder to the new experiment (transfer learning). Set to `null` to train from scratch. |

### Data Selection

| Field | Type | Example | Description |
|-------|------|---------|-------------|
| `dataset_names` | list[string] | `["Ohio2018", "Ohio2020"]` | Dataset identifiers. Must match names used in `standardized_datasets/`. Each entry pairs with the corresponding entry in `dataloaders`. |
| `dataloaders` | list[string] | `["DataloaderOhio", "DataloaderOhio"]` | Dataloader class names. Must match imported class names in `data/handler.py`. Same length as `dataset_names`. |
| `train_participants` | string or list | `"all"` | `"all"` to use all training participants, or a list of specific participant IDs like `["540", "544"]`. |
| `test_participants` | string or list | `"all"` | `"all"` to use all test participants, or a list of specific participant IDs. |

### Feature Configuration

| Field | Type | Example | Description |
|-------|------|---------|-------------|
| `features` | list[string] | `["cbg", "bolus", "carbInput"]` | Column names from the loaded DataFrames to use as model input features. |
| `targets` | list[string] | `["cbg", "bolus", "carbInput"]` | Column names used as prediction targets. For Transformer models with reconstruction, this typically matches `features`. For LSTM, usually just `["cbg"]`. |
| `test_target` | string | `"cbg"` | Which target column is the primary evaluation target. Used to select the correct output channel for RMSE/MAE computation. |
| `disabled_covariates` | list[int] | `[0, 0, 0]` | Binary mask (one per feature). `1` = disable this feature (set to sentinel value), `0` = keep. Used for feature ablation studies. Same length as `features`. |

### Missing Data & Quality

| Field | Type | Example | Description |
|-------|------|---------|-------------|
| `allowed_missing_values_rate` | list[float] | `[1.0, 1.0, 1.0]` | Maximum fraction of NaN values allowed per feature in a training sequence. Sequences exceeding this are dropped. One value per feature. `1.0` = allow all missing, `0.0` = require no missing values. |
| `allowed_missing_values_rate_target` | list[float] | `[1.0, 1.0, 1.0]` | Same as above but for target channels. |
| `fill_types` | list[int] | `[-9, -9, -9]` | Sentinel values used to fill padding at sequence boundaries. One per feature. Common values: `-9` (missing marker), `-5` (future marker). |
| `required_samples_window` | int | `24` | During evaluation, the number of preceding timesteps to check for valid data. Predictions following gaps larger than this window are masked as invalid. Default: `24`. |
| `required_samples_during_test` | list[int] | `[24, 0, 0]` | Per-channel minimum number of valid (non-NaN) samples required in the `required_samples_window`. If not met, the prediction at that timestep is masked. Default: `[24, 1, 1]`. |

### Preprocessing

| Field | Type | Example | Description |
|-------|------|---------|-------------|
| `rolling_mean_window` | list[int] | `[1, 12, 12]` | Gaussian-weighted rolling mean window size per feature. `1` = no smoothing. Higher values smooth noisy features like bolus/carbs. |
| `step_training` | int | `10` | Stride when generating training sequences. `1` = every possible window (dense), `10` = skip 9 between windows (sparser, faster). Does not affect test sequences (always stride 1). |
| `indices_per_day` | int | `288` | Number of data points per day (288 for 5-minute resolution). Informational — not actively used by the pipeline. |
| `scaler` | string | `"StandardScaler"` | Feature normalization class. Loaded via `globals()`. Valid values: `"StandardScaler"` (sklearn), `"IdentityTransformer"` (no scaling). |

### Pipeline Control

| Field | Type | Example | Description |
|-------|------|---------|-------------|
| `train` | bool | `true` | Run the training phase. |
| `test` | bool | `true` | Run the evaluation phase. Can be `true` independently of `train` (to re-evaluate a previously trained model). |
| `trainer` | string | `"Exp_Long_Term_Forecast"` | Trainer class name. Loaded via `globals()`. Valid values: `"TrainerBasic"`, `"Exp_Long_Term_Forecast"`, `"SciKitLinearRegressionModel"`, `"DartsLinearRegressionModel"`. Must match the architecture (see table above). |

---

## `alarm_calibration` — Post-hoc Alarm Recalibration (Optional)

This is a **top-level** config section (sibling of `hp_config` and `run_config`), not nested inside either.

| Field | Type | Example | Description |
|-------|------|---------|-------------|
| `enabled` | bool | `true` | Enable histogram binning recalibration of alarm probabilities during inference. |
| `hyperglycemia_path` | string or null | `"experiments/.../alarm_calibration_metrics_hyperglycemia.json"` | Path to pre-computed hyperglycemia calibration JSON. |
| `hypoglycemia_path` | string or null | `"experiments/.../alarm_calibration_metrics_hypoglycemia.json"` | Path to pre-computed hypoglycemia calibration JSON. |

Omit this section entirely or set `"enabled": false` for default behavior (raw model probabilities).

See [ALARM_RECALIBRATION_USAGE.md](../../ALARM_RECALIBRATION_USAGE.md) for workflow details.

---

## Minimal Config Examples

### LSTM (TidepoolLSTM)

```json
{
    "hp_config": {
        "batch_size": 32,
        "num_epochs": 100,
        "learning_rate": 0.001,
        "architecture": "TidepoolLSTM",
        "hidden_dim": 512,
        "lstm_layers": 5,
        "first_dense_dim": 256,
        "feature_window": 96,
        "forecast_steps": 24,
        "patch_size": 24,
        "forecast_horizons": [6, 12, 24],
        "mask_ratio": 0.0,
        "chance_feature_missing": 0.0,
        "chance_of_smbg": 0.0,
        "history_of_days": 0,
        "hypoglycemia_threshold": 70,
        "hyperglycemia_threshold": 180,
        "context_limit": null,
        "baseline": false
    },
    "run_config": {
        "experiment_path": "/experiments/MyLSTMExperiment",
        "dataset_names": ["Ohio2018", "Ohio2020"],
        "dataloaders": ["DataloaderOhio", "DataloaderOhio"],
        "features": ["cbg", "bolus", "carbInput"],
        "targets": ["cbg"],
        "allowed_missing_values_rate": [0.5, 1.0, 1.0],
        "allowed_missing_values_rate_target": [0.0],
        "fill_types": [-9, -9, -9],
        "rolling_mean_window": [1, 12, 12],
        "disabled_covariates": [0, 0, 0],
        "required_samples_during_test": [24, 0, 0],
        "test_target": "cbg",
        "step_training": 1,
        "train": true,
        "test": true,
        "train_participants": "all",
        "test_participants": "all",
        "parent_model_path": null,
        "trainer": "TrainerBasic",
        "scaler": "StandardScaler"
    }
}
```

### Transformer (iTransformerMasked)

```json
{
    "hp_config": {
        "batch_size": 32,
        "num_epochs": 100,
        "learning_rate": 1e-05,
        "architecture": "iTransformerMasked",
        "token_size": 256,
        "encoder_layers": 6,
        "n_features": 3,
        "feature_window": 2304,
        "forecast_steps": 24,
        "patch_size": 24,
        "forecast_horizons": [6, 12, 24],
        "mask_ratio": 0.0,
        "chance_feature_missing": 0.0,
        "chance_of_smbg": 0.0,
        "history_of_days": 0,
        "hypoglycemia_threshold": 70,
        "hyperglycemia_threshold": 180,
        "context_limit": null,
        "baseline": false,
        "reconstruction": true,
        "freeze_encoder": false,
        "weight_decay": 0.0001,
        "lradj": "cosine_annealing_warmup",
        "lr_update_interval": 2500,
        "model_tag": "Transformer"
    },
    "run_config": {
        "experiment_path": "/experiments/MyTransformerExperiment",
        "dataset_names": ["Tidepool_SAP100"],
        "dataloaders": ["DataloaderTidepoolSAP100"],
        "features": ["cbg", "bolus", "carbInput"],
        "targets": ["cbg", "bolus", "carbInput"],
        "allowed_missing_values_rate": [1.0, 1.0, 1.0],
        "allowed_missing_values_rate_target": [1.0, 1.0, 1.0],
        "fill_types": [-9, -9, -9],
        "rolling_mean_window": [1, 12, 12],
        "disabled_covariates": [0, 0, 0],
        "required_samples_window": 24,
        "required_samples_during_test": [24, 0, 0],
        "test_target": "cbg",
        "step_training": 10,
        "train": true,
        "test": true,
        "train_participants": "all",
        "test_participants": "all",
        "parent_model_path": null,
        "trainer": "Exp_Long_Term_Forecast",
        "scaler": "StandardScaler"
    }
}
```
