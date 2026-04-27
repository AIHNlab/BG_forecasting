# Project Guidelines

## Architecture

This is a blood glucose forecasting ML pipeline. Code is organized into packages:

- `pipeline/` — orchestration (`main()`), training, evaluation
- `data/` — data loading (`DataHandler`), preprocessing (`DataPrepper`), PyTorch datasets (`SequenceDataset`), scaling (`Scaler`)
- `evaluation/` — forecast metrics (RMSE, CG-EGA), alarm metrics, uncertainty calibration
- `dataloaders/` — one loader per dataset, all inherit from `dataloaders.dataloader.Dataloader` (ABC)
- `trainers/` — training loops: `TrainerBasic` (LSTM), `Exp_Long_Term_Forecast` (Transformers), `SciKitLinearRegressionModel`
- `architectures/` — model definitions: `TidepoolLSTM`, `iTransformer`, `iTransformerMasked`, `MTUCT` (paper model; `BGiTransformer` is a legacy alias re-exporting from `MTUCT`)

Root-level files like `ml_pipeline.py`, `datahandler.py`, `dataprepper.py` etc. are **backwards-compatibility shims** that re-export from the packages above. Do not add new code to them.

## Conventions

- **Config-driven**: Experiments are configured via `model_config.json` with `hp_config` and `run_config` sections. Do not change the config schema without discussion.
- **Dynamic class loading**: Trainer, scaler, and architecture classes are loaded via `globals()[class_name_string]`. Any new trainer/architecture class must be imported into the module where `globals()` is called (`pipeline/training.py`, `pipeline/evaluation.py`, `pipeline/orchestrator.py`).
- **Missing data sentinels**: `-9` (missing), `-8` (padding), `-6` (masked patch), `-5` (future). These are checked throughout the pipeline — do not change their values.
- **Project root**: Files in subdirectories use `_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))` to resolve paths relative to the repo root.

## Build and Test

```bash
# Activate virtual environment
.venv\Scripts\activate   # Windows

# Run a single experiment
python ml_pipeline.py

# Run hyperparameter sweep
python run_multiple.py    # LSTM
python run_multiple2.py   # Transformer

# Compile-check all packages
python -m py_compile pipeline/orchestrator.py
python -m py_compile data/handler.py
python -m py_compile evaluation/alarm.py
```

No test suite exists yet. Verify changes by running a small experiment end-to-end.

## Code Style

- Python 3.10+
- No strict linter enforced — follow existing code style
- Prefer explicit imports over `from module import *`
- Use `os.path.join()` and `os.sep` for path construction (cross-platform)
