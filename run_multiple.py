from ml_pipeline import main
import os
import json
import itertools
import copy


config = {
    "hp_config": {
        "batch_size": 32,
        "num_epochs": 100,
        "learning_rate": 1e-05,
        "input_dim": 4,
        "hidden_dim": 128,
        "output_dim": 1,
        "encoder_layers": 6,
        "lstm_layers": 3,
        "patch_size": 24,
        "forecast_steps": 24,
        "architecture": "TidepoolLSTM",
        "token_size": 256,
        "mask_ratio": 0.0,
        "chance_feature_missing": 0.0,
        "chance_of_smbg": 0.0,
        "history_of_days": 0,
        "hypoglycemia_threshold": 70,
        "hyperglycemia_threshold": 180,
        "context_limit": None,
        "baseline": True,
        "reconstruction": True,
        "forecast_horizons": [6, 12, 24],
        "feature_window": 2304,
        "freeze_encoder": False,
        "n_features": 3,
        "n_targets": 1,
        "lr_update_interval": 2500,
        "lradj": "cosine_annealing_warmup",
        "weight_decay": 0.0001,
    },
    "run_config": {
        "experiment_path": "/experiments/LSTMTest",
        "dataset_names": [
            "AI4Food",
            "Glucobench_Colas",
            "Glucobench_Broll",
            "Glucobench_Hall",
            "Glucobench_Dubosson",
            "Glucobench_Weinstock",
            "Shanghai_T1DM",
            "Shanghai_T2DM",
            "Tidepool_HCL150",
            "Ohio2018",
            "Ohio2020",
            "T1DEXI",
            "Tidepool_SAP100",
        ],
        "dataloaders": [
            "DataloaderAI4Food",
            "DataloaderGlucobench",
            "DataloaderGlucobench",
            "DataloaderGlucobench",
            "DataloaderGlucobench",
            "DataloaderGlucobench",
            "DataloaderShanghai",
            "DataloaderShanghai",
            "DataloaderTidepoolHCL150",
            "DataloaderOhio",
            "DataloaderOhio",
            "DataloaderT1DEXI",
            "DataloaderTidepoolSAP100",
        ],
        "indices_per_day": 288,
        "features": ["cbg", "iob", "cob"],
        "allowed_missing_values_rate": [1.0, 1.0, 1.0],
        "targets": ["cbg", "iob", "cob"],
        "allowed_missing_values_rate_target": [1.0, 1.0, 1.0],
        "required_samples_window": 24,
        "required_samples_during_test": [24, 0, 0],
        "disabled_covariates": [0, 0, 0],
        "fill_types": [0, 0, 0],
        "rolling_mean_window": [1, 12, 12],
        "test_target": "cbg",
        "step_training": 10,
        "train": False,
        "test": True,
        "train_participants": "all",
        "test_participants": "all",
        "parent_model_path": "/experiments/LSTMInitialTest/run_002_lr-0.0001_bs-64_fw-48_hd-256_ll-4_arch-TidepoolLSTM_ds-tidepool_sap100",
        "trainer": "TrainerBasic",
        "scaler": "StandardScaler",
    },
}

# enforce which datasets/dataloaders to use together
dataset_configs = {
    # grouped mapping for all Glucobench sites
    "glucobench": {
        "dataset_names": [
            "Glucobench_Colas",
            "Glucobench_Broll",
            "Glucobench_Hall",
            "Glucobench_Dubosson",
            "Glucobench_Weinstock",
        ],
        "dataloaders": [
            "DataloaderGlucobench",
            "DataloaderGlucobench",
            "DataloaderGlucobench",
            "DataloaderGlucobench",
            "DataloaderGlucobench",
        ],
    },
    "ai4food": {
        "dataset_names": ["AI4Food"],
        "dataloaders": ["DataloaderAI4Food"]
    },
    # individual Glucobench sites (one-to-one mappings)
    "glucobench_colas": {
        "dataset_names": ["Glucobench_Colas"],
        "dataloaders": ["DataloaderGlucobench"],
    },
    "glucobench_broll": {
        "dataset_names": ["Glucobench_Broll"],
        "dataloaders": ["DataloaderGlucobench"],
    },
    "glucobench_hall": {
        "dataset_names": ["Glucobench_Hall"],
        "dataloaders": ["DataloaderGlucobench"],
    },
    "glucobench_dubosson": {
        "dataset_names": ["Glucobench_Dubosson"],
        "dataloaders": ["DataloaderGlucobench"],
    },
    "glucobench_weinstock": {
        "dataset_names": ["Glucobench_Weinstock"],
        "dataloaders": ["DataloaderGlucobench"],
    },

    "shanghai_t1dm": {
        "dataset_names": ["Shanghai_T1DM"],
        "dataloaders": ["DataloaderShanghai"],
    },
    "shanghai_t2dm": {
        "dataset_names": ["Shanghai_T2DM"],
        "dataloaders": ["DataloaderShanghai"],
    },

    "tidepool_hcl150": {
        "dataset_names": ["Tidepool_HCL150"],
        "dataloaders": ["DataloaderTidepoolHCL150"],
    },

    "t1dexi": {
        "dataset_names": ["T1DEXI"],
        "dataloaders": ["DataloaderT1DEXI"],
    },

    "tidepool_sap100": {
        "dataset_names": ["Tidepool_SAP100"],
        "dataloaders": ["DataloaderTidepoolSAP100"],
    },

    # ohio kept as the grouped mapping (two datasets)
    "ohio": {
        "dataset_names": ["Ohio2018", "Ohio2020"],
        "dataloaders": ["DataloaderOhio", "DataloaderOhio"],
    },
}


# simple grid (add the params you want to sweep)
param_grid = {
    "learning_rate": [1e-4],
    "batch_size": [64],
    "feature_window": [48],
    "hidden_dim": [256],
    "lstm_layers": [4],
    "architecture": ["TidepoolLSTM"],
    "dataset_config": ["shanghai_t2dm"]
    #"dataset_config": ["t1dexi", "tidepool_sap100", "ohio", "ai4food", "glucobench_colas", "glucobench_broll", "glucobench_hall", "glucobench_dubosson", "glucobench_weinstock", "shanghai_t1dm", "shanghai_t2dm", "tidepool_hcl150"]  # new: dataset configurations
}

# enforce which trainer to use for each architecture
arch_to_trainer = {
    "TidepoolLSTM": "TrainerBasic",
    "LinReg": "SciKitLinearRegressionModel"
}

base_experiment_dir = os.path.join("/experiments", "LSTMInitialTest")
os.makedirs(base_experiment_dir, exist_ok=True)

# short name mapping for hyperparameters to keep experiment folder names compact
hp_name_map = {
    'learning_rate': 'lr',
    'batch_size': 'bs',
    'feature_window': 'fw',
    'hidden_dim': 'hd',
    'lstm_layers': 'll',
    'architecture': 'arch',
    'dataset_config': 'ds'
}

def safe_str(v):
    """Return a filesystem-safe string for a hyperparameter value while preserving readability."""
    if v is None:
        return 'None'
    if isinstance(v, (list, tuple)):
        return '-'.join(safe_str(x) for x in v)
    # keep values readable (e.g. 0.0001) but sanitize path separators and spaces
    return str(v).replace(os.sep, '_').replace(' ', '_')


def make_runs():
    other_keys = [k for k in param_grid if k not in ["architecture", "dataset_config"]]
    other_values = [param_grid[k] for k in other_keys]
    run_idx = 0

    for arch in param_grid["architecture"]:
        trainer = arch_to_trainer[arch]
        for dataset_key in param_grid["dataset_config"]:
            dataset_info = dataset_configs[dataset_key]
            for combo in itertools.product(*other_values):
                run_idx += 1
                run_hp = dict(zip(other_keys, combo))

                cfg = copy.deepcopy(config)
                cfg["hp_config"].update(run_hp)
                cfg["hp_config"]["architecture"] = arch
                cfg["run_config"]["trainer"] = trainer
                cfg["run_config"]["dataset_names"] = dataset_info["dataset_names"]
                cfg["run_config"]["dataloaders"] = dataset_info["dataloaders"]

                # build compact run name from only the parameters present in param_grid
                parts = []
                for k in param_grid.keys():
                    # determine the value for this grid key
                    if k == "architecture":
                        v = arch
                    elif k == "dataset_config":
                        v = dataset_key
                    else:
                        v = run_hp.get(k, None)
                        if v is None:
                            v = cfg.get("hp_config", {}).get(k, cfg.get("run_config", {}).get(k, ""))

                    short_key = hp_name_map.get(k, k)
                    parts.append(f"{short_key}-{safe_str(v)}")

                run_name = f"run_" + "_".join(parts)
                cfg["run_config"]["experiment_path"] = os.path.join(base_experiment_dir, run_name)
                os.makedirs(cfg["run_config"]["experiment_path"], exist_ok=True)
                with open(os.path.join(cfg["run_config"]["experiment_path"], "model_config.json"), "w", encoding="utf-8") as f:
                    json.dump(cfg, f, indent=2)

                print("Starting", run_name)
                try:
                    main(cfg)
                except Exception as e:
                    import traceback
                    tb = traceback.format_exc()
                    print(f"Run {run_name} failed with exception: {e}\n" + tb)
                    # save traceback to experiment folder for inspection
                    err_path = os.path.join(cfg["run_config"]["experiment_path"], "run_error.log")
                    try:
                        with open(err_path, "w", encoding="utf-8") as ef:
                            ef.write(tb)
                    except Exception:
                        # best-effort: don't crash the runner when saving the log
                        pass

if __name__ == "__main__":
    make_runs()