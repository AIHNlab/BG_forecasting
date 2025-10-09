from ml_pipeline import main
import os
import json
import itertools
import copy


# load base config
base_config = 'BestHpLstm'
experiment_path = os.path.join('experiments', base_config)
#experiment_path = os.path.join('experiments','LinRegTest2H')

model_config_path = experiment_path+os.sep+'model_config.json'
config = json.load(open(model_config_path))

# enforce which datasets/dataloaders to use together
dataset_configs = {
    # grouped mapping for all Glucobench sites
    "all": {
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
            "Tidepool_SAP100"
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
            "DataloaderTidepoolSAP100"
        ],
    },
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
    "AI4food": {
        "dataset_names": ["AI4Food"],
        "dataloaders": ["DataloaderAI4Food"]
    },
    # individual Glucobench sites (one-to-one mappings)
    "Colas": {
        "dataset_names": ["Glucobench_Colas"],
        "dataloaders": ["DataloaderGlucobench"],
    },
    "Broll": {
        "dataset_names": ["Glucobench_Broll"],
        "dataloaders": ["DataloaderGlucobench"],
    },
    "Hall": {
        "dataset_names": ["Glucobench_Hall"],
        "dataloaders": ["DataloaderGlucobench"],
    },
    "Dubosson": {
        "dataset_names": ["Glucobench_Dubosson"],
        "dataloaders": ["DataloaderGlucobench"],
    },
    "Weinstock": {
        "dataset_names": ["Glucobench_Weinstock"],
        "dataloaders": ["DataloaderGlucobench"],
    },

    "ShanghaiT1DM": {
        "dataset_names": ["Shanghai_T1DM"],
        "dataloaders": ["DataloaderShanghai"],
    },
    "ShanghaiT2DM": {
        "dataset_names": ["Shanghai_T2DM"],
        "dataloaders": ["DataloaderShanghai"],
    },

    "TidepoolHCL150": {
        "dataset_names": ["Tidepool_HCL150"],
        "dataloaders": ["DataloaderTidepoolHCL150"],
    },

    "T1DEXI": {
        "dataset_names": ["T1DEXI"],
        "dataloaders": ["DataloaderT1DEXI"],
    },

    "TidepoolSAP100": {
        "dataset_names": ["Tidepool_SAP100"],
        "dataloaders": ["DataloaderTidepoolSAP100"],
    },

    # ohio kept as the grouped mapping (two datasets)
    "Ohio": {
        "dataset_names": ["Ohio2018", "Ohio2020"],
        "dataloaders": ["DataloaderOhio", "DataloaderOhio"],
    },
}


# simple grid (add the params you want to sweep)
if config["run_config"]["test"]: #and not config["run_config"]["train"]:
    param_grid = {
        "dataset_config": ["Ohio", "TidepoolHCL150", "TidepoolSAP100", "T1DEXI", "AI4food", "Colas", "Broll", "Hall", "Dubosson", "Weinstock", "ShanghaiT1DM", "ShanghaiT2DM"],
    }
else:
    param_grid = {
        "learning_rate": [1e-3,1e-4,1e-5],
        "batch_size": [32],
        "feature_window": [48,72,96],
        "hidden_dim": [512,256,128],
        "lstm_layers": [3,4,5],
        "first_dense_dim": [512,256,128],
        "dataset_config": ["all"],
        #"dataset_config": ["ohio", "tidepool_hcl150", "tidepool_sap100", "t1dexi", "ai4food", "glucobench_colas", "glucobench_broll", "glucobench_hall", "glucobench_dubosson", "glucobench_weinstock", "shanghai_t1dm", "shanghai_t2dm"],  # new: dataset configurations
        # architecture is taken from the base config and no longer included in the sweep
        # allow specifying an optional model tag to include in run names / filenames
        #"dataset_config": ["shanghai_t2dm"]
    }

# add an optional user-provided model tag (keeps backward compatibility when empty)
param_grid.update({
    "model_tag": ["BestHpLstm"]
})



base_experiment_dir = os.path.join("/experiments", base_config)
os.makedirs(base_experiment_dir, exist_ok=True)

# short name mapping for hyperparameters to keep experiment folder names compact
hp_name_map = {
    'learning_rate': 'lr',
    'batch_size': 'bs',
    'feature_window': 'fw',
    'hidden_dim': 'hd',
    'lstm_layers': 'll',
    'architecture': 'arch',
    'model_tag': 'tag',
    'dataset_config': 'ds',
    'first_dense_dim': 'fd'
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
    # architecture is fixed in the base config; do not sweep it here
    other_keys = [k for k in param_grid if k not in ["dataset_config"]]
    other_values = [param_grid[k] for k in other_keys]
    run_idx = 0

    # read architecture from the base config; fall back to run_config if present
    arch = config.get("hp_config", {}).get("architecture") or config.get("run_config", {}).get("architecture")
    if not arch:
        raise RuntimeError("No architecture found in base config (hp_config.architecture or run_config.architecture)")



    for dataset_key in param_grid["dataset_config"]:
        dataset_info = dataset_configs[dataset_key]
        for combo in itertools.product(*other_values):
            run_idx += 1
            run_hp = dict(zip(other_keys, combo))

            cfg = copy.deepcopy(config)
            cfg["hp_config"].update(run_hp)
            # ensure architecture remains what's in the base config
            cfg["hp_config"]["architecture"] = arch
            cfg["run_config"]["dataset_names"] = dataset_info["dataset_names"]
            cfg["run_config"]["dataloaders"] = dataset_info["dataloaders"]

            # build compact run name from only the parameters present in param_grid
            parts = []
            for k in param_grid.keys():
                # determine the value for this grid key
                if k == "dataset_config":
                    v = dataset_key
                elif k == "model_tag":
                    v = run_hp.get(k, "")
                else:
                    v = run_hp.get(k, None)
                    if v is None:
                        v = cfg.get("hp_config", {}).get(k, cfg.get("run_config", {}).get(k, ""))

                short_key = hp_name_map.get(k, k)
                # for model_tag, add a clearer prefix and skip if empty
                if k == "model_tag":
                    if v:
                        parts.append(f"tag-{safe_str(v)}")
                else:
                    parts.append(f"{short_key}-{safe_str(v)}")

            run_name = f"run_" + "_".join(parts)
            cfg["run_config"]["experiment_path"] = os.path.join(base_experiment_dir, run_name)
            os.makedirs(cfg["run_config"]["experiment_path"], exist_ok=True)
            # include the model tag in the saved config filename if provided
            model_cfg_name = "model_config.json"
            tag = cfg.get("hp_config", {}).get("model_tag", "")
            if tag:
                # safe filename for tag
                model_cfg_name = f"model_config_tag-{safe_str(tag)}.json"

            with open(os.path.join(cfg["run_config"]["experiment_path"], model_cfg_name), "w", encoding="utf-8") as f:
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