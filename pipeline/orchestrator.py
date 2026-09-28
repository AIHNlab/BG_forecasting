"""Experiment orchestration: config handling, data loading, and pipeline dispatch.

The ``main()`` function is the single entry point for running an experiment
end-to-end: it initialises the experiment directory, loads the dataset(s),
configures scalers, and delegates to ``train_model`` / ``evaluate_model``
based on the ``run_config.train`` and ``run_config.test`` flags.
"""

import os
import json
import shutil
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

from sklearn.preprocessing import StandardScaler

from data.handler import DataHandler
from data.prepper import DataPrepper
from data.scaler import Scaler
from utils import IdentityTransformer

_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def init_experiment_directory(config):
    """Create the experiment directory and copy parent model artifacts if specified.

    Copies ``scaler_input.pkl``, ``scaler_target.pkl``, and ``best_model.pth``
    from ``parent_model_path`` when doing fine-tuning.  Also writes a snapshot
    of the current config to ``model_config.json`` in the experiment directory.
    """
    experiment_path = _PROJECT_ROOT + os.sep + config['run_config']['experiment_path']

    os.makedirs(experiment_path, exist_ok=True)
    if config['run_config']['parent_model_path'] is not None:
        parent_model_path = _PROJECT_ROOT + os.sep + config['run_config']['parent_model_path']
        shutil.copy2(os.path.join(parent_model_path, 'scaler_input.pkl'), os.path.join(experiment_path, 'scaler_input.pkl'))
        shutil.copy2(os.path.join(parent_model_path, 'scaler_target.pkl'), os.path.join(experiment_path, 'scaler_target.pkl'))
        shutil.copy2(os.path.join(parent_model_path, 'best_model.pth'), os.path.join(experiment_path, 'best_model.pth'))
    #dump config
    with open(experiment_path+os.sep+'model_config.json', 'w') as f:
        f.write(json.dumps(config, indent=4))

def main(config, train=True, test=True, *, runtime=None):
    """Run a full experiment: load data, optionally train, optionally evaluate.

    Supports both single-dataset and multi-dataset (merged) configurations.
    Trainer and scaler classes are resolved at runtime via ``globals()``.

    Args:
        config: Experiment configuration dict with ``hp_config`` and ``run_config``.
        train: Ignored — controlled by ``config['run_config']['train']``.
        test: Ignored — controlled by ``config['run_config']['test']``.
        runtime: Keyword-only. Optional warm runtime (model, device, bundle digests)
            supplied by the forecast service so the model is not rebuilt per request.
            Ignored unless ``run_config['mode'] == 'forecast'``.
    """
    if config['run_config'].get('mode') == 'forecast':
        # Imported inside the branch so training/evaluation callers never load the
        # service package. Returns before init_experiment_directory, which would
        # otherwise write model_config.json into the shared model directory.
        from service.forecast_run import run_forecast

        return run_forecast(config, runtime=runtime)

    init_experiment_directory(config)
    if len(config['run_config']['dataloaders']) == 1:
        data_handler = DataHandler(config['run_config']['dataloaders'][0], "", dataset_name=config['run_config']['dataset_names'][0])
        data_handler.load_data()
    else:
        data_handler = DataHandler("DataloaderMerged", "", dataset_name="Merged")
        for i, dataloader in enumerate(config['run_config']['dataloaders']):
            data_handler_temp = DataHandler(dataloader, "", dataset_name=config['run_config']['dataset_names'][i])
            data_handler_temp.load_data()
            data_handler.get_train_dataframes().update(data_handler_temp.get_train_dataframes())
            data_handler.get_test_dataframes().update(data_handler_temp.get_test_dataframes())
            data_handler.get_all_dataframes().update(data_handler_temp.get_all_dataframes())
            data_handler.get_train_metadata().update(data_handler_temp.get_train_metadata())
            data_handler.get_test_metadata().update(data_handler_temp.get_test_metadata())
    
    scaler_class_x = globals()[config["run_config"]["scaler"]]()
    scaler_class_y = globals()[config["run_config"]["scaler"]]()
    

    if config['run_config']['train']:
        # Import here to avoid circular imports at module level
        from pipeline.training import train_model

        if config['run_config']['train_participants'] == 'all':
            participants_train = list(data_handler.get_train_dataframes().keys())
        else:
            participants_train = config['run_config']['train_participants']
        prepper = DataPrepper(participants_train, 
                              data_handler.get_train_dataframes(), 
                              feature_list=config['run_config']['features'], 
                              target_list=config['run_config']['targets'], 
                              allowed_missing_values_rate=config['run_config']['allowed_missing_values_rate'],
                              allowed_missing_values_rate_target=config['run_config']['allowed_missing_values_rate_target'],
                              forecast_steps=config['hp_config']['forecast_steps'], 
                              scaler_class_x=scaler_class_x, 
                              scaler_class_y=scaler_class_y, 
                              patch_size=config['hp_config']['patch_size'],
                              fill_types=config['run_config']['fill_types'], 
                              experiment_path= _PROJECT_ROOT+os.sep+config['run_config']['experiment_path'],
                              step=config['run_config']['step_training'],
                              sequence_length=config['hp_config']['feature_window'],
                              history_of_days=config['hp_config']['history_of_days'],
                              mask_prob=config['hp_config']['mask_ratio'],
                              chance_of_smbg=config['hp_config']['chance_of_smbg'],
                              chance_feature_missing=config['hp_config']['chance_feature_missing'],
                              metadata=data_handler.get_train_metadata(),
                              mask_future_target_covariates=True,
                              disabled_covariates=config['run_config']['disabled_covariates'],
                              context_limit=config['hp_config']['context_limit'],
                              baseline= config['hp_config']['baseline'],
                              rolling_mean_window=config['run_config']['rolling_mean_window'])
                              
        dataset = prepper.make_features_and_targetpair()
        #prepper = DataPrepper(participants_test, data_handler, feature_list=config['run_config']['features'], data_type="test", forecast_steps=config['run_config']['forecast_steps'], scaler_class_x=scaler_class_x, scaler_class_y=scaler_class_y, fill_types=config['run_config']['fill_types'], experiment_path=config['run_config']['experiment_path'])
        #features_test, target_test = prepper.make_features_and_targetpair()
        train_model(config, dataset, prepper.hypoglycemia_threshold, prepper.hyperglycemia_threshold)

    if config['run_config']['test']:
        # Import here to avoid circular imports at module level
        from pipeline.evaluation import evaluate_model

        if config['run_config']['test_participants'] == 'all':
            #participants_test = list(data_handler.get_train_dataframes().keys())
            participants_test = list(data_handler.get_test_dataframes().keys())
        else:
            participants_test = config['run_config']['test_participants']

        #evaluate_model(config, data_handler.get_train_dataframes(), scaler_class_x, scaler_class_y, participants=participants_test, metadata=data_handler.get_train_metadata())
        evaluate_model(config, data_handler.get_test_dataframes(), scaler_class_x, scaler_class_y, participants=participants_test, metadata=data_handler.get_test_metadata())
