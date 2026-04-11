import os
import torch
from torch.utils.data import DataLoader, Subset
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from data.prepper import DataPrepper
from data.dataset import CustomDataset, EventBalancedSampler
from data.scaler import Scaler
from utils import IdentityTransformer

# Trainer classes must be in scope for globals() lookup from config
from trainers.trainer_basic import TrainerBasic
from trainers.exp_long_term_forecasting import Exp_Long_Term_Forecast
from trainers.trainer_linreg import DartsLinearRegressionModel, SciKitLinearRegressionModel

from pipeline.orchestrator import init_experiment_directory

_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def create_temporal_split(full_dataset, val_ratio=0.05):
    """
    Creates a temporal train-validation split.
    
    Args:
        full_dataset: The complete dataset
        val_ratio: Ratio of data to use for validation (default: 0.05)
    
    Returns:
        train_indices, val_indices
    """
    # Get total number of sequences
    total_sequences = len(full_dataset)
    
    # Calculate split point
    split_idx = int(total_sequences * (1 - val_ratio))
    
    # Create temporal splits
    train_indices = list(range(0, split_idx))
    val_indices = list(range(split_idx, total_sequences))
    
    return train_indices, val_indices

def train_model(config, full_dataset, hypoglycemia_threshold, hyperglycemia_threshold):
    

    # Get all indices
    indices = list(range(len(full_dataset)))

    # Split indices
    #train_indices, val_indices = train_test_split(indices, test_size=0.2, random_state=42)
    train_indices, val_indices = create_temporal_split(full_dataset)

    # Create Subset Datasets
    train_dataset = Subset(full_dataset, train_indices)
    val_dataset = Subset(full_dataset, val_indices)
    if False:
        train_sampler = EventBalancedSampler(train_dataset, batch_size=config['hp_config']['batch_size'], threshold_low=hypoglycemia_threshold, threshold_high=hyperglycemia_threshold)
        train_loader = DataLoader(train_dataset, batch_size=config['hp_config']['batch_size'], sampler=train_sampler)
    else:
        train_loader = DataLoader(train_dataset, shuffle=True, batch_size=config['hp_config']['batch_size'])
    val_loader = DataLoader(val_dataset, shuffle=False, batch_size=config['hp_config']['batch_size'])
    config['hp_config']['hypoglycemia_threshold'] = hypoglycemia_threshold
    config['hp_config']['hyperglycemia_threshold'] = hyperglycemia_threshold
    # Train the model
    retrain_model = True
    if config['run_config']['parent_model_path'] is not None:
        retrain_model = False
    trainer = globals()[config["run_config"]["trainer"]](config['hp_config'], _PROJECT_ROOT+os.sep+config['run_config']['experiment_path']+os.sep+'best_model.pth',retrain_model=retrain_model)
    trainer.train(train_loader, val_loader)


def train_model_glucobench(config, dataframes_train, dataframes_val, retrain_model=False):
    init_experiment_directory(config)
    if config["run_config"]['train'] == False:
        return
    #if config["run_config"]['scaler'] == "StandardScaler":
    #    scaler_class_x = StandardScaler()
    #    scaler_class_y = StandardScaler()
    scaler_class_x = globals()[config["run_config"]["scaler"]]()
    scaler_class_y = globals()[config["run_config"]["scaler"]]()
    
    train_prepper = DataPrepper(dataframes_train.keys(), 
                                dataframes_train, 
                                feature_list=config['run_config']['features'], 
                                target_list=config['run_config']['targets'], 
                                allowed_missing_values_rate=config['run_config']['allowed_missing_values_rate'],
                                allowed_missing_values_rate_target=config['run_config']['allowed_missing_values_rate_target'],
                                forecast_steps=config['hp_config']['forecast_steps'], 
                                scaler_class_x=scaler_class_x, 
                                scaler_class_y=scaler_class_y, 
                                patch_size=config['hp_config']['patch_size'],
                                fill_types=config['run_config']['fill_types'], 
                                experiment_path=_PROJECT_ROOT+os.sep+config['run_config']['experiment_path'],
                                step=config['run_config']['step_training'],
                                sequence_length=config['hp_config']['feature_window'],
                                history_of_days=config['hp_config']['history_of_days'],
                                mask_future_target_covariates=True,
                                disabled_covariates=config['run_config']['disabled_covariates'],
                                context_limit=config['hp_config']['context_limit'],
                                baseline=config['hp_config']['baseline'])
    features_train, target_train = train_prepper.make_features_and_targetpair()
    
    val_prepper = DataPrepper(dataframes_val.keys(), 
                              dataframes_val, 
                              feature_list=config['run_config']['features'], 
                              target_list=config['run_config']['targets'], 
                              allowed_missing_values_rate=config['run_config']['allowed_missing_values_rate'],
                              allowed_missing_values_rate_target=config['run_config']['allowed_missing_values_rate_target'],
                              forecast_steps=config['hp_config']['forecast_steps'], 
                              scaler_class_x=scaler_class_x, 
                              scaler_class_y=scaler_class_y, 
                              patch_size=config['hp_config']['patch_size'],
                              fill_types=config['run_config']['fill_types'], 
                              experiment_path=_PROJECT_ROOT+os.sep+config['run_config']['experiment_path'],
                              step=config['run_config']['step_training'],
                              sequence_length=config['hp_config']['feature_window'],
                              history_of_days=config['hp_config']['history_of_days'],
                              mask_future_target_covariates=True,
                              disabled_covariates=config['run_config']['disabled_covariates'],
                              context_limit=config['hp_config']['context_limit'],
                              baseline= config['hp_config']['baseline'])
    features_val, target_val = val_prepper.make_features_and_targetpair()
    
    # Split the training data into training and validation sets
    features_train, features_val, target_train, target_val = train_test_split(
        features_train, target_train, test_size=0.2, random_state=42)
    # Create CustomDatasets for the training and validation sets
    train_data = CustomDataset(features_train, target_train, config['run_config']['features'], config['run_config']['targets'], config['hp_config']['history_of_days'], config['hp_config']['forecast_steps'], config['run_config']['test_target'], config['hp_config']['days_to_mask'])
    val_data = CustomDataset(features_val, target_val, config['run_config']['features'], config['run_config']['targets'], config['hp_config']['history_of_days'], config['hp_config']['forecast_steps'], config['run_config']['test_target'], config['hp_config']['days_to_mask'])
    # Create DataLoaders
    train_loader = DataLoader(train_data, shuffle=True, batch_size=config['hp_config']['batch_size'])
    val_loader = DataLoader(val_data, shuffle=False, batch_size=config['hp_config']['batch_size'])

    # Train the model
    #retrain_model = True
    #if config['run_config']['parent_model_path'] is not None:
    #    retrain_model = False
    trainer = globals()[config["run_config"]["trainer"]](config['hp_config'],_PROJECT_ROOT+os.sep+config['run_config']['experiment_path']+os.sep+'best_model.pth',retrain_model=retrain_model)
    trainer.train(train_loader, val_loader)
