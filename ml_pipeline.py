import os
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split
from torch.utils.data import TensorDataset, DataLoader, Dataset
import torch
import torch.nn as nn
import torch.optim as optim
from dataprepper import DataPrepper
from architectures.lstms import MirshekarianLSTM, TidepoolLSTM
from trainers.trainer_basic import TrainerBasic
from trainers.exp_long_term_forecasting import Exp_Long_Term_Forecast
from trainers.trainer_linreg import DartsLinearRegressionModel, SciKitLinearRegressionModel
from datahandler import DataHandler
from dataloaders.dataloader_tidepool_sap100 import Dataloader
from datapreprocessor import DataPreProcessor
from scaler import Scaler
from evaluator import Evaluator
import warnings
from sklearn.preprocessing import StandardScaler
from utils import IdentityTransformer
warnings.simplefilter(action='ignore', category=FutureWarning)
import json
import numpy as np
import shutil
from custom_dataset import CustomDataset, EventBalancedSampler
from alarm_evaluation import run_evaluation
from forecast_evaluation import plot_and_evaluate_horizon, evaluate_cg_ega_horizon, evaluate_uncertainty_calibration, create_calibration_analysis

#def evaluate_model(hp_config,features_test, target_test, model_path):
#
#    test_data = TensorDataset(features_test, target_test)
#    test_loader = DataLoader(test_data, shuffle=False, batch_size=hp_config['batch_size'])
#
#    model =  globals()[config["run_config"]["architecture"]](hp_config)
#    model.load_state_dict(torch.load(model_path))
#
#    evaluator = Evaluator(model, test_loader)
#    return evaluator.evaluate()
def plot_per_horizon(horizons, plot_data, participants, config):
    for horizon in horizons:
        fig, axs = plt.subplots(len(participants), 1, figsize=(30, 6 * len(participants)))
        if len(participants) == 1:
            axs = [axs]
        
        for idx, (ax, participant) in enumerate(zip(axs, participants)):
            if len(plot_data[horizon]) > idx:
                pred_horizon, actual_horizon, std_horizon = plot_data[horizon][idx]  # Modified to unpack std
                
                ax.plot(actual_horizon, label='Actuals', linestyle='-', linewidth=2, color='blue', alpha=0.7)
                ax.plot(pred_horizon, label='Predictions', linestyle='--', linewidth=1, color='red', alpha=0.7)
                
                # Add confidence intervals
                ax.fill_between(
                    range(len(pred_horizon)),
                    pred_horizon - 2 * std_horizon,
                    pred_horizon + 2 * std_horizon,
                    color='red',
                    alpha=0.2,
                    label='95% Confidence Interval'
                )
                
                ax.legend()
                ax.set_title(f'Participant {participant} - Horizon {horizon}')
                ax.set_xlabel('Time')
                ax.set_ylabel('Values')

        plt.tight_layout()
        evaluation_path = os.path.dirname(__file__)+os.path.join(config['run_config']['experiment_path'], 'evaluation')
        os.makedirs(evaluation_path, exist_ok=True)
        plt.savefig(os.path.join(evaluation_path, f'all_participants_horizon_{horizon}.png'))
        plt.close(fig)

def create_temporal_split(full_dataset, val_ratio=0.05):
    """
    Creates a temporal train-validation split.
    
    Args:
        full_dataset: The complete dataset
        val_ratio: Ratio of data to use for validation (default: 0.2)
    
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
    from torch.utils.data import Subset
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
    trainer = globals()[config["run_config"]["trainer"]](config['hp_config'], os.path.dirname(__file__)+os.sep+config['run_config']['experiment_path']+os.sep+'best_model.pth',retrain_model=retrain_model)
    trainer.train(train_loader, val_loader)

def evaluate_model(config, dataframes, scaler_class_x, scaler_class_y, participants, metadata):
    plot_data = {horizon: [] for horizon in config['hp_config']['forecast_horizons']}
    rmses = {horizon: [] for horizon in config['hp_config']['forecast_horizons']}
    cg_ega_metrics = {horizon: [] for horizon in config['hp_config']['forecast_horizons']}
    rmse_df = pd.DataFrame(
        columns=['Participant'] +
        [f'RMSE_{h}' for h in config['hp_config']['forecast_horizons']] +
        [f'RMSE_hypo_{h}' for h in config['hp_config']['forecast_horizons']] +
        [f'RMSE_hyper_{h}' for h in config['hp_config']['forecast_horizons']] +
        [f'RMSE_normo_{h}' for h in config['hp_config']['forecast_horizons']]
    )
    cg_ega_df = pd.DataFrame(columns=['Participant', 'Horizon', 'AP', 'BE', 'EP'])
    forecast_horizons = config['hp_config']['forecast_horizons']
    event_metrics_all = []
    mse_errors = []
    mae_errors = []
    calibration_summary = []
    historic_context = None  # Ensure variable is always defined
    #i=0
    for participant in participants:
        #i+=1
        #if i==3: break
        participants_test = [participant]
        prepper = DataPrepper(participants_test, 
                              dataframes, 
                              feature_list=config['run_config']['features'], 
                              target_list=config['run_config']['targets'], 
                              allowed_missing_values_rate=config['run_config']['allowed_missing_values_rate'],
                              allowed_missing_values_rate_target=config['run_config']['allowed_missing_values_rate_target'],
                              forecast_steps=config['hp_config']['forecast_steps'], 
                              scaler_class_x=scaler_class_x, 
                              scaler_class_y=scaler_class_y, 
                              patch_size=config['hp_config']['patch_size'],
                              fill_types=config['run_config']['fill_types'], 
                              experiment_path=os.path.dirname(__file__)+os.sep+config['run_config']['experiment_path'], 
                              sequence_length=config['hp_config']['feature_window'], 
                              history_of_days=config['hp_config']['history_of_days'],
                              mask_prob=config['hp_config']['mask_ratio'],
                              chance_of_smbg=config['hp_config']['chance_of_smbg'],
                              chance_feature_missing=config['hp_config']['chance_feature_missing'],
                              metadata=metadata,
                              mask_future_target_covariates=False,
                              disabled_covariates=config['run_config']['disabled_covariates'],
                              context_limit=config['hp_config']['context_limit'],
                              baseline= config['hp_config']['baseline'],
                              rolling_mean_window=config['run_config']['rolling_mean_window']
                              )
        testset = prepper.make_features_and_targetpair()
        #test_data = CustomDataset(features_test, target_test, config['run_config']['features'], config['run_config']['targets'], config['hp_config']['history_of_days'], config['hp_config']['forecast_steps'], config['run_config']['test_target'], config['hp_config']['days_to_mask'])
        test_loader = DataLoader(testset, shuffle=False, batch_size=config['hp_config']['batch_size'])
        trainer = globals()[config["run_config"]["trainer"]](config['hp_config'], os.path.dirname(__file__)+os.sep+config['run_config']['experiment_path']+os.sep+'best_model.pth')
        if config['hp_config']['baseline']:
            # Yes, you can pass a scaler for just one of the channels if your trainer/test method supports it.
            # For example, if you only want to scale channel 0, you could do:
            predictions, actuals = trainer.test(test_loader, scaler=prepper.scaler_y.scaler)
            means, stds, hyper_probs, hypo_probs = None, None, None, None
            predictions = np.where(np.isin(predictions, [-8, -9]), np.nan, predictions)
        else:
            predictions, actuals, means, stds, hyper_probs, hypo_probs, historic_context = trainer.test(test_loader, scaler=prepper.scaler_y.scaler, hypoglycemia_threshold=prepper.hypoglycemia_threshold, hyperglycemia_threshold=prepper.hyperglycemia_threshold, scaler_input=prepper.scaler_x.scaler)
        
        if actuals.size == 0:
            continue

        # Ensure means, stds, hyper_probs, hypo_probs are arrays of correct shape if None
        predictions = np.array(predictions)
        actuals = np.array(actuals)
        # Ensure means, stds, hyper_probs, hypo_probs are arrays of correct shape if None
        means = predictions.copy() if means is None else means
        stds = np.zeros_like(predictions) if stds is None else stds
        hyper_probs = np.zeros_like(predictions[:,0].flatten()) if hyper_probs is None else hyper_probs
        hypo_probs = np.zeros_like(predictions[:,0].flatten()) if hypo_probs is None else hypo_probs

        historic_context = np.array(historic_context) if historic_context is not None else None

        rmse_values = {'Participant': participant}
        if config['hp_config']['forecast_steps'] > 1:
            predictions = np.array(predictions)[:, -config['hp_config']['forecast_steps']:, prepper.test_target_index]
            actuals = np.array(actuals)[:, -config['hp_config']['forecast_steps']:, prepper.test_target_index]
            stds = np.array(stds)[:, -config['hp_config']['forecast_steps']:, prepper.test_target_index] 
            means = np.array(means)[:, -config['hp_config']['forecast_steps']:, prepper.test_target_index]

        mse_errors.append(np.mean((actuals[:,:12] - predictions[:,:12])**2, axis=(1)))
        mae_errors.append(np.mean(np.abs(actuals[:,:12] - predictions[:,:12]), axis=(1)))

        # Save participant metadata
        evaluation_path = os.path.dirname(__file__) + os.sep + os.path.join(config['run_config']['experiment_path'], 'evaluation')
        os.makedirs(evaluation_path, exist_ok=True)
        
        if participant in metadata:
            participant_metadata = metadata[participant]
            metadata_filename = f'metadata_participant_{participant}.json'
            participant_path = os.path.join(evaluation_path, str(participant))
            os.makedirs(participant_path, exist_ok=True)
            with open(os.path.join(participant_path, metadata_filename), 'w') as f:
                json.dump(participant_metadata, f, indent=4)

        # Run evaluation for multiple thresholds
        thresholds = [0.20, 0.35, 0.5, 0.65, 0.80]
        for threshold in thresholds:
            event_metrics = run_evaluation(
                actuals, hyper_probs, hypo_probs, participant, config, threshold=threshold,
                historic_context=historic_context,  # pass the multi-channel array
                required_samples=config['run_config'].get('required_samples_during_test', [24, 1, 1])   # pass the per-channel requirements
            )
            if event_metrics is not None:
                # Add threshold info to the metrics
                event_metrics['threshold'] = threshold
                event_metrics_all.append(event_metrics)
        if event_metrics is not None:
            event_metrics_all.append(event_metrics)

        for horizon in forecast_horizons:
            pred_horizon = np.array(predictions)[:, horizon-1].flatten()
            actual_horizon = np.array(actuals)[:, horizon-1].flatten()
            std_horizon = np.array(stds)[:, horizon-1].flatten()
            mean_horizon = np.array(means)[:, horizon-1].flatten()
            
            if historic_context is not None:
                # Use the input given to the model for masking
                nan_mask = np.isnan(actual_horizon)
                window = config['run_config'].get('required_samples_window', 24)

                # Get required samples for each input channel from config
                required_samples = config['run_config'].get('required_samples_during_test', [24, 1, 1])

                nan_window_mask = np.zeros_like(nan_mask, dtype=bool)
                nan_window_mask[:window] = True

                for i in range(window, len(nan_mask)):
                    # Check input channel requirements using the model's input data
                    should_mask = False

                    if i < len(historic_context):
                        #window_start = max(0, i - window)
                        window_inputs = historic_context[i, -window:, :]  # Shape: [window_size, context, channels]

                        # Check requirements for each input channel
                        for channel_idx, required_count in enumerate(required_samples):
                            if channel_idx < window_inputs.shape[1]:  # Check if channel exists
                                # Count valid samples for this channel across the window and context
                                channel_data = window_inputs[:, channel_idx]  # Shape: [window_size, context]
                                valid_count = np.sum(~np.isnan(channel_data))

                                if valid_count < required_count:
                                    should_mask = True
                                    break

                    nan_window_mask[i] = should_mask

                final_mask = nan_mask | nan_window_mask
            else:
                # Filter the arrays using the mask
                nan_mask = np.isnan(actual_horizon)
                # use configured window if present, otherwise default to 24
                window = config['run_config'].get('required_samples_window', 24)
                nan_window_mask = np.zeros_like(nan_mask, dtype=bool)
                # check previous `window` samples for NaNs for every index (handles start correctly)
                for i in range(len(nan_mask)):
                    start = max(0, i - window)
                    if np.any(nan_mask[start:i]):
                        nan_window_mask[i] = True
                final_mask = nan_mask | nan_window_mask

            pred_horizon[final_mask] = np.nan
            mean_horizon[final_mask] = np.nan
            #actual_horizon[final_mask] = np.nan
            std_horizon[final_mask] = np.nan
            plot_data[horizon].append((pred_horizon, actual_horizon, std_horizon))
            rmse = plot_and_evaluate_horizon(pred_horizon, actual_horizon, std_horizon, participant, horizon, config)
            rmses[horizon].append(rmse)
            
            # Store all RMSEs for this horizon in the summary - use explicit np.nan for missing values
            rmse_values[f'RMSE_{horizon}'] = rmse["rmse"] if not pd.isna(rmse["rmse"]) else np.nan
            rmse_values[f'RMSE_hypo_{horizon}'] = rmse["rmse_hypo"] if not pd.isna(rmse["rmse_hypo"]) else np.nan
            rmse_values[f'RMSE_hyper_{horizon}'] = rmse["rmse_hyper"] if not pd.isna(rmse["rmse_hyper"]) else np.nan  
            rmse_values[f'RMSE_normo_{horizon}'] = rmse["rmse_normo"] if not pd.isna(rmse["rmse_normo"]) else np.nan

            # --- CG-EGA per horizon ---
            ap, be, ep = evaluate_cg_ega_horizon(
                pred_horizon, actual_horizon, participant, horizon, config, freq=5, plot_day=1
            )
            cg_ega_metrics[horizon].append((ap, be, ep))
            cg_ega_df = pd.concat([
                cg_ega_df,
                pd.DataFrame([{
                    'Participant': participant,
                    'Horizon': horizon,
                    'AP': ap,
                    'BE': be,
                    'EP': ep
                }])
            ], ignore_index=True)
            # Apply same masking as RMSE eval
            valid_mask = ~np.isnan(pred_horizon) & ~np.isnan(actual_horizon) & ~np.isnan(std_horizon) & ~np.isnan(mean_horizon)
            if valid_mask.sum() > 0:
                calibration_result = evaluate_uncertainty_calibration(
                    y_true=actual_horizon[valid_mask],
                    mu=mean_horizon[valid_mask],
                    sigma=std_horizon[valid_mask],
                    participant=participant,
                    horizon=horizon,
                    config=config,
                    plot=True
                )

                print(f"Participant {participant} | Horizon {horizon} | PICE: {calibration_result['PICE']:.4f}")

                row = {
                    'Participant': participant,
                    'Horizon': horizon,
                    'PICE': calibration_result['PICE']
                }

                for alpha, picp, err in zip(
                    calibration_result['confidence_levels'],
                    calibration_result['picp'],
                    calibration_result['calibration_error_per_level']
                ):
                    level = int(alpha * 100)
                    row[f'PICP_{level}'] = picp
                    row[f'CALERR_{level}'] = err

                calibration_summary.append(row)
        # Append the RMSE values to the summary DataFrame
        rmse_df = pd.concat([rmse_df, pd.DataFrame([rmse_values])], ignore_index=True)
    
    #plot_results(participants, plot_data, config)
    rmse_df.to_csv(os.path.dirname(__file__)+os.sep+os.path.join(config['run_config']['experiment_path'], 'evaluation', 'rmse_summary.csv'), index=False)
    cg_ega_df.to_csv(os.path.dirname(__file__)+os.sep+os.path.join(config['run_config']['experiment_path'], 'evaluation', 'cg_ega_summary.csv'), index=False)
    if False:
        plot_per_horizon(forecast_horizons, plot_data, participants, config)
    # Save the summary of all RMSEs to a CSV file
    # Apply NaN masking and calculate aggregate metrics
    all_mse, all_mae = np.concatenate(mse_errors), np.concatenate(mae_errors)
    
    # Create extended NaN mask (24 steps after each NaN)
    nan_mask = np.isnan(all_mse)
    extended_mask = np.zeros_like(nan_mask, dtype=bool)
    for i in range(len(nan_mask)):
        if nan_mask[i]:
            extended_mask[i:min(i+25, len(nan_mask))] = True
    
    # Apply mask and print results
    all_mse[extended_mask] = np.nan
    all_mae[extended_mask] = np.nan
    
    median_mae = np.nanmedian(all_mae)
    median_mse = np.nanmedian(all_mse)
    
    print("Median MAE: ", median_mae)
    print("Median MSE: ", median_mse)
    
    # Save aggregate stats to file
    aggregate_stats = {
        'median_mae': float(median_mae),
        'median_mse': float(median_mse),
        'mean_mae': float(np.nanmean(all_mae)),
        'mean_mse': float(np.nanmean(all_mse)),
        'std_mae': float(np.nanstd(all_mae)),
        'std_mse': float(np.nanstd(all_mse)),
        'rmse': float(np.sqrt(median_mse)),
        'total_samples': int(np.sum(~np.isnan(all_mae))),
        'masked_samples': int(np.sum(np.isnan(all_mae)))
    }
    
    # Save as JSON
    evaluation_path = os.path.dirname(__file__) + os.sep + os.path.join(config['run_config']['experiment_path'], 'evaluation')
    os.makedirs(evaluation_path, exist_ok=True)
    
    with open(os.path.join(evaluation_path, 'aggregate_metrics.json'), 'w') as f:
        json.dump(aggregate_stats, f, indent=4)
    if event_metrics_all:
        # Flatten metrics into rows
        rows = []
        for entry in event_metrics_all:
            pid = entry['participant']
            threshold = entry.get('threshold', 'default')  # Get threshold or use 'default'
            for condition, metrics in entry.items():
                if condition in ['participant', 'threshold']:
                    continue
                row = {'Participant': pid, 'Threshold': threshold, 'Condition': condition}
                row.update(metrics)
                rows.append(row)

        event_df = pd.DataFrame(rows)
        event_df.to_csv(os.path.dirname(__file__)+os.sep+os.path.join(config['run_config']['experiment_path'], 'evaluation', 'event_level_summary_multi_threshold.csv'), index=False)
        
        # Compute and save summary stats grouped by threshold and condition
        summary_stats = event_df.drop(columns=['Participant']).groupby(['Threshold', 'Condition']).agg(['mean', 'std']).transpose()
        summary_stats.to_csv(os.path.dirname(__file__)+os.sep+os.path.join(config['run_config']['experiment_path'], 'evaluation', 'event_level_summary_stats_multi_threshold.csv'))
        # Save calibration summary
        pd.DataFrame(calibration_summary).to_csv(os.path.join(evaluation_path, 'calibration_summary.csv'), index=False)
        # Compute and save calibration summary stats
        if calibration_summary:
            calibration_df = pd.DataFrame(calibration_summary)
            
            # Create and save calibration statistics and plots
            create_calibration_analysis(calibration_df, evaluation_path, config)

        print("\n=== Aggregated Event-Level Metrics (Multi-Threshold) ===")
        print(summary_stats)



def evaluate_model_glucobench(config, dataframes):
    mse_errors = []
    mae_errors = []
    #if config["run_config"]['scaler'] == "StandardScaler":
    #    scaler_class_x = StandardScaler()
    #    scaler_class_y = StandardScaler()
    scaler_class_x = globals()[config["run_config"]["scaler"]]()
    scaler_class_y = globals()[config["run_config"]["scaler"]]()

    for participant in dataframes.keys():
        participants_test = [participant]
        prepper = DataPrepper(participants_test, 
                              dataframes, 
                              feature_list=config['run_config']['features'], 
                              target_list=config['run_config']['targets'], 
                              allowed_missing_values_rate=config['run_config']['allowed_missing_values_rate'],
                              allowed_missing_values_rate_target=config['run_config']['allowed_missing_values_rate_target'],
                              forecast_steps=config['hp_config']['forecast_steps'], 
                              scaler_class_x=scaler_class_x, 
                              scaler_class_y=scaler_class_y, 
                              patch_size=config['hp_config']['patch_size'],
                              fill_types=config['run_config']['fill_types'], 
                              experiment_path=os.path.dirname(__file__)+os.sep+config['run_config']['experiment_path'],
                              sequence_length=config['hp_config']['feature_window'],
                              history_of_days=config['hp_config']['history_of_days'],
                              token_size=config['hp_config']['token_size'],
                              test_target=config['run_config']['test_target'],
                              tokens_to_mask=config['hp_config']['tokens_to_mask']
                              )
        features_test, target_test = prepper.make_features_and_targetpair()
        test_data = CustomDataset(features_test, target_test, config['run_config']['features'], config['run_config']['targets'], config['hp_config']['history_of_days'], config['hp_config']['forecast_steps'], config['run_config']['test_target'], config['hp_config']['days_to_mask'])
        test_loader = DataLoader(test_data, shuffle=False, batch_size=config['hp_config']['batch_size'])
        trainer = globals()[config["run_config"]["trainer"]](config['hp_config'], os.path.dirname(__file__)+os.sep+config['run_config']['experiment_path']+os.sep+'best_model.pth')
        predictions, actuals = trainer.test(test_loader, scaler=prepper.scaler_y.scaler)
        #For debug
        if False:
                plt.figure(figsize=(12, 6))
                plt.plot(actuals[:,11,0], label='Actuals')
                plt.plot(predictions[:,11,0], label='Predictions')
                plt.title(f'Predictions vs Actuals for participant {participant}')
                plt.legend()
                plt.show()
        mse_errors.append(np.mean((actuals - predictions)**2, axis=(1,2)))
        mae_errors.append(np.mean(np.abs(actuals - predictions), axis=(1,2)))

    print("Median MAE: ", np.median(np.concatenate(mae_errors)))
    print("Median MSE: ", np.median(np.concatenate(mse_errors)))
    #return np.concatenate(mae_errors), np.concatenate(mse_errors)
    return np.vstack((np.concatenate(mse_errors), np.concatenate(mae_errors))).T

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
                                experiment_path=os.path.dirname(__file__)+os.sep+config['run_config']['experiment_path'],
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
                              experiment_path=os.path.dirname(__file__)+os.sep+config['run_config']['experiment_path'],
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
    trainer = globals()[config["run_config"]["trainer"]](config['hp_config'],os.path.dirname(__file__)+os.sep+config['run_config']['experiment_path']+os.sep+'best_model.pth',retrain_model=retrain_model)
    trainer.train(train_loader, val_loader)

def init_experiment_directory(config):
    experiment_path = os.path.dirname(__file__) + os.sep + config['run_config']['experiment_path']

    os.makedirs(experiment_path, exist_ok=True)
    if config['run_config']['parent_model_path'] is not None:
        parent_model_path = os.path.dirname(__file__) + os.sep + config['run_config']['parent_model_path']
        #shutil.copy2(os.path.join(parent_model_path, 'scaler_input.pkl'), os.path.join(experiment_path, 'scaler_input.pkl'))
        #shutil.copy2(os.path.join(parent_model_path, 'scaler_target.pkl'), os.path.join(experiment_path, 'scaler_output.pkl'))
        shutil.copy2(os.path.join(parent_model_path, 'best_model.pth'), os.path.join(experiment_path, 'best_model.pth'))
    #dump config
    with open(experiment_path+os.sep+'model_config.json', 'w') as f:
        f.write(json.dumps(config, indent=4))

def main(config, train=True, test=True):
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
                              experiment_path= os.path.dirname(__file__)+os.sep+config['run_config']['experiment_path'],
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
        if config['run_config']['test_participants'] == 'all':
            participants_test = list(data_handler.get_test_dataframes().keys())
        else:
            participants_test = config['run_config']['test_participants']

        evaluate_model(config, data_handler.get_test_dataframes(), scaler_class_x, scaler_class_y, participants=participants_test, metadata=data_handler.get_test_metadata())

if __name__ == "__main__":
    # Assumes that the dataset is already processed to standardized format

    #Trained on Tidepool, Tested on Ohio
    #experiment_path = os.path.join('experiments','iTransTrainedOnTidepoolOnOhio')

    # Trained and tested on Ohio
    #experiment_path = os.path.join('experiments','iTransTest')

    #Pretrained on Tidepool, retrained on ohio, tested on ohio:
    #experiment_path = os.path.join('experiments','iTransTidepoolRetrainedOnOhio')

    #Pretrained on Tidepool:
    #experiment_path = os.path.join('experiments','iTransTidepool60')

    #Pretrained on Tidepool, retrained on ohio, tested on ohio:
    #experiment_path = os.path.join('experiments','iTransTidepool60TestOhio')

    #Pretrained on Tidepool only CGM data:
    #experiment_path = os.path.join('experiments','iTransTidepool60OnlyCGM')

    #Pretrained on Tidepool only CGM data:
    #experiment_path = os.path.join('experiments','testing')

    #Testing new backbone model
    #experiment_path = os.path.join('experiments','BGiTransformerDevelopMultiTask')
    #experiment_path = os.path.join('experiments','BGiTransformerMultiTaskTidepool')
    #experiment_path = os.path.join('experiments','BGiTransformerDevelop','child_model')
    #experiment_path = os.path.join('experiments','BGiTransformerTidepool')

    #SimpleBaselineModel for Tidepool
    #experiment_path = os.path.join('experiments','LinRegTestTidepool')
    #experiment_path = os.path.join('experiments','SimpleBaselineModelMultivariate')
    #experiment_path = os.path.join('experiments','LstmTestTidepool')
    #experiment_path = os.path.join('experiments','MultiDayCGM2H')
    #experiment_path = os.path.join('experiments','MultiDayLstmCGM2HBigger')
    #experiment_path = os.path.join('experiments','MultiDayCGM2H')
    #data_handler = DataHandler("DataloaderOhio", r"c:\Users\knutj\OneDrive - Universitaet Bern\Datasets\Ohio Data\Ohio_XML", dataset_name="Ohio2018")
    #data_handler.load_data(save_as_csv=True)
    #data_handler = DataHandler("DataloaderOhio", r"c:\Users\knutj\OneDrive - Universitaet Bern\Datasets\Ohio Data\Ohio2020_XML", dataset_name="Ohio2020")
    #data_handler.load_data(save_as_csv=True)
    #data_handler = DataHandler("DataloaderTidepoolSAP100", r"C:\Users\knutj\OneDrive - Universitaet Bern\Datasets\Tidepool Data", dataset_name="Tidepool_SAP100")
    #data_handler.load_data(save_as_csv=True)
    experiment_path = os.path.join('experiments','LSTMTest2')
    #experiment_path = os.path.join('experiments','LinRegTest2H')

    model_config_path = experiment_path+os.sep+'model_config.json'
    config = json.load(open(model_config_path))
    main(config)
