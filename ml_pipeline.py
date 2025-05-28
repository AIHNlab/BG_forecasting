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
from custom_dataset import CustomDataset

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
    rmse_df = pd.DataFrame(columns=['Participant'] + [f'RMSE_{horizon}' for horizon in config['hp_config']['forecast_horizons']])
    forecast_horizons = config['hp_config']['forecast_horizons']
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
                              )
        testset = prepper.make_features_and_targetpair()
        #test_data = CustomDataset(features_test, target_test, config['run_config']['features'], config['run_config']['targets'], config['hp_config']['history_of_days'], config['hp_config']['forecast_steps'], config['run_config']['test_target'], config['hp_config']['days_to_mask'])
        test_loader = DataLoader(testset, shuffle=False, batch_size=config['hp_config']['batch_size'])
        trainer = globals()[config["run_config"]["trainer"]](config['hp_config'], os.path.dirname(__file__)+os.sep+config['run_config']['experiment_path']+os.sep+'best_model.pth')
        predictions, actuals, stds = trainer.test(test_loader, scaler=prepper.scaler_x.scaler, hypoglycemia_threshold=prepper.hypoglycemia_threshold, hyperglycemia_threshold=prepper.hyperglycemia_threshold)
        
        if actuals.size == 0:
            continue
        
        rmse_values = {'Participant': participant}
        if config['hp_config']['forecast_steps'] > 1:
            predictions = np.array(predictions)[:, -config['hp_config']['forecast_steps']:, prepper.test_target_index]
            actuals = np.array(actuals)[:, -config['hp_config']['forecast_steps']:, prepper.test_target_index]
            stds = np.array(stds)[:, -config['hp_config']['forecast_steps']:, prepper.test_target_index]  # Add this line
            
        for horizon in forecast_horizons:
            pred_horizon = np.array(predictions)[:, horizon-1].flatten()
            actual_horizon = np.array(actuals)[:, horizon-1].flatten()
            std_horizon = np.array(stds)[:, horizon-1].flatten()  # Add this line
            
            plot_data[horizon].append((pred_horizon, actual_horizon, std_horizon))  # Modified to include std
            
            # Save plot to a file with confidence intervals
            evaluation_path = os.path.dirname(__file__)+os.path.join(config['run_config']['experiment_path'], 'evaluation', str(participant), f'horizon_{horizon}')
            os.makedirs(evaluation_path, exist_ok=True)
            fig = plt.figure(figsize=(25, 5))
            
            # Plot actual values and predictions
            plt.plot(actual_horizon, label='Actuals', linestyle='-', linewidth=2, color='blue', alpha=0.7)
            plt.plot(pred_horizon, label='Predictions', linestyle='--', linewidth=1, color='red', alpha=0.7)
            
            # Add confidence intervals (±2 standard deviations for 95% confidence)
            plt.fill_between(
                range(len(pred_horizon)),
                pred_horizon - 2 * std_horizon,
                pred_horizon + 2 * std_horizon,
                color='red',
                alpha=0.2,
                label='95% Confidence Interval'
            )
            
            plt.legend()
            plt.title(f'Participant {participant} - Horizon {horizon}')
            plt.savefig(os.path.join(evaluation_path, 'plot.png'))
            plt.close(fig)
            #plt.show()

            # Filter the arrays using the mask
            mask = ~np.isnan(actual_horizon)
            filtered_pred_horizon = pred_horizon[mask]
            filtered_actual_horizon = actual_horizon[mask]            
            rmse = np.sqrt(np.mean((filtered_pred_horizon - filtered_actual_horizon) ** 2))
            rmses[horizon].append(rmse)
            rmse_values[f'RMSE_{horizon}'] = rmse
        
        # Append the RMSE values to the summary DataFrame
        rmse_df = pd.concat([rmse_df, pd.DataFrame([rmse_values])], ignore_index=True)
    
    #plot_results(participants, plot_data, config)
    rmse_df.to_csv(os.path.dirname(__file__)+os.sep+os.path.join(config['run_config']['experiment_path'], 'evaluation', 'rmse_summary.csv'), index=False)
    plot_per_horizon(forecast_horizons, plot_data, participants, config)
    # Save the summary of all RMSEs to a CSV file
    

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
                                history_of_days=config['hp_config']['history_of_days'])
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
                              history_of_days=config['hp_config']['history_of_days'])
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
                              )
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
    experiment_path = os.path.join('experiments','SanityCheck')
    #experiment_path = os.path.join('experiments','LinRegTest2H')

    model_config_path = experiment_path+os.sep+'model_config.json'
    config = json.load(open(model_config_path))
    main(config)
