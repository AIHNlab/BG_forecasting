import os
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split
from torch.utils.data import TensorDataset, DataLoader
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
    """
    This function creates a separate plot for each forecast horizon with predictions and actuals for all participants.
    Each participant will have its own subplot.
    """
    for horizon in horizons:
        fig, axs = plt.subplots(len(participants), 1, figsize=(30, 6 * len(participants)))  # Create subplots
        if len(participants) == 1:
            axs = [axs]  # Ensure axs is iterable when there's only one participant
        
        # Plot data for each participant in their own subplot
        for idx, (ax, participant) in enumerate(zip(axs, participants)):
            if len(plot_data[horizon]) > idx:
                pred_horizon, actual_horizon = plot_data[horizon][idx]
                ax.plot(actual_horizon, label='Actuals', linestyle='-', linewidth=2, color='blue', alpha=0.7)
                ax.plot(pred_horizon, label='Predictions', linestyle='--', linewidth=1, color='red', alpha=0.7)
                ax.legend()
                ax.set_title(f'Participant {participant} - Horizon {horizon}')
                ax.set_xlabel('Time')
                ax.set_ylabel('Values')

        plt.tight_layout()

        # Save the consolidated plot for the current horizon
        evaluation_path = os.path.dirname(__file__)+os.path.join(config['run_config']['experiment_path'], 'evaluation')
        os.makedirs(evaluation_path, exist_ok=True)
        plt.savefig(os.path.join(evaluation_path, f'all_participants_horizon_{horizon}.png'))
        plt.close(fig)  # Close the figure to free up memory

def train_model(hp_config, features_train, target_train, fitted_scaler_y):
    

    # Split the training data into training and validation sets
    features_train, features_val, target_train, target_val = train_test_split(
        features_train, target_train, test_size=0.2, random_state=42)
    # Create TensorDatasets for the training and validation sets
    train_data = TensorDataset(features_train, target_train)
    val_data = TensorDataset(features_val, target_val)
    # Create DataLoaders
    train_loader = DataLoader(train_data, shuffle=True, batch_size=hp_config['batch_size'])
    val_loader = DataLoader(val_data, shuffle=False, batch_size=hp_config['batch_size'])
    


    # Train the model
    retrain_model = True
    if config['run_config']['parent_model_path'] is not None:
        retrain_model = False
    trainer = globals()[config["run_config"]["trainer"]](hp_config, os.path.dirname(__file__)+os.sep+config['run_config']['experiment_path']+os.sep+'best_model.pth',retrain_model=retrain_model)
    trainer.train(train_loader, val_loader)

def evaluate_model(config, dataframes, scaler_class_x, scaler_class_y, participants):
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
                              forecast_steps=config['hp_config']['forecast_steps'], 
                              scaler_class_x=scaler_class_x, 
                              scaler_class_y=scaler_class_y, 
                              indices_per_day=config['run_config']['indices_per_day'],
                              fill_types=config['run_config']['fill_types'], 
                              experiment_path=os.path.dirname(__file__)+os.sep+config['run_config']['experiment_path'], 
                              sequence_length=config['hp_config']['feature_window'], 
                              history_of_days=config['hp_config']['history_of_days'])
        features_test, target_test = prepper.make_features_and_targetpair()
        test_data = TensorDataset(features_test, target_test)
        test_loader = DataLoader(test_data, shuffle=False, batch_size=config['hp_config']['batch_size'])
        trainer = globals()[config["run_config"]["trainer"]](config['hp_config'], os.path.dirname(__file__)+os.sep+config['run_config']['experiment_path']+os.sep+'best_model.pth')
        predictions, actuals = trainer.test(test_loader, scaler=prepper.scaler_y.scaler)
        
        if actuals.size == 0:
            continue
        
        rmse_values = {'Participant': participant}
        
        for horizon in forecast_horizons:
            pred_horizon = np.array(predictions)[:, horizon-1, :].flatten()
            actual_horizon = np.array(actuals)[:, horizon-1, :].flatten()
            
            plot_data[horizon].append((pred_horizon, actual_horizon))
            rmse = np.sqrt(np.mean((pred_horizon - actual_horizon) ** 2))
            rmses[horizon].append(rmse)
            rmse_values[f'RMSE_{horizon}'] = rmse
            
            # Save plot to a file
            evaluation_path = os.path.dirname(__file__)+os.path.join(config['run_config']['experiment_path'], 'evaluation', str(participant), f'horizon_{horizon}')
            os.makedirs(evaluation_path, exist_ok=True)
            fig = plt.figure(figsize=(25, 5))
            plt.plot(actual_horizon, label='Actuals', linestyle='-', linewidth=2, color='blue', alpha=0.7)
            plt.plot(pred_horizon, label='Predictions', linestyle='--', linewidth=1, color='red', alpha=0.7)
            plt.legend()
            plt.title(f'Participant {participant} - Horizon {horizon}')
            plt.savefig(os.path.join(evaluation_path, 'plot.png'))
            plt.close(fig)
            #plt.show()
        
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
                              forecast_steps=config['hp_config']['forecast_steps'], 
                              scaler_class_x=scaler_class_x, 
                              scaler_class_y=scaler_class_y, 
                              indices_per_day=config['run_config']['indices_per_day'],
                              fill_types=config['run_config']['fill_types'], 
                              experiment_path=os.path.dirname(__file__)+os.sep+config['run_config']['experiment_path'],
                              sequence_length=config['hp_config']['feature_window'],
                              history_of_days=config['hp_config']['history_of_days'])
        features_test, target_test = prepper.make_features_and_targetpair()
        test_data = TensorDataset(features_test, target_test)
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
                                forecast_steps=config['hp_config']['forecast_steps'], 
                                scaler_class_x=scaler_class_x, 
                                scaler_class_y=scaler_class_y, 
                                indices_per_day=config['run_config']['indices_per_day'],
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
                              forecast_steps=config['hp_config']['forecast_steps'], 
                              scaler_class_x=scaler_class_x, 
                              scaler_class_y=scaler_class_y, 
                              indices_per_day=config['run_config']['indices_per_day'],
                              fill_types=config['run_config']['fill_types'], 
                              experiment_path=os.path.dirname(__file__)+os.sep+config['run_config']['experiment_path'],
                              step=config['run_config']['step_training'],
                              sequence_length=config['hp_config']['feature_window'],
                              history_of_days=config['hp_config']['history_of_days'])
    features_val, target_val = val_prepper.make_features_and_targetpair()
    
    # Split the training data into training and validation sets
    features_train, features_val, target_train, target_val = train_test_split(
        features_train, target_train, test_size=0.2, random_state=42)
    # Create TensorDatasets for the training and validation sets
    train_data = TensorDataset(features_train, target_train)
    val_data = TensorDataset(features_val, target_val)
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
                              forecast_steps=config['hp_config']['forecast_steps'], 
                              scaler_class_x=scaler_class_x, 
                              scaler_class_y=scaler_class_y, 
                              indices_per_day=config['run_config']['indices_per_day'],
                              fill_types=config['run_config']['fill_types'], 
                              experiment_path= os.path.dirname(__file__)+os.sep+config['run_config']['experiment_path'],
                              step=config['run_config']['step_training'],
                              sequence_length=config['hp_config']['feature_window'],
                              history_of_days=config['hp_config']['history_of_days'])
        features_train, target_train = prepper.make_features_and_targetpair()
        #prepper = DataPrepper(participants_test, data_handler, feature_list=config['run_config']['features'], data_type="test", forecast_steps=config['run_config']['forecast_steps'], scaler_class_x=scaler_class_x, scaler_class_y=scaler_class_y, fill_types=config['run_config']['fill_types'], experiment_path=config['run_config']['experiment_path'])
        #features_test, target_test = prepper.make_features_and_targetpair()
        train_model(config['hp_config'], features_train, target_train, prepper.scaler_y.scaler)

    if config['run_config']['test']:
        if config['run_config']['test_participants'] == 'all':
            participants_test = list(data_handler.get_test_dataframes().keys())
        else:
            participants_test = config['run_config']['test_participants']

        evaluate_model(config, data_handler.get_test_dataframes(), scaler_class_x, scaler_class_y, participants=participants_test)

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
    experiment_path = os.path.join('experiments','MultiDayCGM2HComboAttention')
    #experiment_path = os.path.join('experiments','LinRegTest2H')

    model_config_path = experiment_path+os.sep+'model_config.json'
    config = json.load(open(model_config_path))
    main(config)
