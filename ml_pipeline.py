import os
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split
from torch.utils.data import TensorDataset, DataLoader
import torch
import torch.nn as nn
import torch.optim as optim
from dataprepper import DataPrepper
from architectures.mirshekarian_lstm import MirshekarianLSTM
from trainers.trainer_basic import TrainerBasic
from trainers.exp_long_term_forecasting import Exp_Long_Term_Forecast
from datahandler import DataHandler
from dataloaders.dataloader_tidepool_sap100 import Dataloader
from datapreprocessor import DataPreProcessor
from scaler import Scaler
from evaluator import Evaluator
import warnings
from sklearn.preprocessing import StandardScaler
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

def plot_results(participants, plot_data, config):
    fig, axs = plt.subplots(len(participants), 1, figsize=(10, 5 * len(participants)))
    for i, (ax, data) in enumerate(zip(axs, plot_data)):
        predictions, actuals = data
        ax.plot(actuals, linestyle='-', linewidth=2, color='blue', alpha=0.7)
        ax.plot(predictions, linestyle='--', linewidth=1, color='red', alpha=0.7)
        if i == 0:
            ax.legend(['Actuals', 'Predictions'], loc='lower center', bbox_to_anchor=(0.5, 1.0))
        ax.set_ylim([-10, 450])
    plt.savefig(os.path.join(config['run_config']['experiment_path'], 'evaluation', 'plot_summary.png'))

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
    trainer = globals()[config["run_config"]["trainer"]](hp_config, config['run_config']['experiment_path']+os.sep+'best_model.pth',retrain_model=retrain_model)
    trainer.train(train_loader, val_loader)

def evaluate_model(config, dataframes, scaler_class_x, scaler_class_y, participants):
    plot_data = []
    rmses = []
    rmse_df = pd.DataFrame(columns=['Participant', 'RMSE'])  # DataFrame to store all RMSEs

    for participant in participants:
        participants_test = [participant]
        prepper = DataPrepper(participants_test, dataframes, feature_list=config['run_config']['features'], forecast_steps=config['hp_config']['forecast_steps'], scaler_class_x=scaler_class_x, scaler_class_y=scaler_class_y, fill_types=config['run_config']['fill_types'], experiment_path=config['run_config']['experiment_path'])
        features_test, target_test = prepper.make_features_and_targetpair()
        test_data = TensorDataset(features_test, target_test)
        test_loader = DataLoader(test_data, shuffle=False, batch_size=config['hp_config']['batch_size'])
        trainer = globals()[config["run_config"]["trainer"]](config['hp_config'], config['run_config']['experiment_path']+os.sep+'best_model.pth')
        predictions, actuals = trainer.test(test_loader, scaler=prepper.scaler_y.scaler)

        predictions = np.array(predictions)[:, -1, :].flatten()
        actuals = np.array(actuals)[:, -1, :].flatten()

        plot_data.append((predictions, actuals))
        rmse = np.sqrt(np.mean((predictions - actuals) ** 2))
        rmses.append(rmse)

        # Save RMSE to a DataFrame
        rmse_df_inter = pd.DataFrame({'Participant': [participant], 'RMSE': [rmse]})
        print(rmse_df_inter)

        # Save DataFrame to a CSV file
        evaluation_path = os.path.join(os.path.dirname(__file__),config['run_config']['experiment_path'], 'evaluation', str(participant))
        os.makedirs(evaluation_path, exist_ok=True)
        rmse_df_inter.to_csv(os.path.join(evaluation_path, 'rmse.csv'), index=False)

        # Save plot to a file
        plt.figure()
        plt.plot(predictions, label='Predictions')
        plt.plot(actuals, label='Actuals')
        plt.legend()
        plt.savefig(os.path.join(evaluation_path, 'plot.png'))

        # Append the RMSE to the summary DataFrame
        rmse_df = pd.concat([rmse_df, rmse_df_inter], ignore_index=True)

    plot_results(participants, plot_data, config)

    # Save the summary of all RMSEs to a CSV file
    rmse_df.to_csv(os.path.join(os.path.dirname(__file__),config['run_config']['experiment_path'], 'evaluation', 'rmse_summary.csv'), index=False)

def evaluate_model_glucobench(config, dataframes):
    mse_errors = []
    mae_errors = []
    if config["run_config"]['scaler'] == "StandardScaler":
        scaler_class_x = StandardScaler()
        scaler_class_y = StandardScaler()

    for participant in dataframes.keys():
        participants_test = [participant]
        prepper = DataPrepper(participants_test, dataframes, feature_list=config['run_config']['features'], forecast_steps=config['hp_config']['forecast_steps'], scaler_class_x=scaler_class_x, scaler_class_y=scaler_class_y, fill_types=config['run_config']['fill_types'], experiment_path=config['run_config']['experiment_path'])
        features_test, target_test = prepper.make_features_and_targetpair()
        test_data = TensorDataset(features_test, target_test)
        test_loader = DataLoader(test_data, shuffle=False, batch_size=config['hp_config']['batch_size'])
        trainer = globals()[config["run_config"]["trainer"]](config['hp_config'], os.path.dirname(__file__)+os.sep+config['run_config']['experiment_path']+os.sep+'best_model.pth')
        predictions, actuals = trainer.test(test_loader, scaler=prepper.scaler_y.scaler)
        mse_errors.append(np.mean((actuals - predictions)**2, axis=(1,2)))
        mae_errors.append(np.mean(np.abs(actuals - predictions), axis=(1,2)))

    print("Median MAE: ", np.median(np.concatenate(mae_errors)))
    print("Median MSE: ", np.median(np.concatenate(mse_errors)))
    #return np.concatenate(mae_errors), np.concatenate(mse_errors)
    return np.vstack((np.concatenate(mse_errors), np.concatenate(mae_errors))).T

def train_model_glucobench(config, dataframes_train, dataframes_val):
    init_experiment_directory(config)
    if config["run_config"]['scaler'] == "StandardScaler":
        scaler_class_x = StandardScaler()
        scaler_class_y = StandardScaler()
    train_prepper = DataPrepper(dataframes_train.keys(), dataframes_train, feature_list=config['run_config']['features'], forecast_steps=config['hp_config']['forecast_steps'], scaler_class_x=scaler_class_x, scaler_class_y=scaler_class_y, fill_types=config['run_config']['fill_types'], experiment_path=config['run_config']['experiment_path'],step=config['run_config']['step_training'])
    features_train, target_train = train_prepper.make_features_and_targetpair()
    test_prepper = DataPrepper(dataframes_val.keys(), dataframes_val, feature_list=config['run_config']['features'], forecast_steps=config['hp_config']['forecast_steps'], scaler_class_x=scaler_class_x, scaler_class_y=scaler_class_y, fill_types=config['run_config']['fill_types'], experiment_path=config['run_config']['experiment_path'],step=config['run_config']['step_training'])
    features_val, target_val = test_prepper.make_features_and_targetpair()
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
    retrain_model = True
    if config['run_config']['parent_model_path'] is not None:
        retrain_model = False
    trainer = globals()[config["run_config"]["trainer"]](config['hp_config'],os.path.dirname(__file__)+os.sep+config['run_config']['experiment_path']+os.sep+'best_model.pth',retrain_model=retrain_model)
    trainer.train(train_loader, val_loader)

def init_experiment_directory(config):
    parent_model_path = os.path.dirname(__file__) + os.sep + config['run_config']['parent_model_path']
    experiment_path = os.path.dirname(__file__) + os.sep + config['run_config']['experiment_path']

    os.makedirs(experiment_path, exist_ok=True)
    if parent_model_path is not None:
        shutil.copy2(os.path.join(parent_model_path, 'scaler_input.pkl'), os.path.join(experiment_path, 'scaler_input.pkl'))
        shutil.copy2(os.path.join(parent_model_path, 'scaler_target.pkl'), os.path.join(experiment_path, 'scaler_output.pkl'))
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
    
    if config["run_config"]['scaler'] == "StandardScaler":
        scaler_class_x = StandardScaler()
        scaler_class_y = StandardScaler()
    


    if config['run_config']['train']:
        if config['run_config']['train_participants'] == 'all':
            participants_train = list(data_handler.get_train_dataframes().keys())
        else:
            participants_train = config['run_config']['participants']
        prepper = DataPrepper(participants_train, data_handler.get_train_dataframes(), feature_list=config['run_config']['features'], forecast_steps=config['hp_config']['forecast_steps'], scaler_class_x=scaler_class_x, scaler_class_y=scaler_class_y, fill_types=config['run_config']['fill_types'], experiment_path=config['run_config']['experiment_path'],step=config['run_config']['step_training'])
        features_train, target_train = prepper.make_features_and_targetpair()
        #prepper = DataPrepper(participants_test, data_handler, feature_list=config['run_config']['features'], data_type="test", forecast_steps=config['run_config']['forecast_steps'], scaler_class_x=scaler_class_x, scaler_class_y=scaler_class_y, fill_types=config['run_config']['fill_types'], experiment_path=config['run_config']['experiment_path'])
        #features_test, target_test = prepper.make_features_and_targetpair()
        train_model(config['hp_config'], features_train, target_train, prepper.scaler_y.scaler)

    if config['run_config']['test']:
        if config['run_config']['test_participants'] == 'all':
            participants_test = list(data_handler.get_test_dataframes().keys())
        else:
            participants_test = config['run_config']['participants']

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
    experiment_path = os.path.join('experiments','iTransTidepool60OnlyCGM')

    #Pretrained on Tidepool only CGM data:
    #experiment_path = os.path.join('experiments','testing')

    model_config_path = experiment_path+os.sep+'model_config.json'
    config = json.load(open(model_config_path))
    main(config)
