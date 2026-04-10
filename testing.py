from dataprepper import DataPrepper
from architectures.lstms import MirshekarianLSTM
from trainers.trainer_basic import TrainerBasic
from evaluator import Evaluator

import torch
import torch.nn as nn
import torch.optim as optim

from datahandler import DataHandler
from dataloaders.dataloader_ohio_processed import Dataloader
from datapreprocessor import DataPreProcessor
import os
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from torch.utils.data import TensorDataset, DataLoader
#Getting rid of the annoying warnings
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
from scaler import Scaler

#Read data
participants = [559,563,570,575,588,591]
#directory_path = os.path.join('Ohio Data', 'Ohio2018_processed')
dataset_location = r"C:\Users\knutj\OneDrive - Universitaet Bern\Datasets\Ohio Data\Ohio2018_processed"
data_loader = Dataloader(dataset_location)
data_handler = DataHandler(data_loader, dataset_name='Ohio2018_processed')
data_handler.load_data()
#print(data_handler.get_train_dataframes()['563-ws-training_processed.csv'])
print(data_handler.get_train_dataframes().keys())
#Preprocess data
data_preprocessor = DataPreProcessor(data_handler.get_train_dataframes())
features = ['cbg', 'basal', 'carbInput', 'bolus']
fill_types = ['cubicspline', 0, 0, 0]
data_preprocessor.handle_missing_values(features, fill_types)
#print(data_handler.get_train_dataframes()['563-ws-training_processed.csv'])
scaler = Scaler(data_handler, features)
scaler.fit_and_save('scaler_2018.pkl')

forecast_steps = 6

def train_model_for_programming_task(participants, scaler, model_path='best_model.pth', forecast_steps=6, pretrained_model_path=None):
    features = ['cbg', 'basal', 'carbInput', 'bolus']
    fill_types = ['cubicspline', 0, 0, 0]

    data_loader = Dataloader(dataset_location)
    data_handler = DataHandler(data_loader, dataset_name='Ohio2018_processed')
    data_handler.load_data()
    
    train_preprocessor = DataPreProcessor(data_handler.get_train_dataframes())
    train_preprocessor.handle_missing_values(features, fill_types)
    
    test_preprocessor = DataPreProcessor(data_handler.get_test_dataframes())
    test_preprocessor.handle_missing_values(features, fill_types)


    #initialize data prepper and make features and target pair
    prepper = DataPrepper(participants, data_handler.get_train_dataframes(), forecast_steps=forecast_steps, scaler=scaler.get_scaler())
    features_train, target_train = prepper.make_features_and_targetpair()

    # Create TensorDatasets
    ########################
    # Split the training data into training and validation sets
    features_train, features_val, target_train, target_val = train_test_split(
        features_train, target_train, test_size=0.2, random_state=42)

    # Create TensorDatasets for the training and validation sets
    train_data = TensorDataset(features_train, target_train)
    val_data = TensorDataset(features_val, target_val)

    # Create DataLoaders
    batch_size = 32
    train_loader = DataLoader(train_data, shuffle=True, batch_size=batch_size)
    val_loader = DataLoader(val_data, shuffle=False, batch_size=batch_size)
    ########################

    model = MirshekarianLSTM(input_dim=4, hidden_dim=5, output_dim=1)
    #initialize model
    if pretrained_model_path != None:
        pretrained_model_dir = os.path.dirname(pretrained_model_path)
        if not os.path.exists(pretrained_model_dir):
            os.makedirs(pretrained_model_dir)
        model.load_state_dict(torch.load(pretrained_model_path))


    # Define loss function and optimizer
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.01)

    # Train the model
    trainer = TrainerBasic(model, train_loader, val_loader, criterion, optimizer, num_epochs=100, model_path=model_path)

    trainer.train()

if False:
    model_path = os.path.join('models', 'testing', str(forecast_steps), f'best_model_559.pth')
    participants = ['559-ws-training_processed', '563-ws-training_processed', '570-ws-training_processed', '575-ws-training_processed', '588-ws-training_processed', '591-ws-training_processed']
    train_model_for_programming_task(participants, scaler, model_path, forecast_steps)
if True:
    def evaluate_model_for_programming_task(participants, scaler, model_path='best_model.pth', forecast_steps=6):
        #Read data
        
        features = ['cbg', 'basal', 'carbInput', 'bolus']
        fill_types = ['cubicspline', 0, 0, 0]

        data_loader = Dataloader(dataset_location)
        data_handler = DataHandler(data_loader, dataset_name='Ohio2018_processed')
        data_handler.load_data()
        print(data_handler.get_test_dataframes().keys())

        test_preprocessor = DataPreProcessor(data_handler.get_test_dataframes())
        test_preprocessor.handle_missing_values(features, fill_types)


        #initialize data prepper and make features and target pair

        prepper = DataPrepper(participants, data_handler.get_test_dataframes(), forecast_steps=forecast_steps, scaler=scaler)
        features_test, target_test = prepper.make_features_and_targetpair()

        # Create TensorDatasets
        ########################

        # Create TensorDatasets for the training and validation sets
        test_data = TensorDataset(features_test, target_test)

        # Create DataLoaders
        batch_size = 32
        test_loader = DataLoader(test_data, shuffle=False, batch_size=batch_size)
        ########################

        #initialize model
        model = MirshekarianLSTM(input_dim=4, hidden_dim=5, output_dim=1)

        # Load the state dict previously saved
        model.load_state_dict(torch.load(model_path))

        # Evaluate the model
        evaluator = Evaluator(model, test_loader)
        return evaluator.evaluate()
    participants = ['559-ws-testing_processed', '563-ws-testing_processed', '570-ws-testing_processed', '575-ws-testing_processed', '588-ws-testing_processed', '591-ws-testing_processed']
    plot_data = []
    rmses = []
    for participant in participants:
        test_participants = [participant]
        model_path = os.path.join('models', 'inter-patient', str(forecast_steps), f'best_model_559.pth')
        predictions, actuals, rmse = evaluate_model_for_programming_task(test_participants, scaler.get_scaler(), model_path, forecast_steps)
        plot_data.append((predictions, actuals))
        rmses.append(rmse)

    rmse_df_inter = pd.DataFrame({'Participant': participants, 'RMSE': rmses})
    #display(rmse_df_inter)

    fig, axs = plt.subplots(len(participants), 1, figsize=(10, 5 * len(participants)))
    for i, (ax, data) in enumerate(zip(axs, plot_data)):
        predictions, actuals = data
        ax.plot(actuals, linestyle='-', linewidth=2, color='blue', alpha=0.7)
        ax.plot(predictions, linestyle='--', linewidth=1, color='red', alpha=0.7)
        if i == 0:
            ax.legend(['Actuals', 'Predictions'], loc='lower center', bbox_to_anchor=(0.5, 1.0))
        ax.set_ylim([0, 450])
    plt.show()