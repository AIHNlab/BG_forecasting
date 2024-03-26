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
from datahandler import DataHandler
from dataloaders.dataloader_tidepool_sap100 import Dataloader
from datapreprocessor import DataPreProcessor
from scaler import Scaler
from evaluator import Evaluator
import warnings
from sklearn.preprocessing import StandardScaler
warnings.simplefilter(action='ignore', category=FutureWarning)

def evaluate_model_for_programming_task(data_handler, forecast_steps, features, fill_types, scaler, participants, model_path='best_model.pth'):

    prepper = DataPrepper(participants, data_handler, data_type="test", forecast_steps=forecast_steps, scaler=scaler, fill_types=fill_types)
    features_test, target_test = prepper.make_features_and_targetpair()

    test_data = TensorDataset(features_test, target_test)
    test_loader = DataLoader(test_data, shuffle=False, batch_size=32)

    model = MirshekarianLSTM(input_dim=4, hidden_dim=5, output_dim=1)
    model.load_state_dict(torch.load(model_path))

    evaluator = Evaluator(model, test_loader)
    return evaluator.evaluate()

def train_model(data_handler, participants, features, fill_types, forecast_steps, model_path, batch_size, num_epochs, learning_rate, model, trainer_class, scaler, pretrained_model_path=None):
    
    prepper = DataPrepper(participants, data_handler, data_type="train", forecast_steps=forecast_steps, scaler=scaler, fill_types=fill_types)
    features_train, target_train = prepper.make_features_and_targetpair()

    # Split the training data into training and validation sets
    features_train, features_val, target_train, target_val = train_test_split(
        features_train, target_train, test_size=0.2, random_state=42)

    # Create TensorDatasets for the training and validation sets
    train_data = TensorDataset(features_train, target_train)
    val_data = TensorDataset(features_val, target_val)

    # Create DataLoaders
    train_loader = DataLoader(train_data, shuffle=True, batch_size=batch_size)
    val_loader = DataLoader(val_data, shuffle=False, batch_size=batch_size)

    # Initialize model
    model = MirshekarianLSTM(input_dim=4, hidden_dim=5, output_dim=1)
    if pretrained_model_path != None:
        pretrained_model_dir = os.path.dirname(pretrained_model_path)
        if not os.path.exists(pretrained_model_dir):
            os.makedirs(pretrained_model_dir)
        model.load_state_dict(torch.load(pretrained_model_path))

    # Define loss function and optimizer
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    # Train the model
    trainer = trainer_class(model, train_loader, val_loader, criterion, optimizer, num_epochs=num_epochs, model_path=model_path)
    trainer.train()

    # Return the final loss back to Tune

def plot_results(participants, plot_data):
    fig, axs = plt.subplots(len(participants), 1, figsize=(10, 5 * len(participants)))
    for i, (ax, data) in enumerate(zip(axs, plot_data)):
        predictions, actuals = data
        ax.plot(actuals, linestyle='-', linewidth=2, color='blue', alpha=0.7)
        ax.plot(predictions, linestyle='--', linewidth=1, color='red', alpha=0.7)
        if i == 0:
            ax.legend(['Actuals', 'Predictions'], loc='lower center', bbox_to_anchor=(0.5, 1.0))
        ax.set_ylim([-10, 450])
    plt.show()

def main(config, train, evaluate):
    #data_loader = Dataloader(config['dataset_location'])
    data_handler = DataHandler(config['dataloader'], config['dataset_path'],dataset_name=config['dataset_name'])
    data_handler.load_data()

    if train:
        train_model(
            data_handler=data_handler,
            participants=list(data_handler.get_train_dataframes().keys()),
            features=config['features'],
            fill_types=config['fill_types'],
            forecast_steps=config['forecast_steps'],
            model_path=os.path.join('models', data_handler.get_dataset_name(), 'testing', str(config['forecast_steps']), f'testing.pth'),
            batch_size=config['batch_size'],
            num_epochs=config['num_epochs'],
            learning_rate=config['learning_rate'],
            model=config['model'],
            trainer_class=TrainerBasic,
            scaler=config['scaler'],
            pretrained_model_path=None
        )
    if evaluate:
        participants = list(data_handler.get_test_dataframes().keys())[0:6]
        plot_data = []
        rmses = []
        for participant in participants:
            test_participants = [participant]
            model_path = os.path.join('models', "Tidepool_SAP100",'testing', str(config['forecast_steps']), f'testing.pth')
            #model_path=os.path.join('models', data_handler.get_dataset_name(), 'testing', str(config['forecast_steps']), f'testing.pth')
            predictions, actuals, rmse = evaluate_model_for_programming_task(
                data_handler=data_handler,
                forecast_steps=config['forecast_steps'],
                features=config['features'],
                fill_types=config['fill_types'],
                scaler=config['scaler'],
                participants=test_participants,
                model_path=model_path
            )
            plot_data.append((predictions, actuals))
            rmses.append(rmse)

        rmse_df_inter = pd.DataFrame({'Participant': participants, 'RMSE': rmses})
        print(rmse_df_inter)
        
        plot_results(participants, plot_data)

if __name__ == "__main__":
    config = {
        'dataset_path': r"C:\Users\knutj\OneDrive - Universitaet Bern\Datasets\Ohio Data\Ohio_XML",
        'forecast_steps': 6,
        'features': ['cbg', 'basal', 'carbInput', 'bolus'],
        #'fill_types': ['cubicspline', 0, 0, 0],
        'fill_types': [-5, -5, -5, -5],
        'batch_size': 32,
        'num_epochs': 100,
        'learning_rate': 0.01,
        'model': MirshekarianLSTM(input_dim=4, hidden_dim=5, output_dim=1),
        'scaler': StandardScaler(),
        'dataloader': "DataloaderOhio",
        'dataset_name': 'Ohio2018',

    }
    #main(config, train=True, evaluate=False)
    main(config, train=False, evaluate=True)
