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
from dataloaders.dataloader_ohio_processed import Dataloader
from datapreprocessor import DataPreProcessor
from scaler import Scaler
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)


def train_model(data_handler, participants, features, fill_types, forecast_steps, model_path, batch_size, num_epochs, learning_rate, model, trainer_class, pretrained_model_path=None):
    
    train_preprocessor = DataPreProcessor(data_handler.get_train_dataframes())
    train_preprocessor.handle_missing_values(features, fill_types)
    

    # Initialize data prepper and make features and target pair
    scaler = Scaler(data_handler, features)
    scaler.fit_and_save('scaler_2018.pkl')
    prepper = DataPrepper(participants, data_handler.get_train_dataframes(), forecast_steps=forecast_steps, scaler=scaler.get_scaler())
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

def main():
    # Load and preprocess data
    dataset_location=r"C:\Users\knutj\OneDrive - Universitaet Bern\Datasets\Ohio Data\Ohio2018_processed"
    data_loader = Dataloader(dataset_location)
    data_handler = DataHandler(data_loader, dataset_name='Ohio2018_processed')
    data_handler.load_data()
    forecast_steps = 6
    participants = list(data_handler.get_train_dataframes().keys())
    participant = participants[0]
    train_model(
        data_handler = data_handler,
        participants=participants,
        features=['cbg', 'basal', 'carbInput', 'bolus'],
        fill_types=['cubicspline', 0, 0, 0],
        forecast_steps=forecast_steps,
        model_path=os.path.join('models', 'testing', str(forecast_steps), f'best_model_{participant}.pth'),
        batch_size=32,
        num_epochs=100,
        learning_rate=0.01,
        model = MirshekarianLSTM(input_dim=4, hidden_dim=5, output_dim=1),
        trainer_class = TrainerBasic,
        pretrained_model_path=None
    )

if __name__ == "__main__":
    main()