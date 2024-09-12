import torch
import os
from tqdm import tqdm
import torch.nn as nn
import torch.optim as optim
from architectures.lstms import MirshekarianLSTM, TidepoolLSTM
import numpy as np
import matplotlib.pyplot as plt
from torchsummary import summary

class TrainerBasic:
    def __init__(self, hp_config, model_path, parent_model=None,retrain_model=True):
        model = globals()[hp_config["architecture"]](hp_config)
        if parent_model != None:
            pretrained_model_dir = os.path.dirname(parent_model)
            if not os.path.exists(pretrained_model_dir):
                os.makedirs(pretrained_model_dir)
            model.load_state_dict(torch.load(parent_model))
        self.model = model
        self.criterion = nn.MSELoss()
        self.optimizer = optim.Adam(model.parameters(), lr=hp_config['learning_rate'])
        self.num_epochs = hp_config['num_epochs']
        self.patience = 3
        self.model_path = model_path
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        self.retrain_model = retrain_model

    def validate(self, val_loader):
        self.model.eval() 
        val_loss = 0
        with torch.no_grad():
            for data, targets in val_loader:
                data, targets = data.to(self.device), targets.to(self.device)
                outputs = self.model(data)
                loss = self.criterion(outputs, targets)
                val_loss += loss.item()
        return val_loss / len(val_loader)

    def train(self,train_loader, val_loader):
        if self.retrain_model == False:
            self.model.load_state_dict(torch.load(self.model_path))
        best_val_loss = 1000000000000000
        epochs_no_improve = 0

        for epoch in range(self.num_epochs):
            self.model.train()
            total_loss = 0
            for batch_idx, (data, targets) in tqdm(enumerate(train_loader), desc="Batches", total=len(train_loader)):
                data, targets = data.to(self.device), targets.to(self.device)
                self.optimizer.zero_grad()
                outputs = self.model(data)
                loss = self.criterion(outputs, targets)
                loss.backward()
                self.optimizer.step()
                total_loss += loss.item()

                if False:#batch_idx % 1000 == 0:
                    # Convert outputs and targets to numpy arrays
                    outputs_np = outputs.cpu().detach().numpy()
                    targets_np = targets.cpu().detach().numpy()

                    # Reshape arrays to be 2D with matching dimensions
                    outputs_np = outputs_np.reshape(outputs_np.shape[0], -1)
                    targets_np = targets_np.reshape(targets_np.shape[0], -1)

                    # Plotting
                    plt.figure(figsize=(10, 5))
                    for i in range(outputs_np.shape[1]):
                        if i == 11:
                            plt.plot(targets_np[:, i], label=f'Targets {i}')
                            plt.plot(outputs_np[:, i], label=f'Outputs {i}')
                    plt.xlabel('Sample Index')
                    plt.ylabel('Value')
                    plt.title('Targets vs Outputs')
                    plt.legend()
                    #plt.savefig(f'targets_vs_outputs_epoch{epoch}_batch{batch_idx}.png')  # Save the plot to a file
                    plt.savefig(f'targets_vs_outputs.png')
                    plt.close()  # Close the plot to free up memory

            avg_train_loss = total_loss / len(train_loader)
            avg_val_loss = self.validate(val_loader)

            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                epochs_no_improve = 0

                model_dir = os.path.dirname(self.model_path)
                if not os.path.exists(model_dir):
                    os.makedirs(model_dir)

                torch.save(self.model.state_dict(), self.model_path)
                # Save the model summary
                summary_path = os.path.join(model_dir, 'model_summary.txt')
                with open(summary_path, 'w', encoding='utf-8') as f:
                    summary_str = summary(self.model, input_size=(3, 224, 224))  # Adjust input_size as per your model's requirement
                    f.write(str(summary_str))
            else:
                epochs_no_improve += 1
                if epochs_no_improve == self.patience:
                    print('Early stopping triggered!')
                    break
            
            print(f'Epoch {epoch+1}/{self.num_epochs}, Training Loss: {avg_train_loss}, Validation Loss: {avg_val_loss}, epochs_no_improve: {epochs_no_improve}')


    def test(self, test_loader, scaler=None):
        self.model.load_state_dict(torch.load(self.model_path))
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(device)
        self.model.eval()
        with torch.no_grad():
            predictions, actuals = [], []
            for data, targets in test_loader:
                data, targets = data.to(device), targets.to(device)
                outputs = self.model(data)
                predictions.extend(outputs.cpu().numpy())
                actuals.extend(targets.cpu().numpy())

                # Plotting for every 1000th batch
                if False:
                    outputs_np = outputs.cpu().detach().numpy().reshape(outputs.shape[0], -1)
                    targets_np = targets.cpu().detach().numpy().reshape(targets.shape[0], -1)

                    plt.figure(figsize=(10, 5))
                    for i in range(outputs_np.shape[1]):
                        if i == 11:
                            plt.plot(targets_np[:, i], label=f'Targets {i}')
                            plt.plot(outputs_np[:, i], label=f'Outputs {i}')
                    plt.xlabel('Sample Index')
                    plt.ylabel('Value')
                    plt.title(f'Targets vs Outputs (Test Batch )')
                    plt.legend()
                    plt.savefig(f'test_targets_vs_outputs_batch.png')  # Save the plot to a file
                    plt.close()  # Close the plot to free up memory

        predictions = np.array(predictions).reshape(-1, outputs.shape[1])
        actuals = np.array(actuals).reshape(-1, targets.shape[1])
        # Plot predictions vs actuals
        #plt.figure(figsize=(10, 5))
        #for i in range(predictions.shape[1]):
        #    if i == 11:
        #        plt.plot(predictions[:, i], label=f'Predictions {i}', linestyle='-', alpha=0.7)
        #        plt.plot(actuals[:, i], label=f'Actuals {i}', linestyle='--', alpha=0.7)
        #plt.xlabel('Sample Index')
        #plt.ylabel('Value')
        #plt.title('Predictions vs Actuals')
        #plt.legend()
        #plt.show()

        if scaler is not None:
            predictions = scaler.inverse_transform(predictions)
            actuals = scaler.inverse_transform(actuals)
        
        return np.expand_dims(predictions, axis=-1), np.expand_dims(actuals, axis=-1)
    
    def evaluate(self):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(device)
        self.model.eval()
        with torch.no_grad():
            predictions, actuals = [], []
            for data, targets in self.test_loader:
                data, targets = data.to(device), targets.to(device)
                outputs = self.model(data)
                predictions.extend(outputs[6].cpu().numpy())
                actuals.extend(targets[6].cpu().numpy())

        #fig = plt.figure(figsize=(10, 5))
        #plt.plot(actuals, label='Actuals', linestyle='-', linewidth=2, color='blue', alpha=0.7)
        #plt.plot(predictions, label='Predictions', linestyle='--', linewidth=1, color='red', alpha=0.7)
        #plt.legend()
        #plt.show()


        predictions = torch.tensor(predictions)
        actuals = torch.tensor(actuals)

        rmse  = torch.sqrt(torch.mean((predictions - actuals) ** 2))
        print(f'Root Mean Square Error (RMSE): {rmse:.4f}')
        # Maybe compute MSE, RMSE, MAE, R^2, CC, Fit, and MARD as well
        return predictions, actuals, rmse.item()