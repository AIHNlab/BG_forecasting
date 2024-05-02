import torch
import os
from tqdm import tqdm
import torch.nn as nn
import torch.optim as optim

class TrainerBasic:
    def __init__(self, train_loader, val_loader, hp_config, model_path, parent_model=None):
        model = globals()[hp_config["architecture"]](hp_config)
        if parent_model != None:
            pretrained_model_dir = os.path.dirname(parent_model)
            if not os.path.exists(pretrained_model_dir):
                os.makedirs(pretrained_model_dir)
            model.load_state_dict(torch.load(parent_model))
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.criterion = nn.MSELoss()
        self.optimizer = optim.Adam(model.parameters(), lr=hp_config['learning_rate'])
        self.num_epochs = hp_config['num_epochs']
        self.patience = 3
        self.model_path = model_path
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)

    def validate(self):
        self.model.eval() 
        val_loss = 0
        with torch.no_grad():
            for data, targets in self.val_loader:
                data, targets = data.to(self.device), targets.to(self.device)
                outputs = self.model(data)
                loss = self.criterion(outputs, targets)
                val_loss += loss.item()
        return val_loss / len(self.val_loader)

    def train(self):

        best_val_loss = 1000000000000000
        epochs_no_improve = 0

        for epoch in range(self.num_epochs):
            self.model.train()
            total_loss = 0
            for batch_idx, (data, targets) in tqdm(enumerate(self.train_loader), desc="Batches", total=len(self.train_loader)):
                data, targets = data.to(self.device), targets.to(self.device)
                self.optimizer.zero_grad()
                outputs = self.model(data)
                loss = self.criterion(outputs, targets)
                loss.backward()
                self.optimizer.step()
                total_loss += loss.item()

            avg_train_loss = total_loss / len(self.train_loader)
            avg_val_loss = self.validate()

            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                epochs_no_improve = 0

                model_dir = os.path.dirname(self.model_path)
                if not os.path.exists(model_dir):
                    os.makedirs(model_dir)

                torch.save(self.model.state_dict(), self.model_path)
            else:
                epochs_no_improve += 1
                if epochs_no_improve == self.patience:
                    print('Early stopping triggered!')
                    break
            
            print(f'Epoch {epoch+1}/{self.num_epochs}, Training Loss: {avg_train_loss}, Validation Loss: {avg_val_loss}, epochs_no_improve: {epochs_no_improve}')

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