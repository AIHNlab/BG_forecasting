import torch
import os

class TrainerBasic:
    def __init__(self, model, train_loader, val_loader, criterion, optimizer, num_epochs=100, patience=3, model_path='best_model.pth'):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.criterion = criterion
        self.optimizer = optimizer
        self.num_epochs = num_epochs
        self.patience = patience
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
            for batch_idx, (data, targets) in enumerate(self.train_loader):
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