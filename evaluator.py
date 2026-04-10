import torch

class Evaluator:
    def __init__(self, model, test_loader):
        self.model = model
        self.test_loader = test_loader
    
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

