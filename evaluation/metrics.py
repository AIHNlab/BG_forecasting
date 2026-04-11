"""Simple model evaluator for quick RMSE assessment on a test set."""

import torch


class Evaluator:
    """Run inference on a test loader and compute aggregate RMSE.

    This is a lightweight evaluator primarily used for sanity checks.
    The main evaluation pipeline in ``pipeline.evaluation`` provides
    richer per-horizon and per-participant analysis.

    Args:
        model: A PyTorch ``nn.Module`` in eval-ready state.
        test_loader: A ``DataLoader`` yielding ``(inputs, targets)`` batches.
    """

    def __init__(self, model, test_loader):
        self.model = model
        self.test_loader = test_loader
    
    def evaluate(self):
        """Run inference and return predictions, actuals, and RMSE.

        Returns:
            tuple: ``(predictions, actuals, rmse)`` where predictions and
            actuals are ``torch.Tensor`` and rmse is a float.
        """
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

