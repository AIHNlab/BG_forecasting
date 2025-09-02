import torch
import torch.nn as nn
import torch.nn.functional as F

class MirshekarianLSTM(nn.Module):
    def __init__(self, config):
        super(MirshekarianLSTM, self).__init__()
        self.lstm = nn.LSTM(config['feature_window'], config['hidden_dim'], batch_first=True)
        self.dense = nn.Linear(config['hidden_dim'], config['forecast_steps'])
    
    def forward(self, x):
        lstm_out, (hn, cn) = self.lstm(x)
        out = self.dense(lstm_out[:, -1, :])
        return out



class TidepoolLSTM(nn.Module):
    def __init__(self,config):
        super(TidepoolLSTM, self).__init__()
        
        # LSTM Layer
        self.lstm = nn.LSTM(input_size=config['n_features'], hidden_size=config['hidden_dim'], num_layers=config['lstm_layers'], batch_first=True)
        
        # First dense layer dimension is configurable, others are half of previous
        dense_dim1 = config.get('first_dense_dim', config['hidden_dim']//2)  # Default to half of LSTM if not specified
        dense_dim2 = dense_dim1 // 2
        dense_dim3 = dense_dim2 // 2
        
        # Fully connected layers with Batch Normalization and ReLU activation
        self.fc1 = nn.Linear(config['hidden_dim'], dense_dim1)
        self.bn1 = nn.BatchNorm1d(dense_dim1)
        
        self.fc2 = nn.Linear(dense_dim1, dense_dim2)
        self.bn2 = nn.BatchNorm1d(dense_dim2)
        
        self.fc3 = nn.Linear(dense_dim2, dense_dim3)
        self.bn3 = nn.BatchNorm1d(dense_dim3)
        
        # Output layer
        self.fc4 = nn.Linear(dense_dim3, config['forecast_steps'])
    
    def forward(self, x):
        # LSTM layer
        lstm_out, _ = self.lstm(x)
        
        # Take the output from the last time step
        lstm_out = lstm_out[:, -1, :]
        
        # First dense layer
        x = self.fc1(lstm_out)
        x = self.bn1(x)
        x = F.relu(x)
        
        # Second dense layer
        x = self.fc2(x)
        x = self.bn2(x)
        x = F.relu(x)
        
        # Third dense layer
        x = self.fc3(x)
        x = self.bn3(x)
        x = F.relu(x)
        
        # Output layer
        x = self.fc4(x)
        x = torch.unsqueeze(x, dim=-1)
        return x[:, :, 0]  # Return the output without the last dimension

# Example usage
#model = GlucosePredictionModel()
#input_data = torch.randn(32, 10, 2)  # batch size of 32, sequence length of 10, and 2 input features (glucose and IOB)
#output = model(input_data)
#print(output.shape)  # Should output torch.Size([32, 12])