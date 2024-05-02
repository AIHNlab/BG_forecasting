import torch.nn as nn

class MirshekarianLSTM(nn.Module):
    def __init__(self, config):
        super(MirshekarianLSTM, self).__init__()
        self.lstm = nn.LSTM(config['input_dim'], config['hidden_dim'], batch_first=True)
        self.dense = nn.Linear(config['hidden_dim'], config['output_dim'])
    
    def forward(self, x):
        lstm_out, (hn, cn) = self.lstm(x)
        out = self.dense(lstm_out[:, -1, :])
        return out
