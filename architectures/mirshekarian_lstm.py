import torch.nn as nn

class MirshekarianLSTM(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(MirshekarianLSTM, self).__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, batch_first=True)
        self.dense = nn.Linear(hidden_dim, output_dim)
    
    def forward(self, x):
        lstm_out, (hn, cn) = self.lstm(x)
        out = self.dense(lstm_out[:, -1, :])
        return out
