import torch
import torch.nn as nn

class Model_LSTM(nn.Module):
    
    def __init__(self, input_size, output_size, hidden_dim, n_layers, dropout, device):
        super(Model_LSTM, self).__init__()
        self.device = device
        self.hidden_dim = hidden_dim
        self.n_layers = n_layers

        self.rnn = nn.LSTM(input_size, hidden_dim, n_layers, batch_first=True, dropout=dropout).to(device)   
        self.fc = nn.Linear(hidden_dim, output_size).to(device)
    
    def forward(self, x):
        batch_size = x.size(0)
        h0, c0 = self.init_hidden(batch_size)
        out, (hn, cn) = self.rnn(x, (h0.detach(), c0.detach()))
        out = out[:, -1, :]
        out = self.fc(out)
        return out
    
    def init_hidden(self, batch_size):
        h0 = torch.zeros(self.n_layers, batch_size, self.hidden_dim).to(self.device)
        c0 = torch.zeros(self.n_layers, batch_size, self.hidden_dim).to(self.device)
        return (h0, c0)
