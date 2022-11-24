import torch
import torch.nn as nn

class Model_GRU(nn.Module):
    
    def __init__(self, input_size, output_size, hidden_dim, n_layers, dropout, device):
        super(Model_GRU, self).__init__()
        self.device = device
        self.n_layers = n_layers
        self.hidden_dim = hidden_dim

        self.rnn = nn.GRU(input_size, hidden_dim, n_layers, batch_first=True, dropout=dropout).to(device)
        self.fc = nn.Linear(hidden_dim, output_size).to(device)

    def forward(self, x):
        h0 = torch.zeros(self.n_layers, x.size(0), self.hidden_dim).to(self.device)
        out, _ = self.rnn(x, h0.detach())
        out = out[:, -1, :]
        out = self.fc(out)

        return out
