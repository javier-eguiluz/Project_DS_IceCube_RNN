# LSTM model extended to handle continuous input and output, i.e., one output value for each timestep of the input
# Can run in two modes; either taking long - but finite - sequences as input or in step-by-step mode (where the input length is unlimited)
# The latter mode is much slower and is not suitable for training

import torch
import torch.nn as nn

class Model_Continuous_LSTM(nn.Module):
    
    def __init__(self, input_size, output_size, hidden_dim, n_layers, dropout):
        super(Model_Continuous_LSTM, self).__init__()
        self.hidden_dim = hidden_dim
        self.n_layers = n_layers
        self.step_by_step = False

        self.rnn = nn.LSTM(input_size, hidden_dim, n_layers, batch_first=True, dropout=dropout).to(device)
        self.fc = nn.Linear(hidden_dim, output_size).to(device)

    def step_by_step_mode(self, activate):
        if activate:
            self.h, self.c = self.init_hidden(self.n_layers, self.hidden_dim, 1)  # Batch size = 1 in this case
        self.step_by_step = activate

    def forward(self, x):
        batch_size = x.size(0)

        if self.step_by_step:   # Based on reasoning in the second code block in this page: https://pytorch.org/tutorials/beginner/nlp/sequence_models_tutorial.html
            out, (self.h, self.c) = self.rnn(x, (self.h.detach(), self.c.detach()))
            out = self.fc(out)
        else:
            h0, c0 = self.init_hidden(self.n_layers, self.hidden_dim, batch_size)
            out, (hn, cn) = self.rnn(x, (h0.detach(), c0.detach()))
            out = self.fc(out)

        return out

    def init_hidden(self, n_layers, dim, batch_size):
        h0 = torch.zeros(n_layers, batch_size, dim).to(device)
        c0 = torch.zeros(n_layers, batch_size, dim).to(device)
        return (h0, c0)
