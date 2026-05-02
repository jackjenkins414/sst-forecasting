"""
A basic demo LSTM model, created as a precursor to the actual data-aligned
models for the COMP3242 Group Research Project. 

Please note that this model was made purely as a proof-of-concept for the 
Week 9 meeting, and is unlikely to be viable as an assignment model.

Inspiration taken from:
    lab06.ipynb (Isaac Jaensch & the COMP3242 Teaching Team)

Written by Isaac Jaensch (u7262835), ANU, 2026. 
"""

import torch
import torch.nn as nn
import numpy as np

class SST_LSTM(nn.Module):

    # Initialises the model.
    def __init__(self, input_dim, hidden_dim, output_dim, num_layers):
        super.__init__()

        self.hidden_dim = hidden_dim
        self.num_layers = num_layers

        self.lstm = nn.LSTM(
            input_size = input_dim,
            hidden_size = hidden_dim, 
            num_layers = num_layers,
            batch_first = True
        )

        self.fc = nn.Linear(hidden_dim, output_dim)

    # Completes a forward pass through the model. 
    def forward(self, x, init_hidden):
        out, hidden = self.lstm(x, init_hidden)
        predictions = self.fc(out)
        return predictions, hidden
    
    # Initialises the hidden state for the LSTM. 
    def init_hidden(self, batch_size, device):
        h0 = torch.zeros(self.num_layers, batch_size, self.hidden_dim, device=device)
        c0 = torch.zeros(self.num_layers, batch_size, self.hidden_dim, device=device)
        return (h0, c0)
