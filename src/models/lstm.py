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
        super().__init__()

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


    # Trains the model across a provided number of epochs. 
    # TODO: Check what adaptations need to be made to suit the data structure. 
    def train_model(model, optimiser, criterion, training_loader, num_epochs, device):
        model.to(device)
        # Initialise losses. 
        losses = []

        # Train the model for x epochs. 
        for epoch in range(num_epochs):
            model.train()
            epoch_loss = 0.0

            for x, y in training_loader:

                x = x.to(device)
                y = y.to(device)

                batch_size = x.size(0)

                # Initialise the hidden state for the epoch. 
                hidden = model.init_hidden(batch_size, device)

                # Complete a forward pass.
                predictions, _ = model(x, hidden)

                # Handle a forecasting shape mismatch.
                # Source: Copilot
                if predictions.shape != y.shape: 
                    predictions = predictions[:, -1, :]
                    y = y[:, -1, :]

                # Compute the loss. 
                loss = criterion(predictions, y)

                # Backpropagation. 
                optimiser.zero_grad()
                loss.backward()

                # Step
                optimiser.step()

                # Add the loss. 
                epoch_loss += loss.item()

            # Determine the epoch loss. 
            epoch_loss /= len(training_loader)
            losses.append(epoch_loss)

            # Print the loss for this epoch. 
            print(f"Epoch {epoch+1}/{num_epochs} | Training Loss: {epoch_loss:.6f}")

        return losses
    

if __name__ == "__main__":
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    training_loader = ... # TODO: Figure this out.

    lstm_model = SST_LSTM(
        input_dim=1,
        hidden_dim=64,
        output_dim=1,
        num_layers=1
    )

    # Training
    optimiser = torch.optim.Adam(lstm_model.parameters(), lr=1e-3)
    criterion = nn.MSELoss() #TODO: Replace with a RMSE Loss. 
    losses = SST_LSTM.train_model(
        lstm_model,
        optimiser,
        criterion,
        training_loader,
        num_epochs=10,
        device=device
    )

    print("Training complete.")