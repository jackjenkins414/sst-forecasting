import torch
import torch.nn as nn


class SimpleSstLSTM(nn.Module):
    """
    Simple flattened LSTM model for SST forecasting.

    Input shape:
        batch_size x input_length x input_size

    Output shape:
        batch_size x output_size

    Here:
        input_size  = number of ocean grid points
        output_size = number of ocean grid points being predicted
    """

    def __init__(
        self,
        input_size: int,
        hidden_size: int = 128,
        num_layers: int = 1,
        output_size: int | None = None,
    ):
        super().__init__()

        if output_size is None:
            output_size = input_size

        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
        )

        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        x shape:
            batch_size x input_length x input_size
        """

        lstm_out, _ = self.lstm(x)

        # Use the final LSTM output in the input sequence
        last_hidden = lstm_out[:, -1, :]

        # Map hidden representation to SST values at all ocean points
        y_pred = self.fc(last_hidden)

        return y_pred