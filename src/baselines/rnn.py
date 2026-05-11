import torch
import torch.nn as nn

class RNN(nn.Module):
    """
    A baseline RNN model for grid-based SST forecasting.
    This model should be used for comparison, rather than as an evaluated model. 

    Input shape:
        batch_size x context_len x 1 x H x W
    Output shape:
        batch_size x horizon x H x W

    Architecture:
        flatten spatial    -> batch x context_len x H*W
        Linear + ReLU      -> batch x context_len x d_spatial
        RNN                -> batch x hidden_size  (final hidden state)
        Dropout
        Linear             -> batch x horizon * H*W
        reshape            -> batch x horizon x H x W
    """

    def __init__(
        self,
        H: int,
        W: int,
        context_len: int = 90,
        horizon: int = 7,
        d_spatial: int = 64,
        hidden_size: int = 128,
        num_layers: int = 2,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.H = H
        self.W = W
        self.context_len = context_len
        self.horizon = horizon

        spatial_dim = H * W

        # Spatial encoder
        self.encoder = nn.Sequential(
            nn.Linear(spatial_dim, d_spatial),
            nn.ReLU(inplace=True),
        )

        # Dropout safety check. 
        if (num_layers > 1):
            rnn_dropout = dropout
        else:
            rnn_dropout = 0.0

        self.rnn = nn.RNN(
            input_size=d_spatial,
            hidden_size=hidden_size,
            num_layers=num_layers,
            # TODO: Check relu vs tanh here. 
            nonlinearity="tanh",
            batch_first=True,
            dropout=rnn_dropout,
        )

        # Output head
        self.dropout = nn.Dropout(p=dropout)
        self.decoder = nn.Linear(hidden_size, (horizon * spatial_dim))


    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        x shape:
            batch_size x context_len x 1 x H x W

        Returns SST forecast with shape:
            batch_size x horizon x H x W
        """
        batch_size = x.shape[0]

        # Flatten spatial grid: (B, L, 1, H, W) -> (B, L, H*W)
        x_flat = x.view(batch_size, self.context_len, (self.H * self.W))

        # Spatial encoder: (B, L, H*W) -> (B, L, d_spatial)
        encoded = self.encoder(x_flat)

        # RNN Forward Pass
        # h_n shape: (num_layers, B, hidden_size)
        _, h_n = self.rnn(encoded)
        # Take the final hidden state from the last layer. 
        last_hidden = h_n[-1]

        # Output head: (B, hidden) -> (B, horizon * H * W) -> (B, horizon, H, W)
        out = self.decoder(self.dropout(last_hidden))
        # Reshape to grid forecast. 
        return out.view(batch_size, self.horizon, self.H, self.W)