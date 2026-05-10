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
        RNN       -> batch x hidden_size  (final hidden state)
        Dropout
        Linear             -> batch x horizon * H*W
        reshape            -> batch x horizon x H x W
    """

    # TODO: Complete