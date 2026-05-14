# TODO: This currently is only a framework. Code and comments will be added 
#       in due course. 
# Based on: https://arxiv.org/pdf/2012.07436

import torch
import torch.nn as nn
import math

#TODO: Add head comment.
class CNNSpatialEncoding(nn.Module):
    """
    CNN-based spatial feature extractor for SST grids.

    (B, L, 1, H, W) -> (B, L, d_model)
    """
    def __init__(self, d_model: int):
        """Build the CNN-based spatial encoder.

        Parameters
        ----------
        d_model : int
            Working dimension of the Informer. The CNN compresses
            each spatial grid into a d_model-dimensional vector. 
        """
        super.__init__()
        self.cnn = nn.Sequential(
            # NOTE: Choice of 32 output channels is arbitrary, find optimal value. 
            # TODO: Figure out whether or not to pad. 
            # Extract low-level spatial patterns. 
            nn.Conv2d(1, 32, kernel_size=3, padding=1), 
            nn.ReLU(), 
            # Extract higher-level curvatures and pooling patterns. 
            nn.Conv2d(32, 64, kernel_size=3, padding=1), 
            nn.ReLU(),
            # Collapse H*W to a global 1*1 aggregate. 
            nn.AdaptiveAvgPool2d(1), 
            # Flatten [64, 1, 1] to [64] (vector per timestep). 
            nn.Flatten(),
            # Map a projection to d_model. 
            nn.Linear(64, d_model)
        )
    
    def forward(self, x):
        """Encode each timestep's SST grid into a d_model vector.

        Parameters
        ----------
        x : torch.Tensor
            Input SST sequences, shape (B, L, 1, H, W).
            B = batch size, L = sequence length (90 days),
            1 = C = channel dim, H, W = spatial grid.

        Returns
        -------
        torch.Tensor
            CNN-encoded tensor of shape (B, L, d_model).
        """
        B, L, C, H, W = x.shape
        # Merge batch and time for independent timestep processing. 
        x = x.view(B * L, C, H, W)
        # Apply CNN encoder to each timestep. 
        x = self.cnn(x)
        # Restore structure: (B*L, d_model) -> (B, L, d_model)
        return x.view(B, L, -1)
    
# Imported from Jack's Transformer model. 
class PositionalEncoding(nn.Module):
    """Add sinusoidal positional encodings to an embedded sequence.

    (B, L, d_model) -> (B, L, d_model)
    """

    def __init__(self, d_model: int, seq_len: int, dropout: float) -> None:
        """Build the positional encoding table.

        Parameters
        ----------
        d_model : int
            Working dimension of the Informer. Must match the dimension of the 
            embeddings the encoding is added to.
        seq_len : int
            Maximum sequence length the model will ever see. We pre-compute
            the table up to this length and slice at runtime. For SST we
            always use 90, but will change later. 
        dropout : float
            Dropout probability applied after adding the positional signal.
            Acts as regularisation on the combined embedding+position vector.
        """
        super().__init__()
        self.d_model = d_model
        self.seq_len = seq_len

        # dropout defines the probability. 
        self.dropout = nn.Dropout(dropout)

        # Pre-allocate the encoding table: one d_model-dim vector per position
        # Shape: (seq_len, d_model)
        pe = torch.zeros(seq_len, d_model)

        # Position indices 0, 1, ..., seq_len-1, reshaped to a column vector
        # so it broadcasts cleanly against div_term in the next line.
        # Shape: (seq_len, 1)
        position = torch.arange(0, seq_len, dtype=torch.float).unsqueeze(1)

        # Frequency terms from the paper
        #   PE(pos, 2i)   = sin(pos / 10000^(2i/d_model))
        #   PE(pos, 2i+1) = cos(pos / 10000^(2i/d_model))
        # Computing exp(log(...)) instead of pow() bc it is more numerically stable. 
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )

        # Even indexed dims get sin, odd indexed dims get cos
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        # Add a leading batch dim so we can broadcast against any batch size
        # in forward(). Shape: (seq_len, d_model) -> (1, seq_len, d_model)
        pe = pe.unsqueeze(0)

        # register_buffer not a learnable parameter; pos encodings are deterministic, not learned
        self.register_buffer('pe', pe)

    def forward(self, x):
        """Add positional encodings to the input and apply dropout.

        Parameters
        ----------
        x : torch.Tensor
            Embedded sequence, shape (B, L, d_model). Comes from
            SpatialProjection.

        Returns
        -------
        torch.Tensor
            Same shape (B, L, d_model), now with position information added
            and dropout applied.
        """
        # Slice the pre-computed table to the actual sequence length, then add element-wise
        x = x + self.pe[:, :x.shape[1], :]

        # Dropout on the combined embedding+position signal; regularisation as per paper
        return self.dropout(x)
    
#TODO
class DataEmbedding(nn.Module):
    def __init__():
        #TODO
        return
    
    def forward(self, x):
        #TODO
        return
    
# Imported from Jack's Transformer model.
class LayerNormalisation(nn.Module):
    def __init__(self, d_model: int, eps: float = 10**-6) -> None:
        """Build the layer norm.

        Parameters
        ----------
        d_model : int
            Working dimension of the Informer. The learnable scale and
            shift each have one entry per dimension, so the layer can amplify some
            feature dims and suppress others.
        eps : float
            Small constant added to the denominator for numerical stability,
            preventing division by zero when std is tiny.
        """
        super().__init__()
        # Avoid division blow up 
        self.eps = eps

        # Learnable scale, one per feature dim; initialised to start as the identity
        self.alpha = nn.Parameter(torch.ones(d_model))    # per-dim scale

        # Learnable shift, one per feature dim; initialised to 0 so the layer starts as identity
        self.bias = nn.Parameter(torch.zeros(d_model))    # per-dim shift

    def forward(self, x):
        """Normalise across the last dim, then apply learned scale and shift.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor, shape (B, L, d_model). Comes from any sublayer
            that produces d_model-dim vectors per timestep.

        Returns
        -------
        torch.Tensor
            Same shape (B, L, d_model), normalised per timestep across the feature 
            dim, then rescaled and shifted.
        """
        mean = x.mean(dim=-1, keepdim=True)

        # Each layer normalised by its own stats- not batch stats (hence layer norm not batch norm)
        std = x.std(dim=-1, keepdim=True)

        # standard layernorm formula
        return self.alpha * (x - mean) / (std + self.eps) + self.bias
    
#TODO
class ProbSparseAttention(nn.Module):
    def __init__():
        #TODO
        return
    
    def forward(self, x):
        #TODO
        return
    
    def prob_QK():
        #TODO
        return
    
    def get_initial_context():
        #TODO
        return
    
    def update_context():
        #TODO
        return
    
#TODO
class SelfAttentionLayer(nn.Module):
    def __init__():
        #TODO
        return
    
    def forward(self, x):
        #TODO
        return

#TODO
class CrossAttentionLayer(nn.Module):
    def __init__():
        #TODO
        return
    
    def forward(self, x):
        #TODO
        return
    
# Designed based on Jack's Transformer model.
# Initial inspiration: https://github.com/hkproj/pytorch-transformer/blob/main/model.py
class FeedForwardBlock(nn.Module):
    """Position wise feed-forward network.
    (B, L, d_model) -> (B, L, d_model)
    """

    def __init__(self, d_model: int, d_ff: int, dropout: float) -> None:
        """Build the feed-forward block.

        Parameters
        ----------
        d_model : int
            Working dimension of the Informer. Input and output dim.
        d_ff : int
            Inner hidden dimension. The block expands d_model -> d_ff, applies a 
            nonlinearity, then projects back to d_model. Initial paper uses 
            d_ff = 4 * d_model.
        dropout : float
            Dropout probability applied between the two linear layers,
            after the ReLU. Standard regularisation per the paper.
        """
        super().__init__()

        # TODO: Update comment. 
        self.seq = nn.Sequential(
            nn.Linear(d_model, d_ff), nn.ReLU(), 
            nn.Dropout(dropout), nn.Linear(d_ff, d_model)
        )

    def forward(self, x):
        # TODO: Update comments. 
        return self.seq(x)
    
#TODO
class ResidualConnection(nn.Module):
    def __init__():
        #TODO
        return
    
    def forward(self, x):
        #TODO
        return
    
#TODO
class EncoderLayer(nn.Module):
    def __init__():
        #TODO
        return
    
    def forward(self, x):
        #TODO
        return
    
#TODO
class EncoderDistillation(nn.Module):
    def __init__():
        #TODO
        return
    
    def forward(self, x):
        #TODO
        return
    
#TODO
class InformerEncoder(nn.Module):
    def __init__():
        #TODO
        return
    
    def forward(self, x):
        #TODO
        return
    
#TODO
class DecoderLayer(nn.Module):
    def __init__():
        #TODO
        return
    
    def forward(self, x):
        #TODO
        return
    
#TODO
class InformerDecoder(nn.Module):
    def __init__():
        #TODO
        return
    
    def forward(self, x):
        #TODO
        return
    
#TODO
class ProjectionHead(nn.Module):
    def __init__():
        #TODO
        return
    
    def forward(self, x):
        #TODO
        return

  
#TODO
class ProbSparseInformer(nn.Module):
    def __init__():
        #TODO
        return
    
    def forward(self, x):
        #TODO
        return