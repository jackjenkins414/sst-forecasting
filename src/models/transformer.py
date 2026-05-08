import torch 
import torch.nn as nn
import math 

class SpatialProjection(nn.Module):
    """Project flattened SST grid to d_model.

    (B, L, 1, H, W) -> (B, L, d_model)
    """
    def __init__(self, d_model: int, height: int, width: int):
        """Build the spatial projection layer.

        Parameters
        ----------
        d_model : int
            Working dimension of the Transformer. Every internal layer operates 
            on vectors of this size.
        height : int
            Spatial height of the SST grid (81 for Coral Sea).
        width : int
            Spatial width of the SST grid (121 for Coral Sea).
        """
        super().__init__()
        self.d_model = d_model

        # Spatial dims of the SST grid 
        self.height = height
        self.width = width

        # Learnable linear layer; model learns which spatail parms matter
        # and compresses them into d_model dims
        self.projection = nn.Linear(height * width, d_model)

    def forward(self, x):
        """Project a batch of SST sequences into d_model space.

        Parameters
        ----------
        x : torch.Tensor
            Input SST sequences, shape (B, L, 1, H, W).
            B = batch size, L = sequence length (90 days),
            1 = channel dim, H, W = spatial grid.

        Returns
        -------
        torch.Tensor
            Projected sequences, shape (B, L, d_model), scaled by
            sqrt(d_model) per Vaswani et al. Section 3.4.
        """
        B, L, _, H, W = x.shape

        # Flatten the (1, H, W) per timestep into single H*W vector
        x = x.view(B, L, H * W)

        # Map each H*W vector to a d_model vector by applying learned 
        # projection to every timestep independently
        x = self.projection(x)

        # Scale as per paper implementation 
        return x * math.sqrt(self.d_model)
    
class PositionalEncoding(nn.Module):
    """Add sinusoidal positional encodings to an embedded sequence.

    (B, L, d_model) -> (B, L, d_model)
    """

    def __init__(self, d_model: int, seq_len: int, dropout: float) -> None:
        """Build the positional encoding table.

        Parameters
        ----------
        d_model : int
            Working dimension of the Transformer. Must match the dimension of the 
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

        # nn.Dropout is a module, not a float, has to be instantiated (bug fix here)
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
        # Computing exp(log(...)) instead of pow() bc it is more numerically stable 
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
