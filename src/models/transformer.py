import torch 
import torch.nn as nn
import math 

# This is our input embedding 
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
    

class LayerNormalisation(nn.Module):

    def __init__(self, d_model: int, eps: float = 10**-6) -> None:
        """Build the layer norm.

        Parameters
        ----------
        d_model : int
            Working dimension of the Transformer. The learnable scale and
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

# Reference: https://github.com/hkproj/pytorch-transformer/blob/main/model.py
# Building this transformer from scratch, the feedforward block is one of the components that can be 
# copied across from NLP without modification. Obviously, any part of the transformer architecure that 
# touches the input/output format needs modification (as above), but any component that acts purely 
# on d_model vectors is domain agnositc. 
# We have used their code, but added docstrings and comments 
class FeedForwardBlock(nn.Module):
    """Position wise feed-forward network.
    (B, L, d_model) -> (B, L, d_model)
    """

    def __init__(self, d_model: int, d_ff: int, dropout: float) -> None:
        """Build the feed-forward block.

        Parameters
        ----------
        d_model : int
            Working dimension of the Transformer. Input and output dim.
        d_ff : int
            Inner hidden dimension. The block expands d_model -> d_ff, applies a 
            nonlinearity, then projects back to d_model. Note paper (add ref later)
            uses d_ff = 4 * d_model (e.g. 2048 with d_model=512).
        dropout : float
            Dropout probability applied between the two linear layers,
            after the ReLU. Standard regularisation per the paper.
        """
        super().__init__()
        # First linear: expand from d_model to the wider hidden dim d_ff.
        # Weight shape: (d_ff, d_model), bias shape: (d_ff,); note W1, b1 from paper 
        self.linear_1 = nn.Linear(d_model, d_ff)

        # Dropout after ReLU to reduce co-adaptation of hidden neurons before the output layer
        self.dropout = nn.Dropout(dropout)

        # W2, b2: project d_ff back to d_model for the residual connection 
        self.linear_2 = nn.Linear(d_ff, d_model)

    def forward(self, x):
        """Apply the FFN independently to each timestep.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor, shape (B, L, d_model). One d_model vector per
            timestep per batch element.

        Returns
        -------
        torch.Tensor
            Same shape (B, L, d_model). Each timestep's vector has been
            transformed by the same FFN weights, independently.
        """
        # Three step here, and all operating on the last dim only:
        #   (B, L, d_model) -> linear_1 -> (B, L, d_ff)
        #   (B, L, d_ff)    -> relu     -> (B, L, d_ff)     (nonlinearity)
        #   (B, L, d_ff)    -> dropout  -> (B, L, d_ff)     (regularisation)
        #   (B, L, d_ff)    -> linear_2 -> (B, L, d_model)
        # The same weights are applied to every timestep, hence the "position-wise"
        # from the paper's terminology. No mixing across timesteps happens here; 
        # that's attention's job.
        return self.linear_2(self.dropout(torch.relu(self.linear_1(x))))

# Reference: https://github.com/hkproj/pytorch-transformer/blob/main/model.py
# Multi-head attention where the model learns which timesteps to attend to when building 
# each output representation. We continue to build it from scratch (rather than using 
# nn.MultiheadAttention) for two reasons. First, to stay consistent with the rest of the 
# from scratch implementation, since building everything except the most important component 
# feels wrong, and would leave a gap in my understanding. Second, this hand-rolling approach 
# makes self.attention_scores trivially accessible after a forward pass, which is useful for 
# visualising what the model actually attends to across the 90-day window. Like the FFN, MHA 
# operates purely on d_model vectors and is also domain agnostic, no SST specific changes are 
# needed here. If attention ever becomes a runtime bottleneck (unlikely and in theory the more
# expensive candidate is Linear in SpatialProjection), we can swap to nn.MultiheadAttention 
# for its optimised kernels (Flash Attention, fused ops); the surrounding code wouldn't 
# need to change. But for now we stick with this implementation. 
class MultiHeadAttentionBlock(nn.Module):
    """Multi-head self-attention.

    (B, L, d_model) -> (B, L, d_model)
    """

    def __init__(self, d_model: int, h: int, dropout: float) -> None:
        """Build the multi-head attention block.

        Parameters
        ----------
        d_model : int
            Working dimension of the Transformer. Must be divisible by h.
        h : int
            Number of attention heads. Each head sees d_model/h dims.
        dropout : float
            Dropout probability applied to attention weights after softmax.
        """
        super().__init__()
        self.d_model = d_model
        self.h = h

        # Each head sees d_k = d_model / h dims; lets heads specialise on
        # different subspaces of the representation
        self.d_k = d_model // h

        # Wq, Wk, Wv: project input to query/key/value spaces; bias=False
        # per common reference implementations (paper unspecified)
        self.w_q = nn.Linear(d_model, d_model, bias=False)
        self.w_k = nn.Linear(d_model, d_model, bias=False)
        self.w_v = nn.Linear(d_model, d_model, bias=False)

        # Wo: combines all h heads' outputs back into d_model
        self.w_o = nn.Linear(d_model, d_model, bias=False)

        # Dropout applied to attention scores after softmax (paper Section 5.4)
        self.dropout = nn.Dropout(dropout)

    @staticmethod
    def attention(query, key, value, mask, dropout: nn.Dropout):
        """Compute scaled dot-product attention.

        Parameters
        ----------
        query, key, value : torch.Tensor
            Shape (B, h, L, d_k). Already split into heads.
        mask : torch.Tensor or None
            Optional mask, broadcastable to (B, h, L, L). Positions where
            mask == 0 get -inf scores (zero attention after softmax).
        dropout : nn.Dropout or None
            Applied to attention weights after softmax.

        Returns
        -------
        tuple
            (attention output, attention weights). Output is (B, h, L, d_k);
            weights are (B, h, L, L), kept for visualisation.
        """
        d_k = query.shape[-1]

        # Scaled dot-product per paper: QK^T / sqrt(d_k). Scaling prevents dot products 
        # from growing large and pushing softmax into saturation
        # (B, h, L, d_k) @ (B, h, d_k, L) -> (B, h, L, L)
        attention_scores = (query @ key.transpose(-2, -1)) / math.sqrt(d_k)

        # Mask out invalid positions before softmax; -inf becomes 0 after softmax
        if mask is not None:
            attention_scores.masked_fill_(mask == 0, -1e9)

        # Softmax over last dim: each query gets a distribution over keys
        attention_scores = attention_scores.softmax(dim=-1)

        # Dropout on attention weights; standard regularisation per paper
        if dropout is not None:
            attention_scores = dropout(attention_scores)

        # Weighted sum of values using attention weights
        # (B, h, L, L) @ (B, h, L, d_k) -> (B, h, L, d_k)
        # Also return the weights themselves for later visualisation
        return (attention_scores @ value), attention_scores

    def forward(self, q, k, v, mask):
        """Run multi-head attention on the input.

        For self-attention, q == k == v. The block keeps the q/k/v
        arguments separate to stay general (would support cross-attention
        if we ever need it).

        Parameters
        ----------
        q, k, v : torch.Tensor
            All shape (B, L, d_model). For self-attention all three are the
            same tensor.
        mask : torch.Tensor or None
            Optional mask. None for our encoder-only forecaster.

        Returns
        -------
        torch.Tensor
            Shape (B, L, d_model). Each timestep's vector is now a weighted
            combination of all other timesteps' values, refined by Wo.
        """
        # Project inputs to Q, K, V spaces; shape unchanged (B, L, d_model)
        query = self.w_q(q)
        key = self.w_k(k)
        value = self.w_v(v)

        # Split into h heads: (B, L, d_model) -> (B, L, h, d_k) -> (B, h, L, d_k)
        # Transpose puts the head dim before the sequence dim so attention()
        # can treat each head independently
        query = query.view(query.shape[0], query.shape[1], self.h, self.d_k).transpose(1, 2)
        key = key.view(key.shape[0], key.shape[1], self.h, self.d_k).transpose(1, 2)
        value = value.view(value.shape[0], value.shape[1], self.h, self.d_k).transpose(1, 2)

        # Run attention; store scores on self for later visualisation
        x, self.attention_scores = MultiHeadAttentionBlock.attention(query, key, value, mask, self.dropout)

        # Recombine heads: (B, h, L, d_k) -> (B, L, h, d_k) -> (B, L, d_model)
        # .contiguous() is needed because transpose leaves the tensor non-contiguous
        # and view requires contiguous memory
        x = x.transpose(1, 2).contiguous().view(x.shape[0], -1, self.h * self.d_k)

        # Final Wo projection mixes head outputs back into a single d_model rep
        return self.w_o(x)