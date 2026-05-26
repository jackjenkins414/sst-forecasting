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
    
# Reference: https://github.com/hkproj/pytorch-transformer/blob/main/model.py
# Domain agnostic again, no changes needed here. Each sublayer in the encoder (attention and FFN) 
# is wrapped in a residual connection. This version is different from the paper since it uses a 
# the pre-norm variant (LayerNorm applied before the sublayer, not after as in the original paper),
# which is more stable. 
class ResidualConnection(nn.Module):
    """Residual connection wrapping a sublayer with pre-norm and dropout.

    (B, L, d_model) -> (B, L, d_model)
    """

    def __init__(self, d_model: int, dropout: float) -> None:
        """Build residual connection wrapper.

        Parameters
        ----------
        d_model : int
            Working dimension of the Transformer. Passed through to the internal LayerNorm 
            so its scale and shift parameters are the right size.
        dropout : float
            Dropout probability applied to the sublayer's output before adding back to 
            the residual stream.
        """
        super().__init__()
        # Dropout applied to the sublayer's output; standard regularisation on the 
        # contribution this block adds to the residual stream
        self.dropout = nn.Dropout(dropout)

        # LayerNorm applied to the input before the sublayer runs (pre-norm)
        self.norm = LayerNormalisation(d_model)

    def forward(self, x, sublayer):
        """Run x through the sublayer with a residual skip and pre-norm.

        Parameters
        ----------
        x : torch.Tensor
            Input from the previous layer, shape (B, L, d_model). This is
            the residual stream that the sublayer's output is added to.
        sublayer : callable
            The sublayer to wrap, e.g. attention or FFN. Called as
            sublayer(normalised_x) and expected to return shape (B, L, d_model).

        Returns
        -------
        torch.Tensor
            Same shape (B, L, d_model). The original input plus the
            dropout-regularised output of the sublayer applied to its
            normalised version.
        """
        # Pre-norm residual block: normalise x, pass through sublayer, apply dropout, 
        # then add back to the unchanged skip path for stable deep stack training.
        return x + self.dropout(sublayer(self.norm(x)))
    
# Reference: https://github.com/hkproj/pytorch-transformer/blob/main/model.py
# One repeating unit of the encoder, wires together one self-attention sublayer and one FFN 
# sublayer, each wrapped in its own ResidualConnection. The class takes the already constructed 
# attention and FFN blocks as arguments (rather than their hyperparameters), which makes the block 
# composable and lets us swap in different attention or FFN variants later without touching this 
# class. Domain agnostic, no SST specific changes are needed here. 
class EncoderBlock(nn.Module):
    """One encoder block: self attention sublayer + FFN sublayer, each wrapped 
    in a ResidualConnection.

    (B, L, d_model) -> (B, L, d_model)
    """

    def __init__(
        self,
        d_model: int,
        self_attention_block: MultiHeadAttentionBlock,
        feed_forward_block: FeedForwardBlock,
        dropout: float,
    ) -> None:
        """Build one encoder block from pre-constructed sublayers.

        Parameters
        ----------
        d_model : int
            Working dimension of the Transformer. Passed through to the
            internal ResidualConnections so their LayerNorms are sized
            correctly.
        self_attention_block : MultiHeadAttentionBlock
            The attention sublayer. Already constructed externally so
            this class doesn't need to know attention hyperparameters.
        feed_forward_block : FeedForwardBlock
            The FFN sublayer. Same composability reasoning as above.
        dropout : float
            Dropout probability for both ResidualConnections.
        """
        super().__init__()
        # Store the two sublayers; each gets called inside its own residual wrapper
        self.self_attention_block = self_attention_block
        self.feed_forward_block = feed_forward_block

        # Two residuals, one per sublayer; ModuleList registers both as submodules
        # so their parameters show up in .parameters() and they move with .to(device)
        self.residual_connections = nn.ModuleList(
            [ResidualConnection(d_model, dropout) for _ in range(2)]
        )

    def forward(self, x, mask):
        """Run x through self attention then FFN, each with a residual wrapper.

        Parameters
        ----------
        x : torch.Tensor
            Input from the previous layer (or from positional encoding for
            the first block), shape (B, L, d_model).
        mask : torch.Tensor or None
            Optional attention mask. None for our encoder-only forecaster
            since every timestep should attend to every other timestep.

        Returns
        -------
        torch.Tensor
            Same shape (B, L, d_model). Representation refined by one
            round of self attention and one FFN, both residually wrapped.
        """
        # First sublayer: self attention, wrapped in residual
        # The lambda adapts attention's (q, k, v, mask) signature to the 
        # single-argument callable ResidualConnection expects
        # For self attention, q = k = v = the same tensor
        x = self.residual_connections[0](x, lambda x: self.self_attention_block(x, x, x, mask))

        # Second sublayer: FFN, wrapped in residual
        # No lambda needed since FFN naturally takes just x
        x = self.residual_connections[1](x, self.feed_forward_block)

        return x


# Reference: https://github.com/hkproj/pytorch-transformer/blob/main/model.py
# Stack of N EncoderBlocks plus a final LayerNorm. The final norm is needed because we use 
# pre-norm, with pre-norm the residual stream itself never gets normalised after the last 
# block, so we apply one final LayerNorm to clean up that output before whatever comes next 
# (so the output head here). Again, mostly domain agnostic code.
class Encoder(nn.Module):
    """Stack of N EncoderBlocks with a final LayerNorm.

    (B, L, d_model) -> (B, L, d_model)
    """

    def __init__(self, d_model: int, layers: nn.ModuleList) -> None:
        """Build the encoder from a list of pre-constructed blocks.

        Parameters
        ----------
        d_model : int
            Working dimension of the Transformer. Used to size the final LayerNorm.
        layers : nn.ModuleList
            The encoder blocks, constructed externally. 
        """
        super().__init__()
        self.layers = layers

        # Final LayerNorm; only needed because we use pre-norm residuals 
        # In pre-norm the residual stream isn't normalised after the last 
        # sublayer, so this cleans up the output before it reaches the head
        self.norm = LayerNormalisation(d_model)

    def forward(self, x, mask):
        """Run x through every block in sequence then apply final norm.

        Parameters
        ----------
        x : torch.Tensor
            Input from positional encoding, shape (B, L, d_model).
        mask : torch.Tensor or None
            Optional attention mask, passed unchanged to every block.

        Returns
        -------
        torch.Tensor
            Same shape (B, L, d_model). Fully encoded representation,
            ready for the output head.
        """
        # Apply each encoder block in sequence; each block sees the output of the previous
        # one and refines it further
        for layer in self.layers:
            x = layer(x, mask)

        # Normalise
        return self.norm(x)
    
# Last token decoding: take only the final timestep's encoder output and project directly 
# to all 7 forecast horizons at once, mirroring the LSTM's decoding strategy so the temporal 
# mechanism is the only architectural variable between the two models.
# Note: Obviously this somewhat defeats the point of attention entirely since this throws away 
# 89/90 of the encoder's input, but we include it as motivation for this model's predecessor; 
# a temporal fusion transformer, and add per-horizon learned queries that cross-attend to the 
# encoded history. 
class OutputHead(nn.Module):
    """Decode the last encoder timestep into a 7 day forecast grid.

    (B, L, d_model) -> (B, h, H, W)
    """

    def __init__(self, d_model: int, horizon: int, height: int, width: int):
        """Build the output head.

        Parameters
        ----------
        d_model : int
            Working dimension of the Transformer. Input dim of the head.
        horizon : int
            Number of forecast days to predict (7 for our problem).
        height : int
            Spatial height of the output grid (81 for Coral Sea).
        width : int
            Spatial width of the output grid (121 for Coral Sea).
        """
        super().__init__()
        # Store dims; needed in forward() for the reshape
        self.d_model = d_model
        self.horizon = horizon
        self.height = height
        self.width = width

        # Single learned linear: d_model -> horizon * H * W
        # Symmetric with SpatialProjection at the input boundary, just in reverse
        self.projection = nn.Linear(d_model, horizon * height * width)

    def forward(self, x):
        """Decode the encoder's final timestep into a multi horizon forecast.

        Parameters
        ----------
        x : torch.Tensor
            Encoder output, shape (B, L, d_model). All L timesteps are present but we 
            only use the last one. We could try and flatten-all decode but this would lead 
            to an impractical parameter count. For example: 
                L=90, d_model=128: 90 * 128 * 7 * 9801 = 790M...

        Returns
        -------
        torch.Tensor
            Forecast grids, shape (B, horizon, H, W). No activation applied;
            outputs are raw z scored anomalies which can be positive or
            negative.
        """
        # Take the final timestep's d_model vector; via self attention this summarises the 
        # whole 90 day history from day 90's perspective
        x = x[:, -1, :]

        # Project to the flattened multi-horizon forecast
        x = self.projection(x)

        # Reshape to the (B, horizon, H, W) we use 
        return x.view(-1, self.horizon, self.height, self.width)
    
class SstFlatTransformer(nn.Module):
    """Encoder only Transformer for SST forecasting.

    (B, L, 1, H, W) -> (B, h, H, W)
    """

    def __init__(
        self,
        height: int,
        width: int,
        seq_len: int = 90,
        horizon: int = 7,
        d_model: int = 128,
        n_blocks: int = 2,
        n_heads: int = 4,
        d_ff: int = 512,
        dropout: float = 0.1,
    ):
        """Build the full Transformer from its component classes.

        Parameters
        ----------
        height : int
            Spatial height of the SST grid (81 for Coral Sea).
        width : int
            Spatial width of the SST grid (121 for Coral Sea).
        seq_len : int
            Length of the input history window in days. Sets the size of the positional 
            encoding table.
        horizon : int
            Number of forecast days to predict.
        d_model : int
            Working dimension of the Transformer. Used everywhere.
        n_blocks : int
            Number of stacked EncoderBlocks in the Encoder.
        n_heads : int
            Number of attention heads per MultiHeadAttentionBlock. Must divide d_model.
        d_ff : int
            Hidden dim inside each FeedForwardBlock
        dropout : float
            Dropout probability used in positional encoding, attention, FFN, and residual
            connections.
        """
        super().__init__()

        # Input boundary: flattened grid (H*W) projected to d_model
        self.input_projection = SpatialProjection(d_model, height, width)

        # Positional encoding added after the input projection; tells the model where each 
        # timestep sits in the 90 day window
        self.positional_encoding = PositionalEncoding(d_model, seq_len, dropout)

        # Build n_blocks identical EncoderBlocks, each with its own freshly constructed 
        # MultiHeadAttentionBlock and FeedForwardBlock so weights aren't shared 
        encoder_blocks = nn.ModuleList([
            EncoderBlock(
                d_model,
                MultiHeadAttentionBlock(d_model, n_heads, dropout),
                FeedForwardBlock(d_model, d_ff, dropout),
                dropout,
            )
            for _ in range(n_blocks)
        ])
        self.encoder = Encoder(d_model, encoder_blocks)

        # Output boundary; last timestep's d_model vector -> 7 day forecast grid
        self.output_head = OutputHead(d_model, horizon, height, width)

    def forward(self, x):
        """Run a batch of SST history windows through the full Transformer.

        Parameters
        ----------
        x : torch.Tensor
            Input SST windows, shape (B, L, 1, H, W). Matches the contract
            used by SstWindowDataset and the rest of the pipeline.

        Returns
        -------
        torch.Tensor
            Forecast grids, shape (B, horizon, H, W).
        """
        # Flatten grid and project to d_model space
        # (B, L, 1, H, W) -> (B, L, d_model)
        x = self.input_projection(x)

        # (B, L, d_model) -> (B, L, d_model)
        x = self.positional_encoding(x)

        # Stack of self attention + FFN blocks
        # (B, L, d_model) -> (B, L, d_model)
        x = self.encoder(x, mask=None)

        # Decode the last timestep into a multi horizon forecast
        # (B, L, d_model) -> (B, horizon, H, W)
        x = self.output_head(x)

        return x