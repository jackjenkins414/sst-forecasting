# Based on: https://arxiv.org/pdf/2012.07436
# TODO: Add file head comment. 

import torch
import torch.nn as nn
import math

#TODO: RE-EVALUATE AND REPLACE
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
        super().__init__()
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
    
#TODO: RE-EVALUATE AND REPLACE 
class ProbSparseAttention(nn.Module):
    """ProbSparse Informer attention.

    Selects the top-u queries by sparsity measurement to compute attention, 
    reducing complexity from O(L^2) to O(L log L). 

    (B, L, d_model) -> (B, L, d_model)
    """
    def __init__(self, dropout, factor):
        """Build the ProbSparse attention module.

        Parameters
        ----------
        d_model : int
            Working dimension of the Transformer. Must be divisible by h.
        dropout : float
            Dropout probability applied to attention weights after softmax.
        factor : float
            Controls how many queries are selected. 
        """
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        self.factor = factor
    
    def forward(self, Q, K, V):
        """Compute ProbSparse attention.

        Parameters
        ----------
        Q, K, V : torch.Tensor
            Query, key, value tensors.
            All of shape (B, L, d_model).

        Returns
        ----------
        torch.Tensor
            Shape (B, L, d_model). 
            Attention output after selecting the top-u queries.
        """
        B, H, L, D = Q.shape

        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(D)

        # Importance calculation per the paper (#TODO: ADD DIRECT REFERENCE)
        # TODO: Review MaxMean vs LogSumExp to determine correct implementation. 
        # Currently using MaxMean based on https://github.com/zhouhaoyi/Informer2020/blob/main/models/attn.py
        # LogSumExp: importance = torch.logsumexp(scores, dim=-1) - scores.max(dim=-1).values
        importance = scores.max(dim=-1).values - scores.mean(dim=-1)

        # The number of queries to keep (the core ProbSparse mechanism).
        # Added safety to ensure that u is never 0. 
        u = min(max(1, int(self.factor * math.log(L))), L)
        # Indices of the top-u queries. 
        top_u_indices = importance.topk(u, dim=-1)[1]

        # The selected top-u queries.
        Q_top = torch.gather(
            Q,
            dim=2,
            index=top_u_indices.unsqueeze(-1).expand(-1, -1, -1, D)
        )

        # Full attention computation for the selected queries. 
        attn_scores = torch.matmul(Q_top, K.transpose(-2, -1)) / math.sqrt(D)
        attn_weights = torch.softmax(attn_scores, dim=-1)
        attn_weights = self.dropout(attn_weights)
        attn_out = torch.matmul(attn_weights, V)

        # Bug fix - create full length context tensor with original query positions. 
        context = torch.zeros_like(Q)
        context.scatter_(
            dim=2,
            index=top_u_indices.unsqueeze(-1).expand(-1, -1, -1, D),
            src=attn_out
        )

        return context
    
#TODO: RE-EVALUATE AND REPLACE
class SelfAttentionLayer(nn.Module):
    """Multi-head ProbSparse self-attention layer.

    Implements the ProbSparse attention mechanism for efficient self-attention. 

    # TODO: Confirm if this is L or u. 
    (B, L, d_model) -> (B, L, d_model)
    """
    def __init__(self, d_model, n_heads, dropout, factor):
        """Build the attention.

        Parameters
        ----------
        d_model : int
            Working dimension of the Informer. Must be divisible by n_heads.
        h_heads : int
            Number of attention heads. Each head sees d_model // h dims.
        dropout : float
            Dropout probability applied to attention weights after softmax.
        """
        super().__init__()
        self.h = n_heads
        self.d_k = d_model // n_heads

        self.q = nn.Linear(d_model, d_model)
        self.k = nn.Linear(d_model, d_model)
        self.v = nn.Linear(d_model, d_model)
        self.proj = nn.Linear(d_model, d_model)
        self.attn = ProbSparseAttention(dropout, factor)
    
    def forward(self, x):
        """Compute self-attention on input sequence.

        Parameters
        ----------
        x : torch.Tensor
            Input of shape (B, L, d_model)

        Returns
        -------
        torch.Tensor
            Output of shape (B, L, d_model)
        """
        B, L, _ = x.shape

        # Project and reshape. 
        Q = self.q(x).view(B, L, self.h, self.d_k).transpose(1, 2)
        K = self.k(x).view(B, L, self.h, self.d_k).transpose(1, 2)
        V = self.v(x).view(B, L, self.h, self.d_k).transpose(1, 2)

        # Apply ProbSparse attention. 
        out = self.attn(Q, K, V).transpose(1, 2).contiguous().view(B, L, -1)

        return self.proj(out)

#TODO: RE-EVALUATE AND REPLACE
class CrossAttentionLayer(nn.Module):
    """Multihead cross-attention layer.

    (B, L, d_model) -> (B, L, d_model)
    """
    def __init__(self, d_model, n_heads):
        """Build the multi-head attention block.

        Parameters
        ----------
        d_model : int
            Working dimension of the Informer. Must be divisible by n_heads.
        n_heads : int
            Number of attention heads. Each head sees d_model // n_heads dims.
        """
        super().__init__()
        self.h = n_heads
        self.d_k = d_model // n_heads

        self.q = nn.Linear(d_model, d_model)
        self.k = nn.Linear(d_model, d_model)
        self.v = nn.Linear(d_model, d_model)
        self.proj = nn.Linear(d_model, d_model)
    
    def forward(self, x, mem):
        """Run multi-head cross-attention.

        Parameters
        ----------
        x : torch.Tensor
            Shape: (B, L, d_model). Query sequence.
        mem : torch.Tensor
            Shape: (B, S, d_model). Encoder memory.

        Returns
        -------
        torch.Tensor
            Shape: (B, L, d_model).
        """
        B, L, _ = x.shape

        # Project and reshape.
        Q = self.q(x).view(B, L, self.h, self.d_k).transpose(1, 2)
        K = self.k(mem).view(B, mem.size(1), self.h, self.d_k).transpose(1, 2)
        V = self.v(mem).view(B, mem.size(1), self.h, self.d_k).transpose(1, 2)

        # Compute scaled dot product attention, apply softmax,
        # get weighted sum, recombine heads. 
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.d_k)
        weights = torch.softmax(scores, dim=-1)
        out = torch.matmul(weights, V).transpose(1, 2).contiguous().view(B, L, -1)

        return self.proj(out)
    
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
        return self.seq(x)
    
# Imported from Jack's Transformer model.
class ResidualConnection(nn.Module):
    """Residual connection wrapping a sublayer with pre-norm and dropout.

    (B, L, d_model) -> (B, L, d_model)
    """

    def __init__(self, d_model: int, dropout: float) -> None:
        """Build residual connection wrapper.

        Parameters
        ----------
        d_model : int
            Working dimension of the Informer. Passed through to the internal 
            LayerNorm so its scale and shift parameters are the right size.
        dropout : float
            Dropout probability applied to the sublayer's output before adding 
            back to the residual stream.
        """
        super().__init__()
        # Dropout applied to the sublayer's output; standard regularisation on 
        # the contribution this block adds to the residual stream
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
        # Pre-norm residual block: normalise x, pass through sublayer, 
        # apply dropout, then add back to the unchanged skip path for 
        # stable deep stack training.
        return x + self.dropout(sublayer(self.norm(x)))
    
#TODO: RE-EVALUATE AND REPLACE
class EncoderLayer(nn.Module):
    """A single encoder layer for the Informer.

    Combines ProbSparse self-attention and feed-forward with 
    residual connections and layer normalisation.

    (B, L, d_model) -> (B, L, d_model)
    """
    def __init__(self, attn, ff, d_model, dropout):
        """Initialises the encoder layer.

        Parameters
        ----------
        attn : nn.Module
            Self-attention module (ProbSparse or full attention).
        ff : nn.Module
            Feed-Forward block.
        d_model : int
            Working dimension of the Informer.
        dropout : float
            Dropout probability for residual connections.
        """
        super().__init__()
        self.attn = attn
        self.ff = ff

        self.res1 = ResidualConnection(d_model, dropout)
        self.res2 = ResidualConnection(d_model, dropout)

    def forward(self, x):
        """Completes a forward pass through the encoder layer. 

        Parameters
        ----------
        x : torch.Tensor
            Input from the previous layer (or from positional encoding for
            the first block) of shape (B, L, d_model).

        Returns
        -------
        torch.Tensor
            Shape: (B, L, d_model). 
        """
        x = self.res1(x, self.attn)
        x = self.res2(x, self.ff)
        return x
    
#TODO: RE-EVALUATE AND REPLACE
class EncoderDistillation(nn.Module):
    """Self attention distillation in line with the Informer paper. 
    Reduces sequence length by factor of 2.

    (B, L, d_model) -> (B, L//2, d_model)
    """
    def __init__(self, d_model):
        """Build distillation layer.

        Parameters
        ----------
        d_model : int
            Dimension of input/output embeddings (working dim of the Informer).
        """
        super().__init__()
        # 1D Convolution with MaxPool Stride Length 2 to halve the sequence length. 
        self.conv = nn.Conv1d(d_model, d_model, kernel_size=3, stride=1, padding=1)
        self.elu = nn.ELU()
        self.pool = nn.MaxPool1d(kernel_size=2, stride=2, padding=0)

    def forward(self, x):
        """Complete a distillation.

        Parameters
        ----------
        x : torch.Tensor
            Input from the previous layer of shape (B, L, d_model).
        """
        # Transpose L and d_model for Conv and MaxPool.
        x = x.transpose(1, 2)
        x = self.conv(x)
        x = self.elu(x)
        x = self.pool(x)
        # Restore transposition. 
        return x.transpose(1, 2)
    
#TODO: RE-EVALUATE AND REPLACE
class InformerEncoder(nn.Module):
    """Informer encoder stack with ProbSparse self-attention and distillation.

    A (#NOTE: ONCE COMPLETED) strict implementation of the paper's ProbSparse
    encoding methodology. 

    (B, L, d_model) -> (B, L, d_model)
    """
    # TODO: See whether to use ModuleList or just an int for no of layers. 
    def __init__(self, layers: list[nn.Module], d_model: int):
        """Construct the encoder stack.

        Parameters
        ----------
        # TODO: See whether to use ModuleList or just an int for no of layers. 
        layers : list[nn.Module]
            A list of encoder layers. Each layer is expected to implement
            ProbSparse self-attention and feed-forward with residuals.
        d_model : int
            Working dimension of the encoder. 
        """
        super().__init__()
        # TODO: See whether to use ModuleList or just an int for no of layers. 
        self.layers = nn.ModuleList(layers)
        self.norm = LayerNormalisation(d_model)
        self.distill = EncoderDistillation(d_model)

    def forward(self, x):
        """Forward pass through the encoder stack.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor of shape (B, L, d_model).

        Returns
        -------
        torch.Tensor
            Encoder output of shape (B, L_new, d_model), with L_new <= L 
            if distillation reduces sequence length.
        """
        for i, layer in enumerate(self.layers):
            x = layer(x)
            # Apply distillation after every two layers.
            if ((i % 2) == 1):
                x = self.distill(x)
        return self.norm(x)
    
#TODO: RE-EVALUATE AND REPLACE
class DecoderLayer(nn.Module):
    def __init__(self, self_attn, cross_attn, ff, d_model, dropout):
        super().__init__()
        self.self_attn = self_attn
        self.cross_attn = cross_attn
        self.ff = ff

        self.res1 = ResidualConnection(d_model, dropout)
        self.res2 = ResidualConnection(d_model, dropout)
        self.res3 = ResidualConnection(d_model, dropout)

    def forward(self, x, memory):
        # Self-attention over the decoder input. 
        x = self.res1(x, self.self_attn)

        # Cross-attention to the encoder output.
        x = self.res2(x, lambda x: self.cross_attn(x, memory))

        # Feed-Forward block
        x = self.res3(x, self.ff)
        return x
    
#TODO: RE-EVALUATE AND REPLACE
class InformerDecoder(nn.Module):
    # TODO: See whether to use ModuleList or just an int for no of layers.
    def __init__(self, layers: list[nn.Module], d_model: int):
        """Construct the decoder stack.

        Parameters
        ----------
        # TODO: See whether to use ModuleList or just an int for no of layers. 
        layers : list[nn.Module]
            A list of decoder layers. Each layer does self-attn -> cross-attn
            -> feed-forward with residuals. 
        d_model : int
            Working dimension of the decoder. 
        """
        super().__init__()
        # TODO: See whether to use ModuleList or just an int for no of layers.
        self.layers = nn.ModuleList(layers)
        self.norm = LayerNormalisation(d_model)

    def forward(self, x, memory):
        """Pass input through decoder stack.

        Parameters
        ----------
        x : torch.Tensor
            Decoder input tensor of shape (B, L_dec, d_model).
        memory : torch.Tensor
            Encoder output tensor of shape (B, L_enc, d_model). Used for
            cross-attention to inject encoder information into the decoder.

        Returns
        -------
        torch.Tensor
            Shape: (B, L_dec, d_model). Each position contains the
            decoder's final representation (ready for projection).
        """
        # TODO: See whether to use ModuleList or just an int for no of layers.
        for layer in self.layers:
            x = layer(x, memory)
        return self.norm(x)

#TODO: RE-EVALUATE AND REPLACE
class OutputProjectionHead(nn.Module):
    """Linearly project d_model to reshape to H*W SST grid. 

    (B, L, d_model) -> (B, L, H, W)
    """
    def __init__(self, d_model, H, W):
        super().__init__()
        self.H = H
        self.W = W
        self.proj = nn.Linear(d_model, H*W)

    def forward(self, x):
        x = self.proj(x)
        return x.view(x.size(0), x.size(1), self.H, self.W)
  
#TODO: RE-EVALUATE, UPDATE, AND VERIFY
class ProbSparseInformer(nn.Module):
    """ProbSparse Attention Informer for SST forecasting. 

    (B, L, 1, H, W) -> (B, h, H, W)
    """
    def __init__(
        self,
        height: int,
        width: int,
        context_len: int = 90,
        horizon: int = 7,
        d_model: int = 128,
        n_heads: int = 4,
        n_encoder_layers: int = 3,
        n_decoder_layers: int = 2,
        d_ff: int = 512,
        dropout: float = 0.1,
        factor: int = 5
    ):
        """Build the full ProbSparse Informer from its component classes.

        Parameters
        ----------
        height : int
            Spatial height of the SST grid (81 for Coral Sea).
        width : int
            Spatial width of the SST grid (121 for Coral Sea).
        context_len : int
            Length of the input history window in days. Sets the size of the 
            positional encoding table.
        horizon : int
            Number of forecast days to predict.
        d_model : int
            Working dimension of the Transformer. Used everywhere.
        n_heads : int
            Number of attention heads per MultiHeadAttentionBlock. Must divide d_model.
        n_encoder_layers : int
            Number of encoder layers. 
        n_decoder_layers : int
            Number of decoder layers.
        d_ff : int
            Hidden dim inside each FeedForwardBlock
        dropout : float
            Dropout probability used in positional encoding, attention, FFN, 
            and residual connections.
        factor : float
            Factor applied to control ratio of selected queries. 
        """
        super().__init__()
        self.h = height
        self.w = width
        self.context_len = context_len
        self.horizon = horizon
        self.d_model = d_model

        # Spatial encoding. 
        self.spatial_enc = CNNSpatialEncoding(d_model)

        # Positional encoding. 
        self.pos_enc = PositionalEncoding(d_model, context_len, dropout)

        # Full encoder stack with ProbSparse self-attention. 
        encoder_layers = [
            EncoderLayer(
                SelfAttentionLayer(d_model, n_heads, dropout, factor),
                FeedForwardBlock(d_model, d_ff, dropout),
                d_model, dropout
            ) for _ in range(n_encoder_layers)
        ]
        self.encoder = InformerEncoder(encoder_layers, d_model)

        # Full decoder stack with self-attention and cross-attention. 
        decoder_layers = [
            DecoderLayer(
                SelfAttentionLayer(d_model, n_heads, dropout, factor),
                CrossAttentionLayer(d_model, n_heads),
                FeedForwardBlock(d_model, d_ff, dropout),
                d_model, dropout
            ) for _ in range(n_decoder_layers)
        ]
        self.decoder = InformerDecoder(decoder_layers, d_model)

        # Creates a learnable tensor of zeros. 
        self.dec_input_init = nn.Parameter(torch.zeros(1, horizon, d_model))

        # Projection from decoder output to the SST grid.
        self.proj_head = OutputProjectionHead(d_model, height, width)
    
    def forward(self, x):
        """Run a batch of SST history windows through the Informer.

        Parameters
        ----------
        x : torch.Tensor
            Input SST windows, shape (B, L, 1, H, W). Matches the contract
            used by SstWindowDataset and the rest of the pipeline.

        Returns
        -------
        torch.Tensor
            Forecast grids of shape (B, horizon, H, W).
        """
        B = x.shape[0]

        # Run the encoding. 
        x = self.spatial_enc(x)
        x = self.pos_enc(x)
        encoder_out = self.encoder(x)

        # Initialise the decoder input. 
        decoder_in = self.dec_input_init.expand(B, -1, -1)

        # Run the decoding. 
        decoder_out = self.decoder(decoder_in, encoder_out)

        # Project to the grid. 
        pred = self.proj_head(decoder_out)

        # Return the predictions. 
        return pred