# Based on: https://arxiv.org/pdf/2012.07436
# TODO: Add file head comment. 

import torch
import torch.nn as nn
import math


# Input embedding, imported from Jack's spatial-flat transformer model.  
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


# Handles calendar time to allow the model to learn seasonal dynamics. 
class TemporalEmbedding(nn.Module):
    """Maps day-of-year indices to d_model vectors, allowing the model to learn 
    annual sea surface temperature periodicity and seasonal structuring.
    (B, L) -> (B, L, d_model)
    """
    def __init__(self, d_model: int):
        """Build the temporal embedding lookup.

        Parameter
        ----------
        d_model : int
            Working dimension of the Informer.
        """ 
        super().__init__()
        # Creates a learnable look up table with a vector corresponding to each 
        # day. Each day has its own trainable vector. 
        self.day = nn.Embedding(366, d_model)

    def forward(self, date):
        """Convert day-of-year indices into learnable embeddings.

        Parameter
        ----------
        date : torch.Tensor
            Day-of-year indices of shape (B, L). Each entry should be an 
            integer in the range [0, 365].

        Returns
        -------
        torch.Tensor
            Learnable temporal embeddings of shape (B, L, d_model). 
            Each timestep carries explicit seasonal information for the Informer.
        """
        # Note: date expressed as the day of the year between 0 and 365.
        # Returns a learnable seasonal representation.  
        return self.day(date)


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
    

# Implements the paper's unified input representation from appendix item B.  
class DataEmbedding(nn.Module):
    """Combines spatial SST embeddings, positional encodings, and
    learnable temporal embeddings into the final input representation.

    Implements Xt_feed = alpha u + PE + SE

    (B, L, 1, H, W) + (B, L) -> (B, L, d_model)
    """
    def __init__(self, d_model: int, height: int, width: int, dropout: float, seq_len: int):
        """Build the Informer embedding pipeline.

        Parameters
        ----------
        d_model : int
            Working dimension of the Informer.
        height : int
            Spatial height of the SST grid.
        width : int
            Spatial width of the SST grid.
        dropout : float
            Dropout probability applied after combining all embeddings.
        seq_len : int
            Maximum sequence length the model will ever see through 
            positional encoding. 
        """
        self.val = SpatialProjection(d_model, height, width)
        self.pos = PositionalEncoding(d_model, seq_len, dropout)
        self.temporal = TemporalEmbedding(d_model)

        self.dropout = nn.Dropout(dropout)

    def forward(self, x, date):
        """Implement the full Informer input embedding.

        Parameters
        ----------
        x : torch.Tensor
            SST input tensor of shape (B, L, 1, H, W).
        date : torch.Tensor
            Day-of-year indices of shape (B, L).

        Returns
        -------
        torch.Tensor
            Combined embeddings of shape (B, L, d_model).
        """
        val = self.val(x)
        pos = self.pos(val)
        temporal = self.temporal(date)

        x = val + pos + temporal
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
    

# Implements the paper's recommended ProbSparse Attention mechanism. 
class ProbSparseAttention(nn.Module):
    """ProbSparse Informer attention, as seen in the paper.

    Selects the top-u queries by sparsity measurement to compute attention, 
    reducing complexity from O(L^2) to O(L log L). 

    (B, L, d_model) -> (B, L, d_model) using (B, num_heads, L, D) internally,
    where D = d_model // num_heads
    """
    def __init__(self, masked: bool, dropout: float, factor: int):
        """Build the ProbSparse attention module.

        Parameters
        ----------
        masked : bool
            Controls whether or not a causal mask is applied. 
            # NOTE: This is likely true for decoder, false for encoder?
        d_model : int
            Working dimension of the Transformer. Must be divisible by h.
        dropout : float
            Dropout probability applied to attention weights after softmax.
        factor : float
            Controls how many queries are selected. 
        """
        super().__init__()
        self.masked = masked
        self.dropout = nn.Dropout(dropout)
        self.factor = factor

    def prob_QK(self, Q, K, sample_k, top_n):
        """Samples query-key pairs and computes sparsity metrics for said pairs.

        Parameters
        ----------
        Q : torch.Tensor
            Query tensor of shape (B, H, L_Q, D).

        K : torch.Tensor
            Key tensor of shape (B, H, L_K, D).

        sample_k : int
            The number of keys randomly sampled per query. 

        top_n : int
            The number of dominant queries selected based on the sparsity metrics.

        Returns
        -------
        Q_K :
            Full attention scores for only the selected Top-u queries.
            (B, H, top_n, L_K)

        M_top :
            Indices of the selected Top-u queries. 
            (B, H, top_n)
        """
        # Extract dimensions from K and Q.
        B, H, L_K, D = K.shape
        _, _, L_Q, _ = Q.shape

        # Randomly sample indices for queries, with each query having sample_k keys. 
        i_sample = torch.randint(L_K, (L_Q, sample_k), device=Q.device)

        # Expand keys such that every query can access every sampled key. 
        K_expand = K.unsqueeze(-3).expand(B, H, L_Q, L_K, D)

        # Gather the random key sample. 
        K_sample = K_expand[
            :, :, torch.arange(L_Q, device=Q.device).unsqueeze(1), 
            i_sample, :
            ]
        
        # Compute the sampled query-key dot products for each pair. 
        Q_K_sample = torch.matmul(Q.unsqueeze(-2), K_sample.transpose(-2, -1)).squeeze(-2)

        # Implements the paper's sparsity measurement. 
        # It is an empirical max-mean approximation of Kullback-Leibler divergence. 
        M = Q_K_sample.max(dim=-1).values - Q_K_sample.mean(dim=-1)

        # Based on this, select the top-u queries. 
        M_top = M.topk(top_n, sorted=False).indices

        # Gather the most dominant queries based on M_top for full
        # query-key attention score production. 
        Q_reduced = Q[
            torch.arange(B)[:, None, None], 
            torch.arange(H)[None, :, None], M_top, :
            ]
        Q_K = torch.matmul(Q_reduced, K.transpose(-2, -1))

        return Q_K, M_top
    
    def initialise_queries(self, V, L_Q):
        """Initialise context for the non-selected queries.

        Parameters
        ----------
        V : torch.Tensor
            Value tensor of shape (B, H, L_V, D).

        L_Q : int
            Query sequence length.

        Returns
        -------
        torch.Tensor
            Initial context tensor of shape (B, H, L_Q, D).
        """
        # Extract dimensions from V.
        B, H, _, D = V.shape

        # Self-attention based on the causal mask. 
        # NOTE: Confirm that true is for decoder, false is for encoder. 
        if self.masked:
            # DECODER
            # Cumulative representation from timesteps <= t. 
            context = V.cumsum(dim=-2)
        else: 
            # ENCODER
            # Take the mean of the value vectors across the time dimension. 
            V_mean = V.mean(dim=-2)

            # Broadcast the mean across all query positions.
            context = V_mean.unsqueeze(-2).expand(B, H, L_Q, D).clone()

        return context
    
    def update_context(self, context, V, scores, index):
        """Update only the top-u queries with full attention.

        Parameters
        ----------
        context : torch.Tensor
            The initial context tensor.

        V : torch.Tensor
            The value tensor.

        scores : torch.Tensor
            Full attention scores for the selected queries.

        index : torch.Tensor
            Indices of the top-u selected queries.

        Returns
        -------
        torch.Tensor
            The updated ProbSparse attention context tensor.
        """
        # Extract dimensions from V.
        B, H, _, D = V.shape

        # Update based on the causal mask. 
        # NOTE: Confirm that true is for decoder, false is for encoder. 
        if self.masked:
            # Decoder Mask
            # An upper triangular mask removes access to all future positions. 
            mask = torch.triu(torch.ones(scores.shape[-2], 
                                         scores.shape[-1], 
                                         device=scores.device), diagonal=1
                            ).bool()
            
            # Replace masked positions with -inf before softmax.
            scores = scores.masked_fill(mask, float("-inf"))

        # Scaled dot-product attention scores. 
        attn = torch.softmax((scores / math.sqrt(D)), dim=-1)

        # Regularise.
        attn = self.dropout(attn)

        # Get the weighted sum of values. 
        context_update = torch.matmul(attn, V)

        # Update the top-u query position attention outputs. 
        context[
            torch.arange(B)[:, None, None], 
            torch.arange(H)[None, :, None], 
            index, :
            ] = context_update

        return context
    
    
    def forward(self, Q, K, V):
        """Compute ProbSparse attention.

        Parameters
        ----------
        Q, K, V : torch.Tensor
            Query, key, value tensors.
            All of shape (B, L_x, d_model).

        Returns
        ----------
        torch.Tensor
            Shape (B, L_Q, d_model). 
            Attention output after selecting the top-u queries.
        """
        # Extract dimensions. 
        _, _, L_Q, _ = Q.shape
        _, _, L_K, _ = K.shape

        # Sampled keys and dominant queries. 
        # Determined through the paper's formula _ = c * ln(L_x)
        sample_k = min(L_K, (int(self.factor * math.ceil(math.log(L_K + 1)))))
        top_n = min(L_Q, (int(self.factor * math.ceil(math.log(L_Q + 1)))))

        # Compute the ProbSparse query-key scores for the top-u queries.
        scores_top, index = self.prob_QK(Q, K, sample_k, top_n)

        # Initialise the context tensor for all queries.
        context = self.initialise_queries(V, L_Q)

        # Update the dominant queries using full attention. 
        context = self.update_context(context, V, scores_top, index)

        return context.contiguous()
    

# Full attention for cross attention.  
class FullAttention(nn.Module):
    """Computes full scaled dot-product attention across every query-key pair. 
    Used for encoder-decoder cross-attention. 

    (B, L, d_model) -> (B, L, d_model)
    """
    def __init__(self, masked: bool, dropout: float):
        """Build the full attention module.

        Parameters
        ----------
        masked : bool
            Controls whether or not a causal mask is applied. 
        dropout : float
            Dropout probability applied to attention weights after softmax.
        """
        super().__init__()
        self.masked = masked
        self.dropout = nn.Dropout(dropout)

    def forward(self, Q, K, V):
        """Compute full scaled dot-product attention.

        Parameters
        ----------
        Q, K, V : torch.Tensor
            Query, key, value tensors.
            All of shape (B, L_x, d_model).

        Returns
        ----------
        torch.Tensor
            Shape (B, L_Q, d_model). 
            Attention output after selecting the top-u queries.
        """
        # Per-head feature dimension.
        D = Q.shape[-1]

        # Scaled dot-product scores according to the transformer paper upon
        # which the informer is based. 
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(D)

        # Apply the causal mask if required. 
        if self.masked:
            # An upper triangular mask removes access to all future positions. 
            mask = torch.triu(torch.ones(scores.shape[-2], 
                                         scores.shape[-1], 
                                         device=scores.device), diagonal=1
                            ).bool()
            
            # Replace masked positions with -inf before softmax.
            # NOTE: Consider using -1e9 instead of -inf for stability. 
            scores = scores.masked_fill(mask, float("-inf"))

        # Scaled dot-product attention scores. 
        attn = torch.softmax(scores, dim=-1)

        # Regularise.
        attn = self.dropout(attn)

        # Return the weighted sum of values. 
        return torch.matmul(attn, V)

    
# Computes the self attention for a given attention mechanism. 
class SelfAttentionLayer(nn.Module):
    """Multi-head ProbSparse self-attention layer.

    Implements the ProbSparse attention mechanism for efficient self-attention. 

    # TODO: Confirm if this is L or u. 
    (B, L, d_model) -> (B, L, d_model)
    """
    def __init__(self, attention, d_model, n_heads):
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
        self.attention = attention
    
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

        # Apply attention mechanism. 
        out = self.attn(Q, K, V).transpose(1, 2).contiguous().view(B, L, -1)

        return self.proj(out)


# Encoder-Decoder Cross Attention
class CrossAttentionLayer(nn.Module):
    """Multihead encoder-decoder cross-attention layer.

    (B, L_enc, d_model) x (B, L_dec, d_model) -> (B, L, d_model)
    """
    def __init__(self, d_model, n_heads, dropout):
        """Build the multi-head attention block.

        Parameters
        ----------
        d_model : int
            Working dimension of the Informer. Must be divisible by n_heads.
        n_heads : int
            Number of attention heads. Each head sees d_model // n_heads dims.
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

        # Full attention for cross attention, as per the paper.
        self.attention = FullAttention(False, dropout)
    
    def forward(self, x, mem):
        """Run multi-head cross-attention.

        Parameters
        ----------
        x : torch.Tensor
            Shape: (B, L_dec, d_model). Decoder input tensor.
        mem : torch.Tensor
            Shape: (B, L_enc, d_model). Encoder memory tensor.

        Returns
        -------
        torch.Tensor
            Shape: (B, L_dec, d_model). Cross-attention decoder representation. 
        """
        B, L_Q, _ = x.shape
        _, L_K, _ = mem.shape

        # Project decoder states to queries. 
        Q = self.q(x).view(B, L_Q, self.h, self.d_k).transpose(1, 2)
        
        # Project encoder memory to keys. 
        K = self.k(mem).view(B, L_K, self.h, self.d_k).transpose(1, 2)

        # Project encoder memory to values.
        V = self.v(mem).view(B, L_K, self.h, self.d_k).transpose(1, 2)

        # Apply full attention. 
        attn = self.attention(Q, K, V)

        # Recombine all attention heads.
        out = out.transpose(1, 2).contiguous().view(B, L_Q, -1)

        # Return final learned projection. 
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
    
# A single encoder block. 
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
    
# Runs an encoder distillation per the paper. 
class EncoderDistillation(nn.Module):
    """Self attention distillation in line with the Informer paper. 
    Reduces sequence length by factor of 2 according to MaxPool(ELU(Conv1d(x))). 

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
        self.conv = nn.Conv1d(d_model, d_model, kernel_size=3, padding=1)
        self.elu = nn.ELU()
        self.pool = nn.MaxPool1d(kernel_size=2, stride=2)

    def forward(self, x):
        """Complete a distillation.

        Parameters
        ----------
        x : torch.Tensor
            Input from the previous layer of shape (B, L, d_model).

        Returns
        -------
        torch.Tensor
            Distilled encoder sequence of shape (B, L/2, d_model). 
        """
        # Transpose L and d_model for Conv and MaxPool.
        x = x.transpose(1, 2)
        x = self.conv(x)
        x = self.elu(x)
        x = self.pool(x)
        # Restore transposition. 
        return x.transpose(1, 2)
    
# A full encoder informer stack. 
class InformerEncoder(nn.Module):
    """Informer encoder stack with ProbSparse self-attention and distillation. 

    (B, L, d_model) -> (B, L_new, d_model), where L_new is the new length 
    after distillation. 
    """
    # TODO: See whether to use ModuleList or just an int for no of layers. 
    def __init__(self, layers: list[nn.Module], distill_layers: list[nn.Module], d_model: int):
        """Construct the encoder stack.

        Parameters
        ----------
        # TODO: See whether to use ModuleList or just an int for no of layers. 
        layers : list[nn.Module]
            A list of encoder blocks. Each layer is expected to implement
            ProbSparse self-attention and feed-forward with residuals.
        distill_layers : list[nn.Module]
            Distillation layers inserted between encoder blocks.
        d_model : int
            Working dimension of the encoder. 
        """
        super().__init__()
        # TODO: See whether to use ModuleList or just an int for no of layers. 
        self.layers = nn.ModuleList(layers)
        self.distill_layers = nn.ModuleList(distill_layers)
        self.norm = LayerNormalisation(d_model)

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
            # Run ProbSparse Attention and FF. 
            x = layer(x)
            # Distill sequence after each layer except final layer.
            if i < len(self.distill_layers):
                x = self.distill_layers[i](x)
        
        # Final normalisation for the encoder. 
        return self.norm(x)
    
# A single decoder block. 
class DecoderLayer(nn.Module):
    """A single decoder layer for the Informer.

    Combines ProbSparse self-attention, cross attention, and feed-forward. 

    (B, L, d_model) -> (B, L, d_model)
    """
    def __init__(self, self_attn, cross_attn, ff, d_model, dropout):
        """Initialises the decoder layer.

        Parameters
        ----------
        self_attn : nn.Module
            ProbSparse self-attention module.
        cross_attn: nn.Module
            Encoder-decoder attention. 
        ff : nn.Module
            Feed-Forward block.
        d_model : int
            Working dimension of the Informer.
        dropout : float
            Dropout probability for residual connections.
        """
        super().__init__()
        self.self_attn = self_attn
        self.cross_attn = cross_attn
        self.ff = ff

        self.res1 = ResidualConnection(d_model, dropout)
        self.res2 = ResidualConnection(d_model, dropout)
        self.res3 = ResidualConnection(d_model, dropout)

    def forward(self, x, mem):
        """Completes a forward pass through the decoder layer. 

        Parameters
        ----------
        x : torch.Tensor
            Hidden decoder states. 
        mem : torch.Tensor
            Encoder output representations. 

        Returns
        -------
        torch.Tensor
            Shape: (B, L, d_model). 
        """
        # Self-attention over the decoder input. 
        x = self.res1(x, self.self_attn)

        # Cross-attention to the encoder output.
        x = self.res2(x, lambda x: self.cross_attn(x, mem))

        # Feed-Forward block
        x = self.res3(x, self.ff)
        return x
    
# A full decoder stack. 
class InformerDecoder(nn.Module):
    """Informer decoder stack. 

    (B, L, d_model) -> (B, L, d_model)
    """
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

# Project decoder vectors back to the SST grid. 
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
        B, L, _ = x.shape

        # Project to flattened spatial grids. 
        x = self.proj(x)

        # Reshape to the 2D SST map. 
        return x.view(B, L, self.H, self.W)
  
# The final combined ProbSparse Informer implementation. 
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
        factor: int = 5,
        label_len: int = None
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
        label_len : int
            Number of decoder start tokens. 
            If provided, it will be concat(start_tokens, zeros). 
            Else, it will be (context_len // 2).
        """
        super().__init__()
        self.h = height
        self.w = width
        self.context_len = context_len
        self.horizon = horizon
        self.d_model = d_model

        # Decoder start token length. 
        # Informer decoder receives historical tokens plus zero placeholders 
        # (if required). 
        if label_len is None:
            label_len = (context_len // 2)
        self.label_len = label_len

        # Encoder Embedding. 
        self.enc_embedding = DataEmbedding(d_model, height, width, dropout)

        # Full encoder stack with ProbSparse self-attention. 
        encoder_layers = []
        for _ in range(n_encoder_layers):
            encoder_layers.append(
                EncoderLayer(
                    # Multi-head ProbSparse self-attention.
                    SelfAttentionLayer(
                        # Encoder uses unmasked attention, 
                        # allowing every timestep to attend globally.
                        ProbSparseAttention(False, factor, dropout),
                        d_model, n_heads
                    ),

                    # Position-wise feed-forward.
                    FeedForwardBlock(d_model, d_ff, dropout), 
                    
                    d_model, dropout
                )
            )

        # Encoder Distillation Layers 
        distill_layers = []
        for _ in range((n_encoder_layers - 1)):
            distill_layers.append(EncoderDistillation(d_model))

        self.encoder = InformerEncoder(encoder_layers, distill_layers, d_model)

        # Full decoder stack with self-attention and cross-attention. 
        decoder_layers = []
        for _ in range(n_decoder_layers):
            decoder_layers.append(
                DecoderLayer(
                    # Masked Decoder ProbSparse self-attention. 
                    # Causal masking prevents future leakage.
                    SelfAttentionLayer(
                        ProbSparseAttention(True, factor, dropout),
                        d_model, n_heads,
                    ),

                    # Cross-attention allows the decoder to attend to encoder memory.
                    CrossAttentionLayer(d_model, n_heads, dropout),

                    # Position-wise feed-forward.
                    FeedForwardBlock(d_model, d_ff, dropout),

                    d_model, dropout
                )
            )
        
        self.decoder = InformerDecoder(decoder_layers, d_model)

        # Projection from decoder output to the SST grid.
        self.proj_head = OutputProjectionHead(d_model, height, width)
    

    def build_decoder_input(self, x, dates):
        """Constructs the Informer's generative decoder input.

        The decoder receives:
            1. Known historical SST observations. 
            2. Zero placeholders for future predictions. 

        This enables efficient one-shot generative forecasting.

        Parameters
        ----------
        x : torch.Tensor
            SST input sequence of shape (B, L, 1, H, W)

        dates : torch.Tensor
            Seasonal indices for each timestep of shape (B, L). 

        Returns
        -------
        dec_x:
            (B, (label_len + horizon), 1, H, W)
        dec_day:
            (B, (label_len + horizon))
        """
        # Batch size. 
        B = x.size(0)

        # Use the final known SST observations as decoder start tokens.
        token_x = x[:, -self.label_len:]

        # Corresponding seasonal indices.
        token_day = dates[:, -self.label_len:]

        # Apply zeroed SST placeholders for future prediction steps.
        # The decoder learns to replace these with forecasts.
        zeros = torch.zeros(B, self.horizon, 1, self.h, self.w, device=x.device)

        # Placeholder future temporal indices to replace during training. 
        zero_day = torch.zeros(B, self.horizon, dtype=torch.long, device=x.device)

        # Concatenate history and future placeholders. 
        dec_x = torch.cat([token_x, zeros], dim=1)
        dec_day = torch.cat([token_day, zero_day], dim=1)

        return dec_x, dec_day
    

    def forward(self, x, dates):
        """Run a batch of SST history windows through the Informer.

        Parameters
        ----------
        x : torch.Tensor
            Input SST windows, shape (B, L, 1, H, W). Matches the contract
            used by SstWindowDataset and the rest of the pipeline.
        dates : torch.Tensor
            Seasonal indices for each timestep of shape (B, L). 

        Returns
        -------
        torch.Tensor
            Forecast grids of shape (B, horizon, H, W).
        """
        enc_x = self.enc_embedding(x, dates)

        # Generate encoder memory representations. 
        mem = self.encoder(enc_x)

        # Construct Informer generative decoder inputs.
        dec_x_raw, dec_day = self.build_decoder_input(x, dates)

        # Embed the decoder's SST inputs.
        dec_x = self.dec_embedding(dec_x_raw, dec_day)

        # Generate the decoder's hidden states.
        dec_out = self.decoder(dec_x, mem)

        # Keep only the future timesteps (predictions). 
        dec_out = dec_out[:, -self.horizon:]

        # Project to the grid. 
        pred = self.proj_head(dec_out)

        # Return the predictions. 
        return pred