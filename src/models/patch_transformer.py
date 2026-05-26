import torch
import torch.nn as nn
import math

# We can reuse most of the components from the vanilla Transformer; only the input/output boundaries
#  and the encoder block structure will change
from src.models.transformer import (
    PositionalEncoding,
    LayerNormalisation,
    FeedForwardBlock,
    MultiHeadAttentionBlock,
    ResidualConnection,
)


class PatchProjection(nn.Module):
    """Split each timestep's grid into spatial patches and project each to d_model.

    (B, L, 1, H, W) -> (B, L, N_patches, d_model)
    """
    def __init__(
        self,
        d_model: int,
        height: int,
        width: int,
        patch_height: int,
        patch_width: int,
    ):
        """Build the patch projection layer.

        Parameters
        ----------
        d_model : int
            Working dimension of the Transformer; each patch is projected to a vector of this size.
        height : int
            Spatial height of the SST grid (81 for Coral Sea).
        width : int
            Spatial width of the SST grid (121 for Coral Sea).
        patch_height : int
            Height of each spatial patch. Must divide height exactly.
        patch_width : int
            Width of each spatial patch. Must divide width exactly.
        """
        super().__init__()
        self.d_model = d_model
        self.height = height
        self.width = width
        self.patch_height = patch_height
        self.patch_width = patch_width

        # Grid divisions; with H=81, W=121, ph=9, pw=11 we get 9 x 11 = 99 patches
        self.n_patches_h = height // patch_height
        self.n_patches_w = width // patch_width
        self.n_patches = self.n_patches_h * self.n_patches_w

        # Patch cell count; We will use 9 * 11 = 99 cells per patch
        patch_dim = patch_height * patch_width

        # Each patch (patch_h * patch_w cells) projected to d_model independently
        self.projection = nn.Linear(patch_dim, d_model)

    def forward(self, x):
        """Split each timestep's grid into patches and project to d_model.

        Parameters
        ----------
        x : torch.Tensor
            Input SST sequences, shape (B, L, 1, H, W).

        Returns
        -------
        torch.Tensor
            Patch tokens, shape (B, L, N_patches, d_model), scaled by
            sqrt(d_model) per Vaswani et al.
        """
        B, L, _, H, W = x.shape

        # Drop the singleton channel dim; (B, L, 1, H, W) -> (B, L, H, W)
        x = x.squeeze(2)

        # Reshape to expose patch structure; so split H into (n_patches_h, patch_h) and W into 
        # (n_patches_w, patch_w). The grid (H, W) becomes a 4D layout (n_patches_h, patch_h, 
        # n_patches_w, patch_w) where the patch_h and patch_w dims are the cells inside each patch
        # (B, L, H, W) -> (B, L, n_patches_h, patch_h, n_patches_w, patch_w)
        x = x.view(
            B, L,
            self.n_patches_h, self.patch_height,
            self.n_patches_w, self.patch_width,
        )

        # Move patch index dims (n_patches_h, n_patches_w) next to each other, and the inside-patch dims 
        # (patch_h, patch_w) next to each other; (B, L, n_patches_h, patch_h, n_patches_w, patch_w)
        # -> (B, L, n_patches_h, n_patches_w, patch_h, patch_w)
        x = x.permute(0, 1, 2, 4, 3, 5).contiguous()

        # Flatten the two patch index dims into a single token dim, and the two inside-patch dims into a 
        # single feature dim; -> (B, L, n_patches, patch_h * patch_w)
        x = x.view(B, L, self.n_patches, self.patch_height * self.patch_width)

        # Project each patch independently to d_model; linear broadcasts over all leading dims, so the same 
        # weights apply to every (batch, time, patch); -> (B, L, n_patches, d_model)
        x = self.projection(x)

        # Same scaling as in the our basic Transformers SpatialProjection
        return x * math.sqrt(self.d_model)
    
class SpaceTimePositionalEncoding(nn.Module):
    """Add spatial and temporal positional encodings to patch tokens.

    (B, L, N_patches, d_model) -> (B, L, N_patches, d_model)
    """

    def __init__(
        self,
        d_model: int,
        seq_len: int,
        n_patches: int,
        dropout: float,
    ) -> None:
        """Build spatial and temporal positional encoding tables.

        Parameters
        ----------
        d_model : int
            Working dimension of the Transformer. Must match the patch token dimension produced by 
            PatchProjection.
        seq_len : int
            Length of temporal axis in days. Sets the size of the temporal positional encoding table.
        n_patches : int
            Number of spatial patches per timestep. Sets the size of the spatial positional encoding 
            table.
        dropout : float
            Dropout probability applied once after both encodings have been added to the patch tokens
        """
        super().__init__()
        self.d_model = d_model

        # 2 separate sinusoidal tables, one per axis. Can reuse PositionalEncoding implementation; 
        # dropout=0 tho here because we apply a single shared dropout at the end of forward() rather 
        # than once per axis
        self.temporal_pe = PositionalEncoding(d_model, seq_len, dropout=0.0)
        self.spatial_pe = PositionalEncoding(d_model, n_patches, dropout=0.0)

        # Single dropout pass over the combined (token + spatial + temporal) signal
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        """Add spatial and temporal positional signals to patch tokens.

        Parameters
        ----------
        x : torch.Tensor
            Patch tokens, shape (B, L, N_patches, d_model). Comes from PatchProjection.

        Returns
        -------
        torch.Tensor
            Same shape (B, L, N_patches, d_model), with positional information added along both axes
            and dropout applied.
        """
        B, L, N, _ = x.shape

        # Note our PositionalEncoding expects (B, seq, d_model) and adds its own (1, seq, d_model) buffer

        # GenAI Reference: Consulted GenAI to trick PositionalEncoding into addressing the right axis 
        # with reshape. GenAI adivsed me of a method to do this, since initially I was unsure. It said: 
        # For temporal:
        #   collapse patches into batch so each patch row across time gets the temporal signal added 
        #   independently. We then have (B, L, N, d_model) -> (B*N, L, d_model) -> +temporal_pe -> back to 4D
        x_t = x.permute(0, 2, 1, 3).contiguous().view(B * N, L, self.d_model)
        x_t = self.temporal_pe(x_t)
        x = x_t.view(B, N, L, self.d_model).permute(0, 2, 1, 3).contiguous()

        # For Spatial: 
        #   collapse time into batch so each time row across patches gets the spatial signal added 
        #   independently; (B, L, N, d_model) -> (B*L, N, d_model) -> +spatial_pe -> back to 4D
        x_s = x.view(B * L, N, self.d_model)
        x_s = self.spatial_pe(x_s)
        x = x_s.view(B, L, N, self.d_model)

        # Correctness: I verified the patch projection and positional encoding modules with a small sanity check; 
        # the projected tensor had the expected shape (2, 90, 99, 128), the positional encoding preserved this 
        # shape, and the positional encoding introduced no trainable parameters

        # Not GenAI 
        # Single dropout pass on the combined signal
        return self.dropout(x)
    
class SpaceTimeEncoderBlock(nn.Module):
    """One factorised space-time encoder block.

    Three sublayers, each in its own residual; spatial self-attention (across patches within a timestep), temporal
    self-attention (across timesteps within a patch), then position-wise FFN.

    (B, L, N_patches, d_model) -> (B, L, N_patches, d_model)
    """

    def __init__(
        self,
        d_model: int,
        spatial_attention_block: MultiHeadAttentionBlock,
        temporal_attention_block: MultiHeadAttentionBlock,
        feed_forward_block: FeedForwardBlock,
        dropout: float,
    ) -> None:
        """Build one space-time encoder block from pre-constructed sublayers.

        Parameters
        ----------
        d_model : int
            Working dimension of the Transformer. Passed through to the three ResidualConnections so their 
            LayerNorms are sized correctly.
        spatial_attention_block : MultiHeadAttentionBlock
            Attention sublayer that mixes information across the 99 patches within each timestep.
        temporal_attention_block : MultiHeadAttentionBlock
            Attention sublayer that mixes information across the 90 timesteps ithin each patch.
        feed_forward_block : FeedForwardBlock
            Position-wise FFN sublayer, applied independently to every (time, patch) token.
        dropout : float
            Dropout probability used inside each ResidualConnection.
        """
        super().__init__()
        self.spatial_attention_block = spatial_attention_block
        self.temporal_attention_block = temporal_attention_block
        self.feed_forward_block = feed_forward_block

        # Three residuals, one per sublayer; ModuleList so parameters register and move with .to(device)
        self.residual_connections = nn.ModuleList(
            [ResidualConnection(d_model, dropout) for _ in range(3)]
        )

    def forward(self, x, mask):
        """Run x through spatial attention, temporal attention, then FFN.

        Parameters
        ----------
        x : torch.Tensor
            Patch tokens, shape (B, L, N_patches, d_model).
        mask : torch.Tensor or None
            Optional attention mask. None for our encoder-only forecaster.

        Returns
        -------
        torch.Tensor
            Same shape (B, L, N_patches, d_model). Representation refined by one spatial attention pass,
            one temporal attention pass, and one FFN, each residually wrapped.
        """
        B, L, N, D = x.shape

        # Sublayer 1: spatial attention. Collapse (B, L) into batch so each timestep's 99 patches form an
        # independent attention sequence. Every batch time slice attends across all 99 patches in parallel
        # (B, L, N, D) -> (B*L, N, D)
        x = x.view(B * L, N, D)
        x = self.residual_connections[0](
            x, lambda t: self.spatial_attention_block(t, t, t, mask)
        )

        # Sublayer 2: temporal attention. Reshape so each patch's 90-day history forms an independent attention
        # sequence. Need to swap N and L axes so L is the sequence dim, then collapse (B, N) into batch
        # (B*L, N, D) -> (B, L, N, D) -> (B, N, L, D) -> (B*N, L, D)
        x = x.view(B, L, N, D).permute(0, 2, 1, 3).contiguous().view(B * N, L, D)
        x = self.residual_connections[1](
            x, lambda t: self.temporal_attention_block(t, t, t, mask)
        )

        # Sublayer 3: FFN. Operates per-token on the d_model dim only, so the exact layout of leading dims doesnt
        # matter. Restore (B, L, N, D) for the next block; (B*N, L, D) -> (B, N, L, D) -> (B, L, N, D)
        x = x.view(B, N, L, D).permute(0, 2, 1, 3).contiguous()
        x = self.residual_connections[2](x, self.feed_forward_block)

        return x
    
class SpaceTimeEncoder(nn.Module):
    """Stack of N SpaceTimeEncoderBlocks with a final LayerNorm.

    (B, L, N_patches, d_model) -> (B, L, N_patches, d_model)
    """

    def __init__(self, d_model: int, layers: nn.ModuleList) -> None:
        """Build the encoder from a list of pre-constructed blocks.

        Parameters
        ----------
        d_model : int
            Working dimension of the Transformer. Used to size the final LayerNorm.
        layers : nn.ModuleList
            The space-time encoder blocks, constructed externally so this class stays composable.
        """
        super().__init__()
        self.layers = layers

        # Final LayerNorm; only needed because we use pre-norm residuals. Operates on the last dim (d_model)
        # and broadcasts over all leading dims, so it works fine on the 4D (B, L, N, D) tensor without any reshape
        self.norm = LayerNormalisation(d_model)

    def forward(self, x, mask):
        """Run x through every block in sequence then apply final norm.

        Parameters
        ----------
        x : torch.Tensor
            Patch tokens from positional encoding, shape (B, L, N_patches, d_model).
        mask : torch.Tensor or None
            Optional attention mask, passed unchanged to every block.

        Returns
        -------
        torch.Tensor
            Same shape (B, L, N_patches, d_model). Fully encoded representation, ready for the output head.
        """
        for layer in self.layers:
            x = layer(x, mask)
        return self.norm(x)

class PatchOutputHead(nn.Module):
    """Decode last-timestep per-patch features into a forecast grid.

    (B, L, N_patches, d_model) -> (B, h, H, W)
    """

    def __init__(
        self,
        d_model: int,
        horizon: int,
        height: int,
        width: int,
        patch_height: int,
        patch_width: int,
    ):
        """Build the patch output head.

        Parameters
        ----------
        d_model : int
            Working dimension of the Transformer. Input dim of the per-patch projection.
        horizon : int
            Number of forecast days to predict (7 for our problem).
        height : int
            Spatial height of the SST grid (81 for Coral Sea).
        width : int
            Spatial width of the SST grid (121 for Coral Sea).
        patch_height : int
            Height of each spatial patch. Must match PatchProjection.
        patch_width : int
            Width of each spatial patch. Must match PatchProjection.
        """
        super().__init__()
        self.d_model = d_model
        self.horizon = horizon
        self.height = height
        self.width = width
        self.patch_height = patch_height
        self.patch_width = patch_width

        # Patch grid dims; H=81/ph= 9 -> n_patches_h = 9, W=121/pw =11 -> n_patches_w = 11
        self.n_patches_h = height // patch_height
        self.n_patches_w = width // patch_width
        self.n_patches = self.n_patches_h * self.n_patches_w

        # Shared projection; each d_model patch vector -> a (horizon, patch_h, patch_w) tile; symmetric with 
        # PatchProjections Linear(patch_h * patch_w, d_model) at the input
        self.projection = nn.Linear(d_model, horizon * patch_height * patch_width)

    def forward(self, x):
        """Decode last-timestep patch features into a multi-horizon forecast grid.

        Parameters
        ----------
        x : torch.Tensor
            Encoder output, shape (B, L, N_patches, d_model). Only the last timestep is used; via self-attention
            it already summarises the whole history from day 90's perspective.

        Returns
        -------
        torch.Tensor
            Forecast grids, shape (B, horizon, H, W). No activation applied; outputs are raw z-scored anomalies.
        """
        B = x.shape[0]

        # Take the final timestep's patch tokens; via spatial + temporal attention each patch vector already 
        # summarises the whole 90 day history of that patch and its spatial neighbours
        # (B, L, N, d_model) -> (B, N, d_model)
        x = x[:, -1, :, :]

        # Project each patch independently to a (horizon, patch_h, patch_w) tile; linear broadcasts over leading
        # dims, so the same weights apply to every (batch, patch). This is the inverse of PatchProjection's 
        # per-patch projection; (B, N, d_model) -> (B, N, horizon * patch_h * patch_w)
        x = self.projection(x)

        # Reshape to expose the patch and tile dims separately. The token dim N splits back into (n_patches_h,
        # n_patches_w), and the flat output dim splits into (horizon, patch_h, patch_w)
        # (B, N, horizon * ph * pw) -> (B, n_ph, n_pw, horizon, ph, pw)
        x = x.view(
            B,
            self.n_patches_h, self.n_patches_w,
            self.horizon, self.patch_height, self.patch_width,
        )

        # Permute so horizon comes first (after batch), then the spatial dims pair up; (n_ph, ph) become rows of the
        # full H axis, (n_pw, pw) become cols of W; (B, n_ph, n_pw, h, ph, pw) -> (B, h, n_ph, ph, n_pw, pw)
        x = x.permute(0, 3, 1, 4, 2, 5).contiguous()

        # Collapse the (patch_index, inside_patch) pairs into the original H and W; this is the exact inverse of 
        # the reshape sequence in PatchProjection; (B, h, n_ph, ph, n_pw, pw) -> (B, h, H, W)
        x = x.view(B, self.horizon, self.height, self.width)

        return x
    
class SstPatchTransformer(nn.Module):
    """Patch-tokenised space-time Transformer for SST forecasting.

    (B, L, 1, H, W) -> (B, h, H, W)
    """

    def __init__(
        self,
        height: int,
        width: int,
        patch_height: int = 9,
        patch_width: int = 11,
        seq_len: int = 90,
        horizon: int = 7,
        d_model: int = 128,
        n_blocks: int = 2,
        n_heads: int = 4,
        d_ff: int = 512,
        dropout: float = 0.1,
    ):
        """Build the full patch Transformer from its component classes.

        Parameters
        ----------
        height : int
            Spatial height of the SST grid (81 for Coral Sea).
        width : int
            Spatial width of the SST grid (121 for Coral Sea).
        patch_height : int
            Height of each spatial patch. Must divide height exactly.
        patch_width : int
            Width of each spatial patch. Must divide width exactly.
        seq_len : int
            Length of the input history window in days.
        horizon : int
            Number of forecast days to predict.
        d_model : int
            Working dimension of the Transformer.
        n_blocks : int
            Number of stacked SpaceTimeEncoderBlocks.
        n_heads : int
            Number of attention heads per MultiHeadAttentionBlock. Note this must divide d_model.
        d_ff : int
            Hidden dim inside each FeedForwardBlock.
        dropout : float
            Dropout probability used throughout.
        """
        super().__init__()

        # Patch grid; H = 81/ph = 9 -> 9 rows of patches, W = 121/pw = 11 -> 11 cols
        n_patches = (height // patch_height) * (width // patch_width)

        # Input boundary; grid -> per-patch d_model tokens
        self.input_projection = PatchProjection(
            d_model=d_model,
            height=height,
            width=width,
            patch_height=patch_height,
            patch_width=patch_width,
        )

        # Positional information along both axes
        self.positional_encoding = SpaceTimePositionalEncoding(
            d_model=d_model,
            seq_len=seq_len,
            n_patches=n_patches,
            dropout=dropout,
        )

        # Build n_blocks space-time encoder blocks; each gets its own freshly constructed pair of 
        # attention modules and FFN so weights arent shared
        encoder_blocks = nn.ModuleList([
            SpaceTimeEncoderBlock(
                d_model=d_model,
                spatial_attention_block=MultiHeadAttentionBlock(d_model, n_heads, dropout),
                temporal_attention_block=MultiHeadAttentionBlock(d_model, n_heads, dropout),
                feed_forward_block=FeedForwardBlock(d_model, d_ff, dropout),
                dropout=dropout,
            )
            for _ in range(n_blocks)
        ])
        self.encoder = SpaceTimeEncoder(d_model, encoder_blocks)

        # Output boundary; last-timestep per-patch features -> forecast grid
        self.output_head = PatchOutputHead(
            d_model=d_model,
            horizon=horizon,
            height=height,
            width=width,
            patch_height=patch_height,
            patch_width=patch_width,
        )

    def forward(self, x):
        """Run a batch of SST history windows through the full patch Transformer.

        Parameters
        ----------
        x : torch.Tensor
            Input SST windows, shape (B, L, 1, H, W). Matches the contract used by SstWindowDataset and 
            the rest of the pipeline.

        Returns
        -------
        torch.Tensor
            Forecast grids, shape (B, horizon, H, W).
        """
        # (B, L, 1, H, W) -> (B, L, N_patches, d_model)
        x = self.input_projection(x)

        # (B, L, N_patches, d_model) -> (B, L, N_patches, d_model)
        x = self.positional_encoding(x)

        # (B, L, N_patches, d_model) -> (B, L, N_patches, d_model)
        x = self.encoder(x, mask=None)

        # (B, L, N_patches, d_model) -> (B, horizon, H, W)
        x = self.output_head(x)

        return x
    

    
