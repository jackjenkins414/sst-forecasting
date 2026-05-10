"""Unit tests: forward-pass shape and output sanity for E1 models.

All tests run on CPU with a tiny synthetic input — no Zarr store required.
Target: < 5 s total on any machine.
"""

from __future__ import annotations

import torch
import pytest


@pytest.fixture(scope="module")
def tiny_input() -> torch.Tensor:
    """Tiny (B=2, L=10, 1, H=16, W=12) random input."""
    torch.manual_seed(0)
    return torch.randn(2, 10, 1, 16, 12)


# ─────────────────────────────────────────────────────────────────────────────
# SpatialFlatLSTM
# ─────────────────────────────────────────────────────────────────────────────


class TestSpatialFlatLSTM:
    H, W, L, h = 16, 12, 10, 7

    def _model(self, num_layers: int = 2, **kwargs):
        from sst_forecasting.models.lstm import SpatialFlatLSTM
        return SpatialFlatLSTM(
            H=self.H, W=self.W,
            context_len=self.L, horizon=self.h,
            d_spatial=16, hidden_size=32, num_layers=num_layers,
            **kwargs,
        )

    def test_output_shape(self, tiny_input):
        model = self._model()
        out = model(tiny_input)
        assert out.shape == (2, self.h, self.H, self.W)

    def test_output_dtype_float32(self, tiny_input):
        model = self._model()
        out = model(tiny_input)
        assert out.dtype == torch.float32

    def test_no_nan_in_output(self, tiny_input):
        model = self._model()
        out = model(tiny_input)
        assert not torch.isnan(out).any(), "LSTM output contains NaN"

    def test_single_layer_no_dropout(self, tiny_input):
        """num_layers=1 must not crash (LSTM dropout=0 guard)."""
        model = self._model(num_layers=1)
        out = model(tiny_input)
        assert out.shape == (2, self.h, self.H, self.W)

    def test_parameter_count_positive(self):
        model = self._model()
        assert model.count_parameters() > 0

    def test_gradients_flow(self, tiny_input):
        model = self._model()
        out = model(tiny_input)
        loss = out.mean()
        loss.backward()
        for name, p in model.named_parameters():
            if p.requires_grad:
                assert p.grad is not None, f"No grad for {name}"

    def test_eval_mode_deterministic(self, tiny_input):
        """In eval mode, two forward passes must be identical (no MC dropout)."""
        model = self._model()
        model.eval()
        with torch.no_grad():
            o1 = model(tiny_input)
            o2 = model(tiny_input)
        assert torch.allclose(o1, o2)

    def test_batch_independence(self):
        """Changing one batch element must not affect others (no batch-norm leakage)."""
        from sst_forecasting.models.lstm import SpatialFlatLSTM
        model = SpatialFlatLSTM(H=self.H, W=self.W, context_len=self.L, horizon=self.h,
                                d_spatial=16, hidden_size=32)
        model.eval()
        x = torch.randn(2, self.L, 1, self.H, self.W)
        with torch.no_grad():
            out_pair = model(x)
            x_single = x[:1]
            out_single = model(x_single)
        assert torch.allclose(out_pair[:1], out_single, atol=1e-5), (
            "Batch element 0 output differs when evaluated alone vs in a batch"
        )


# ─────────────────────────────────────────────────────────────────────────────
# SpatialFlatTransformer
# ─────────────────────────────────────────────────────────────────────────────


class TestSpatialFlatTransformer:
    H, W, L, h = 16, 12, 10, 7

    def _model(self, num_encoder_layers: int = 2, **kwargs):
        from sst_forecasting.models.transformer import SpatialFlatTransformer
        return SpatialFlatTransformer(
            H=self.H, W=self.W,
            context_len=self.L, horizon=self.h,
            d_model=32, nhead=4, num_encoder_layers=num_encoder_layers,
            dim_feedforward=64,
            **kwargs,
        )

    def test_output_shape(self, tiny_input):
        model = self._model()
        out = model(tiny_input)
        assert out.shape == (2, self.h, self.H, self.W)

    def test_output_dtype_float32(self, tiny_input):
        model = self._model()
        out = model(tiny_input)
        assert out.dtype == torch.float32

    def test_no_nan_in_output(self, tiny_input):
        model = self._model()
        out = model(tiny_input)
        assert not torch.isnan(out).any(), "Transformer output contains NaN"

    def test_single_layer(self, tiny_input):
        model = self._model(num_encoder_layers=1)
        out = model(tiny_input)
        assert out.shape == (2, self.h, self.H, self.W)

    def test_parameter_count_positive(self):
        model = self._model()
        assert model.count_parameters() > 0

    def test_gradients_flow(self, tiny_input):
        model = self._model()
        out = model(tiny_input)
        loss = out.mean()
        loss.backward()
        for name, p in model.named_parameters():
            if p.requires_grad:
                assert p.grad is not None, f"No grad for {name}"

    def test_eval_mode_deterministic(self, tiny_input):
        model = self._model()
        model.eval()
        with torch.no_grad():
            o1 = model(tiny_input)
            o2 = model(tiny_input)
        assert torch.allclose(o1, o2)

    def test_positional_encoding_changes_output(self, tiny_input):
        """Output with PE must differ from a model where PE embeddings are zeroed."""
        from sst_forecasting.models.transformer import SpatialFlatTransformer, SinusoidalPE
        model = self._model()
        model.eval()
        with torch.no_grad():
            out_with_pe = model(tiny_input)

        # Zero out PE buffer to disable positional information
        model.pos_enc.pe.zero_()
        model.eval()
        with torch.no_grad():
            out_no_pe = model(tiny_input)

        assert not torch.allclose(out_with_pe, out_no_pe), (
            "Zeroing positional encoding had no effect — PE may not be wired in"
        )


# ─────────────────────────────────────────────────────────────────────────────
# Cross-architecture sanity
# ─────────────────────────────────────────────────────────────────────────────


def test_lstm_transformer_same_io_interface():
    """Both models accept the same (B, L, 1, H, W) input and return (B, h, H, W)."""
    from sst_forecasting.models.lstm import SpatialFlatLSTM
    from sst_forecasting.models.transformer import SpatialFlatTransformer

    x = torch.randn(2, 10, 1, 16, 12)
    for Model, kwargs in [
        (SpatialFlatLSTM,
         dict(d_spatial=16, hidden_size=32)),
        (SpatialFlatTransformer,
         dict(d_model=32, nhead=4, num_encoder_layers=2, dim_feedforward=64)),
    ]:
        m = Model(H=16, W=12, context_len=10, horizon=7, **kwargs)
        out = m(x)
        assert out.shape == (2, 7, 16, 12), f"{Model.__name__} output shape wrong"


# ─────────────────────────────────────────────────────────────────────────────
# SpatialConvLSTM
# ─────────────────────────────────────────────────────────────────────────────


class TestSpatialConvLSTM:
    H, W, L, h = 16, 12, 10, 7

    def _model(self, hidden_channels=None, **kwargs):
        from sst_forecasting.models.convlstm import SpatialConvLSTM
        return SpatialConvLSTM(
            H=self.H, W=self.W,
            context_len=self.L, horizon=self.h,
            hidden_channels=hidden_channels or [8, 16],
            **kwargs,
        )

    def test_output_shape(self, tiny_input):
        model = self._model()
        out = model(tiny_input)
        assert out.shape == (2, self.h, self.H, self.W)

    def test_output_dtype_float32(self, tiny_input):
        model = self._model()
        out = model(tiny_input)
        assert out.dtype == torch.float32

    def test_no_nan_in_output(self, tiny_input):
        model = self._model()
        out = model(tiny_input)
        assert not torch.isnan(out).any(), "ConvLSTM output contains NaN"

    def test_single_layer(self, tiny_input):
        """Single ConvLSTM layer must not crash."""
        model = self._model(hidden_channels=[16])
        out = model(tiny_input)
        assert out.shape == (2, self.h, self.H, self.W)

    def test_three_layers(self, tiny_input):
        """Three-layer stack must not crash."""
        model = self._model(hidden_channels=[8, 16, 16])
        out = model(tiny_input)
        assert out.shape == (2, self.h, self.H, self.W)

    def test_parameter_count_positive(self):
        model = self._model()
        assert model.count_parameters() > 0

    def test_parameter_count_much_smaller_than_flat_lstm(self):
        """hidden_channels=[32,64] ConvLSTM has fewer params than flat LSTM."""
        from sst_forecasting.models.lstm import SpatialFlatLSTM
        from sst_forecasting.models.convlstm import SpatialConvLSTM
        conv_model = SpatialConvLSTM(
            H=self.H, W=self.W, context_len=self.L, horizon=self.h,
            hidden_channels=[32, 64],
        )
        flat_model = SpatialFlatLSTM(
            H=self.H, W=self.W, context_len=self.L, horizon=self.h,
            d_spatial=64, hidden_size=128, num_layers=2,
        )
        assert conv_model.count_parameters() < flat_model.count_parameters()

    def test_gradients_flow(self, tiny_input):
        model = self._model()
        out = model(tiny_input)
        loss = out.mean()
        loss.backward()
        for name, p in model.named_parameters():
            if p.requires_grad:
                assert p.grad is not None, f"No grad for {name}"

    def test_eval_mode_deterministic(self, tiny_input):
        model = self._model()
        model.eval()
        with torch.no_grad():
            o1 = model(tiny_input)
            o2 = model(tiny_input)
        assert torch.allclose(o1, o2)

    def test_batch_independence(self):
        """Changing one batch element must not affect others (no shared state)."""
        from sst_forecasting.models.convlstm import SpatialConvLSTM
        model = SpatialConvLSTM(
            H=self.H, W=self.W, context_len=self.L, horizon=self.h,
            hidden_channels=[8, 16],
        )
        model.eval()
        x = torch.randn(2, self.L, 1, self.H, self.W)
        with torch.no_grad():
            out_pair = model(x)
            out_single = model(x[:1])
        assert torch.allclose(out_pair[:1], out_single, atol=1e-5), (
            "Batch element 0 output differs when evaluated alone vs in a batch"
        )

    def test_spatial_locality(self, tiny_input):
        """Zeroing one half of the grid should leave the other half mostly unchanged."""
        from sst_forecasting.models.convlstm import SpatialConvLSTM
        # short context + single layer keeps receptive field small (≤5 cells)
        model = SpatialConvLSTM(
            H=self.H, W=self.W, context_len=3, horizon=self.h,
            hidden_channels=[4], kernel_size=3,
        )
        model.eval()
        x = torch.randn(1, 3, 1, self.H, self.W)
        x_patched = x.clone()
        # Zero out the entire left half of the spatial grid
        x_patched[:, :, :, :, : self.W // 2] = 0.0
        with torch.no_grad():
            out_orig = model(x)
            out_patched = model(x_patched)
        # The right half of the output should be unchanged (or very similar)
        right_half_diff = (out_orig[0, :, :, self.W // 2:] -
                           out_patched[0, :, :, self.W // 2:]).abs().max()
        assert right_half_diff < 0.5, "right half output changed too much when only left half was zeroed"


# ─────────────────────────────────────────────────────────────────────────────
# All-model I/O interface parity
# ─────────────────────────────────────────────────────────────────────────────


def test_all_models_same_io_interface():
    """LSTM, Transformer, and ConvLSTM all accept (B,L,1,H,W) → (B,h,H,W)."""
    from sst_forecasting.models.lstm import SpatialFlatLSTM
    from sst_forecasting.models.transformer import SpatialFlatTransformer
    from sst_forecasting.models.convlstm import SpatialConvLSTM

    H, W, L, h = 16, 12, 10, 7
    x = torch.randn(2, L, 1, H, W)
    models = [
        SpatialFlatLSTM(H=H, W=W, context_len=L, horizon=h,
                        d_spatial=16, hidden_size=32),
        SpatialFlatTransformer(H=H, W=W, context_len=L, horizon=h,
                               d_model=32, nhead=4, num_encoder_layers=2,
                               dim_feedforward=64),
        SpatialConvLSTM(H=H, W=W, context_len=L, horizon=h,
                        hidden_channels=[8, 16]),
    ]
    for m in models:
        out = m(x)
        assert out.shape == (2, h, H, W), f"{type(m).__name__} output shape wrong"
