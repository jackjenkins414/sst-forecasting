"""Tests for SSTWindowDataset and related data utilities."""

from __future__ import annotations

import numpy as np
import pytest
import torch

from sst_forecasting.data.dataset import SSTWindowDataset
from sst_forecasting.data.splits import SPLITS, date_mask, split_indices
from sst_forecasting.data.transforms import FillLand, Standardize, SpatialPatchify


# ---------------------------------------------------------------------------
# splits.py
# ---------------------------------------------------------------------------


class TestSplits:
    def test_split_keys(self):
        assert set(SPLITS) == {"train", "val", "test"}

    def test_date_mask_shapes(self):
        import pandas as pd

        time = pd.date_range("1981-09-01", periods=7305, freq="D")
        for split in ("train", "val", "test"):
            mask = date_mask(time, split)
            assert mask.dtype == bool
            assert mask.shape == (len(time),)
            assert mask.sum() > 0, f"No timesteps for split={split}"

    def test_splits_non_overlapping(self):
        import pandas as pd

        time = pd.date_range("1981-09-01", "2000-12-31", freq="D")
        masks = {s: date_mask(time, s) for s in SPLITS}
        # No timestep in two splits at once
        for a, b in [("train", "val"), ("train", "test"), ("val", "test")]:
            assert not np.any(masks[a] & masks[b]), f"Overlap between {a} and {b}"

    def test_invalid_split_raises(self):
        import pandas as pd

        time = pd.date_range("1990-01-01", periods=10, freq="D")
        with pytest.raises(ValueError):
            date_mask(time, "unknown_split")

    def test_split_indices_contiguous(self):
        import pandas as pd

        time = pd.date_range("1981-09-01", "2000-12-31", freq="D")
        for split in SPLITS:
            idx = split_indices(time, split)
            assert len(idx) > 0
            diffs = np.diff(idx)
            assert np.all(diffs == 1), f"Split '{split}' indices are not contiguous"


# ---------------------------------------------------------------------------
# SSTWindowDataset — in-memory (data_array) path
# ---------------------------------------------------------------------------


class TestSSTWindowDatasetInMemory:
    L, h = 30, 7

    def test_len(self, tiny_array: np.ndarray):
        T = len(tiny_array)
        ds = SSTWindowDataset(data_array=tiny_array, context_len=self.L, horizon=self.h)
        assert len(ds) == T - self.L - self.h + 1

    def test_output_shapes(self, tiny_array: np.ndarray):
        _, H, W = tiny_array.shape
        ds = SSTWindowDataset(data_array=tiny_array, context_len=self.L, horizon=self.h)
        x, y = ds[0]
        assert x.shape == torch.Size([self.L, 1, H, W])
        assert y.shape == torch.Size([self.h, H, W])

    def test_output_dtype(self, tiny_array: np.ndarray):
        ds = SSTWindowDataset(data_array=tiny_array, context_len=self.L, horizon=self.h)
        x, y = ds[0]
        assert x.dtype == torch.float32
        assert y.dtype == torch.float32

    def test_no_nans_in_output(self, tiny_array: np.ndarray):
        ds = SSTWindowDataset(data_array=tiny_array, context_len=self.L, horizon=self.h)
        x, y = ds[0]
        assert not torch.isnan(x).any()
        assert not torch.isnan(y).any()

    def test_window_sliding_by_one(self, tiny_array: np.ndarray):
        """Consecutive windows should overlap by (L-1) timesteps."""
        ds = SSTWindowDataset(data_array=tiny_array, context_len=self.L, horizon=self.h)
        x0, _ = ds[0]
        x1, _ = ds[1]
        # x1[:-1] should equal x0[1:]
        assert torch.allclose(x1[:, :, :, :][:-1], x0[:, :, :, :][1:])

    def test_index_out_of_range(self, tiny_array: np.ndarray):
        ds = SSTWindowDataset(data_array=tiny_array, context_len=self.L, horizon=self.h)
        with pytest.raises(IndexError):
            _ = ds[len(ds)]

    def test_not_enough_timesteps_raises(self):
        arr = np.zeros((5, 4, 4), dtype=np.float32)
        with pytest.raises(ValueError):
            SSTWindowDataset(data_array=arr, context_len=30, horizon=7)

    def test_requires_zarr_or_array(self):
        with pytest.raises(ValueError):
            SSTWindowDataset()

    def test_both_zarr_and_array_raises(self, tiny_zarr: str, tiny_array: np.ndarray):
        with pytest.raises(ValueError):
            SSTWindowDataset(zarr_path=tiny_zarr, data_array=tiny_array)


# ---------------------------------------------------------------------------
# SSTWindowDataset — Zarr path
# ---------------------------------------------------------------------------


class TestSSTWindowDatasetZarr:
    L, h = 30, 7

    def test_zarr_train_split_len(self, tiny_zarr: str):
        ds = SSTWindowDataset(tiny_zarr, split="train", context_len=self.L, horizon=self.h)
        assert len(ds) > 0

    def test_zarr_output_shapes(self, tiny_zarr: str):
        ds = SSTWindowDataset(tiny_zarr, split="train", context_len=self.L, horizon=self.h)
        x, y = ds[0]
        assert x.shape[0] == self.L
        assert x.shape[1] == 1   # channel dim
        assert y.shape[0] == self.h
        assert x.shape[2:] == y.shape[1:]  # H and W match

    def test_zarr_no_nans(self, tiny_zarr: str):
        ds = SSTWindowDataset(tiny_zarr, split="train", context_len=self.L, horizon=self.h)
        x, y = ds[0]
        assert not torch.isnan(x).any()
        assert not torch.isnan(y).any()

    def test_missing_zarr_raises(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            SSTWindowDataset(tmp_path / "nonexistent.zarr", split="train")

    def test_spatial_shape_property(self, tiny_zarr: str):
        ds = SSTWindowDataset(tiny_zarr, split="train", context_len=self.L, horizon=self.h)
        H, W = ds.spatial_shape
        assert H == 16 and W == 16


# ---------------------------------------------------------------------------
# Transforms
# ---------------------------------------------------------------------------


class TestTransforms:
    def test_standardize_roundtrip(self):
        norm = Standardize(mean=1.5, std=2.0)
        x = torch.randn(4, 1, 8, 8)
        assert torch.allclose(norm.inverse(norm(x)), x, atol=1e-5)

    def test_fill_land_removes_nan(self):
        fill = FillLand(fill_value=-1.0)
        x = torch.tensor([float("nan"), 0.5, float("nan"), 1.0])
        out = fill(x)
        assert not torch.isnan(out).any()
        assert (out[0] == -1.0) and (out[2] == -1.0)

    def test_spatial_patchify_shape(self):
        patchify = SpatialPatchify(patch_h=4, patch_w=4)
        x = torch.randn(10, 1, 16, 16)   # (L, C, H, W)
        out = patchify(x)
        # num_patches = (16//4) * (16//4) = 16, patch_size = 4*4 = 16
        assert out.shape == torch.Size([10, 1, 16, 16])

    def test_spatial_patchify_bad_dims(self):
        patchify = SpatialPatchify(patch_h=8, patch_w=8)
        x = torch.randn(4, 15, 15)   # 15 not divisible by 8
        with pytest.raises(ValueError):
            patchify(x)
