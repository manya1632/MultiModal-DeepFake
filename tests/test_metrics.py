"""
Property-based and unit tests for utils/metrics.py.

Properties tested:
  - Property 5: PSNR identity and correctness (Requirements 8.1, 8.4, 16.3)
  - Property 6: NC identity, range, and correctness (Requirements 8.2, 8.5, 16.4)
  - Property 7: Metrics ValueError on shape mismatch (Requirements 8.6)
"""

import math
import pytest
import numpy as np
import torch
from hypothesis import given, settings, assume
from hypothesis import strategies as st

from utils.metrics import (
    compute_psnr,
    compute_nc,
    compute_auc,
    compute_eer,
    compute_acc,
    compute_iou_mean,
    compute_iou_at_50,
    compute_iou_at_75,
    compute_iou_at_95,
    compute_token_f1,
    compute_token_precision,
    compute_token_recall,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _rand_tensor(*shape):
    return torch.rand(*shape)


# ---------------------------------------------------------------------------
# Property 5: PSNR identity and correctness
# Validates: Requirements 8.1, 8.4, 16.3
# ---------------------------------------------------------------------------

class TestComputePSNR:
    """Property 5: PSNR identity and correctness"""

    @given(
        n=st.integers(min_value=1, max_value=4),
        size=st.integers(min_value=1, max_value=256),
    )
    @settings(max_examples=100)
    def test_psnr_identity_returns_inf(self, n: int, size: int):
        """compute_psnr(T, T) must return float('inf') for any tensor T."""
        T = _rand_tensor(n, size)
        result = compute_psnr(T, T)
        assert result == float("inf"), (
            f"Expected inf for identical tensors, got {result}"
        )

    @given(
        n=st.integers(min_value=1, max_value=4),
        size=st.integers(min_value=1, max_value=256),
    )
    @settings(max_examples=100)
    def test_psnr_formula_correctness(self, n: int, size: int):
        """PSNR = 10 * log10(1 / MSE) for known non-zero MSE."""
        original = torch.zeros(n, size)
        # watermarked has a known constant offset so MSE is predictable
        offset = 0.1
        watermarked = original + offset
        mse = offset ** 2
        expected = 10.0 * math.log10(1.0 / mse)
        result = compute_psnr(original, watermarked)
        assert abs(result - expected) < 1e-4, (
            f"PSNR formula mismatch: expected {expected:.4f}, got {result:.4f}"
        )

    def test_psnr_returns_float(self):
        """Return type must be a Python float."""
        T = _rand_tensor(2, 8)
        noise = T + 0.05
        result = compute_psnr(T, noise)
        assert isinstance(result, float)

    def test_psnr_positive_for_different_tensors(self):
        """PSNR should be a positive finite value for non-identical tensors."""
        a = torch.zeros(3, 224, 224)
        b = torch.ones(3, 224, 224) * 0.01
        result = compute_psnr(a, b)
        assert math.isfinite(result) and result > 0


# ---------------------------------------------------------------------------
# Property 6: NC identity, range, and correctness
# Validates: Requirements 8.2, 8.5, 16.4
# ---------------------------------------------------------------------------

class TestComputeNC:
    """Property 6: NC identity, range, and correctness"""

    @given(
        n=st.integers(min_value=1, max_value=4),
        size=st.integers(min_value=1, max_value=256),
    )
    @settings(max_examples=100)
    def test_nc_identity_returns_one(self, n: int, size: int):
        """compute_nc(T, T) must return exactly 1.0 for any non-zero tensor T."""
        T = _rand_tensor(n, size) + 0.01  # avoid all-zero tensor
        result = compute_nc(T, T)
        assert abs(result - 1.0) < 1e-5, (
            f"Expected 1.0 for identical tensors, got {result}"
        )

    @given(
        n=st.integers(min_value=1, max_value=4),
        size=st.integers(min_value=1, max_value=256),
    )
    @settings(max_examples=100)
    def test_nc_range(self, n: int, size: int):
        """NC must always be in [-1.0, 1.0]."""
        a = _rand_tensor(n, size)
        b = _rand_tensor(n, size)
        result = compute_nc(a, b)
        assert -1.0 <= result <= 1.0, (
            f"NC out of range [-1, 1]: got {result}"
        )

    @given(
        n=st.integers(min_value=1, max_value=4),
        size=st.integers(min_value=1, max_value=256),
    )
    @settings(max_examples=100)
    def test_nc_anti_identity_returns_minus_one(self, n: int, size: int):
        """compute_nc(T, -T) must return -1.0 for any non-zero tensor T."""
        T = _rand_tensor(n, size) + 0.01
        result = compute_nc(T, -T)
        assert abs(result - (-1.0)) < 1e-5, (
            f"Expected -1.0 for negated tensor, got {result}"
        )

    def test_nc_returns_float(self):
        """Return type must be a Python float."""
        a = _rand_tensor(128)
        b = _rand_tensor(128)
        assert isinstance(compute_nc(a, b), float)


# ---------------------------------------------------------------------------
# Property 7: Metrics ValueError on shape mismatch
# Validates: Requirements 8.6
# ---------------------------------------------------------------------------

class TestShapeMismatch:
    """Property 7: Metrics ValueError on shape mismatch"""

    @given(
        shape_a=st.lists(st.integers(1, 8), min_size=1, max_size=3),
        shape_b=st.lists(st.integers(1, 8), min_size=1, max_size=3),
    )
    @settings(max_examples=100)
    def test_psnr_raises_on_shape_mismatch(self, shape_a, shape_b):
        """compute_psnr must raise ValueError when shapes differ."""
        assume(tuple(shape_a) != tuple(shape_b))
        a = torch.rand(*shape_a)
        b = torch.rand(*shape_b)
        with pytest.raises(ValueError, match="[Ss]hape"):
            compute_psnr(a, b)

    @given(
        shape_a=st.lists(st.integers(1, 8), min_size=1, max_size=3),
        shape_b=st.lists(st.integers(1, 8), min_size=1, max_size=3),
    )
    @settings(max_examples=100)
    def test_nc_raises_on_shape_mismatch(self, shape_a, shape_b):
        """compute_nc must raise ValueError when shapes differ."""
        assume(tuple(shape_a) != tuple(shape_b))
        a = torch.rand(*shape_a)
        b = torch.rand(*shape_b)
        with pytest.raises(ValueError, match="[Ss]hape"):
            compute_nc(a, b)

    def test_psnr_explicit_mismatch(self):
        """Explicit shape mismatch: (2, 3) vs (3, 2)."""
        with pytest.raises(ValueError):
            compute_psnr(torch.rand(2, 3), torch.rand(3, 2))

    def test_nc_explicit_mismatch(self):
        """Explicit shape mismatch: (128,) vs (64,)."""
        with pytest.raises(ValueError):
            compute_nc(torch.rand(128), torch.rand(64))


# ---------------------------------------------------------------------------
# Unit tests for detection metrics
# ---------------------------------------------------------------------------

class TestDetectionMetrics:
    """Unit tests for AUC, EER, ACC."""

    def test_auc_perfect(self):
        y_true = np.array([0, 0, 1, 1])
        y_score = np.array([0.1, 0.2, 0.8, 0.9])
        assert compute_auc(y_true, y_score) == 1.0

    def test_auc_random(self):
        y_true = np.array([0, 1, 0, 1])
        y_score = np.array([0.5, 0.5, 0.5, 0.5])
        auc = compute_auc(y_true, y_score)
        assert 0.0 <= auc <= 1.0

    def test_eer_range(self):
        y_true = np.array([0, 0, 1, 1])
        y_score = np.array([0.1, 0.4, 0.6, 0.9])
        eer = compute_eer(y_true, y_score)
        assert 0.0 <= eer <= 1.0

    def test_acc_perfect(self):
        y_true = np.array([0, 1, 0, 1])
        y_pred = np.array([0, 1, 0, 1])
        assert compute_acc(y_true, y_pred) == 1.0

    def test_acc_zero(self):
        y_true = np.array([0, 1, 0, 1])
        y_pred = np.array([1, 0, 1, 0])
        assert compute_acc(y_true, y_pred) == 0.0


class TestIoUMetrics:
    """Unit tests for IoU metrics."""

    def test_iou_mean(self):
        assert abs(compute_iou_mean([0.4, 0.6, 0.8]) - (0.4 + 0.6 + 0.8) / 3) < 1e-6

    def test_iou_mean_empty(self):
        assert compute_iou_mean([]) == 0.0

    def test_iou_at_50(self):
        scores = [0.3, 0.5, 0.7, 0.9]
        # 3 out of 4 are >= 0.50
        assert abs(compute_iou_at_50(scores) - 0.75) < 1e-6

    def test_iou_at_75(self):
        scores = [0.3, 0.5, 0.7, 0.9]
        # only 0.9 >= 0.75 → 1 out of 4 = 0.25
        assert abs(compute_iou_at_75(scores) - 0.25) < 1e-6

    def test_iou_at_95(self):
        scores = [0.3, 0.5, 0.7, 0.9]
        # 0 out of 4 are >= 0.95
        assert compute_iou_at_95(scores) == 0.0

    def test_iou_at_95_all_pass(self):
        scores = [0.95, 0.97, 1.0]
        assert compute_iou_at_95(scores) == 1.0


class TestTokenMetrics:
    """Unit tests for token-level F1, precision, recall."""

    def test_precision_normal(self):
        assert abs(compute_token_precision(3, 1) - 0.75) < 1e-6

    def test_precision_zero_denom(self):
        assert compute_token_precision(0, 0) == 0.0

    def test_recall_normal(self):
        assert abs(compute_token_recall(3, 1) - 0.75) < 1e-6

    def test_recall_zero_denom(self):
        assert compute_token_recall(0, 0) == 0.0

    def test_f1_normal(self):
        # precision = 3/4 = 0.75, recall = 3/4 = 0.75, f1 = 0.75
        assert abs(compute_token_f1(3, 1, 1) - 0.75) < 1e-6

    def test_f1_zero(self):
        assert compute_token_f1(0, 0, 0) == 0.0

    def test_f1_perfect(self):
        assert abs(compute_token_f1(5, 0, 0) - 1.0) < 1e-6
