"""
Tests for TextWatermarkDecoder.

Validates: Requirements 5.1
"""

import torch
import pytest

from models.watermark_text_decoder import TextWatermarkDecoder


# ---------------------------------------------------------------------------
# Unit tests
# ---------------------------------------------------------------------------

def test_output_shape_valid_input():
    """Output shape is (B, 128) for valid 3D input."""
    model = TextWatermarkDecoder()
    model.eval()

    embeddings_w = torch.randn(4, 32, 768)
    with torch.no_grad():
        m_I_hat = model(embeddings_w)

    assert m_I_hat.shape == (4, 128), f"Expected (4, 128), got {m_I_hat.shape}"


def test_output_values_in_range():
    """Output values are in [0, 1] due to Sigmoid activation."""
    model = TextWatermarkDecoder()
    model.eval()

    embeddings_w = torch.randn(2, 16, 768)
    with torch.no_grad():
        m_I_hat = model(embeddings_w)

    assert m_I_hat.min().item() >= 0.0, "Values should be >= 0"
    assert m_I_hat.max().item() <= 1.0, "Values should be <= 1"


def test_value_error_on_2d_input():
    """ValueError raised when input is 2D (non-3D)."""
    model = TextWatermarkDecoder()
    with pytest.raises(ValueError, match="3D"):
        model(torch.randn(4, 768))


def test_value_error_on_1d_input():
    """ValueError raised when input is 1D."""
    model = TextWatermarkDecoder()
    with pytest.raises(ValueError, match="3D"):
        model(torch.randn(768))


def test_value_error_on_4d_input():
    """ValueError raised when input is 4D."""
    model = TextWatermarkDecoder()
    with pytest.raises(ValueError, match="3D"):
        model(torch.randn(2, 4, 16, 768))


def test_configurable_hidden_dim():
    """Configurable hidden_dim works correctly."""
    hidden_dim = 512
    model = TextWatermarkDecoder(hidden_dim=hidden_dim)
    model.eval()

    embeddings_w = torch.randn(3, 20, hidden_dim)
    with torch.no_grad():
        m_I_hat = model(embeddings_w)

    assert m_I_hat.shape == (3, 128), f"Expected (3, 128), got {m_I_hat.shape}"


def test_batch_size_one():
    """Works correctly with batch size of 1."""
    model = TextWatermarkDecoder()
    model.eval()

    embeddings_w = torch.randn(1, 10, 768)
    with torch.no_grad():
        m_I_hat = model(embeddings_w)

    assert m_I_hat.shape == (1, 128)


def test_seq_len_one():
    """Works correctly with seq_len of 1 (single token)."""
    model = TextWatermarkDecoder()
    model.eval()

    embeddings_w = torch.randn(2, 1, 768)
    with torch.no_grad():
        m_I_hat = model(embeddings_w)

    assert m_I_hat.shape == (2, 128)
    assert m_I_hat.min().item() >= 0.0
    assert m_I_hat.max().item() <= 1.0
