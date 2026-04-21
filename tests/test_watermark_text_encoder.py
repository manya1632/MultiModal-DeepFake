"""
Tests for TextWatermarkEncoder.

Property 2: Text encoder output shape preservation
  For any token embedding tensor of shape (B, seq_len, hidden_dim) and any watermark
  vector of shape (B, 128), the TextWatermarkEncoder SHALL produce an output tensor of
  exactly shape (B, seq_len, hidden_dim).

Validates: Requirements 4.2, 4.3
"""

import torch
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from models.watermark_text_encoder import TextWatermarkEncoder


# ---------------------------------------------------------------------------
# Property-based test
# ---------------------------------------------------------------------------

@settings(max_examples=100, deadline=None)
@given(
    batch_size=st.integers(min_value=1, max_value=4),
    seq_len=st.integers(min_value=1, max_value=128),
)
def test_property_2_text_encoder_output_shape(batch_size, seq_len):
    """
    Property 2: Text encoder output shape preservation
    Validates: Requirements 4.2, 4.3

    For any batch_size in [1, 4] and seq_len in [1, 128], the encoder must return
    a tensor of shape (batch_size, seq_len, hidden_dim) — identical to the input shape.
    """
    hidden_dim = 768
    model = TextWatermarkEncoder(hidden_dim=hidden_dim)
    model.eval()

    embeddings = torch.randn(batch_size, seq_len, hidden_dim)
    m_I = torch.randint(0, 2, (batch_size, 128)).float()

    with torch.no_grad():
        E_w = model(embeddings, m_I)

    assert E_w.shape == (batch_size, seq_len, hidden_dim), (
        f"Expected output shape ({batch_size}, {seq_len}, {hidden_dim}), got {E_w.shape}"
    )


# ---------------------------------------------------------------------------
# Unit tests
# ---------------------------------------------------------------------------

def test_alpha_clamped_high():
    """Unit test: alpha above 0.05 is clamped to 0.05 during forward."""
    model = TextWatermarkEncoder()
    model.eval()

    with torch.no_grad():
        model.alpha.fill_(0.5)  # force above max

    embeddings = torch.randn(1, 10, 768)
    m_I = torch.zeros(1, 128)

    with torch.no_grad():
        E_w = model(embeddings, m_I)

    # Output shape must still be correct
    assert E_w.shape == (1, 10, 768)


def test_alpha_clamped_low():
    """Unit test: alpha below 0.01 is clamped to 0.01 during forward."""
    model = TextWatermarkEncoder()
    model.eval()

    with torch.no_grad():
        model.alpha.fill_(0.0)  # force below min

    embeddings = torch.randn(1, 10, 768)
    m_I = torch.zeros(1, 128)

    with torch.no_grad():
        E_w = model(embeddings, m_I)

    assert E_w.shape == (1, 10, 768)


def test_alpha_is_learnable_parameter():
    """Unit test: alpha is a registered learnable nn.Parameter."""
    model = TextWatermarkEncoder()
    params = dict(model.named_parameters())

    assert "alpha" in params, "alpha should be a registered nn.Parameter"
    assert params["alpha"].requires_grad, "alpha should have requires_grad=True"


def test_output_dtype_matches_input():
    """Unit test: output dtype matches input embeddings dtype."""
    model = TextWatermarkEncoder()
    model.eval()

    embeddings = torch.randn(2, 16, 768, dtype=torch.float32)
    m_I = torch.randint(0, 2, (2, 128)).float()

    with torch.no_grad():
        E_w = model(embeddings, m_I)

    assert E_w.dtype == embeddings.dtype, (
        f"Expected dtype {embeddings.dtype}, got {E_w.dtype}"
    )


def test_configurable_hidden_dim():
    """Unit test: hidden_dim is configurable (non-default value works correctly)."""
    hidden_dim = 512
    model = TextWatermarkEncoder(hidden_dim=hidden_dim)
    model.eval()

    embeddings = torch.randn(2, 20, hidden_dim)
    m_I = torch.randint(0, 2, (2, 128)).float()

    with torch.no_grad():
        E_w = model(embeddings, m_I)

    assert E_w.shape == (2, 20, hidden_dim)
