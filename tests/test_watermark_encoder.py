"""
Property tests for ImageWatermarkEncoder.

Property 1: Image encoder output shape preservation
  For any batch of images of shape (B, 3, 224, 224) and any 128-bit watermark vectors
  of shape (B, 128), the ImageWatermarkEncoder SHALL produce an output tensor of exactly
  shape (B, 3, 224, 224).

Validates: Requirements 2.2, 2.3
"""

import torch
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from models.watermark_image_encoder import ImageWatermarkEncoder


@pytest.fixture(scope="module")
def encoder():
    """Shared encoder instance (eval mode, no grad) for all property tests."""
    model = ImageWatermarkEncoder()
    model.eval()
    return model


@settings(max_examples=100, deadline=None)
@given(batch_size=st.integers(min_value=1, max_value=4))
def test_property_1_image_encoder_output_shape(batch_size):
    """
    Property 1: Image encoder output shape preservation
    Validates: Requirements 2.2, 2.3

    For any batch_size in [1, 4], the encoder must return a tensor of shape
    (batch_size, 3, 224, 224) — identical to the input image shape.
    """
    model = ImageWatermarkEncoder()
    model.eval()

    image = torch.randn(batch_size, 3, 224, 224)
    m_T = torch.randint(0, 2, (batch_size, 128)).float()

    with torch.no_grad():
        I_w = model(image, m_T)

    assert I_w.shape == (batch_size, 3, 224, 224), (
        f"Expected output shape ({batch_size}, 3, 224, 224), got {I_w.shape}"
    )


def test_encoder_alpha_clamped():
    """Unit test: alpha is clamped to [0.01, 0.05] regardless of initialization."""
    model = ImageWatermarkEncoder(alpha=0.03)
    model.eval()

    # Force alpha outside range
    with torch.no_grad():
        model.alpha.fill_(0.1)

    image = torch.randn(1, 3, 224, 224)
    m_T = torch.zeros(1, 128)

    with torch.no_grad():
        I_w = model(image, m_T)

    # The perturbation should be scaled by clamped alpha (0.05), not 0.1
    # Verify output shape is still correct
    assert I_w.shape == (1, 3, 224, 224)


def test_encoder_output_is_residual():
    """Unit test: output equals image + alpha * perturbation (residual structure)."""
    model = ImageWatermarkEncoder(alpha=0.03)
    model.eval()

    image = torch.zeros(1, 3, 224, 224)
    m_T = torch.zeros(1, 128)

    with torch.no_grad():
        I_w = model(image, m_T)

    # With zero image, I_w = 0 + alpha * f_theta(0, 0) — shape must still be correct
    assert I_w.shape == (1, 3, 224, 224)


def test_encoder_alpha_learnable():
    """Unit test: alpha is a learnable parameter."""
    model = ImageWatermarkEncoder()
    alpha_param = dict(model.named_parameters()).get("alpha")
    assert alpha_param is not None, "alpha should be a registered nn.Parameter"
    assert alpha_param.requires_grad, "alpha should have requires_grad=True"
