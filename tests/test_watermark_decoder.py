"""
Unit tests for ImageWatermarkDecoder.

Covers:
  - Output shape is (B, 128) for valid input
  - Output values are in [0, 1]
  - ValueError raised on wrong input shape (non-4D or C != 3)

Requirements: 3.1
"""

import torch
import pytest

from models.watermark_image_decoder import ImageWatermarkDecoder


@pytest.fixture(scope="module")
def decoder():
    model = ImageWatermarkDecoder()
    model.eval()
    return model


# ---------------------------------------------------------------------------
# Output shape tests
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("batch_size", [1, 2, 4])
def test_output_shape(decoder, batch_size):
    """Output shape must be (B, 128) for valid (B, 3, 224, 224) input."""
    image_w = torch.randn(batch_size, 3, 224, 224)
    with torch.no_grad():
        m_T_hat = decoder(image_w)
    assert m_T_hat.shape == (batch_size, 128), (
        f"Expected ({batch_size}, 128), got {m_T_hat.shape}"
    )


def test_output_shape_single_image(decoder):
    """Single image (B=1) produces shape (1, 128)."""
    image_w = torch.randn(1, 3, 224, 224)
    with torch.no_grad():
        out = decoder(image_w)
    assert out.shape == (1, 128)


# ---------------------------------------------------------------------------
# Output value range tests
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("batch_size", [1, 3])
def test_output_values_in_range(decoder, batch_size):
    """All output values must be in [0, 1] (Sigmoid output)."""
    image_w = torch.randn(batch_size, 3, 224, 224)
    with torch.no_grad():
        m_T_hat = decoder(image_w)
    assert m_T_hat.min().item() >= 0.0, "Output contains values below 0"
    assert m_T_hat.max().item() <= 1.0, "Output contains values above 1"


def test_output_values_extreme_input(decoder):
    """Values stay in [0, 1] even for extreme input values."""
    image_w = torch.full((1, 3, 224, 224), 1e6)
    with torch.no_grad():
        out = decoder(image_w)
    assert out.min().item() >= 0.0
    assert out.max().item() <= 1.0


# ---------------------------------------------------------------------------
# ValueError on wrong input shape
# ---------------------------------------------------------------------------

def test_raises_on_3d_input(decoder):
    """ValueError raised when input is 3D (missing batch dimension)."""
    bad_input = torch.randn(3, 224, 224)
    with pytest.raises(ValueError, match="4D"):
        decoder(bad_input)


def test_raises_on_2d_input(decoder):
    """ValueError raised when input is 2D."""
    bad_input = torch.randn(224, 224)
    with pytest.raises(ValueError, match="4D"):
        decoder(bad_input)


def test_raises_on_wrong_channels(decoder):
    """ValueError raised when input has C != 3."""
    bad_input = torch.randn(2, 1, 224, 224)
    with pytest.raises(ValueError, match="3"):
        decoder(bad_input)


def test_raises_on_4_channels(decoder):
    """ValueError raised when input has 4 channels (e.g. RGBA)."""
    bad_input = torch.randn(1, 4, 224, 224)
    with pytest.raises(ValueError, match="3"):
        decoder(bad_input)


def test_raises_on_5d_input(decoder):
    """ValueError raised when input is 5D."""
    bad_input = torch.randn(1, 1, 3, 224, 224)
    with pytest.raises(ValueError, match="4D"):
        decoder(bad_input)
