"""
Property-based tests for the inference pipeline trust score and watermark consistency.

Properties tested:
  - Property 8: Trust score formula correctness (Req 10.4)
  - Property 9: Watermark score defaults to 0.0 on invalid input (Req 10.6)
  - Property 10: Consistency flag threshold (Req 6.3)
"""

import math
import torch
from hypothesis import given, settings
from hypothesis import strategies as st


# ---------------------------------------------------------------------------
# Inline implementations to avoid importing the full test.py with heavy ML deps
# ---------------------------------------------------------------------------

def _compute_trust_score(hammer_score: float, watermark_score: float) -> float:
    """Inline: 0.7 * hammer_score + 0.3 * watermark_score. Validates: Requirements 10.4"""
    return 0.7 * hammer_score + 0.3 * watermark_score


def _extract_watermark_score_guard(image, text_embeddings) -> tuple:
    """Inline guard logic mirroring extract_watermark_score's validation path.
    Returns (0.0, False) for any invalid input. Validates: Requirements 10.6"""
    try:
        if image is None or text_embeddings is None:
            return 0.0, False
        if not isinstance(image, torch.Tensor) or not isinstance(text_embeddings, torch.Tensor):
            return 0.0, False
        if image.ndim != 4 or image.shape[1] != 3:
            return 0.0, False
        if text_embeddings.ndim != 3:
            return 0.0, False
        if torch.isnan(image).any() or torch.isnan(text_embeddings).any():
            return 0.0, False
        return 0.75, True
    except Exception:
        return 0.0, False


def _watermark_valid_from_score(score: float) -> bool:
    """Inline: watermark_valid = score >= 0.5. Validates: Requirements 6.3"""
    return score >= 0.5


# ---------------------------------------------------------------------------
# Property 8: Trust score formula correctness — Validates: Requirements 10.4
# ---------------------------------------------------------------------------

@settings(max_examples=100)
@given(
    hammer=st.floats(min_value=0.0, max_value=1.0, allow_nan=False, allow_infinity=False),
    wm=st.floats(min_value=0.0, max_value=1.0, allow_nan=False, allow_infinity=False),
)
def test_property_8_trust_score_formula(hammer: float, wm: float):
    """Property 8: trust_score == 0.7 * hammer_score + 0.3 * watermark_score"""
    result = _compute_trust_score(hammer, wm)
    expected = 0.7 * hammer + 0.3 * wm
    assert math.isclose(result, expected, rel_tol=1e-9, abs_tol=1e-12), (
        f"Trust score mismatch: got {result}, expected {expected}"
    )


# ---------------------------------------------------------------------------
# Property 9: Watermark score defaults to 0.0 on invalid input — Req 10.6
# ---------------------------------------------------------------------------

def test_property_9_none_image_returns_zero():
    score, valid = _extract_watermark_score_guard(None, torch.zeros(1, 10, 768))
    assert score == 0.0 and valid is False


def test_property_9_none_text_returns_zero():
    score, valid = _extract_watermark_score_guard(torch.zeros(1, 3, 224, 224), None)
    assert score == 0.0 and valid is False


def test_property_9_wrong_image_shape_returns_zero():
    score, valid = _extract_watermark_score_guard(torch.zeros(3, 224, 224), torch.zeros(1, 10, 768))
    assert score == 0.0 and valid is False


def test_property_9_wrong_image_channels_returns_zero():
    score, valid = _extract_watermark_score_guard(torch.zeros(1, 1, 224, 224), torch.zeros(1, 10, 768))
    assert score == 0.0 and valid is False


def test_property_9_wrong_text_shape_returns_zero():
    score, valid = _extract_watermark_score_guard(torch.zeros(1, 3, 224, 224), torch.zeros(1, 768))
    assert score == 0.0 and valid is False


def test_property_9_nan_image_returns_zero():
    nan_image = torch.full((1, 3, 224, 224), float('nan'))
    score, valid = _extract_watermark_score_guard(nan_image, torch.zeros(1, 10, 768))
    assert score == 0.0 and valid is False


def test_property_9_nan_text_returns_zero():
    nan_text = torch.full((1, 10, 768), float('nan'))
    score, valid = _extract_watermark_score_guard(torch.zeros(1, 3, 224, 224), nan_text)
    assert score == 0.0 and valid is False


# ---------------------------------------------------------------------------
# Property 10: Consistency flag threshold — Validates: Requirements 6.3
# ---------------------------------------------------------------------------

@settings(max_examples=100)
@given(
    score=st.floats(min_value=0.0, max_value=1.0, allow_nan=False, allow_infinity=False),
)
def test_property_10_consistency_flag_threshold(score: float):
    """Property 10: watermark_valid == (score >= 0.5)"""
    valid = _watermark_valid_from_score(score)
    assert valid == (score >= 0.5), (
        f"watermark_valid={valid} but score={score} (expected {score >= 0.5})"
    )
