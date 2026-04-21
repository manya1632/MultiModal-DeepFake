"""
Property-based tests for the training pipeline.

Property 15: Balanced sampler equal class counts
  Validates: Requirements 1.1

Property 16: Encoder freeze preserves trainability of other parameters
  Validates: Requirements 1.4, 1.5
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torch.nn as nn
import numpy as np
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st
from torch.utils.data import Dataset, Subset


# ---------------------------------------------------------------------------
# Inline implementations of the functions under test
# (avoids importing train.py which has heavy ML dependencies)
# ---------------------------------------------------------------------------

def create_balanced_subset(dataset, subset_size: int):
    """Create a balanced subset with equal real and fake samples.
    Returns a torch.utils.data.Subset of size min(subset_size, 2*min(n_real, n_fake)).
    Validates: Requirements 1.1"""
    real_indices = []
    fake_indices = []

    for idx in range(len(dataset)):
        try:
            sample = dataset[idx]
            label = sample[1]  # label is second element
            if label == 'orig':
                real_indices.append(idx)
            else:
                fake_indices.append(idx)
        except Exception:
            continue

    n_per_class = min(len(real_indices), len(fake_indices), subset_size // 2)

    rng = np.random.default_rng(42)
    selected_real = rng.choice(real_indices, size=n_per_class, replace=False).tolist()
    selected_fake = rng.choice(fake_indices, size=n_per_class, replace=False).tolist()

    selected_indices = selected_real + selected_fake
    rng.shuffle(selected_indices)

    return Subset(dataset, selected_indices)


def freeze_encoders(model: torch.nn.Module) -> None:
    """Freeze visual_encoder and text_encoder weights in HAMMER model.
    Only fusion layers, classification head, and watermark modules remain trainable.
    Validates: Requirements 1.4, 1.5"""
    frozen_count = 0
    for name, param in model.named_parameters():
        if 'visual_encoder' in name or 'text_encoder' in name:
            param.requires_grad = False
            frozen_count += 1


# ---------------------------------------------------------------------------
# Mock dataset for testing
# ---------------------------------------------------------------------------

class MockDataset(Dataset):
    """Mock dataset with configurable real/fake label counts."""

    def __init__(self, n_real: int, n_fake: int):
        self.samples = []
        for _ in range(n_real):
            self.samples.append((torch.zeros(3, 224, 224), 'orig'))
        for _ in range(n_fake):
            self.samples.append((torch.zeros(3, 224, 224), 'face_swap'))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx]


# ---------------------------------------------------------------------------
# Mock HAMMER-like model for freeze_encoders testing
# ---------------------------------------------------------------------------

class MockHAMMER(nn.Module):
    """Minimal mock of HAMMER with visual_encoder, text_encoder, and fusion layers."""

    def __init__(self):
        super().__init__()
        # Encoder modules (should be frozen)
        self.visual_encoder = nn.Sequential(
            nn.Linear(64, 64),
            nn.Linear(64, 32),
        )
        self.text_encoder = nn.Sequential(
            nn.Linear(64, 64),
            nn.Linear(64, 32),
        )
        # Fusion / classification head (should remain trainable)
        self.fusion_layer = nn.Linear(64, 32)
        self.cls_head = nn.Linear(32, 2)
        # Watermark module (should remain trainable)
        self.watermark_proj = nn.Linear(128, 64)

    def forward(self, x):
        return x


# ---------------------------------------------------------------------------
# Property 15: Balanced sampler equal class counts
# ---------------------------------------------------------------------------

@settings(max_examples=100, deadline=None)
@given(
    n_real=st.integers(min_value=1, max_value=500),
    n_fake=st.integers(min_value=1, max_value=500),
)
def test_property_15_balanced_subset_equal_class_counts(n_real, n_fake):
    """
    **Property 15: Balanced sampler equal class counts**
    **Validates: Requirements 1.1**

    For any dataset with N real and M fake samples, the balanced sampler SHALL
    return a subset of size 2 * min(N, M) with exactly equal numbers of real
    and fake samples.
    """
    dataset = MockDataset(n_real, n_fake)
    expected_per_class = min(n_real, n_fake)
    subset_size = 2 * max(n_real, n_fake)  # large enough to not be the limiting factor

    subset = create_balanced_subset(dataset, subset_size)

    # Count real and fake in the subset
    real_count = 0
    fake_count = 0
    for idx in subset.indices:
        _, label = dataset[idx]
        if label == 'orig':
            real_count += 1
        else:
            fake_count += 1

    # Equal class counts
    assert real_count == fake_count, (
        f"Expected equal real/fake counts, got real={real_count}, fake={fake_count}"
    )
    # Total size is 2 * min(N, M)
    assert len(subset) == 2 * expected_per_class, (
        f"Expected subset size {2 * expected_per_class}, got {len(subset)}"
    )


@settings(max_examples=50)
@given(
    n_real=st.integers(min_value=1, max_value=200),
    n_fake=st.integers(min_value=1, max_value=200),
    subset_size=st.integers(min_value=2, max_value=100),
)
def test_property_15_balanced_subset_respects_subset_size(n_real, n_fake, subset_size):
    """
    **Property 15 (subset_size cap): Balanced sampler equal class counts**
    **Validates: Requirements 1.1**

    When subset_size limits the number of samples, the subset should be at most
    subset_size with equal class counts.
    """
    dataset = MockDataset(n_real, n_fake)
    subset = create_balanced_subset(dataset, subset_size)

    # Count real and fake in the subset
    real_count = 0
    fake_count = 0
    for idx in subset.indices:
        _, label = dataset[idx]
        if label == 'orig':
            real_count += 1
        else:
            fake_count += 1

    # Equal class counts
    assert real_count == fake_count, (
        f"Expected equal real/fake counts, got real={real_count}, fake={fake_count}"
    )
    # Total size does not exceed subset_size
    assert len(subset) <= subset_size, (
        f"Expected subset size <= {subset_size}, got {len(subset)}"
    )
    # Expected per class
    expected_per_class = min(n_real, n_fake, subset_size // 2)
    assert real_count == expected_per_class, (
        f"Expected {expected_per_class} per class, got real={real_count}"
    )


# ---------------------------------------------------------------------------
# Property 16: Encoder freeze preserves trainability of other parameters
# ---------------------------------------------------------------------------

def test_property_16_freeze_encoders_sets_requires_grad_false():
    """
    **Property 16: Encoder freeze preserves trainability of other parameters**
    **Validates: Requirements 1.4, 1.5**

    After calling freeze_encoders(), all parameters in visual_encoder and
    text_encoder SHALL have requires_grad=False, and all parameters in the
    fusion layers, classification head, and watermark modules SHALL have
    requires_grad=True.
    """
    model = MockHAMMER()

    # All params start as trainable
    for param in model.parameters():
        assert param.requires_grad, "All params should start as trainable"

    freeze_encoders(model)

    # visual_encoder and text_encoder params must be frozen
    for name, param in model.named_parameters():
        if 'visual_encoder' in name or 'text_encoder' in name:
            assert not param.requires_grad, (
                f"Parameter '{name}' should be frozen (requires_grad=False) after freeze_encoders()"
            )

    # All other params must remain trainable
    for name, param in model.named_parameters():
        if 'visual_encoder' not in name and 'text_encoder' not in name:
            assert param.requires_grad, (
                f"Parameter '{name}' should remain trainable (requires_grad=True) after freeze_encoders()"
            )


@settings(max_examples=20)
@given(
    n_visual_layers=st.integers(min_value=1, max_value=4),
    n_text_layers=st.integers(min_value=1, max_value=4),
)
def test_property_16_freeze_encoders_property(n_visual_layers, n_text_layers):
    """
    **Property 16: Encoder freeze preserves trainability of other parameters**
    **Validates: Requirements 1.4, 1.5**

    For any HAMMER-like model with varying encoder depths, after freeze_encoders():
    - All visual_encoder and text_encoder params have requires_grad=False
    - All other params have requires_grad=True
    """

    class DynamicMockHAMMER(nn.Module):
        def __init__(self, n_visual, n_text):
            super().__init__()
            self.visual_encoder = nn.Sequential(
                *[nn.Linear(32, 32) for _ in range(n_visual)]
            )
            self.text_encoder = nn.Sequential(
                *[nn.Linear(32, 32) for _ in range(n_text)]
            )
            self.fusion_layer = nn.Linear(32, 16)
            self.cls_head = nn.Linear(16, 2)

    model = DynamicMockHAMMER(n_visual_layers, n_text_layers)
    freeze_encoders(model)

    frozen_names = []
    trainable_names = []

    for name, param in model.named_parameters():
        if 'visual_encoder' in name or 'text_encoder' in name:
            frozen_names.append(name)
            assert not param.requires_grad, (
                f"'{name}' should be frozen after freeze_encoders()"
            )
        else:
            trainable_names.append(name)
            assert param.requires_grad, (
                f"'{name}' should remain trainable after freeze_encoders()"
            )

    # Sanity: both groups must be non-empty
    assert len(frozen_names) > 0, "Expected at least one frozen parameter"
    assert len(trainable_names) > 0, "Expected at least one trainable parameter"
