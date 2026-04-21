"""
Image Watermark Decoder
Extracts a 128-bit watermark from a (possibly degraded) watermarked image.

Architecture:
  4-layer strided conv → GlobalAvgPool → Linear(512, 128) → Sigmoid

Input:  (B, 3, 224, 224) watermarked image tensor
Output: (B, 128) predicted watermark, values in [0, 1]

Requirements: 3.1
"""

import torch
import torch.nn as nn


class ImageWatermarkDecoder(nn.Module):
    """
    Lightweight CNN that extracts the 128-bit watermark from a watermarked image.

    Architecture:
        Conv(3→64, 3×3, stride=2)   → (B, 64, 112, 112)
        Conv(64→128, 3×3, stride=2) → (B, 128, 56, 56)
        Conv(128→256, 3×3, stride=2)→ (B, 256, 28, 28)
        Conv(256→512, 3×3, stride=2)→ (B, 512, 14, 14)
        GlobalAvgPool               → (B, 512)
        Linear(512, 128)            → (B, 128)
        Sigmoid                     → (B, 128), values in [0, 1]
    """

    def __init__(self):
        super().__init__()

        self.features = nn.Sequential(
            # Layer 1: (B, 3, 224, 224) → (B, 64, 112, 112)
            nn.Conv2d(3, 64, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            # Layer 2: (B, 64, 112, 112) → (B, 128, 56, 56)
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            # Layer 3: (B, 128, 56, 56) → (B, 256, 28, 28)
            nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            # Layer 4: (B, 256, 28, 28) → (B, 512, 14, 14)
            nn.Conv2d(256, 512, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
        )

        # GlobalAvgPool + classifier
        self.pool = nn.AdaptiveAvgPool2d(1)  # (B, 512, 1, 1)
        self.classifier = nn.Sequential(
            nn.Linear(512, 128),
            nn.Sigmoid(),
        )

    def forward(self, image_w: torch.Tensor) -> torch.Tensor:
        """
        Args:
            image_w: (B, 3, 224, 224) watermarked image tensor
        Returns:
            m_T_hat: (B, 128) predicted watermark, values in [0, 1]
        Raises:
            ValueError: if input is not 4D or does not have 3 channels
        """
        if image_w.ndim != 4:
            raise ValueError(
                f"Expected 4D input tensor (B, C, H, W), got {image_w.ndim}D tensor "
                f"with shape {tuple(image_w.shape)}"
            )
        if image_w.shape[1] != 3:
            raise ValueError(
                f"Expected 3 input channels (C=3), got C={image_w.shape[1]} "
                f"with shape {tuple(image_w.shape)}"
            )

        x = self.features(image_w)       # (B, 512, H', W')
        x = self.pool(x)                 # (B, 512, 1, 1)
        x = x.flatten(1)                 # (B, 512)
        m_T_hat = self.classifier(x)     # (B, 128)
        return m_T_hat
