"""
Image Watermark Encoder
Embeds a 128-bit watermark into an image via residual addition.

Architecture:
  - ResNet-18 encoder path (pretrained=False) as feature extractor
  - 128-bit watermark projected to spatial feature map via linear layer, concatenated at bottleneck
  - 4-layer transposed-conv decoder with skip connections produces residual perturbation
  - I_w = image + alpha * f_theta(image, m_T)
  - alpha is a learnable scalar initialized to 0.03, clamped to [0.01, 0.05] in forward

Requirements: 2.1, 2.2, 2.3
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Minimal ResNet-18 building blocks (no torchvision dependency)
# ---------------------------------------------------------------------------

class BasicBlock(nn.Module):
    """ResNet-18 basic residual block (2 × 3×3 conv)."""

    def __init__(self, in_channels: int, out_channels: int, stride: int = 1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)

        self.downsample = None
        if stride != 1 or in_channels != out_channels:
            self.downsample = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels),
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = x
        out = F.relu(self.bn1(self.conv1(x)), inplace=True)
        out = self.bn2(self.conv2(out))
        if self.downsample is not None:
            identity = self.downsample(x)
        return F.relu(out + identity, inplace=True)


def _make_layer(in_channels: int, out_channels: int, blocks: int, stride: int = 1) -> nn.Sequential:
    layers = [BasicBlock(in_channels, out_channels, stride=stride)]
    for _ in range(1, blocks):
        layers.append(BasicBlock(out_channels, out_channels))
    return nn.Sequential(*layers)


# ---------------------------------------------------------------------------
# Image Watermark Encoder
# ---------------------------------------------------------------------------

class ImageWatermarkEncoder(nn.Module):
    """
    UNet-style CNN that embeds a 128-bit watermark into an image via residual addition.

    Encoder: ResNet-18 backbone (pretrained=False)
    Decoder: 4-layer transposed-conv with skip connections
    Watermark: projected to spatial feature map and concatenated at bottleneck

    Args:
        watermark_dim: dimension of the watermark vector (default 128)
        alpha: initial perturbation scale, clamped to [0.01, 0.05] in forward
    """

    def __init__(self, watermark_dim: int = 128, alpha: float = 0.03):
        super().__init__()

        # Learnable alpha scalar, initialized to 0.03
        self.alpha = nn.Parameter(torch.tensor(float(alpha)))

        # --- Encoder: ResNet-18 backbone (pretrained=False) ---
        # Stem: conv1 + bn1 + relu  → (B, 64, 112, 112)
        self.enc0 = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
        )
        # MaxPool                   → (B, 64, 56, 56)
        self.pool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        # Layer1                    → (B, 64, 56, 56)
        self.enc1 = _make_layer(64, 64, blocks=2, stride=1)
        # Layer2                    → (B, 128, 28, 28)
        self.enc2 = _make_layer(64, 128, blocks=2, stride=2)
        # Layer3                    → (B, 256, 14, 14)
        self.enc3 = _make_layer(128, 256, blocks=2, stride=2)
        # Layer4 (bottleneck)       → (B, 512, 7, 7)
        self.enc4 = _make_layer(256, 512, blocks=2, stride=2)

        # --- Watermark projection ---
        # Project (B, 128) → (B, 512) then expand to (B, 512, 7, 7)
        self.wm_proj = nn.Linear(watermark_dim, 512)

        # --- Decoder: 4-layer transposed-conv with skip connections ---
        # Input: 512 (bottleneck) + 512 (watermark) = 1024 → 256, 14×14
        self.dec4 = nn.Sequential(
            nn.ConvTranspose2d(1024, 256, kernel_size=2, stride=2),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
        )
        # Skip from enc3 (256 ch) → concat → 512 → 128, 28×28
        self.dec3 = nn.Sequential(
            nn.ConvTranspose2d(512, 128, kernel_size=2, stride=2),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
        )
        # Skip from enc2 (128 ch) → concat → 256 → 64, 56×56
        self.dec2 = nn.Sequential(
            nn.ConvTranspose2d(256, 64, kernel_size=2, stride=2),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
        )
        # Skip from enc1 (64 ch) → concat → 128 → 64, 112×112
        self.dec1 = nn.Sequential(
            nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
        )
        # Final upsample 112→224 and project to 3 channels (residual perturbation)
        self.dec0 = nn.Sequential(
            nn.ConvTranspose2d(64, 32, kernel_size=2, stride=2),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 3, kernel_size=3, padding=1),
            nn.Tanh(),
        )

    def forward(self, image: torch.Tensor, m_T: torch.Tensor) -> torch.Tensor:
        """
        Args:
            image: (B, 3, 224, 224) normalized image tensor
            m_T:   (B, 128) binary watermark vector (float32 0/1)
        Returns:
            I_w:   (B, 3, 224, 224) watermarked image, same shape as input
        Formula: I_w = image + alpha * f_theta(image, m_T)
        """
        # Clamp alpha to [0.01, 0.05]
        alpha = torch.clamp(self.alpha, 0.01, 0.05)

        # Encoder forward pass (collect skip features)
        s0 = self.enc0(image)          # (B, 64, 112, 112)
        s1 = self.enc1(self.pool(s0))  # (B, 64, 56, 56)
        s2 = self.enc2(s1)             # (B, 128, 28, 28)
        s3 = self.enc3(s2)             # (B, 256, 14, 14)
        bottleneck = self.enc4(s3)     # (B, 512, 7, 7)

        # Project watermark and expand to spatial map
        wm_feat = self.wm_proj(m_T)                                    # (B, 512)
        wm_feat = wm_feat.view(-1, 512, 1, 1).expand_as(bottleneck)   # (B, 512, 7, 7)

        # Concatenate watermark at bottleneck
        x = torch.cat([bottleneck, wm_feat], dim=1)  # (B, 1024, 7, 7)

        # Decoder with skip connections
        x = self.dec4(x)                   # (B, 256, 14, 14)
        x = torch.cat([x, s3], dim=1)      # (B, 512, 14, 14)
        x = self.dec3(x)                   # (B, 128, 28, 28)
        x = torch.cat([x, s2], dim=1)      # (B, 256, 28, 28)
        x = self.dec2(x)                   # (B, 64, 56, 56)
        x = torch.cat([x, s1], dim=1)      # (B, 128, 56, 56)
        x = self.dec1(x)                   # (B, 64, 112, 112)
        perturbation = self.dec0(x)        # (B, 3, 224, 224)

        # Residual addition: I_w = image + alpha * f_theta(image, m_T)
        I_w = image + alpha * perturbation
        return I_w
