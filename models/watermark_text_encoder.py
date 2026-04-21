"""
Text Watermark Encoder
Embeds a watermark into BERT token embeddings via a learned linear projection.

Architecture:
  - P: Linear(watermark_dim=128, hidden_dim=768) learned projection
  - alpha: learnable scalar initialized to 0.03, clamped to [0.01, 0.05] in forward
  - E_w = embeddings + alpha * P(m_I).unsqueeze(1).expand_as(embeddings)

Requirements: 4.1, 4.2, 4.3
"""

import torch
import torch.nn as nn


class TextWatermarkEncoder(nn.Module):
    """
    Projection-layer encoder that adds a watermark perturbation to BERT token embeddings.

    Args:
        hidden_dim: dimension of the token embeddings (default 768 for BERT)
        watermark_dim: dimension of the watermark vector (default 128)
        alpha: initial perturbation scale, clamped to [0.01, 0.05] in forward
    """

    def __init__(self, hidden_dim: int = 768, watermark_dim: int = 128, alpha: float = 0.03):
        super().__init__()

        # Learnable alpha scalar, initialized to 0.03
        self.alpha = nn.Parameter(torch.tensor(float(alpha)))

        # Learned linear projection: (B, 128) -> (B, hidden_dim)
        self.projection = nn.Linear(watermark_dim, hidden_dim)

    def forward(self, embeddings: torch.Tensor, m_I: torch.Tensor) -> torch.Tensor:
        """
        Args:
            embeddings: (B, seq_len, hidden_dim) BERT token embeddings
            m_I:        (B, 128) watermark vector derived from image features
        Returns:
            E_w: (B, seq_len, hidden_dim) watermarked embeddings, same shape as input
        Formula: E_w = embeddings + alpha * P(m_I).unsqueeze(1).expand_as(embeddings)
        """
        # Clamp alpha to [0.01, 0.05]
        alpha = torch.clamp(self.alpha, 0.01, 0.05)

        # Project watermark: (B, 128) -> (B, hidden_dim)
        wm_proj = self.projection(m_I)  # (B, hidden_dim)

        # Expand to match embeddings shape: (B, hidden_dim) -> (B, 1, hidden_dim) -> (B, seq_len, hidden_dim)
        wm_expanded = wm_proj.unsqueeze(1).expand_as(embeddings)

        # Residual addition
        E_w = embeddings + alpha * wm_expanded
        return E_w
