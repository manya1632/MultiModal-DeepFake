"""
Text Watermark Decoder
Extracts a 128-bit watermark from watermarked token embeddings via mean pooling + MLP.

Architecture:
  MeanPool(seq_len) → Linear(hidden_dim, 256) → GELU → Linear(256, 128) → Sigmoid

Requirements: 5.1
"""

import torch
import torch.nn as nn


class TextWatermarkDecoder(nn.Module):
    """
    Extracts the 128-bit watermark from watermarked token embeddings.

    Args:
        hidden_dim: dimension of the token embeddings (default 768 for BERT)
        watermark_dim: dimension of the output watermark vector (default 128)
    """

    def __init__(self, hidden_dim: int = 768, watermark_dim: int = 128):
        super().__init__()

        self.mlp = nn.Sequential(
            nn.Linear(hidden_dim, 256),
            nn.GELU(),
            nn.Linear(256, watermark_dim),
            nn.Sigmoid(),
        )

    def forward(self, embeddings_w: torch.Tensor) -> torch.Tensor:
        """
        Args:
            embeddings_w: (B, seq_len, hidden_dim) watermarked token embeddings
        Returns:
            m_I_hat: (B, 128) predicted watermark, values in [0, 1]
        Raises:
            ValueError: if embeddings_w is not a 3D tensor
        """
        if embeddings_w.dim() != 3:
            raise ValueError(
                f"Expected 3D input tensor of shape (B, seq_len, hidden_dim), "
                f"got {embeddings_w.dim()}D tensor with shape {tuple(embeddings_w.shape)}"
            )

        # Mean pool over seq_len: (B, seq_len, hidden_dim) -> (B, hidden_dim)
        pooled = embeddings_w.mean(dim=1)

        # MLP: (B, hidden_dim) -> (B, 128)
        m_I_hat = self.mlp(pooled)
        return m_I_hat
