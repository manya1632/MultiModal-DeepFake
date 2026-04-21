"""
CLIP-based user authentication module.

Users register with a profile image + password; authentication computes cosine
similarity between the query embedding and the stored embedding.

Requirements: 11.1, 11.2, 11.3, 11.4, 11.5, 11.6, 11.7
"""

import datetime
import os
from typing import Optional

import numpy as np
import PIL.Image

import jwt

from auth.user_db import ConflictError, UserDB


class AuthenticationError(Exception):
    """Raised when authentication fails (similarity ≤ threshold)."""


class CLIPAuthenticator:
    """
    CLIP-based access control.

    Registers users by storing a unit-normalized embedding derived from
    normalize(CLIP_image(image) + CLIP_text(password)).

    Authenticates by comparing cosine similarity to the stored embedding.
    Returns a signed JWT on success.
    """

    def __init__(
        self,
        model_name: str = "openai/clip-vit-base-patch32",
        threshold: float = 0.85,
        db: Optional[UserDB] = None,
    ) -> None:
        """
        Load frozen CLIP model. Weights are never updated.

        Args:
            model_name: HuggingFace model identifier for CLIP.
            threshold: Cosine similarity threshold for authentication.
            db: Optional UserDB instance (creates a default one if not provided).
        """
        from transformers import CLIPModel, CLIPProcessor  # lazy import

        self._threshold = threshold
        self._db = db if db is not None else UserDB()

        self._processor = CLIPProcessor.from_pretrained(model_name)
        self._model = CLIPModel.from_pretrained(model_name)
        self._model.eval()
        # Freeze all CLIP weights — never fine-tuned
        for param in self._model.parameters():
            param.requires_grad = False

    # ------------------------------------------------------------------
    # Core embedding
    # ------------------------------------------------------------------

    def compute_embedding(self, image: PIL.Image.Image, password: str) -> np.ndarray:
        """
        Compute a unit-normalized float32 embedding of shape (512,).

        Formula: normalize(CLIP_image(image) + CLIP_text(password))

        Args:
            image: PIL Image (any mode; will be converted to RGB internally).
            password: Plain-text password string.

        Returns:
            Unit-normalized float32 numpy array of shape (512,).
        """
        import torch

        inputs = self._processor(
            text=[password],
            images=image,
            return_tensors="pt",
            padding=True,
        )

        with torch.no_grad():
            image_features = self._model.get_image_features(
                pixel_values=inputs["pixel_values"]
            )  # (1, 512)
            text_features = self._model.get_text_features(
                input_ids=inputs["input_ids"],
                attention_mask=inputs["attention_mask"],
            )  # (1, 512)

        combined = image_features[0].float().numpy() + text_features[0].float().numpy()
        norm = np.linalg.norm(combined)
        if norm < 1e-10:
            # Degenerate case: return a zero-safe unit vector
            combined = np.ones(combined.shape, dtype=np.float32)
            norm = np.linalg.norm(combined)
        return (combined / norm).astype(np.float32)

    # ------------------------------------------------------------------
    # Registration & authentication
    # ------------------------------------------------------------------

    def register(self, username: str, image: PIL.Image.Image, password: str) -> None:
        """
        Register a new user.

        Raises:
            ConflictError: if username already exists.
        """
        embedding = self.compute_embedding(image, password)
        self._db.save(username, embedding)
        self._db.persist()

    def authenticate(self, username: str, image: PIL.Image.Image, password: str) -> str:
        """
        Authenticate a user and return a signed JWT token.

        Args:
            username: The username to authenticate.
            image: Profile image for embedding computation.
            password: Password string for embedding computation.

        Returns:
            Signed JWT string (expiry: 24 hours).

        Raises:
            KeyError: if username is not registered.
            AuthenticationError: if cosine similarity ≤ threshold.
        """
        stored = self._db.lookup(username)  # raises KeyError if not found
        query = self.compute_embedding(image, password)

        similarity = float(np.dot(query, stored))  # both are unit-normalized

        if similarity > self._threshold:
            return self._issue_jwt(username)
        else:
            raise AuthenticationError(
                f"Authentication failed for '{username}': "
                f"similarity {similarity:.4f} ≤ threshold {self._threshold:.4f}."
            )

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _issue_jwt(self, username: str) -> str:
        """Issue a signed JWT with 24-hour expiry."""
        secret = os.environ.get("JWT_SECRET", "change-me-in-production")
        now = datetime.datetime.now(tz=datetime.timezone.utc)
        payload = {
            "sub": username,
            "iat": now,
            "exp": now + datetime.timedelta(hours=24),
        }
        return jwt.encode(payload, secret, algorithm="HS256")
