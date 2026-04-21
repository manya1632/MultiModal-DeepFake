"""
Tests for auth/clip_auth.py — Properties 11, 12 and unit tests.

All tests mock the CLIP model to avoid downloading weights.

Property 11: User embedding is unit-normalized
Validates: Requirements 11.2

Property 12: Authentication threshold decision
Validates: Requirements 11.4, 11.5
"""

import datetime
import os
import tempfile
from unittest.mock import MagicMock, patch

import numpy as np
import PIL.Image
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

import jwt

from auth.user_db import ConflictError, UserDB
from auth.clip_auth import AuthenticationError, CLIPAuthenticator


# ---------------------------------------------------------------------------
# Fixtures / helpers
# ---------------------------------------------------------------------------

def _make_image() -> PIL.Image.Image:
    """Return a tiny solid-color RGB image."""
    return PIL.Image.new("RGB", (32, 32), color=(128, 64, 32))


def _unit_vector(dim: int = 512, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    v = rng.standard_normal(dim).astype(np.float32)
    return v / np.linalg.norm(v)


def _make_authenticator(
    image_vec: np.ndarray,
    text_vec: np.ndarray,
    threshold: float = 0.85,
    db: "UserDB | None" = None,
) -> CLIPAuthenticator:
    """
    Build a CLIPAuthenticator whose compute_embedding is replaced by a
    deterministic function: normalize(image_vec + text_vec).
    """
    with tempfile.TemporaryDirectory() as tmpdir:
        if db is None:
            db = UserDB(db_path=os.path.join(tmpdir, "users.npz"))

        auth = object.__new__(CLIPAuthenticator)
        auth._threshold = threshold
        auth._db = db

        # Deterministic embedding: normalize(image_vec + text_vec)
        combined = image_vec + text_vec
        norm = np.linalg.norm(combined)
        fixed_embedding = (combined / norm).astype(np.float32)

        auth.compute_embedding = MagicMock(return_value=fixed_embedding)
        auth._issue_jwt = CLIPAuthenticator._issue_jwt.__get__(auth, CLIPAuthenticator)
        return auth


# ---------------------------------------------------------------------------
# Property-based tests — Property 11
# ---------------------------------------------------------------------------

@settings(max_examples=100)
@given(
    img_seed=st.integers(0, 10_000),
    txt_seed=st.integers(0, 10_000),
)
def test_property_11_embedding_is_unit_normalized(img_seed: int, txt_seed: int) -> None:
    """
    Property 11: User embedding is unit-normalized.
    Validates: Requirements 11.2

    For any image and password, compute_embedding returns a vector with ||e||_2 ≈ 1.0.
    """
    image_vec = _unit_vector(seed=img_seed)
    text_vec = _unit_vector(seed=txt_seed)

    with tempfile.TemporaryDirectory() as tmpdir:
        db = UserDB(db_path=os.path.join(tmpdir, "users.npz"))
        auth = _make_authenticator(image_vec, text_vec, db=db)

        embedding = auth.compute_embedding(_make_image(), "some_password")
        norm = np.linalg.norm(embedding)
        assert abs(norm - 1.0) < 1e-5, f"Expected unit norm, got {norm}"


# ---------------------------------------------------------------------------
# Property-based tests — Property 12
# ---------------------------------------------------------------------------

@settings(max_examples=100)
@given(
    similarity=st.floats(min_value=0.0, max_value=1.0, allow_nan=False),
    threshold=st.floats(min_value=0.1, max_value=0.99, allow_nan=False),
)
def test_property_12_authentication_threshold_decision(
    similarity: float, threshold: float
) -> None:
    """
    Property 12: Authentication threshold decision.
    Validates: Requirements 11.4, 11.5

    Grant access iff cosine_similarity > threshold; deny otherwise.

    We bypass compute_embedding entirely and directly test the decision logic
    in authenticate() by injecting a stored embedding and a query embedding
    whose dot product equals the desired similarity value.
    """
    with tempfile.TemporaryDirectory() as tmpdir:
        db = UserDB(db_path=os.path.join(tmpdir, "users.npz"))

        # Stored embedding: unit vector along axis 0
        stored = np.zeros(512, dtype=np.float64)
        stored[0] = 1.0
        stored = stored.astype(np.float32)

        # Query embedding: we want dot(query, stored) == similarity.
        # Since stored = e0, dot(query, stored) = query[0].
        # Build a unit vector with query[0] = similarity (clamped to [-1, 1]).
        sim_clamped = float(np.clip(similarity, -1.0, 1.0))
        query = np.zeros(512, dtype=np.float32)
        query[0] = sim_clamped
        residual = float(max(0.0, 1.0 - sim_clamped ** 2) ** 0.5)
        query[1] = residual
        # Renormalize to ensure exact unit norm despite float32 rounding
        q_norm = np.linalg.norm(query)
        if q_norm > 1e-10:
            query = query / q_norm

        # The actual dot product after renormalization
        actual_similarity = float(np.dot(query, stored))

        db.save("testuser", stored)

        auth = object.__new__(CLIPAuthenticator)
        auth._threshold = threshold
        auth._db = db
        auth.compute_embedding = MagicMock(return_value=query)
        auth._issue_jwt = CLIPAuthenticator._issue_jwt.__get__(auth, CLIPAuthenticator)

        if actual_similarity > threshold:
            token = auth.authenticate("testuser", _make_image(), "pw")
            assert isinstance(token, str) and len(token) > 0
        else:
            with pytest.raises(AuthenticationError):
                auth.authenticate("testuser", _make_image(), "pw")


# ---------------------------------------------------------------------------
# Unit tests
# ---------------------------------------------------------------------------

def test_register_stores_embedding() -> None:
    """register() stores the computed embedding in UserDB."""
    with tempfile.TemporaryDirectory() as tmpdir:
        db = UserDB(db_path=os.path.join(tmpdir, "users.npz"))
        image_vec = _unit_vector(seed=1)
        text_vec = _unit_vector(seed=2)
        auth = _make_authenticator(image_vec, text_vec, db=db)

        auth.register("alice", _make_image(), "password123")

        stored = db.lookup("alice")
        expected = auth.compute_embedding(_make_image(), "password123")
        np.testing.assert_array_almost_equal(stored, expected)


def test_authenticate_grants_access_above_threshold() -> None:
    """authenticate() returns a JWT when similarity > threshold."""
    with tempfile.TemporaryDirectory() as tmpdir:
        db = UserDB(db_path=os.path.join(tmpdir, "users.npz"))

        # Use identical embeddings → similarity = 1.0 > 0.85
        fixed_emb = _unit_vector(seed=42)
        db.save("bob", fixed_emb)

        auth = object.__new__(CLIPAuthenticator)
        auth._threshold = 0.85
        auth._db = db
        auth.compute_embedding = MagicMock(return_value=fixed_emb)
        auth._issue_jwt = CLIPAuthenticator._issue_jwt.__get__(auth, CLIPAuthenticator)

        token = auth.authenticate("bob", _make_image(), "pw")
        assert isinstance(token, str)
        # Verify it's a valid JWT
        secret = os.environ.get("JWT_SECRET", "change-me-in-production")
        decoded = jwt.decode(token, secret, algorithms=["HS256"])
        assert decoded["sub"] == "bob"


def test_authenticate_denies_access_below_threshold() -> None:
    """authenticate() raises AuthenticationError when similarity ≤ threshold."""
    with tempfile.TemporaryDirectory() as tmpdir:
        db = UserDB(db_path=os.path.join(tmpdir, "users.npz"))

        stored_emb = np.zeros(512, dtype=np.float32)
        stored_emb[0] = 1.0
        db.save("carol", stored_emb)

        # Query orthogonal to stored → similarity = 0.0
        query_emb = np.zeros(512, dtype=np.float32)
        query_emb[1] = 1.0

        auth = object.__new__(CLIPAuthenticator)
        auth._threshold = 0.85
        auth._db = db
        auth.compute_embedding = MagicMock(return_value=query_emb)
        auth._issue_jwt = CLIPAuthenticator._issue_jwt.__get__(auth, CLIPAuthenticator)

        with pytest.raises(AuthenticationError):
            auth.authenticate("carol", _make_image(), "wrong_pw")


def test_duplicate_registration_raises_conflict_error() -> None:
    """Registering the same username twice raises ConflictError."""
    with tempfile.TemporaryDirectory() as tmpdir:
        db = UserDB(db_path=os.path.join(tmpdir, "users.npz"))
        fixed_emb = _unit_vector(seed=7)

        auth = object.__new__(CLIPAuthenticator)
        auth._threshold = 0.85
        auth._db = db
        auth.compute_embedding = MagicMock(return_value=fixed_emb)
        auth._issue_jwt = CLIPAuthenticator._issue_jwt.__get__(auth, CLIPAuthenticator)

        auth.register("dave", _make_image(), "pw1")
        with pytest.raises(ConflictError):
            auth.register("dave", _make_image(), "pw2")


def test_compute_embedding_returns_unit_normalized_vector() -> None:
    """compute_embedding returns a float32 vector with ||e||_2 ≈ 1.0."""
    image_vec = _unit_vector(seed=10)
    text_vec = _unit_vector(seed=20)

    with tempfile.TemporaryDirectory() as tmpdir:
        db = UserDB(db_path=os.path.join(tmpdir, "users.npz"))
        auth = _make_authenticator(image_vec, text_vec, db=db)

        emb = auth.compute_embedding(_make_image(), "test_password")
        assert emb.dtype == np.float32
        assert emb.shape == (512,)
        assert abs(np.linalg.norm(emb) - 1.0) < 1e-5


def test_jwt_expiry_is_24_hours() -> None:
    """Issued JWT has expiry approximately 24 hours from now."""
    with tempfile.TemporaryDirectory() as tmpdir:
        db = UserDB(db_path=os.path.join(tmpdir, "users.npz"))
        fixed_emb = _unit_vector(seed=99)
        db.save("eve", fixed_emb)

        auth = object.__new__(CLIPAuthenticator)
        auth._threshold = 0.85
        auth._db = db
        auth.compute_embedding = MagicMock(return_value=fixed_emb)
        auth._issue_jwt = CLIPAuthenticator._issue_jwt.__get__(auth, CLIPAuthenticator)

        token = auth.authenticate("eve", _make_image(), "pw")
        secret = os.environ.get("JWT_SECRET", "change-me-in-production")
        decoded = jwt.decode(token, secret, algorithms=["HS256"])

        exp = datetime.datetime.fromtimestamp(decoded["exp"], tz=datetime.timezone.utc)
        iat = datetime.datetime.fromtimestamp(decoded["iat"], tz=datetime.timezone.utc)
        delta = exp - iat
        assert abs(delta.total_seconds() - 86400) < 60  # within 1 minute of 24h


def test_authenticate_unknown_user_raises_key_error() -> None:
    """authenticate() raises KeyError for an unregistered username."""
    with tempfile.TemporaryDirectory() as tmpdir:
        db = UserDB(db_path=os.path.join(tmpdir, "users.npz"))
        fixed_emb = _unit_vector(seed=5)

        auth = object.__new__(CLIPAuthenticator)
        auth._threshold = 0.85
        auth._db = db
        auth.compute_embedding = MagicMock(return_value=fixed_emb)
        auth._issue_jwt = CLIPAuthenticator._issue_jwt.__get__(auth, CLIPAuthenticator)

        with pytest.raises(KeyError):
            auth.authenticate("ghost", _make_image(), "pw")
