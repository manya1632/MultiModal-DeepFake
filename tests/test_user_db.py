"""
Tests for auth/user_db.py — Property 13 and unit tests.

Property 13: Username conflict detection
Validates: Requirements 12.4
"""

import os
import tempfile

import numpy as np
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from auth.user_db import ConflictError, UserDB


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _random_embedding(dim: int = 512) -> np.ndarray:
    v = np.random.randn(dim).astype(np.float32)
    return v / np.linalg.norm(v)


def _make_db(tmp_path: str) -> UserDB:
    """Return a fresh UserDB backed by a non-existent file in tmp_path."""
    return UserDB(db_path=os.path.join(tmp_path, "users.npz"))


# ---------------------------------------------------------------------------
# Property-based tests — Property 13
# ---------------------------------------------------------------------------

@settings(max_examples=100)
@given(
    username=st.text(min_size=1, max_size=64, alphabet=st.characters(whitelist_categories=("Lu", "Ll", "Nd"))),
)
def test_property_13_conflict_raises_and_preserves_original(username: str) -> None:
    """
    Property 13: Username conflict detection.
    Validates: Requirements 12.4

    Register a username, attempt to register again, assert ConflictError
    and that the original embedding is unchanged.
    """
    with tempfile.TemporaryDirectory() as tmpdir:
        db = _make_db(tmpdir)

        original_embedding = _random_embedding()
        db.save(username, original_embedding)

        second_embedding = _random_embedding()
        with pytest.raises(ConflictError):
            db.save(username, second_embedding)

        # Original embedding must be unchanged
        stored = db.lookup(username)
        np.testing.assert_array_equal(stored, original_embedding)


# ---------------------------------------------------------------------------
# Unit tests
# ---------------------------------------------------------------------------

def test_persist_and_reload_cycle() -> None:
    """Persist/reload cycle preserves all records."""
    with tempfile.TemporaryDirectory() as tmpdir:
        db_path = os.path.join(tmpdir, "users.npz")
        db = UserDB(db_path=db_path)

        emb_alice = _random_embedding()
        emb_bob = _random_embedding()
        db.save("alice", emb_alice)
        db.save("bob", emb_bob)
        db.persist()

        # Reload from disk
        db2 = UserDB(db_path=db_path)
        np.testing.assert_array_almost_equal(db2.lookup("alice"), emb_alice)
        np.testing.assert_array_almost_equal(db2.lookup("bob"), emb_bob)


def test_lookup_missing_user_raises_key_error() -> None:
    """Looking up a non-existent username raises KeyError."""
    with tempfile.TemporaryDirectory() as tmpdir:
        db = _make_db(tmpdir)
        with pytest.raises(KeyError):
            db.lookup("nonexistent_user")


def test_corrupted_file_initializes_empty_store(caplog: pytest.LogCaptureFixture) -> None:
    """A corrupted .npz file causes UserDB to initialize an empty store and log a warning."""
    with tempfile.TemporaryDirectory() as tmpdir:
        db_path = os.path.join(tmpdir, "users.npz")
        # Write garbage bytes to simulate corruption
        with open(db_path, "wb") as f:
            f.write(b"this is not a valid npz file \x00\xff\xfe")

        import logging
        with caplog.at_level(logging.WARNING, logger="auth.user_db"):
            db = UserDB(db_path=db_path)

        assert len(db) == 0
        assert any("failed to load" in record.message or "Initializing empty" in record.message
                   for record in caplog.records)


def test_missing_file_initializes_empty_store(caplog: pytest.LogCaptureFixture) -> None:
    """A missing .npz file causes UserDB to initialize an empty store and log a warning."""
    with tempfile.TemporaryDirectory() as tmpdir:
        db_path = os.path.join(tmpdir, "does_not_exist.npz")

        import logging
        with caplog.at_level(logging.WARNING, logger="auth.user_db"):
            db = UserDB(db_path=db_path)

        assert len(db) == 0
        assert any("not found" in record.message or "Initializing empty" in record.message
                   for record in caplog.records)


def test_save_and_lookup_roundtrip() -> None:
    """Saved embedding can be retrieved via lookup."""
    with tempfile.TemporaryDirectory() as tmpdir:
        db = _make_db(tmpdir)
        emb = _random_embedding()
        db.save("user1", emb)
        retrieved = db.lookup("user1")
        np.testing.assert_array_almost_equal(retrieved, emb)


def test_persist_is_atomic(tmp_path: "os.PathLike") -> None:
    """persist() writes atomically — no partial file left on success."""
    db_path = str(tmp_path / "users.npz")
    db = UserDB(db_path=db_path)
    db.save("charlie", _random_embedding())
    db.persist()
    assert os.path.exists(db_path)
    # No leftover .tmp files
    tmp_files = [f for f in os.listdir(str(tmp_path)) if f.endswith(".tmp")]
    assert tmp_files == []


def test_duplicate_registration_does_not_overwrite() -> None:
    """ConflictError is raised and the original record is preserved."""
    with tempfile.TemporaryDirectory() as tmpdir:
        db = _make_db(tmpdir)
        original = _random_embedding()
        db.save("dave", original)

        intruder = _random_embedding()
        with pytest.raises(ConflictError):
            db.save("dave", intruder)

        stored = db.lookup("dave")
        np.testing.assert_array_equal(stored, original)
