"""
Tests for integrity/model_integrity.py — Property 14 and unit tests.

Property 14: Model integrity hash verification
Validates: Requirements 13.2, 13.3
"""

import json
import os
import tempfile

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from integrity.model_integrity import (
    ModelIntegrityError,
    compute_file_hash,
    save_hashes,
    verify_model,
)


# ---------------------------------------------------------------------------
# Property-based tests — Property 14
# ---------------------------------------------------------------------------

@settings(max_examples=100)
@given(content=st.binary(min_size=1, max_size=4096))
def test_property_14_modified_file_raises_integrity_error(content: bytes) -> None:
    """Property 14: Modified file raises ModelIntegrityError. Validates: Req 13.2, 13.3"""
    with tempfile.NamedTemporaryFile(delete=False) as f:
        f.write(content)
        path = f.name
    try:
        expected_hash = compute_file_hash(path)
        modified = bytes([content[0] ^ 0xFF]) + content[1:]
        with open(path, "wb") as f:
            f.write(modified)
        with pytest.raises(ModelIntegrityError):
            verify_model(path, expected_hash)
    finally:
        os.unlink(path)


@settings(max_examples=100)
@given(content=st.binary(min_size=0, max_size=4096))
def test_property_14_unmodified_file_passes(content: bytes) -> None:
    """Property 14: Unmodified file passes without raising. Validates: Req 13.2, 13.3"""
    with tempfile.NamedTemporaryFile(delete=False) as f:
        f.write(content)
        path = f.name
    try:
        expected_hash = compute_file_hash(path)
        verify_model(path, expected_hash)  # must not raise
    finally:
        os.unlink(path)


# ---------------------------------------------------------------------------
# Unit tests
# ---------------------------------------------------------------------------

def test_save_hashes_writes_valid_json() -> None:
    with tempfile.TemporaryDirectory() as tmpdir:
        model_a = os.path.join(tmpdir, "model_a.pt")
        model_b = os.path.join(tmpdir, "model_b.pt")
        with open(model_a, "wb") as f:
            f.write(b"weights_a")
        with open(model_b, "wb") as f:
            f.write(b"weights_b")
        hash_file = os.path.join(tmpdir, "hashes.json")
        save_hashes({"model_a": model_a, "model_b": model_b}, hash_file)
        with open(hash_file) as f:
            data = json.load(f)
        assert set(data.keys()) == {"model_a", "model_b"}
        assert data["model_a"] == compute_file_hash(model_a)
        assert data["model_b"] == compute_file_hash(model_b)


def test_verify_model_passes_for_unmodified_file() -> None:
    with tempfile.NamedTemporaryFile(delete=False) as f:
        f.write(b"model weights data")
        path = f.name
    try:
        expected = compute_file_hash(path)
        verify_model(path, expected)
    finally:
        os.unlink(path)


def test_verify_model_raises_on_mismatch() -> None:
    with tempfile.NamedTemporaryFile(delete=False) as f:
        f.write(b"original data")
        path = f.name
    try:
        wrong_hash = "a" * 64
        with pytest.raises(ModelIntegrityError, match="Integrity check failed"):
            verify_model(path, wrong_hash)
    finally:
        os.unlink(path)


def test_missing_file_raises_file_not_found() -> None:
    with pytest.raises(FileNotFoundError):
        compute_file_hash("/nonexistent/path/model.pt")


def test_verify_model_missing_file_raises() -> None:
    with pytest.raises((FileNotFoundError, OSError)):
        verify_model("/nonexistent/path/model.pt", "a" * 64)
