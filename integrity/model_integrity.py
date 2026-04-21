"""
Model integrity verification via SHA-256 hashing.

Verifies that model weight files have not been tampered with since training.
Requirements: 13.1, 13.2, 13.3, 13.4
"""

import hashlib
import json
from pathlib import Path


class ModelIntegrityError(Exception):
    """Raised when a model file's SHA-256 hash does not match the expected hash."""


def compute_file_hash(path: str) -> str:
    """Compute the SHA-256 hex digest of a file using 8KB streaming reads.

    Args:
        path: Path to the file to hash.

    Returns:
        Lowercase hex string of the SHA-256 digest (64 characters).

    Raises:
        FileNotFoundError: If the file does not exist.
    """
    sha256 = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(8192):
            sha256.update(chunk)
    return sha256.hexdigest()


def save_hashes(model_paths: dict, hash_file: str = "model_hashes.json") -> None:
    """Compute and save SHA-256 hashes for all model weight files to a JSON file.

    Args:
        model_paths: Mapping of model name → file path.
        hash_file:   Path to the output JSON file (default: "model_hashes.json").
    """
    hashes = {name: compute_file_hash(path) for name, path in model_paths.items()}
    with open(hash_file, "w") as f:
        json.dump(hashes, f, indent=2)


def verify_model(path: str, expected_hash: str) -> None:
    """Verify a model weight file against its expected SHA-256 hash.

    Args:
        path:          Path to the model weight file.
        expected_hash: Expected SHA-256 hex digest.

    Raises:
        ModelIntegrityError: If the computed hash does not match expected_hash.
        FileNotFoundError:   If the file does not exist.
    """
    actual_hash = compute_file_hash(path)
    if actual_hash != expected_hash:
        raise ModelIntegrityError(
            f"Integrity check failed for '{path}': "
            f"expected {expected_hash}, got {actual_hash}"
        )
