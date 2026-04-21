"""
Secure user credential storage using numpy .npz format.

Stores unit-normalized float32 embeddings alongside usernames.
Never stores raw images, raw passwords, or reversible representations.

Requirements: 12.1, 12.2, 12.3, 12.4, 12.5
"""

import logging
import os
import tempfile
from typing import Optional

import numpy as np

logger = logging.getLogger(__name__)


class ConflictError(Exception):
    """Raised when attempting to register a username that already exists."""


class UserDB:
    """
    Persistent store for user embeddings.

    Storage format: numpy .npz with:
      - 'usernames': np.array of str, shape (N,)
      - 'embeddings': np.array of float32, shape (N, 512)
    """

    def __init__(self, db_path: str = "auth/users.npz") -> None:
        """
        Load from db_path on startup.
        If missing or corrupted: initialize empty store and log WARNING.
        """
        self._db_path = db_path
        self._usernames: list[str] = []
        self._embeddings: list[np.ndarray] = []

        if os.path.exists(db_path):
            try:
                data = np.load(db_path, allow_pickle=False)
                self._usernames = list(data["usernames"].astype(str))
                self._embeddings = [data["embeddings"][i] for i in range(len(self._usernames))]
                logger.info("UserDB loaded %d records from %s", len(self._usernames), db_path)
            except Exception as exc:  # noqa: BLE001
                logger.warning(
                    "UserDB: failed to load '%s' (%s). Initializing empty store.",
                    db_path,
                    exc,
                )
                self._usernames = []
                self._embeddings = []
        else:
            logger.warning(
                "UserDB: file '%s' not found. Initializing empty store.",
                db_path,
            )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def save(self, username: str, embedding: np.ndarray) -> None:
        """
        Store a user embedding.

        Raises:
            ConflictError: if username already exists.
        """
        if username in self._usernames:
            raise ConflictError(f"Username '{username}' is already registered.")
        self._usernames.append(username)
        self._embeddings.append(np.array(embedding, dtype=np.float32))

    def lookup(self, username: str) -> np.ndarray:
        """
        Return the stored embedding for username.

        Raises:
            KeyError: if username is not found.
        """
        try:
            idx = self._usernames.index(username)
        except ValueError:
            raise KeyError(f"Username '{username}' not found in UserDB.")
        return self._embeddings[idx].copy()

    def persist(self) -> None:
        """
        Write current state to db_path atomically (temp file + rename).
        """
        os.makedirs(os.path.dirname(self._db_path) or ".", exist_ok=True)

        usernames_arr = np.array(self._usernames, dtype=str)
        if self._embeddings:
            embeddings_arr = np.stack(self._embeddings, axis=0).astype(np.float32)
        else:
            embeddings_arr = np.empty((0, 512), dtype=np.float32)

        # Write to a temp file in the same directory, then rename for atomicity.
        # Note: np.savez automatically appends ".npz" if the path doesn't end with it,
        # so we use a base name without extension and track the actual written path.
        dir_name = os.path.dirname(self._db_path) or "."
        fd, tmp_base = tempfile.mkstemp(dir=dir_name, suffix=".tmp")
        os.close(fd)
        os.unlink(tmp_base)  # remove the placeholder; np.savez will create tmp_base + ".npz"
        tmp_path = tmp_base + ".npz"
        try:
            np.savez(tmp_base, usernames=usernames_arr, embeddings=embeddings_arr)
            os.replace(tmp_path, self._db_path)
        except Exception:
            # Clean up temp file on failure
            if os.path.exists(tmp_path):
                os.unlink(tmp_path)
            raise

    # ------------------------------------------------------------------
    # Properties (read-only)
    # ------------------------------------------------------------------

    @property
    def db_path(self) -> str:
        return self._db_path

    def __len__(self) -> int:
        return len(self._usernames)
