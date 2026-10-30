"""Sender-only nonce reservation; never imported by the pure receiver.

All senders sharing a key must use the SAME durable ledger, or coordinate an
external globally unique nonce allocation. Copying/resetting/rolling back the
ledger, using separate ledgers, or device crashes outside filesystem guarantees
can defeat uniqueness. A reserved nonce is burned even if generation fails.
The low-level pure codec cannot enforce this deployment obligation.
"""
from __future__ import annotations

import hmac
import os
from pathlib import Path
import secrets

from .canonical_channel_v2 import ContractError
from .framed_interval_public_v1 import DOMAIN, VERSION, canonical_json


class NonceAlreadyReserved(ContractError):
    pass


def reserve_nonce(directory, key, nonce):
    """Atomically burn one (key, nonce), independent of spec/context/plaintext.

    Returns a non-secret reservation receipt. No key or payload is stored.
    Exceptions do not authorize removal/retry of an existing reservation.
    """
    if type(key) is not bytes or len(key) != 32: raise ContractError("key must be32 bytes")
    if type(nonce) is not bytes or len(nonce) != 12: raise ContractError("nonce must be12 bytes")
    path = Path(directory)
    # The parent must already exist; persist a newly created ledger name too.
    path.mkdir(mode=0o700, exist_ok=True)
    if path.is_symlink() or not path.is_dir(): raise ContractError("real nonce ledger directory required")
    parent_fd = os.open(path.parent, os.O_RDONLY | os.O_DIRECTORY)
    try: os.fsync(parent_fd)
    finally: os.close(parent_fd)
    tag = hmac.digest(key, DOMAIN + b"nonce-ledger-reservation\x00" + nonce, "sha256").hex()
    name = tag + ".json"
    record = {"schema": VERSION + "-nonce-reservation", "reservation_id": tag,
              "nonce_hex": nonce.hex(), "key_saved": False, "state": "burned_before_generation"}
    directory_fd = os.open(path, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)
    try:
        try:
            fd = os.open(name, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW, 0o600, dir_fd=directory_fd)
        except FileExistsError as exc:
            raise NonceAlreadyReserved("same key/nonce already reserved; never retry or delete it") from exc
        with os.fdopen(fd, "wb") as output:
            output.write(canonical_json(record) + b"\n")
            output.flush(); os.fsync(output.fileno())
        os.fsync(directory_fd)
    finally:
        os.close(directory_fd)
    return record


def reserve_random_nonce(directory, key):
    """OS CSPRNG nonce; reserve before returning. No collision retry is hidden."""
    nonce = secrets.token_bytes(12)
    return nonce, reserve_nonce(directory, key, nonce)
