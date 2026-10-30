"""Crash-conservative, process-locked accounting for one <= USD 20 campaign.

Amounts are integer nanodollars, rounded upward. Reservations are committed
before generation. An unknown attempt stays reserved and is never auto-retried.
This ledger cannot impose a provider-side/account-wide billing limit.
"""
from __future__ import annotations

from contextlib import contextmanager
from decimal import Decimal, InvalidOperation, ROUND_CEILING
import hashlib
import json
import os
from pathlib import Path
import tempfile
import time

UNIT = 1_000_000_000
SCHEMA = "synsteg-budget-ledger-recovery-v1"


class BudgetError(RuntimeError):
    pass


def money(value) -> int:
    try:
        number = Decimal(str(value))
    except (InvalidOperation, ValueError, TypeError):
        raise BudgetError("Invalid monetary value") from None
    if not number.is_finite() or number < 0:
        raise BudgetError("Monetary values must be finite and nonnegative")
    return int((number * UNIT).to_integral_value(rounding=ROUND_CEILING))


def token_cost(input_tokens, output_tokens, input_rate, output_rate) -> int:
    if any(type(v) is not int or v < 0 for v in (input_tokens, output_tokens)):
        raise BudgetError("Invalid token usage")
    # Rates are USD per million; round the combined charge upward only once.
    for rate in (input_rate, output_rate):
        money(rate)
    cost = (Decimal(input_tokens) * Decimal(str(input_rate))
            + Decimal(output_tokens) * Decimal(str(output_rate))) / Decimal(1_000_000)
    return money(cost)


def canonical_hash(value) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=True,
                                    separators=(",", ":")).encode()).hexdigest()


def atomic_json(path: Path, value) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile("w", encoding="utf-8", dir=path.parent,
                                         prefix=path.name + ".", delete=False) as stream:
            temporary = stream.name
            json.dump(value, stream, ensure_ascii=True, sort_keys=True, indent=2)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
        temporary = None
        if os.name != "nt":
            fd = os.open(path.parent, os.O_RDONLY)
            try:
                os.fsync(fd)
            finally:
                os.close(fd)
    finally:
        if temporary:
            try:
                os.unlink(temporary)
            except FileNotFoundError:
                pass


class CampaignLedger:
    def __init__(self, directory, budget_usd="20"):
        self.directory = Path(directory)
        self.path = self.directory / "ledger.json"
        self.cap = money(budget_usd)
        if not 0 < self.cap <= 20 * UNIT:
            raise BudgetError("Campaign budget must be in (0, 20] USD")

    @contextmanager
    def locked(self):
        """OS locks release on process exit; do not delete the lock file."""
        self.directory.mkdir(parents=True, exist_ok=True)
        with (self.directory / "campaign.lock").open("a+b") as lock:
            lock.seek(0, os.SEEK_END)
            if lock.tell() == 0:
                lock.write(b"0")
                lock.flush()
            lock.seek(0)
            acquired = False
            try:
                if os.name == "nt":
                    import msvcrt
                    msvcrt.locking(lock.fileno(), msvcrt.LK_NBLCK, 1)
                else:
                    import fcntl
                    fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
                acquired = True
            except OSError:
                raise BudgetError("Campaign is locked by another process; no requests sent") from None
            try:
                yield self
            finally:
                if acquired:
                    lock.seek(0)
                    if os.name == "nt":
                        import msvcrt
                        msvcrt.locking(lock.fileno(), msvcrt.LK_UNLCK, 1)
                    else:
                        import fcntl
                        fcntl.flock(lock.fileno(), fcntl.LOCK_UN)

    def read(self):
        if not self.path.exists():
            state = {"schema": SCHEMA, "budget_nanodollars": self.cap,
                     "created_at_unix": time.time(), "attempts": {},
                     "provider_billing_limit": False}
            atomic_json(self.path, state)
        state = json.loads(self.path.read_text(encoding="utf-8"))
        if state.get("schema") != SCHEMA or state.get("budget_nanodollars") != self.cap:
            raise BudgetError("Existing campaign schema or budget differs; refusing replacement")
        if not isinstance(state.get("attempts"), dict):
            raise BudgetError("Malformed ledger")
        self.total(state)
        return state

    @staticmethod
    def total(state):
        amount = 0
        for attempt in state["attempts"].values():
            charged = attempt.get("accounted_nanodollars")
            if type(charged) is not int or charged < 0:
                raise BudgetError("Malformed attempt accounting")
            amount += charged
        return amount

    def reserve(self, identity, fingerprint, ceiling, metadata):
        state = self.read()
        if state.get("blocked_reason"):
            raise BudgetError("Campaign is blocked after an accounting-bound violation")
        old = state["attempts"].get(identity)
        if old is not None:
            if old.get("request_fingerprint") != fingerprint:
                raise BudgetError("Record already exists with different frozen request settings")
            return old, False
        if any(a["status"] != "completed" for a in state["attempts"].values()):
            raise BudgetError("Campaign contains a failed/unknown attempt; review required before any new request")
        if type(ceiling) is not int or ceiling < 0:
            raise BudgetError("Invalid reservation")
        if self.total(state) + ceiling > self.cap:
            raise BudgetError("Campaign reservation budget exhausted; no request sent")
        attempt = {"status": "reserved_outcome_unknown", "identity": identity,
                   "request_fingerprint": fingerprint, "reserved_nanodollars": ceiling,
                   "accounted_nanodollars": ceiling, "created_at_unix": time.time(),
                   "metadata": metadata, "automatic_retries": 0}
        state["attempts"][identity] = attempt
        atomic_json(self.path, state)
        return attempt, True

    def finish(self, identity, result, actual_cost=None):
        state = self.read()
        attempt = state["attempts"][identity]
        if attempt["status"] != "reserved_outcome_unknown":
            raise BudgetError("Attempt is already terminal")
        attempt.update(finished_at_unix=time.time(), result=result)
        if actual_cost is None:
            attempt["status"] = "failed_or_unknown_no_retry"
        elif type(actual_cost) is not int or not 0 <= actual_cost <= attempt["reserved_nanodollars"]:
            # Preserve observed excess rather than pretending the cap held.
            if type(actual_cost) is int and actual_cost >= 0:
                attempt["accounted_nanodollars"] = actual_cost
            attempt["status"] = "accounting_bound_violation_stop"
            state["blocked_reason"] = "Observed usage/cost violated configured reservation"
            atomic_json(self.path, state)
            raise BudgetError("Usage/cost exceeded configured bound; campaign blocked")
        else:
            attempt["status"] = "completed"
            attempt["accounted_nanodollars"] = actual_cost
        atomic_json(self.path, state)
        return attempt

    def summary(self):
        state = self.read()
        return {"budget_usd": self.cap / UNIT,
                "accounted_or_reserved_usd": self.total(state) / UNIT,
                "remaining_reservation_usd": max(0, self.cap - self.total(state)) / UNIT,
                "attempt_count": len(state["attempts"]),
                "unknown_or_failed_attempts": sum(a["status"] != "completed" for a in state["attempts"].values()),
                "provider_billing_limit": False}
