"""Code-enforced trust boundaries, independent of any model output."""
from __future__ import annotations

import os
from pathlib import Path

from .models import Identity


class TrustGate:
    def __init__(self, primary_id: int, secondary_id: int):
        if type(primary_id) is not int or type(secondary_id) is not int or primary_id <= 0 or secondary_id <= 0 or primary_id == secondary_id:
            raise ValueError("Two distinct positive Telegram user IDs required")
        self.primary_id, self.secondary_id = primary_id, secondary_id

    def identify(self, sender_id: int) -> Identity | None:
        if type(sender_id) is not int:
            return None
        if sender_id == self.primary_id:
            return Identity.PRIMARY
        if sender_id == self.secondary_id:
            return Identity.SECONDARY
        return None


class ActionGate:
    """Only a local configuration grant can enable PRIMARY's host tools."""

    def __init__(self, primary_id: int | None = None, *, full_access: bool = False):
        self.primary_id = primary_id
        self.full_access = full_access

    def authorize(self, sender_id: int, proposal: object) -> None:
        if (not self.full_access or self.primary_id is None or sender_id != self.primary_id
                or not isinstance(proposal, dict) or proposal.get("kind") not in {"run_command", "write_file"}):
            raise PermissionError("PC actions require PRIMARY and pc_access=full in local configuration")


def owned_path(root: Path, *parts: str) -> Path:
    root = root.resolve()
    candidate = root.joinpath(*parts).resolve()
    if candidate == root or not candidate.is_relative_to(root):
        raise ValueError("Path leaves agent-owned storage")
    return candidate


class InstanceLock:
    """OS lock is released even after a crash; no stale PID guessing."""

    def __init__(self, root: Path):
        root.mkdir(parents=True, exist_ok=True)
        self.handle = owned_path(root, "daemon.lock").open("a+b")
        try:
            if os.fstat(self.handle.fileno()).st_size == 0:
                self.handle.write(b"0")
                self.handle.flush()
            self.handle.seek(0)
            if os.name == "nt":
                import msvcrt
                msvcrt.locking(self.handle.fileno(), msvcrt.LK_NBLCK, 1)
            else:
                import fcntl
                fcntl.flock(self.handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError:
            self.handle.close()
            raise RuntimeError("Another daemon already uses this data_dir") from None

    def close(self) -> None:
        if self.handle.closed:
            return
        if os.name == "nt":
            import msvcrt
            self.handle.seek(0)
            msvcrt.locking(self.handle.fileno(), msvcrt.LK_UNLCK, 1)
        else:
            import fcntl
            fcntl.flock(self.handle, fcntl.LOCK_UN)
        self.handle.close()
