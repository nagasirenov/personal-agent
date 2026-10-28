"""Owner-approved sharing of context with the shared SECONDARY account.

The owner's data stays private by default.  Only text the owner explicitly approved ("notes") reaches the
SECONDARY account, and the SECONDARY side can only ask: a permission request grants nothing by itself.
Everything is persisted in MemoryStore state, so notes and requests survive a restart.
"""
from __future__ import annotations

import re
from datetime import datetime, timedelta, timezone

STATE_KEY = "owner_sharing"
REQUEST_TTL = timedelta(hours=24)
MAX_REQUEST_CHARS = 500
MAX_NOTE_CHARS = 1000
MAX_ACTIVE_NOTES = 20
MAX_EXPIRES_HOURS = 24 * 365
DEFAULT_NOTE_HOURS = 24 * 7
_KEEP_FINISHED = timedelta(days=7)
_CONTROL = re.compile(r"[\x00-\x08\x0b-\x1f\x7f]")


class SharingError(Exception):
    def __init__(self, code: str) -> None:
        self.code = code
        super().__init__(code)


def clean_text(value: object, maximum: int) -> str:
    """Single bounded text field: no control characters, no quote marks that could break the owner's quoting."""
    if not isinstance(value, str):
        raise SharingError("invalid_arguments")
    text = _CONTROL.sub(" ", value).replace("«", '"').replace("»", '"').strip()
    if not text:
        raise SharingError("invalid_arguments")
    return text[:maximum]


def _now() -> datetime:
    return datetime.now(timezone.utc)


def _parse(value) -> datetime | None:
    if not isinstance(value, str):
        return None
    try:
        parsed = datetime.fromisoformat(value)
    except ValueError:
        return None
    return parsed.replace(tzinfo=timezone.utc) if parsed.tzinfo is None else parsed.astimezone(timezone.utc)


class SharingStore:
    """Notes the owner approved for the secondary account and the secondary account's permission requests."""

    def __init__(self, store) -> None:
        self.store = store

    # -- raw state ---------------------------------------------------------------------------------
    def _load(self) -> dict:
        state = self.store.get_state(STATE_KEY, None)
        if not isinstance(state, dict):
            state = {}
        return {"seq": int(state.get("seq", 0)) if type(state.get("seq", 0)) is int else 0,
                "notes": [n for n in state.get("notes", []) if isinstance(n, dict)],
                "requests": [r for r in state.get("requests", []) if isinstance(r, dict)]}

    def _save(self, state: dict) -> None:
        self.store.set_state(STATE_KEY, state)

    def _refresh(self, state: dict, now: datetime) -> None:
        """Expire stale pending requests and drop long-finished entries."""
        for request in state["requests"]:
            created = _parse(request.get("created_at"))
            if request.get("status") == "pending" and (created is None or now - created >= REQUEST_TTL):
                request["status"] = "expired"
                request["answered_at"] = now.isoformat()
        state["requests"] = [
            r for r in state["requests"]
            if r.get("status") == "pending" or (_parse(r.get("answered_at") or r.get("created_at")) or now) > now - _KEEP_FINISHED]
        state["notes"] = [
            n for n in state["notes"]
            if not n.get("revoked_at") and (_parse(n.get("expires_at")) is None or _parse(n["expires_at"]) > now - _KEEP_FINISHED)]

    @staticmethod
    def _is_active(note: dict, now: datetime) -> bool:
        expires = _parse(note.get("expires_at"))
        return not note.get("revoked_at") and (expires is None or expires > now)

    # -- notes -------------------------------------------------------------------------------------
    def add_note(self, text: object, *, source: str, expires_hours: object = None, now: datetime | None = None) -> dict:
        now = now or _now()
        text = clean_text(text, MAX_NOTE_CHARS)
        if expires_hours is None:
            expires_at = None
        elif type(expires_hours) in (int, float) and 0 < expires_hours <= MAX_EXPIRES_HOURS:
            expires_at = (now + timedelta(hours=expires_hours)).isoformat()
        else:
            raise SharingError("invalid_arguments")
        state = self._load()
        self._refresh(state, now)
        if sum(self._is_active(n, now) for n in state["notes"]) >= MAX_ACTIVE_NOTES:
            raise SharingError("too_many_notes")
        state["seq"] += 1
        note = {"id": state["seq"], "text": text, "created_at": now.isoformat(), "expires_at": expires_at,
                "source": source}
        state["notes"].append(note)
        self._save(state)
        return dict(note)

    def active_notes(self, now: datetime | None = None) -> list[dict]:
        now = now or _now()
        return [dict(n) for n in self._load()["notes"] if self._is_active(n, now)]

    def revoke_note(self, note_id: object, now: datetime | None = None) -> bool:
        now = now or _now()
        state = self._load()
        for note in state["notes"]:
            if note.get("id") == note_id and self._is_active(note, now):
                note["revoked_at"] = now.isoformat()
                self._refresh(state, now)
                self._save(state)
                return True
        return False

    # -- requests ----------------------------------------------------------------------------------
    def create_request(self, text: object, now: datetime | None = None) -> dict:
        now = now or _now()
        text = clean_text(text, MAX_REQUEST_CHARS)
        state = self._load()
        self._refresh(state, now)
        state["seq"] += 1
        request = {"id": state["seq"], "from": "secondary", "request_text": text, "created_at": now.isoformat(),
                   "status": "pending"}
        state["requests"].append(request)
        self._save(state)
        return dict(request)

    def drop_request(self, request_id: int) -> None:
        state = self._load()
        state["requests"] = [r for r in state["requests"] if r.get("id") != request_id]
        self._save(state)

    def pending_requests(self, now: datetime | None = None) -> list[dict]:
        now = now or _now()
        state = self._load()
        self._refresh(state, now)
        self._save(state)
        return [dict(r) for r in state["requests"] if r.get("status") == "pending"]

    def recent_requests(self, now: datetime | None = None, limit: int = 5) -> list[dict]:
        now = now or _now()
        state = self._load()
        self._refresh(state, now)
        self._save(state)
        return [dict(r) for r in state["requests"]][-limit:]

    def get_request(self, request_id: object, now: datetime | None = None) -> dict | None:
        now = now or _now()
        state = self._load()
        self._refresh(state, now)
        self._save(state)
        return next((dict(r) for r in state["requests"] if type(request_id) is int and r.get("id") == request_id), None)

    def resolve_request(self, request_id: object, status: str, now: datetime | None = None,
                        *, owner_note: str = "") -> dict:
        """Mark a still-pending request approved or denied (owner_note stays on the owner's side)."""
        if status not in {"approved", "denied"}:
            raise ValueError(status)
        now = now or _now()
        state = self._load()
        self._refresh(state, now)
        request = next((r for r in state["requests"] if type(request_id) is int and r.get("id") == request_id), None)
        if request is None:
            self._save(state)
            raise SharingError("request_not_found")
        if request.get("status") != "pending":
            self._save(state)
            raise SharingError("request_not_pending")
        request["status"] = status
        request["answered_at"] = now.isoformat()
        if owner_note:
            request["owner_note"] = owner_note
        self._save(state)
        return dict(request)
