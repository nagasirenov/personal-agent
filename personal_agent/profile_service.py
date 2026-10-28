"""Sender-owned attachment reads and persistent user-profile saves for the native tools."""
from __future__ import annotations

from datetime import datetime, timedelta, timezone
from pathlib import Path

from .memory import MAX_PROFILE_CHARACTERS
from .observation_requests import MAX_FILE_BYTES, TextChunkError, read_text_chunk


_TEXT_ATTACHMENT_SUFFIXES = {".txt", ".md", ".csv", ".log"}
_ATTACHMENT_AGE = timedelta(days=30)
_ATTACHMENT_CANDIDATES = 5


class UserProfileService:
    def __init__(self, store, data_dir: Path) -> None:
        self.store = store
        self.data_dir = Path(data_dir)

    def attachment_candidates(self, user_id: int) -> list[dict]:
        """List a few recent, sender-owned text attachments without reading their contents."""
        now = datetime.now(timezone.utc)
        root = (self.data_dir / "memory" / "attachments").resolve()
        candidates = []
        for metadata in self.store.recent_attachments(user_id, limit=50):
            if metadata.get("sender_id") != user_id or metadata.get("kind") != "document":
                continue
            name = str(metadata.get("original_name", ""))
            if Path(name).suffix.casefold() not in _TEXT_ATTACHMENT_SUFFIXES:
                continue
            try:
                attached_at = datetime.fromisoformat(metadata["time"])
                if attached_at.tzinfo is None:
                    attached_at = attached_at.replace(tzinfo=timezone.utc)
                age = now - attached_at.astimezone(timezone.utc)
                path = Path(metadata["path"]).resolve()
                attachment_id = str(metadata["id"])
                if (not attachment_id or len(attachment_id) > 100 or "\x00" in attachment_id
                        or age < timedelta(0) or age > _ATTACHMENT_AGE
                        or not path.is_relative_to(root) or not path.is_file()):
                    continue
            except (KeyError, OSError, TypeError, ValueError):
                continue
            candidates.append({"attachment_id": attachment_id, "name": name,
                               "time": metadata["time"]})
            if len(candidates) >= _ATTACHMENT_CANDIDATES:
                break
        return candidates

    def load_attachment(self, user_id: int, attachment_id: str, offset: int = 0,
                        max_bytes: int = MAX_FILE_BYTES) -> dict:
        """Read one chunk of a recent sender-owned text attachment chosen by the model."""
        def failure(code: str) -> dict:
            return {"kind": "attachment_source", "ok": False, "sender_id": user_id,
                    "source_attachment_id": attachment_id, "error": {"code": code}}

        if not isinstance(attachment_id, str):
            return failure("attachment_not_available")
        candidate = next((item for item in self.attachment_candidates(user_id)
                          if item["attachment_id"] == attachment_id), None)
        if candidate is None:
            return failure("attachment_not_available")
        metadata = next((item for item in self.store.recent_attachments(user_id, limit=50)
                         if item.get("sender_id") == user_id and str(item.get("id")) == attachment_id), None)
        if metadata is None:
            return failure("attachment_not_available")
        try:
            path = Path(metadata["path"]).resolve()
            root = (self.data_dir / "memory" / "attachments").resolve()
            if not path.is_relative_to(root):
                return failure("source_outside_storage")
            chunk = read_text_chunk(path, offset, max_bytes)
        except TextChunkError as exc:
            return failure("source_not_utf8" if exc.code == "non_utf8_file" else "source_invalid_text")
        except (KeyError, OSError, ValueError):
            return failure("source_unreadable")
        if chunk["total_bytes"] == 0 or (chunk["offset"] == 0 and chunk["eof"] and not chunk["text"].strip()):
            return failure("source_invalid_text")
        return {"kind": "attachment_source", "ok": True, "has_image": False,
                "sender_id": user_id, "attachment_id": attachment_id,
                "source_attachment_id": attachment_id, "name": candidate["name"], **chunk,
                "time": candidate["time"]}

    def save_from_model(self, user_id: int, text: str, *, append: bool = True,
                        source_attachment_id: str | None = None,
                        authorized_source_attachment_ids=()) -> dict:
        """Persist the profile text selected by the model tool, then verify it by reading it back."""
        if source_attachment_id and source_attachment_id not in authorized_source_attachment_ids:
            return self._profile_failure(user_id, "profile_source_not_loaded", source_attachment_id)
        if not isinstance(text, str) or not text.strip() or "\x00" in text:
            return self._profile_failure(user_id, "profile_source_invalid_text", source_attachment_id)
        if len(text.strip()) > MAX_PROFILE_CHARACTERS:
            return self._profile_failure(user_id, "profile_too_large", source_attachment_id)
        return self._save_profile(user_id, text, source_attachment_id=source_attachment_id or "", append=append)

    def _profile_failure(self, user_id: int, code: str, source_attachment_id: str | None = None) -> dict:
        return {"kind": "profile_save_result", "ok": False, "sender_id": user_id,
                "source_attachment_id": source_attachment_id, "error": {"code": code},
                "max_characters": MAX_PROFILE_CHARACTERS, "read_back_verified": False}

    def _save_profile(self, user_id: int, text: str, *, source_attachment_id: str, append: bool) -> dict:
        try:
            saved = self.store.save_user_profile(
                user_id, text, source_attachment_id=source_attachment_id, append=append,
            )
        except ValueError:
            return self._profile_failure(user_id, "profile_too_large", source_attachment_id or None)
        receipt = {
            "kind": "profile_save_result",
            "ok": True,
            "sender_id": user_id,
            "source_attachment_id": saved["source_attachment_id"] or None,
            "characters": saved["characters"],
            "sha256": saved["sha256"],
            "read_back_verified": True,
            "saved_at": saved["saved_at"],
        }
        self.store.set_state(f"profile_save_receipt:{user_id}", receipt)
        from .models import TimelineEvent
        self.store.record_event(TimelineEvent("memory", "USER_PROFILE_SAVED", receipt))
        return receipt
