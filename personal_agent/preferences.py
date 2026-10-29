"""Persistent, account-scoped user preferences with conservative extraction."""
from __future__ import annotations

from datetime import datetime, timezone
import re
from typing import Any

from .reply_context import strip_memory_prefix


DEFAULT_PREFERENCES: dict[str, Any] = {
    "response_format": "auto",
    "markdown_allowed": True,
    "screenshot_delivery": "explicit_only",
}

_STATE_PREFIX = "user_preferences"
_LEGACY_STATE_PREFIX = "conversation_preferences"
_RECEIPT_PREFIX = "preference_save"
_MAX_NOTES = 20
_MAX_NOTE_CHARS = 500

_HYPOTHETICAL = re.compile(
    r"^\s*(?:если\s+(?:бы|я\s+попрошу)|допустим|представь|например|гипотетическ\w*|"
    r"как\s+будто|когда\s+я\s+попрошу|would\s+you|suppose|hypothetically)\b",
    re.IGNORECASE,
)
_NEGATED_STYLE_COMMAND = re.compile(
    r"\bне\s+(?:отвечай|говори|общайся|пиши|обращайся)\b",
    re.IGNORECASE,
)
_STYLE_COMMAND = re.compile(r"\b(?:отвечай|говори|общайся|пиши|обращайся)\b", re.IGNORECASE)
_FRIENDLY = re.compile(r"\b(?:проще|попроще|дружелюб\w*|неформаль\w*)\b", re.IGNORECASE)
_INFORMAL = re.compile(r"\bна\s+ты\b", re.IGNORECASE)
_PLAIN_TEXT = re.compile(
    r"(?:\b(?:отвечай|пиши|ответы?)\b.{0,45}\b(?:обычн\w*|прост\w*|plain)\s+(?:текст\w*|text)\b|"
    r"\bбез\s+(?:markdown|маркдаун\w*|форматирован\w*)\b|"
    r"\bне\s+используй\s+(?:markdown|маркдаун\w*|форматирован\w*)\b|"
    r"\bне\s+форматируй\s+(?:ответ\w*|сообщен\w*|текст\w*)\b|"
    r"\b(?:обычн\w*|прост\w*|plain)\s+(?:текст\w*|text)\s+(?:в|для)\s+(?:telegram|телеграм\w*)\b|"
    r"\b(?:telegram|телеграм\w*)\b.{0,30}\b(?:plain\s+text|обычн\w*\s+текст\w*)\b|"
    r"\b(?:запомни|сохрани|запиши(?:ть)?(?:\s+в\s+память)?)\b.{0,100}\b(?:не\s+(?:нужно|надо)|без)\s+(?:ответ\w*\s+)?форматирован\w*\b)",
    re.IGNORECASE,
)
_STICKER_LOOKUP = re.compile(
    r"(?:\b(?:в\s+следующ(?:ий|ие)\s+раз|впредь)\b.{0,70}\b(?:найди|ищи|определи|назови)\b.{0,45}\bбез\s+(?:лишних\s+)?вопросов\b|"
    r"\b(?:определяй|узнавай|находи|ищи|опознавай)\b.{0,80}\b(?:персонаж\w*|стикер\w*|кто\s+изображ\w*)\b.{0,70}\b(?:сам\w*|без\s+вопросов)\b)",
    re.IGNORECASE,
)
_SCREENSHOT_EXPLICIT = re.compile(
    r"(?:\b(?:скриншот\w*|сним\w*\s+экрана)\b.{0,55}\b(?:присылай|отправляй|показывай)?\s*только\s+(?:по\s+(?:запросу|прямой\s+просьбе|моей\s+просьбе)|если\s+(?:я\s+)?попрошу)\b|"
    r"\b(?:скриншот\w*|сним\w*\s+экрана)\b.{0,25}\b(?:мне\s+)?не\s+(?:присылай|отправляй|показывай)\b.{0,45}\bесли\s+(?:я\s+)?(?:прямо\s+)?не\s+(?:прошу|попрошу)\b|"
    r"\bне\s+(?:присылай|отправляй|показывай)\b.{0,35}\b(?:скриншот\w*|сним\w*\s+экрана)\b.{0,30}\bбез\s+(?:моей\s+)?(?:явной\s+)?просьбы\b)",
    re.IGNORECASE,
)
_REMEMBER_NOTE = re.compile(
    r"^\s*(?:пожалуйста[,\s]+)?запомни(?:\s+пожалуйста)?\s*[:,]?\s*(?:что\s+)?(.+?)\s*$",
    re.IGNORECASE | re.DOTALL,
)
_INSTRUCTION_IN_NOTE = re.compile(
    r"\b(?:отвечай|пиши|говори|общайся|обращайся|присылай|отправляй|показывай|используй)\b",
    re.IGNORECASE,
)

def _clean_text(text: str) -> str:
    return text.casefold().replace("ё", "е")


class PreferenceActionDenied(ValueError):
    pass


_ALLOWED_UPDATES = {
    "response_format": {"plain_text", "auto"}, "tone": {"conversational", "friendly", "formal"},
    "address": {"informal", "formal"}, "screenshot_delivery": {"explicit_only"},
    "sticker_identity_lookup": {"search_without_asking"},
}


def validate_preference_updates(updates: object) -> dict[str, str]:
    """Validate the arguments of the save_preferences tool as a whole; any invalid field rejects all."""
    if (not isinstance(updates, dict) or not updates or any(
        key not in _ALLOWED_UPDATES or not isinstance(value, str) or value not in _ALLOWED_UPDATES[key]
        for key, value in updates.items()
    )):
        raise PreferenceActionDenied("invalid_arguments")
    return dict(updates)


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _without_quoted_or_code(text: str) -> str:
    """Remove quoted/example material so it cannot become a preference command."""
    value = re.sub(r"```.*?```", " ", text, flags=re.DOTALL)
    value = re.sub(r"`[^`]*`", " ", value)
    value = re.sub(r"«[^»]*»", " ", value, flags=re.DOTALL)
    value = re.sub(r'"[^"\n]*"', " ", value)
    value = re.sub(r"'[^'\n]*'", " ", value)
    value = "\n".join(line for line in value.splitlines() if not line.lstrip().startswith(">"))
    return value


def _legacy_style_update(text: str) -> dict[str, Any]:
    """Recover old, direct style commands during one-time preference migration only."""
    unquoted = _without_quoted_or_code(text)
    clean = _clean_text(unquoted)
    if _HYPOTHETICAL.search(clean):
        return {}
    update: dict[str, Any] = {}
    has_style_command = bool(_STYLE_COMMAND.search(clean)) and not bool(_NEGATED_STYLE_COMMAND.search(clean))
    if has_style_command and _FRIENDLY.search(clean):
        update["tone"] = "friendly"
    if has_style_command and _INFORMAL.search(clean):
        update["address"] = "informal"

    # The old parser also treated mentions like “обычный текст в Telegram” as
    # preferences. Migrate only unambiguous commands, not bare discussion.
    direct_plain_command = re.match(
        r"^\s*(?:отвечай\b|пиши\b|ответы?\b|без\s+(?:markdown|маркдаун\w*|форматирован\w*)\b|"
        r"не\s+используй\b|не\s+форматируй\b|запомни\b)",
        clean,
    )
    if direct_plain_command and _PLAIN_TEXT.search(clean):
        update.update(response_format="plain_text", markdown_allowed=False)
    return update


def parse_preference_update(text: str) -> dict[str, Any]:
    """Extract deterministic non-style settings; tone and formatting use semantic classification."""
    if not isinstance(text, str) or not text.strip():
        return {}
    unquoted = _without_quoted_or_code(text)
    clean = _clean_text(unquoted)
    if _HYPOTHETICAL.search(clean):
        return {}

    update: dict[str, Any] = {}
    if _SCREENSHOT_EXPLICIT.search(clean):
        update["screenshot_delivery"] = "explicit_only"
    if _STICKER_LOOKUP.search(clean):
        update["sticker_identity_lookup"] = "search_without_asking"

    note_match = _REMEMBER_NOTE.fullmatch(unquoted)
    if note_match and not _INSTRUCTION_IN_NOTE.search(note_match.group(1)):
        note = re.sub(r"\s+", " ", note_match.group(1)).strip(" .")[:_MAX_NOTE_CHARS]
        if note:
            update["user_memory_note"] = note
    return update


def style_from_message(text: str) -> dict[str, str]:
    """Legacy style-only parser retained for compatibility; runtime uses semantic classification."""
    update = _legacy_style_update(text)
    return {key: update[key] for key in ("tone", "address") if key in update}


class PreferenceStore:
    """Persist supported preferences and a per-turn save receipt for one backing store."""

    def __init__(self, store) -> None:
        self.store = store

    @staticmethod
    def state_key(user_id: int) -> str:
        return f"{_STATE_PREFIX}:{int(user_id)}"

    @staticmethod
    def legacy_state_key(user_id: int) -> str:
        return f"{_LEGACY_STATE_PREFIX}:{int(user_id)}"

    @staticmethod
    def receipt_key(user_id: int) -> str:
        return f"{_RECEIPT_PREFIX}:{int(user_id)}"

    def _historical_update(self, user_id: int) -> dict[str, Any]:
        result: dict[str, Any] = {}
        # recent_messages is newest-first; replay oldest-first so the latest explicit
        # preference wins if the user changed their mind more than once.
        for message in reversed(self.store.recent_messages(int(user_id), limit=100)):
            if message.get("role") != "user":
                continue
            text = strip_memory_prefix(str(message.get("text", "")))
            update = {**parse_preference_update(text), **_legacy_style_update(text)}
            update.pop("user_memory_note", None)
            result.update(update)
        return result

    @staticmethod
    def _normalize_saved(saved: Any) -> dict[str, Any]:
        result = dict(DEFAULT_PREFERENCES)
        if isinstance(saved, dict):
            response_format = saved.get("response_format")
            markdown_allowed = saved.get("markdown_allowed")
            if response_format == "plain_text" or markdown_allowed is False:
                result["response_format"] = "plain_text"
                result["markdown_allowed"] = False
            elif response_format == "auto" or markdown_allowed is True:
                result["response_format"] = "auto"
                result["markdown_allowed"] = True
            if saved.get("screenshot_delivery") == "explicit_only":
                result["screenshot_delivery"] = "explicit_only"
            if saved.get("sticker_identity_lookup") == "search_without_asking":
                result["sticker_identity_lookup"] = "search_without_asking"
            if isinstance(saved.get("tone"), str) and saved["tone"] in {"friendly", "conversational", "formal"}:
                result["tone"] = saved["tone"]
            if isinstance(saved.get("address"), str) and saved["address"] in {"informal", "formal"}:
                result["address"] = saved["address"]
            notes = saved.get("user_memory_notes")
            if isinstance(notes, list):
                result["user_memory_notes"] = [str(note)[:_MAX_NOTE_CHARS] for note in notes[-_MAX_NOTES:] if str(note).strip()]
        return result

    @staticmethod
    def _legacy_view(saved: dict[str, Any]) -> dict[str, str]:
        return {key: str(saved[key]) for key in ("tone", "address") if key in saved}

    def get(self, user_id: int) -> dict[str, Any]:
        """Return merged preferences, migrating legacy state and historical user messages once."""
        key = self.state_key(user_id)
        raw = self.store.get_state(key, None)
        migration_key = f"user_preferences_history_migrated:{int(user_id)}:2"
        if raw is None:
            legacy = self.store.get_state(self.legacy_state_key(user_id), None)
            source = self._historical_update(user_id)
            if isinstance(legacy, dict):
                source = {**source, **legacy}
            saved = self._normalize_saved(source)
            self.store.set_state(key, saved)
            self.store.set_state(self.legacy_state_key(user_id), self._legacy_view(saved))
            self.store.set_state(migration_key, True)
            return dict(saved)

        saved = self._normalize_saved(raw)
        if not self.store.get_state(migration_key, False):
            # Earlier versions initialized defaults before they understood these requests.
            # Recover older explicit preferences once, then keep new user updates normally.
            historical = self._historical_update(user_id)
            legacy = self.store.get_state(self.legacy_state_key(user_id), {}) or {}
            historical = {**historical, **{key: value for key, value in legacy.items() if key in {"tone", "address"}}}
            for name, value in historical.items():
                if name not in saved or saved.get(name) == DEFAULT_PREFERENCES.get(name):
                    saved[name] = value
            saved = self._normalize_saved(saved)
            self.store.set_state(migration_key, True)
        if saved != raw:
            self.store.set_state(key, saved)
        return dict(saved)

    def legacy_preferences(self, user_id: int) -> dict[str, str]:
        saved = self.get(user_id)
        legacy = self._legacy_view(saved)
        if self.store.get_state(self.legacy_state_key(user_id), None) != legacy:
            self.store.set_state(self.legacy_state_key(user_id), legacy)
        return legacy

    def remember(
        self, user_id: int, text: str, *, semantic_update: dict[str, Any] | None = None
    ) -> dict[str, Any]:
        """Persist a direct supported update and record whether this turn actually saved one."""
        saved = self.get(user_id)
        parsed = parse_preference_update(text)
        note = parsed.pop("user_memory_note", None)
        update = dict(parsed)

        if isinstance(semantic_update, dict):
            response_format = semantic_update.get("response_format")
            markdown_allowed = semantic_update.get("markdown_allowed")
            if response_format == "plain_text" or markdown_allowed is False:
                update.update(response_format="plain_text", markdown_allowed=False)
            elif (isinstance(response_format, str) and response_format in {"auto", "markdown"}) or markdown_allowed is True:
                update.update(response_format="auto", markdown_allowed=True)
            tone = semantic_update.get("tone")
            if isinstance(tone, str) and tone in {"friendly", "conversational", "formal"}:
                update["tone"] = semantic_update["tone"]
            address = semantic_update.get("address")
            if isinstance(address, str) and address in {"informal", "formal"}:
                update["address"] = semantic_update["address"]
            if semantic_update.get("screenshot_delivery") == "explicit_only":
                update["screenshot_delivery"] = "explicit_only"
            if semantic_update.get("sticker_identity_lookup") == "search_without_asking":
                update["sticker_identity_lookup"] = "search_without_asking"

        if note:
            notes = list(saved.get("user_memory_notes", []))
            if note not in notes:
                notes.append(note)
            notes = notes[-_MAX_NOTES:]
            update["user_memory_notes"] = notes

        receipt: dict[str, Any] = {"ok": False, "updated": {}, "time": _utc_now()}
        if update:
            new_saved = {**saved, **update}
            self.store.set_state(self.state_key(user_id), new_saved)
            self.store.set_state(self.legacy_state_key(user_id), self._legacy_view(new_saved))
            if self.store.get_state(self.state_key(user_id), None) != new_saved:
                raise RuntimeError("preference_readback_failed")
            saved = new_saved
            receipt.update(ok=True, updated=update)
        else:
            receipt["reason"] = "no_supported_update"
        self.store.set_state(self.receipt_key(user_id), receipt)
        return dict(saved)

    def last_receipt(self, user_id: int) -> dict[str, Any] | None:
        value = self.store.get_state(self.receipt_key(user_id), None)
        return dict(value) if isinstance(value, dict) else None


def preferences_for(store, user_id: int) -> dict[str, Any]:
    return PreferenceStore(store).get(user_id)


def remember_preferences(
    store, user_id: int, text: str, *, semantic_update: dict[str, Any] | None = None
) -> dict[str, Any]:
    return PreferenceStore(store).remember(user_id, text, semantic_update=semantic_update)
