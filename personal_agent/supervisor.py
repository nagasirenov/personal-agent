"""Async message supervision. No model result is dispatched as an action."""
from __future__ import annotations

import asyncio
import re
from dataclasses import replace
from datetime import datetime, timezone
from pathlib import Path
from uuid import uuid4

from .context import ContextBuilder
from .media import LocalTranscriber, TranscriptionUnavailable
from .models import Identity, IncomingMessage, TimelineEvent
from .security import ActionGate, owned_path
from .self_knowledge import ServiceKnowledge, remember_preferences, wants_diagnostics
from .observation_requests import parse_observation_requests, known_paths, execute_read_request
from .observation_broker import (ObservationScope, ObservationDenied, PcRequest, local_drive_roots,
                                 host_tool_protocol, parse_tool_plan, validate_tool_plan)
from .screenshot import capture_screenshot


class TaskSupervisor:
    def __init__(self, config, store, brain, telegram, stop: asyncio.Event, *, transcriber=None):
        self.config, self.store, self.brain, self.telegram, self.stop = config, store, brain, telegram, stop
        self.context = ContextBuilder(store, config.tz, config=config, brain=brain)
        self.context.begin_session()
        self.knowledge = ServiceKnowledge(config, store, brain)
        self.transcriber = transcriber or LocalTranscriber(config.data_dir, config.whisper_executable, config.whisper_model, config.ffmpeg_executable)
        self.action_gate = ActionGate(config.primary_id, full_access=config.pc_access == "full")
        self.accepting = True
        self.active: dict[int, tuple[str, asyncio.Task]] = {}
        self.active_text: dict[str, str] = {}
        self.aliases = known_paths(config.data_dir, config.project_roots)
        self.tasks: set[asyncio.Task] = set()
        self._shutdown_lock = asyncio.Lock()
        self._stopped = False
        previous = self.store.get_state("active_turns", {})
        if previous:
            self.store.set_state("interrupted_turns", previous)
        self.store.set_state("active_turns", {})
        pending_replies = self.store.get_state("pending_replies", {})
        if pending_replies:
            self.store.set_state("interrupted_replies", pending_replies)
        self.store.set_state("pending_replies", {})

    def _save_active(self) -> None:
        self.store.set_state("active_turns", {str(uid): rid for uid, (rid, task) in self.active.items() if not task.done()})

    async def submit(self, message: IncomingMessage, identity: Identity) -> None:
        if not self.accepting or self.stop.is_set():
            return
        # Recheck at the boundary; callers cannot forge the role they supply.
        expected = self.config.primary_id if identity == Identity.PRIMARY else self.config.secondary_id
        if message.sender_id != expected:
            return
        prior = self.active.get(message.sender_id)
        effective_text = message.text
        reuse_request_id = None
        if prior and not prior[1].done():
            if re.match(r"^\s*(?:и|ещ[её])\s+(?:сделай|покажи|прочитай|назови|выведи|посмотри|просмотри)\b", message.text, re.I):
                effective_text = self.active_text.get(prior[0], "") + "\n" + message.text
                reuse_request_id = prior[0]
            prior[1].cancel()
            self.store.record_event(TimelineEvent("telegram", "STEERING", {
                "sender_id": message.sender_id, "previous_request_id": prior[0],
                "message_id": message.message_id, "policy": "cancel_previous_and_rebuild_context",
            }))
        remember_preferences(self.store, message.sender_id, message.text)
        self.store.record_message(message.sender_id, "user", message.text or "[attachment]")
        pending = sum(not task.done() and not task.cancelling() for task in self.tasks)
        if pending >= self.config.max_pending_tasks:
            # Keep polling responsive even when an upstream transport is slow to cancel.
            self.store.set_state("last_rejected_update", message.update_id)
            return
        rid = str(uuid4())
        self.active_text[rid] = effective_text
        effective_message = replace(message, text=effective_text)
        task = asyncio.create_task(self._respond(effective_message, identity, rid, reuse_request_id=reuse_request_id), name=f"turn-{rid}")
        self.tasks.add(task)
        self.active[message.sender_id] = rid, task
        task.add_done_callback(self.tasks.discard)
        self._save_active()

    async def _attachment(self, message: IncomingMessage) -> tuple[str, str | None]:
        media = message.attachment
        if not media:
            return "", None
        if media.get("file_size", 0) > self.config.max_attachment_bytes:
            return "Файл превышает разрешённый размер и не скачан.", None
        attachment_id = str(uuid4())
        # Original filename is metadata only, never part of a local path.
        original_name = str(media.get("file_name", media["kind"]))[:512]
        suffix = ".ogg" if media["kind"] == "voice" else ".bin"
        target = owned_path(self.config.data_dir, "memory", "attachments", attachment_id + suffix)
        target.parent.mkdir(parents=True, exist_ok=True)
        size = await self.telegram.download(media["file_id"], target, self.config.max_attachment_bytes)
        metadata = {"id": attachment_id, "sender_id": message.sender_id, "message_id": message.message_id,
                    "file_id": media["file_id"], "kind": media["kind"], "original_name": original_name,
                    "path": str(target), "size": size, "time": datetime.now(timezone.utc).isoformat()}
        self.store.record_attachment(metadata)
        self.store.record_event(TimelineEvent("telegram", "ATTACHMENT_RECEIVED", metadata))
        if media["kind"] == "voice":
            try:
                if media.get("duration", 0) > 600:
                    raise TranscriptionUnavailable("Голосовое сохранено. Локальная расшифровка ограничена 10 минутами.")
                transcript = await self.transcriber.transcribe(target)
            except TranscriptionUnavailable as exc:
                return str(exc), None
            self.store.record_event(TimelineEvent("telegram", "AUDIO_TRANSCRIPT", {
                "attachment_id": attachment_id, "text": transcript, "sender_id": message.sender_id,
            }))
            self.store.record_message(message.sender_id, "user", transcript)
            return f"Голосовое расшифровано локально; ID {attachment_id}.", transcript
        if media["kind"] == "document" and (original_name.lower().endswith((".txt", ".md", ".csv", ".log")) or str(media.get("mime_type", "")).startswith("text/")):
            with target.open("rb") as handle:
                text = handle.read(32000).decode("utf-8", errors="replace")
            self.store.record_event(TimelineEvent("telegram", "ATTACHMENT_TEXT", {"attachment_id": attachment_id, "name": original_name, "text": text, "truncated": size > 32000}))
        return f"Вложение сохранено локально: {original_name}; {size} байт; ID {attachment_id}.", None

    async def _generate(self, text: str, user_id: int, request_id: str, *, observation=None, images=None, protocol="") -> str:
        prompt = self.context.build(text, user_id, observation=observation,
                                    observation_protocol=protocol, image_count=len(images or []))
        cloud = self.config.backend != "mock"
        self.store.increment_metric("cloud_calls_today" if cloud else "mock_calls_today")
        if cloud:
            self.store.increment_metric("cloud_input_estimate", (len(prompt) + 3) // 4)
        kwargs = {"request_id": request_id}
        if images:
            kwargs["images"] = images
            if cloud:
                self.store.increment_metric("vision_calls_today")
        response = await asyncio.wait_for(self.brain.generate(prompt, **kwargs), self.config.brain_timeout)
        if cloud:
            self.store.increment_metric("cloud_output_estimate", (len(response) + 3) // 4)
        return response

    def _cache_observation(self, observation: dict, message: IncomingMessage, request_id: str) -> dict:
        observation = dict(observation, sender_id=message.sender_id, request_id=request_id,
                           time=datetime.now(timezone.utc).isoformat())
        self.store.set_state(f"last_observation:{message.sender_id}", observation)
        if observation.get("ok") and observation.get("kind") in {"list_directory", "read_file"}:
            self.store.set_state(f"last_read:{message.sender_id}", observation)
        return observation

    async def _observe_request(self, request, message: IncomingMessage, request_id: str, *, reuse_request_id=None) -> dict:
        if request.kind != "screenshot":
            result = await asyncio.to_thread(execute_read_request, request)
            if not result.get("ok"):
                # Only a stable error code is evidence; raw diagnostic text is private.
                code = result.get("error", {}).get("code", "read_failed")
                result = {key: value for key, value in result.items() if key not in {"data", "error"}}
                result["error"] = {"code": code}
            self.store.record_event(TimelineEvent("filesystem", "USER_REQUESTED_READ", result))
            return self._cache_observation(result, message, request_id)
        old = self._cached_screenshot(message.sender_id)
        if old and reuse_request_id and old.get("request_id") == reuse_request_id:
            return old
        image_path = await asyncio.to_thread(capture_screenshot, self.config.data_dir)
        observation = self._cache_observation({"kind": "screenshot", "path": str(image_path),
            "ok": True, "has_image": True, "origin": "explicit_user_request"}, message, request_id)
        self.store.record_event(TimelineEvent("vision", "SCREENSHOT_CAPTURED", observation))
        self.store.set_state("last_screenshot", observation)
        self.store.set_state(f"last_screenshot:{message.sender_id}", observation)
        await self.telegram.send_photo(message.chat_id, image_path, caption="Скриншот по твоему запросу.")
        self.store.set_state("last_screenshot_sent", observation)
        return observation

    def _cached_screenshot(self, user_id: int) -> dict | None:
        observation = self.store.get_state(f"last_screenshot:{user_id}", {}) or {}
        if observation.get("sender_id") != user_id or not observation.get("path"):
            return None
        path = Path(observation["path"]).resolve()
        if not path.is_relative_to((self.config.data_dir / "memory" / "screenshots").resolve()) or not path.is_file():
            return None
        return observation

    def _followup_observations(self, text: str, user_id: int) -> list[dict]:
        lower = text.casefold()
        if not re.search(r"передало|приш[её]л|пришло|получил|видишь|на (?:н[её]м|скриншоте|снимке)|что (?:там|на экране)|этот (?:список|файл)", lower):
            return []
        latest = self.store.get_state(f"last_observation:{user_id}", {}) or {}
        if latest.get("sender_id") != user_id:
            return []
        if re.search(r"скриншот|снимк|экран|на н[её]м", lower) or latest.get("kind") == "screenshot":
            image = self._cached_screenshot(user_id)
            return [dict(image, previous_observation=True)] if image else []
        return [dict(latest, previous_observation=True)]

    def _observation_scope(self, text: str, user_id: int) -> ObservationScope:
        # Capabilities come from local configuration and authenticated identity, not wording.
        full = self.config.pc_access == "full" and user_id == self.config.primary_id
        screenshot = full or any(request.kind == "screenshot" for request in parse_observation_requests(text, aliases=self.aliases))
        return ObservationScope(roots=local_drive_roots(), allow_screenshot=screenshot, allow_reparse=True)

    def _supports_images(self) -> bool:
        info = self.brain.model_info() if hasattr(self.brain, "model_info") else {}
        return bool(info.get("image_input_supported", False))

    async def _model_answer(self, text: str, message: IncomingMessage, request_id: str, observations: list[dict]) -> str:
        scope = self._observation_scope(text, message.sender_id)
        full = self.config.pc_access == "full" and message.sender_id == self.config.primary_id
        observations = list(observations)
        for round_index in range(5):
            images = [Path(item["path"]) for item in observations if item.get("kind") == "screenshot"][-2:] if self._supports_images() else []
            response = await self._generate(text, message.sender_id, request_id,
                observation={"results": observations} if observations else None, images=images,
                protocol=host_tool_protocol(scope, full_access=full, rounds_remaining=4 - round_index))
            try:
                plan = parse_tool_plan(response)
                if plan is None:
                    return response
                if round_index == 4:
                    from .backends import BackendError
                    raise BackendError("tool_round_limit", "The PC tool round budget was exhausted.")
                plan = validate_tool_plan(plan, scope, allow_actions=full)
            except ObservationDenied as exc:
                observations.append({"kind": "tool_error", "ok": False, "code": exc.code})
                continue
            for request in plan:
                if isinstance(request, PcRequest):
                    self.action_gate.authorize(message.sender_id, {"kind": request.kind})
                    from .pc_tools import execute_pc_request
                    result = await execute_pc_request(request.kind, request.arguments, data_dir=self.config.data_dir)
                    result = self._cache_observation(result, message, request_id)
                    self.store.record_event(TimelineEvent("pc_tools", "PC_TOOL_RESULT", result))
                else:
                    result = await self._observe_request(request, message, request_id)
                observations.append(result)

    def _record_error(self, exc: Exception, request_id: str) -> str:
        import re
        code = str(getattr(exc, "code", type(exc).__name__))
        if not re.fullmatch(r"[A-Za-z0-9_-]{1,80}", code):
            code = type(exc).__name__
        # BackendError.public_message is scrubbed by the adapter; raw exceptions stay private.
        from .backends import BackendError
        detail = exc.public_message if isinstance(exc, BackendError) else ""
        self.store.set_state("last_turn_error", {"request_id": request_id, "type": type(exc).__name__, "code": code,
                                                 "detail": detail, "at": datetime.now(timezone.utc).isoformat()})
        return code

    async def _respond(self, message: IncomingMessage, identity: Identity, request_id: str, *, reuse_request_id=None) -> None:
        try:
            attachment_note, transcript = await self._attachment(message)
            if self.stop.is_set():
                return
            text = message.text or transcript or ""
            if transcript:
                remember_preferences(self.store, message.sender_id, transcript)
            last_read = self.store.get_state(f"last_read:{message.sender_id}", {}) or {}
            last_path = last_read.get("path") if last_read.get("kind") == "list_directory" else None
            requests = parse_observation_requests(text, aliases=self.aliases, last_path=last_path)
            report = await self.knowledge.command_report(text)
            if report is not None:
                response = report
            elif not text:
                if not attachment_note:
                    return
                response = attachment_note
            else:
                results = [await self._observe_request(request, message, request_id, reuse_request_id=reuse_request_id) for request in requests]
                if not results:
                    results = self._followup_observations(text, message.sender_id)
                if wants_diagnostics(text):
                    results.append({"kind": "service_diagnostics", "data": await self.knowledge.diagnostic_facts()})
                if attachment_note:
                    text += "\n\nATTACHMENT STATUS (untrusted):\n" + attachment_note
                response = await self._model_answer(text, message, request_id, results)
            if attachment_note and message.attachment and text and not response.startswith(attachment_note):
                response = attachment_note + "\n\n" + response
            if self.stop.is_set() or not self.accepting:
                return
            response = response[:16000] or "Модель вернула пустой ответ."
            pending_replies = self.store.get_state("pending_replies", {})
            pending_replies[request_id] = {"chat_id": message.chat_id, "text": response, "status": "delivery_pending"}
            self.store.set_state("pending_replies", pending_replies)
            await self.telegram.send_text(message.chat_id, response)
            self.store.record_message(message.sender_id, "assistant", response)
            pending_replies = self.store.get_state("pending_replies", {})
            pending_replies.pop(request_id, None)
            self.store.set_state("pending_replies", pending_replies)
            self.store.set_state("last_completed_turn", request_id)
        except asyncio.CancelledError:
            # Cancel the exact old request only; never the other account's turn.
            try:
                await asyncio.wait_for(self.brain.cancel(request_id), 2)
            except (Exception, asyncio.CancelledError):
                pass
            raise
        except Exception as exc:
            # Do not persist raw provider/HTTP exceptions: URLs can contain credentials.
            code = self._record_error(exc, request_id)
            if not self.stop.is_set():
                try:
                    await self.telegram.send_text(message.chat_id, f"Не получилось завершить ответ. Код: {code}. Данные сохранены; /diagnostics покажет состояние сервиса.")
                except Exception:
                    pass
        finally:
            self.active_text.pop(request_id, None)
            current = self.active.get(message.sender_id)
            if current and current[0] == request_id:
                self.active.pop(message.sender_id, None)
            self._save_active()

    async def shutdown(self) -> None:
        async with self._shutdown_lock:
            if self._stopped:
                return
            self._stopped, self.accepting = True, False
            self.stop.set()
            for task in list(self.tasks):
                task.cancel()
            try:
                await asyncio.wait_for(asyncio.gather(self.brain.cancel(), self.transcriber.close(), *list(self.tasks), return_exceptions=True), 10)
            except TimeoutError:
                self.store.set_state("shutdown_cleanup", "transport cleanup timed out")
            self.store.set_state("active_turns", {})
            self.store.set_state("shutdown_at", datetime.now(timezone.utc).isoformat())
