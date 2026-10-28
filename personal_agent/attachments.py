"""Incoming attachment preparation for local storage, transcription, and vision."""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
import struct
from typing import BinaryIO
from uuid import uuid4

from .animated_stickers import StickerRenderError, render_sticker
from .media import TranscriptionUnavailable
from .models import IncomingMessage, TimelineEvent
from .security import owned_path
from .telegram import TelegramError
from .video import VIDEO_SUFFIXES, VideoAnalyzer, VideoError

_VIDEO_KINDS = {"video", "video_note", "animation"}


@dataclass(frozen=True)
class AttachmentResult:
    notice: str = ""
    transcript: str | None = None
    images: list[Path] = field(default_factory=list)
    observations: list[dict] = field(default_factory=list)
    document_text: str | None = None


@dataclass(frozen=True)
class _ImageInfo:
    format: str
    width: int
    height: int
    animated: bool = False


class AttachmentPipeline:
    """Normalize one Telegram attachment without relying on its filename."""

    MAX_IMAGE_BYTES = 10 * 1024 * 1024
    MAX_IMAGE_DIMENSION = 16_384
    MAX_IMAGE_PIXELS = 40_000_000

    def __init__(self, config, store, telegram, transcriber, video_analyzer: VideoAnalyzer | None = None) -> None:
        self.config = config
        self.store = store
        self.telegram = telegram
        self.transcriber = transcriber
        self._video_analyzer = video_analyzer

    @property
    def video_analyzer(self) -> VideoAnalyzer:
        if self._video_analyzer is None:
            self._video_analyzer = VideoAnalyzer(
                self.config.data_dir, ffmpeg=str(getattr(self.config, "ffmpeg_executable", "") or ""),
                transcriber=self.transcriber)
        return self._video_analyzer

    async def prepare(self, message: IncomingMessage) -> AttachmentResult:
        media = message.attachment
        if not media:
            return AttachmentResult()

        limit = int(self.config.max_attachment_bytes)
        reported_size = media.get("file_size", 0)
        if isinstance(reported_size, int) and reported_size > limit:
            return AttachmentResult(notice="Файл превышает разрешённый размер и не скачан.")

        kind = str(media.get("kind", "attachment"))
        image_limit = min(limit, self.MAX_IMAGE_BYTES)
        sticker_animated = kind == "sticker" and bool(media.get("is_animated"))
        sticker_video = kind == "sticker" and bool(media.get("is_video"))
        direct_image = kind == "photo" or (kind == "sticker" and not sticker_animated and not sticker_video)
        if direct_image and isinstance(reported_size, int) and reported_size > image_limit:
            return AttachmentResult(notice="Изображение превышает лимит 10 МиБ для анализа и не скачано.")

        attachment_id = str(uuid4())
        original_name = str(media.get("file_name", self._default_name(media, kind)))[:512]
        suffix = self._suffix(media, kind)
        target = owned_path(self.config.data_dir, "memory", "attachments", attachment_id + suffix)
        target.parent.mkdir(parents=True, exist_ok=True)
        download_limit = image_limit if direct_image else limit
        try:
            size = await self.telegram.download(str(media["file_id"]), target, download_limit)
        except TelegramError as exc:
            target.unlink(missing_ok=True)
            return AttachmentResult(notice=f"Не удалось скачать вложение: {exc}")
        actual_size = target.stat().st_size
        size = actual_size if actual_size != size else size
        if size > download_limit:
            target.unlink(missing_ok=True)
            return AttachmentResult(notice="Вложение превышает разрешённый размер и удалено.")

        now = datetime.now(timezone.utc).isoformat()
        metadata = {
            "id": attachment_id,
            "sender_id": message.sender_id,
            "message_id": message.message_id,
            "file_id": media["file_id"],
            "kind": kind,
            "original_name": original_name,
            "path": str(target),
            "size": size,
            "time": now,
        }
        if kind == "sticker":
            metadata["sticker"] = {
                "format": str(media.get("sticker_format", "webm" if sticker_video else "tgs" if sticker_animated else "webp")),
                "is_animated": sticker_animated,
                "is_video": sticker_video,
                "emoji": media.get("emoji"),
                "set_name": media.get("set_name"),
                "width": media.get("width"),
                "height": media.get("height"),
            }

        image_info = None if kind == "voice" else self._inspect_image(target)
        if image_info is not None:
            metadata["image"] = {
                "format": image_info.format,
                "width": image_info.width,
                "height": image_info.height,
                "animated": image_info.animated,
            }

        self.store.record_attachment(metadata)
        self.store.record_event(TimelineEvent("telegram", "ATTACHMENT_RECEIVED", metadata))

        if kind == "voice":
            return await self._prepare_voice(message, media, target, attachment_id)

        if kind == "sticker" and (sticker_animated or sticker_video):
            return await self._prepare_animated_sticker(
                message, target, attachment_id, now,
                sticker_format="webm" if sticker_video else "tgs",
            )

        if self._is_video(media, kind, original_name):
            return await self._prepare_video(message, target, attachment_id, now, original_name)

        if image_info is not None:
            if size > image_limit:
                return AttachmentResult(
                    notice=f"Изображение сохранено локально: {original_name}; {size} байт; ID {attachment_id}. Для анализа оно превышает лимит 10 МиБ.",
                )
            if not self._safe_dimensions(image_info):
                return AttachmentResult(
                    notice=f"Изображение сохранено локально: {original_name}; ID {attachment_id}. Размеры изображения слишком велики для безопасного анализа.",
                )
            observation = {
                "kind": "attachment_image",
                "path": str(target),
                "ok": True,
                "has_image": True,
                "origin": "telegram_attachment",
                "attachment_id": attachment_id,
                "sender_id": message.sender_id,
                "message_id": message.message_id,
                "source_kind": kind,
                "image_format": image_info.format,
                "width": image_info.width,
                "height": image_info.height,
                "time": now,
            }
            self.store.set_state(f"last_attachment_image:{message.sender_id}", observation)
            self.store.record_event(TimelineEvent("telegram", "ATTACHMENT_IMAGE", observation))
            return AttachmentResult(
                notice=f"Изображение сохранено локально и подготовлено для анализа: {original_name}; {size} байт; ID {attachment_id}.",
                images=[target],
                observations=[observation],
            )

        if kind == "document" and (
            original_name.lower().endswith((".txt", ".md", ".csv", ".log"))
            or str(media.get("mime_type", "")).startswith("text/")
        ):
            with target.open("rb") as handle:
                text = handle.read(32_000).decode("utf-8-sig", errors="replace")
            self.store.record_event(TimelineEvent("telegram", "ATTACHMENT_TEXT", {
                "attachment_id": attachment_id,
                "sender_id": message.sender_id,
                "name": original_name,
                "text": text,
                "truncated": size > 32_000,
            }))
            observation = {
                "kind": "attachment_text", "ok": True, "has_image": False,
                "path": str(target), "attachment_id": attachment_id,
                "sender_id": message.sender_id, "message_id": message.message_id,
                "name": original_name, "text": text,
                "truncated": size > 32_000, "time": now,
            }
            self.store.set_state(f"last_text_attachment:{message.sender_id}", observation)
            self.store.record_event(TimelineEvent("telegram", "ATTACHMENT_TEXT_CONTENT", observation))
            return AttachmentResult(
                notice=f"Текстовый файл сохранён локально: {original_name}; {size} байт; ID {attachment_id}.",
                observations=[observation], document_text=text,
            )

        if direct_image:
            return AttachmentResult(
                notice=f"Вложение сохранено локально: {original_name}; {size} байт; ID {attachment_id}. Формат изображения не распознан, поэтому в анализ оно не передано.",
            )
        return AttachmentResult(
            notice=f"Вложение сохранено локально: {original_name}; {size} байт; ID {attachment_id}.",
        )

    async def _prepare_animated_sticker(
        self,
        message: IncomingMessage,
        target: Path,
        attachment_id: str,
        received_at: str,
        *,
        sticker_format: str,
    ) -> AttachmentResult:
        try:
            rendered = await render_sticker(
                target,
                root=self.config.data_dir,
                format=sticker_format,
                ffmpeg=str(getattr(self.config, "ffmpeg_executable", "") or ""),
            )
            if not rendered.images or len(rendered.images) != len(rendered.timestamps) or len(rendered.images) > 4:
                raise StickerRenderError("sticker_invalid_frames")

            frame_root = (Path(self.config.data_dir) / "memory" / "attachments").resolve()
            images: list[Path] = []
            observations: list[dict] = []
            timestamps = [float(value) for value in rendered.timestamps]
            total = len(rendered.images)
            duration = float(rendered.duration)
            for index, (frame, frame_time) in enumerate(zip(rendered.images, timestamps)):
                frame = Path(frame).resolve()
                if not frame.is_relative_to(frame_root) or not frame.is_file():
                    raise StickerRenderError("sticker_invalid_frames")
                info = self._inspect_image(frame)
                if info is None or info.format != "png" or not self._safe_dimensions(info):
                    raise StickerRenderError("sticker_invalid_frames")
                images.append(frame)
                observations.append({
                    "kind": "attachment_image",
                    "path": str(frame),
                    "ok": True,
                    "has_image": True,
                    "origin": "telegram_attachment",
                    "attachment_id": attachment_id,
                    "sender_id": message.sender_id,
                    "message_id": message.message_id,
                    "source_kind": "sticker",
                    "sticker_format": sticker_format,
                    "image_format": "png",
                    "width": info.width,
                    "height": info.height,
                    "time": received_at,
                    "frame_index": index,
                    "frame_time": frame_time,
                    "duration": duration,
                    "total": total,
                })

            cache = dict(
                observations[0],
                path=str(images[0]),
                paths=[str(path) for path in images],
                timestamps=timestamps,
                duration=duration,
            )
            self.store.set_state(f"last_attachment_image:{message.sender_id}", cache)
            for observation in observations:
                self.store.record_event(TimelineEvent("telegram", "ATTACHMENT_IMAGE", observation))
            return AttachmentResult(
                notice=f"Стикер {sticker_format.upper()} сохранён локально; подготовлено {total} кадр(а) для анализа; ID {attachment_id}.",
                images=images,
                observations=observations,
            )
        except StickerRenderError as exc:
            observation = {
                "kind": "attachment_render_error",
                "media_kind": "sticker",
                "format": sticker_format,
                "has_image": False,
                "path": str(target),
                "attachment_id": attachment_id,
                "sender_id": message.sender_id,
                "message_id": message.message_id,
                "time": received_at,
                "ok": False,
                "status": exc.code,
            }
            self.store.set_state(f"last_attachment_image:{message.sender_id}", observation)
            return AttachmentResult(
                notice=f"{exc.public_message} Исходный стикер {sticker_format.upper()} сохранён локально; ID {attachment_id}.",
                observations=[observation],
            )

    async def _prepare_video(self, message: IncomingMessage, target: Path, attachment_id: str, received_at: str,
                             name: str) -> AttachmentResult:
        """Frames (contact sheets), transcript and metadata of a sent video, shown to the model on this turn."""
        base = {"attachment_id": attachment_id, "sender_id": message.sender_id, "message_id": message.message_id,
                "origin": "telegram_attachment", "time": received_at}
        try:
            meta = await self.video_analyzer.prepare_file(message.sender_id, target, attachment_id, name=name)
            view = await self.video_analyzer.view(message.sender_id, meta)
        except VideoError as exc:
            observation = {**base, "kind": "attachment_render_error", "media_kind": "video", "has_image": False,
                           "ok": False, "status": exc.code, "path": str(target)}
            return AttachmentResult(
                notice=f"{exc.public_message} Исходное видео сохранено локально; ID {attachment_id}.",
                observations=[observation])
        observation = {**base, **view.to_result(), "kind": "attachment_video", "ok": True,
                       "has_image": bool(view.images), "source_kind": "video"}
        transcript = view.transcript_text[:4000]
        self.store.record_event(TimelineEvent("telegram", "ATTACHMENT_VIDEO", {
            **base, "name": name, "metadata": view.metadata, "transcript_source": view.transcript_source,
            "transcript": transcript, "sheets": len(view.sheets)}))
        return AttachmentResult(
            notice=f"Видео сохранено локально, подготовлено {len(view.sheets)} лист(а) кадров; ID {attachment_id}.",
            images=list(view.images), observations=[observation])

    @staticmethod
    def _is_video(media: dict, kind: str, name: str) -> bool:
        if kind in _VIDEO_KINDS:
            return True
        return kind == "document" and (str(media.get("mime_type", "")).lower().startswith("video/")
                                       or Path(name.lower()).suffix in VIDEO_SUFFIXES - {".gif"})

    async def _prepare_voice(self, message: IncomingMessage, media: dict, target: Path, attachment_id: str) -> AttachmentResult:
        try:
            if media.get("duration", 0) > 600:
                raise TranscriptionUnavailable("Голосовое сохранено. Локальная расшифровка ограничена 10 минутами.")
            transcript = await self.transcriber.transcribe(target)
        except TranscriptionUnavailable as exc:
            return AttachmentResult(notice=str(exc))
        self.store.record_event(TimelineEvent("telegram", "AUDIO_TRANSCRIPT", {
            "attachment_id": attachment_id,
            "text": transcript,
            "sender_id": message.sender_id,
        }))
        self.store.record_message(message.sender_id, "user", transcript)
        return AttachmentResult(notice=f"Голосовое расшифровано локально; ID {attachment_id}.", transcript=transcript)

    @staticmethod
    def _default_name(media: dict, kind: str) -> str:
        if kind != "sticker":
            return kind
        sticker_format = str(media.get("sticker_format", "webp")).lower()
        return f"sticker.{sticker_format}"

    @staticmethod
    def _suffix(media: dict, kind: str) -> str:
        if kind == "voice":
            return ".ogg"
        if kind == "sticker":
            return "." + str(media.get("sticker_format", "webp")).lower()
        name_ext = Path(str(media.get("file_name", ""))).suffix.lower()
        if kind in _VIDEO_KINDS:
            return name_ext if name_ext in VIDEO_SUFFIXES else ".mp4"
        if kind == "document":
            mime = str(media.get("mime_type", "")).lower()
            if mime.startswith("video/") or name_ext in VIDEO_SUFFIXES:
                return name_ext if name_ext in VIDEO_SUFFIXES else ".mp4"
        return ".bin"

    def _safe_dimensions(self, info: _ImageInfo) -> bool:
        return (
            0 < info.width <= self.MAX_IMAGE_DIMENSION
            and 0 < info.height <= self.MAX_IMAGE_DIMENSION
            and info.width * info.height <= self.MAX_IMAGE_PIXELS
        )

    @classmethod
    def _inspect_image(cls, path: Path) -> _ImageInfo | None:
        try:
            with path.open("rb") as handle:
                head = handle.read(64 * 1024)
                if head.startswith(b"\x89PNG\r\n\x1a\n") and len(head) >= 24 and head[12:16] == b"IHDR":
                    width, height = struct.unpack(">II", head[16:24])
                    return _ImageInfo("png", width, height)
                if head[:6] in {b"GIF87a", b"GIF89a"} and len(head) >= 10:
                    width, height = struct.unpack("<HH", head[6:10])
                    return _ImageInfo("gif", width, height)
                if head.startswith(b"\xff\xd8\xff"):
                    handle.seek(0)
                    dimensions = cls._jpeg_dimensions(handle)
                    return _ImageInfo("jpeg", *dimensions) if dimensions else None
                if head[:4] == b"RIFF" and head[8:12] == b"WEBP":
                    webp = cls._webp_info(head)
                    return _ImageInfo("webp", *webp) if webp else None
        except (OSError, ValueError, struct.error):
            return None
        return None

    @staticmethod
    def _jpeg_dimensions(handle: BinaryIO) -> tuple[int, int] | None:
        if handle.read(2) != b"\xff\xd8":
            return None
        sof_markers = {0xC0, 0xC1, 0xC2, 0xC3, 0xC5, 0xC6, 0xC7, 0xC9, 0xCA, 0xCB, 0xCD, 0xCE, 0xCF}
        scanned = 2
        while scanned < 2 * 1024 * 1024:
            byte = handle.read(1)
            scanned += 1
            if byte != b"\xff":
                if not byte:
                    return None
                continue
            marker = handle.read(1)
            scanned += 1
            while marker == b"\xff":
                marker = handle.read(1)
                scanned += 1
            if not marker:
                return None
            code = marker[0]
            if code in {0x01, 0xD8, 0xD9} or 0xD0 <= code <= 0xD7:
                continue
            length_raw = handle.read(2)
            scanned += 2
            if len(length_raw) != 2:
                return None
            length = struct.unpack(">H", length_raw)[0]
            if length < 2:
                return None
            if code in sof_markers:
                frame = handle.read(5)
                if len(frame) != 5:
                    return None
                height, width = struct.unpack(">HH", frame[1:5])
                return width, height
            handle.seek(length - 2, 1)
            scanned += length - 2
        return None

    @staticmethod
    def _webp_info(data: bytes) -> tuple[int, int, bool] | None:
        if len(data) < 20 or data[:4] != b"RIFF" or data[8:12] != b"WEBP":
            return None
        offset = 12
        while offset + 8 <= len(data):
            chunk = data[offset:offset + 4]
            size = int.from_bytes(data[offset + 4:offset + 8], "little")
            payload = data[offset + 8:offset + 8 + min(size, 10)]
            if chunk == b"VP8X" and len(payload) >= 10:
                width = 1 + int.from_bytes(payload[4:7], "little")
                height = 1 + int.from_bytes(payload[7:10], "little")
                return width, height, bool(payload[0] & 0x02)
            if chunk == b"VP8 " and len(payload) >= 10 and payload[3:6] == b"\x9d\x01\x2a":
                width = int.from_bytes(payload[6:8], "little") & 0x3FFF
                height = int.from_bytes(payload[8:10], "little") & 0x3FFF
                return width, height, False
            if chunk == b"VP8L" and len(payload) >= 5 and payload[0] == 0x2F:
                bits = int.from_bytes(payload[1:5], "little")
                width = 1 + (bits & 0x3FFF)
                height = 1 + ((bits >> 14) & 0x3FFF)
                return width, height, False
            offset += 8 + size + (size & 1)
        return None
