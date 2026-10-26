"""Opt-in real backend check using synthetic files/images and an offline Telegram sink."""
from __future__ import annotations

import asyncio
from dataclasses import replace
from pathlib import Path
import struct
import sys
import tempfile
from unittest.mock import patch
import zlib

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from personal_agent.backends import CodexSubscriptionBackend
from personal_agent.config import load_config
from personal_agent.memory import MemoryStore
from personal_agent.models import Identity, IncomingMessage
from personal_agent.observation_broker import parse_tool_plan
from personal_agent.observation_requests import known_paths
from personal_agent.supervisor import TaskSupervisor


class OfflineTelegram:
    def __init__(self):
        self.answers = []
        self.photos = 0

    async def send_text(self, chat_id, text):
        self.answers.append(text)

    async def send_photo(self, chat_id, path, caption=""):
        self.photos += 1


class NoTranscriber:
    async def close(self):
        pass


class TracedBackend(CodexSubscriptionBackend):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.image_counts = []
        self.tool_kinds = []

    async def generate(self, prompt, request_id=None, *, images=None):
        self.image_counts.append(len(images or []))
        answer = await super().generate(prompt, request_id, images=images)
        try:
            self.tool_kinds.extend(request.kind for request in (parse_tool_plan(answer) or []))
        except Exception:
            pass
        return answer


def red_png(path: Path):
    def chunk(kind, data):
        return struct.pack(">I", len(data)) + kind + data + struct.pack(">I", zlib.crc32(kind + data))
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", struct.pack(">IIBBBBB", 64, 64, 8, 2, 0, 0, 0))
                     + chunk(b"IDAT", zlib.compress((b"\x00" + b"\xff\x00\x00" * 64) * 64)) + chunk(b"IEND", b""))


async def main():
    source = load_config(Path("config.local.toml"))
    with tempfile.TemporaryDirectory(prefix="personal-agent-conversation-") as folder:
        root = Path(folder)
        project = root / "fixture"
        project.mkdir()
        desktop = root / "desktop"
        desktop.mkdir()
        (desktop / "DESKTOP_FOUND_8372.txt").write_text("fixture", encoding="utf-8")
        control = "CONTROL_READ_OK_82437"
        (project / "README.md").write_text("Control line: " + control, encoding="utf-8")
        config = replace(source, data_dir=root, token="", primary_id=101, secondary_id=202,
                         project_roots={"fixture": project}, brain_timeout=150, pc_access="full")
        brain = TracedBackend(model=config.model, codex_command=config.codex_executable,
                              reasoning_effort=config.reasoning, work_root=root)
        telegram = OfflineTelegram()
        store = MemoryStore(root)
        supervisor = TaskSupervisor(config, store, brain, telegram, asyncio.Event(), transcriber=NoTranscriber())

        async def ask(text):
            index = len(telegram.answers) + 1
            await supervisor.submit(IncomingMessage(index, 101, 101, text, index), Identity.PRIMARY)
            await asyncio.gather(*list(supervisor.tasks))
            return telegram.answers[-1]

        try:
            if "--desktop-only" in sys.argv:
                actual_desktop = Path(known_paths(root)["desktop"])
                expected = [entry.name for entry in actual_desktop.iterdir()][:100]
                answer = await ask("Ну слава богу, скажи какие файлы на рабочем столе есть")
                matched = sum(name.casefold() in answer.casefold() for name in expected)
                print(f"actual_desktop_entries={len(expected)}; names_in_model_answer={matched}", flush=True)
                passed = matched >= min(3, len(expected)) and "list_directory" in brain.tool_kinds
                print("original_desktop_phrase=" + ("PASS" if passed else "FAIL"), flush=True)
                print("telegram_network_calls=0", flush=True)
                if not passed:
                    raise SystemExit(1)
                return
            image = root / "memory" / "screenshots" / "synthetic-red.png"
            red_png(image)
            with patch("personal_agent.supervisor.capture_screenshot", return_value=image):
                first = await ask("Сделай скриншот и скажи, какого цвета изображение. Ответь одним словом.")
            second = await ask("И ты его видишь? Назови основной цвет одним словом.")
            read = await ask("Проанализируй проект fixture: прочитай README.md и назови контрольную строку из него.")
            memory = await ask("Назови только путь твоей папки памяти.")
            with patch("personal_agent.observation_requests._desktop_path", return_value=desktop):
                store.record_message(101, "user", "Можешь посмотреть, что у меня на рабочем столе?")
                store.record_message(101, "assistant", "Путь известен, но списка содержимого у меня сейчас нет.")
                recovered = await ask("Так открой сам и перечисли содержимое.")
            target = root / "host-action.txt"
            action = await ask(f"Создай файл {target} с текстом HOST_WRITE_OK_713, затем через PowerShell проверь его содержимое. Сообщи результат.")
            checks = {
                "first_image_understood": "красн" in first.casefold() or "red" in first.casefold(),
                "followup_image_understood": "красн" in second.casefold() or "red" in second.casefold(),
                "one_photo_two_image_inputs": telegram.photos == 1 and brain.image_counts[:2] == [1, 1],
                "model_read_plan_executed": control in read,
                "real_memory_path_answered": str(root / "memory").casefold() in memory.casefold(),
                "vague_followup_reads_desktop": "DESKTOP_FOUND_8372.txt" in recovered,
                "real_model_writes_and_runs_powershell": target.exists() and target.read_text(encoding="utf-8-sig") == "HOST_WRITE_OK_713"
                    and "run_command" in brain.tool_kinds and "HOST_WRITE_OK_713" in action,
            }
            for name, passed in checks.items():
                print(f"{name}={'PASS' if passed else 'FAIL'}", flush=True)
            print(f"model_calls={len(brain.image_counts)}; telegram_network_calls=0", flush=True)
            if not all(checks.values()):
                print("Synthetic answers:", repr([first, second, read, memory, recovered, action]), flush=True)
                raise SystemExit(1)
        finally:
            await supervisor.shutdown()
            await brain.close()
            store.close()


if __name__ == "__main__":
    asyncio.run(main())
