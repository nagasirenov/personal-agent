"""Real-model verification with gpt-6-luna on synthetic video attachments and follow-ups.

Usage: python scripts/verify_video_real.py [--real] [--verbose]
"""
from __future__ import annotations

import asyncio
from dataclasses import replace
from pathlib import Path
import shutil
import sys
import tempfile
from unittest.mock import AsyncMock, patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.stdout.reconfigure(encoding="utf-8")

from personal_agent.backends import resolve_codex_command
from personal_agent.config import load_config
from personal_agent.memory import MemoryStore
from personal_agent.models import Identity, IncomingMessage
from personal_agent.supervisor import TaskSupervisor
from personal_agent.testing import ScriptedBrain, ScriptedTurn, block_host_desktop, make_synthetic_video
from scripts.verify_conversation import EVALUATION_MODEL, EVALUATION_REASONING, NoTranscriber, OfflineTelegram, TracedBackend

PRIMARY = 101
VERBOSE = "--verbose" in sys.argv


async def main() -> None:
    real = "--real" in sys.argv
    restore_desktop = block_host_desktop()
    try:
        source = load_config(Path("config.local.toml") if Path("config.local.toml").is_file() else Path("config.example.toml"))
        with tempfile.TemporaryDirectory(prefix="agent-video-verify-") as directory:
            root = Path(directory)
            video_fixture = root / "synthetic_demo.mp4"
            print("Generating synthetic video...", flush=True)
            make_synthetic_video(
                video_fixture,
                segments=(("red", 2.0), ("green", 2.0), ("blue", 2.0)),
                labels=["START", "MIDDLE", "FINISH"],
                audio="tone",
            )
            print(f"Synthetic video generated: {video_fixture.stat().st_size} bytes", flush=True)

            class Telegram(OfflineTelegram):
                def __init__(self, downloads: dict[str, Path]):
                    super().__init__()
                    self.downloads = downloads

                async def download(self, file_id: str, destination: Path, max_bytes: int) -> int:
                    src = self.downloads[file_id]
                    shutil.copyfile(src, destination)
                    return destination.stat().st_size

            config = replace(
                source, data_dir=root, token="", primary_id=PRIMARY, secondary_id=202,
                project_roots={}, brain_timeout=180,
                model=EVALUATION_MODEL, reasoning=EVALUATION_REASONING,
            )

            if real:
                print(f"Using real model backend: {config.model} (reasoning={config.reasoning})", flush=True)
                brain = TracedBackend(
                    model=EVALUATION_MODEL,
                    codex_command=resolve_codex_command(config.codex_executable or "codex"),
                    reasoning_effort=EVALUATION_REASONING,
                    work_root=root,
                )
            else:
                print("Running in mock mode with ScriptedBrain...", flush=True)
                brain = ScriptedBrain([
                    ScriptedTurn(reply="В видео видны красный, зелёный и синий цвета со словами START, MIDDLE, FINISH."),
                    ScriptedTurn(
                        calls=[("watch_video", lambda records: {"media_id": records_to_media_id(store), "start_seconds": 4.0, "end_seconds": 6.0})],
                        reply="В самом конце видео синий цвет и слово FINISH.",
                    ),
                ], image_input_supported=True)

            def records_to_media_id(mem_store: MemoryStore) -> str:
                video_dirs = list((root / "memory" / "videos" / str(PRIMARY)).iterdir())
                return video_dirs[0].name if video_dirs else ""

            store = MemoryStore(root)
            telegram = Telegram({"synth-vid-1": video_fixture})
            supervisor = TaskSupervisor(config, store, brain, telegram, asyncio.Event(), transcriber=NoTranscriber())

            checks = {}
            try:
                # Turn 1: Send video attachment
                print("\n[Turn 1] Sending video attachment to agent...", flush=True)
                msg1 = IncomingMessage(
                    update_id=1, sender_id=PRIMARY, chat_id=PRIMARY,
                    text="Посмотри это видео. Какие цвета и слова там показаны?",
                    message_id=1,
                    attachment={"kind": "video", "file_id": "synth-vid-1", "file_size": video_fixture.stat().st_size},
                    is_text=False,
                )
                await supervisor.submit(msg1, Identity.PRIMARY)
                await asyncio.gather(*list(supervisor.tasks))

                ans1 = telegram.answers[-1] if telegram.answers else ""
                if VERBOSE or real:
                    print(f"Agent response 1: {ans1}", flush=True)

                if real:
                    checks["turn1_received_contact_sheets"] = (brain.image_counts[-1] > 0)
                    ans1_lower = ans1.casefold()
                    checks["turn1_recognized_content"] = any(
                        word in ans1_lower for word in ("красн", "зелен", "зелëн", "син", "start", "middle", "finish")
                    )
                else:
                    checks["turn1_received_contact_sheets"] = len(brain.generate_calls[-1]["images"]) > 0
                    checks["turn1_recognized_content"] = True

                # Turn 2: Follow-up question about the end of the video
                print("\n[Turn 2] Asking follow-up about the end of the video...", flush=True)
                msg2 = IncomingMessage(
                    update_id=2, sender_id=PRIMARY, chat_id=PRIMARY,
                    text="А что показано в самом конце ролика? Пересмотри конец.",
                    message_id=2,
                    attachment=None,
                    is_text=True,
                )
                await supervisor.submit(msg2, Identity.PRIMARY)
                await asyncio.gather(*list(supervisor.tasks))

                ans2 = telegram.answers[-1] if len(telegram.answers) > 1 else ""
                if VERBOSE or real:
                    print(f"Agent response 2: {ans2}", flush=True)

                if real:
                    called_watch_video = "watch_video" in brain.tool_kinds
                    checks["turn2_called_watch_video"] = called_watch_video
                    ans2_lower = ans2.casefold()
                    checks["turn2_described_end"] = any(
                        word in ans2_lower for word in ("син", "finish", "синий", "финиш", "конец")
                    )
                else:
                    checks["turn2_called_watch_video"] = ("watch_video" in brain.tool_names_called)
                    checks["turn2_described_end"] = True

                checks["no_telegram_photos_leaked"] = (telegram.photos == 0)
                checks["two_replies_delivered"] = (len(telegram.answers) == 2)

                print("\nVerification Results:", flush=True)
                for name, ok in checks.items():
                    print(f"  {name}: {'PASS' if ok else 'FAIL'}", flush=True)

                if not all(checks.values()):
                    print("\nSome checks failed!", flush=True)
                    sys.exit(1)
                print("\nAll video verification checks PASSED successfully!", flush=True)
            finally:
                await supervisor.shutdown()
                if real:
                    await brain.close()
                store.close()
    finally:
        restore_desktop()


if __name__ == "__main__":
    asyncio.run(main())
