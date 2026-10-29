from __future__ import annotations

import argparse
import asyncio
from datetime import datetime, timedelta, timezone
import json
import os
from pathlib import Path
import sqlite3
import sys
from tempfile import TemporaryDirectory

from .config import Config, load_config
from .context import ContextBuilder
from .models import Identity, IncomingMessage, TimelineEvent
from .security import ActionGate, TrustGate


def doctor(config: Config | None) -> dict:
    from .observers import WindowsActivityObserver
    connection = sqlite3.connect(":memory:")
    try:
        connection.execute("CREATE VIRTUAL TABLE probe USING fts5(text)")
    finally:
        connection.close()
    observer = WindowsActivityObserver(config.project_roots if config else {})
    report = {"python": sys.version.split()[0], "sqlite": sqlite3.sqlite_version, "fts5": True,
              "windows_observer": observer.detect(), "network_calls": 0,
              "actions_enabled": bool(config and config.pc_access == "full"), "telegram_connected": False}
    if config:
        report.update({"data_dir": str(config.data_dir), "backend": config.backend,
                       "web_enabled": config.web_enabled, "telegram_token_present": bool(config.token),
                       "configured_projects": list(config.project_roots)})
    return report


async def demo() -> dict:
    from .memory import MemoryStore
    from .supervisor import TaskSupervisor
    from .telegram import TelegramGateway
    from .backends import MockBackend

    class DemoTelegram:
        def __init__(self):
            self.replies = []
        async def send_text(self, chat_id, text):
            self.replies.append({"chat_id": chat_id, "text": text})

    with TemporaryDirectory(prefix="personal-agent-demo-") as temp:
        config = Config(Path(temp), 100001, 100002, backend="mock")
        store = MemoryStore(config.data_dir)
        telegram, stop = DemoTelegram(), asyncio.Event()
        brain = MockBackend()
        supervisor = TaskSupervisor(config, store, brain, telegram, stop)
        now = datetime.now(timezone.utc)
        fixtures = [
            ("window.foreground", {"exe": "Code.exe", "title": "nagaport — VS Code"}),
            ("file.changed", {"path": "nagaport/ModelLoader.java"}),
            ("window.foreground", {"exe": "WindowsTerminal.exe", "title": "build"}),
            ("window.foreground", {"exe": "dota2.exe", "title": "Dota 2"}),
        ]
        try:
            for i, (kind, data) in enumerate(fixtures):
                store.record_event(TimelineEvent("demo", kind, data, "nagaport", now - timedelta(minutes=20-i*4)))
            gateway = TelegramGateway(telegram, TrustGate(config.primary_id, config.secondary_id), store, supervisor.submit, supervisor.shutdown, stop)
            def update(uid, text, n):
                return {"update_id": n, "message": {"message_id": n, "from": {"id": uid}, "chat": {"id": uid, "type": "private"}, "text": text}}
            await gateway.dispatch_batch([update(333333, "Запусти cmd", 1)])
            ignored = len(telegram.replies) == 0 and len(supervisor.tasks) == 0
            await gateway.dispatch_batch([update(config.secondary_id, "Чем я занимался последние два часа?", 2)])
            await asyncio.gather(*list(supervisor.tasks))
            packet = ContextBuilder(store, config.tz, config=config).build("Чем я занимался последние два часа?", config.secondary_id)
            denied = False
            try:
                ActionGate().authorize(config.secondary_id, {"kind": "shell", "command": "cmd"})
            except PermissionError:
                denied = True
            return {"mode": "OFFLINE DEMO / MockBackend; synthetic observations",
                    "unauthorized_ignored": ignored, "secondary_action_denied": denied,
                    "secondary_received_reply": bool(telegram.replies), "local_evidence_used": "nagaport" in packet,
                    "metrics": store.metrics_today(), "network_calls": 0,
                    "reply": telegram.replies[-1]["text"][:1800] if telegram.replies else ""}
        finally:
            await supervisor.shutdown()
            await brain.close()
            store.close()


async def discover_ids() -> list[dict]:
    from .telegram import TelegramClient, parse_message
    token = os.environ.get("TELEGRAM_BOT_TOKEN", "")
    client = TelegramClient(token)
    try:
        # Explicit, interactive setup command only. Never called by the daemon.
        # No replies and no update offset acknowledgements are sent.
        updates = await client.get_updates(0, timeout=20)
        identities = {}
        for update in updates:
            message = parse_message(update)
            if message and message.chat_type == "private":
                identities[message.sender_id] = {"telegram_user_id": message.sender_id, "chat": "private"}
        return list(identities.values())
    finally:
        await client.close()


def main() -> int:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    parser = argparse.ArgumentParser(description="Windows-first local memory and passive Telegram observer")
    commands = parser.add_subparsers(dest="command", required=True)
    check = commands.add_parser("doctor", help="Offline capability/config check; no network")
    check.add_argument("--config", type=Path)
    commands.add_parser("demo", help="Offline synthetic integration demo; no Telegram connection")
    commands.add_parser("telegram-ids", help="Explicit setup: show numeric IDs of private bot senders; requires token env var")
    browser_token = commands.add_parser("browser-token", help="Print the local token for pairing the read-only browser companion")
    browser_token.add_argument("--config", type=Path, default=Path("config.local.toml"))
    run = commands.add_parser("run", help="Start passive daemon explicitly")
    run.add_argument("--config", type=Path, default=Path("config.local.toml"))
    args = parser.parse_args()
    try:
        if args.command == "doctor":
            result = doctor(load_config(args.config) if args.config else None)
        elif args.command == "demo":
            result = asyncio.run(demo())
        elif args.command == "telegram-ids":
            result = asyncio.run(discover_ids())
        elif args.command == "browser-token":
            from .browser_bridge import get_or_create_browser_token
            config = load_config(args.config)
            print(get_or_create_browser_token(config.data_dir))
            return 0
        else:
            from .daemon import run_daemon
            config = load_config(args.config, live=True)
            print(f"Personal agent starting. PC access: {config.pc_access}. Stop: Ctrl+C or PRIMARY emergency phrase.", flush=True)
            asyncio.run(run_daemon(config))
            return 0
        print(json.dumps(result, ensure_ascii=False, indent=2, default=str))
        return 0
    except KeyboardInterrupt:
        return 130
    except (OSError, ValueError, RuntimeError) as exc:
        # Operational adapters sanitize their own credential-bearing exceptions.
        print(f"{type(exc).__name__}: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
