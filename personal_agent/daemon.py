"""Explicitly started daemon. OBSERVE never invokes THINK."""
from __future__ import annotations

import asyncio
from datetime import datetime, timedelta, timezone
import signal
import time

from .memory import MemoryStore
from .browser_bridge import BrowserContextReceiver
from .models import TimelineEvent
from .observers import WindowsActivityObserver, summarize_activity
from .security import InstanceLock, TrustGate
from .supervisor import TaskSupervisor
from .telegram import TelegramClient, TelegramGateway


def build_backend(config):
    from .backends import MockBackend, CodexSubscriptionBackend, CodexChatGPTWebBackend, resolve_codex_command
    if config.backend == "mock":
        return MockBackend()
    codex_command = resolve_codex_command(config.codex_executable)
    # Backend adapters retain their own no-tools boundary and cancellation.
    if config.backend == "codex_subscription":
        return CodexSubscriptionBackend(model=config.model, codex_command=codex_command,
                                        reasoning_effort=config.reasoning, work_root=config.data_dir)
    web_model = config.model if config.model.startswith("chatgpt-web/") else "chatgpt-web/" + config.model
    return CodexChatGPTWebBackend(enabled=config.web_enabled, base_url=config.web_base_url, model=web_model,
                                codex_command=codex_command, reasoning_effort=config.reasoning,
                                work_root=config.data_dir)


async def observe_loop(observer, store, config, stop: asyncio.Event, agent_observers=None) -> None:
    agent_observers = agent_observers or {}
    last_export = 0.0
    next_retention_cleanup = 0.0
    prior_day = None
    while not stop.is_set():
        monotonic_now = time.monotonic()
        if monotonic_now >= next_retention_cleanup:
            cleanup_time = datetime.now(timezone.utc)
            try:
                counts = await asyncio.to_thread(
                    store.cleanup_retained_data, now=cleanup_time, tz=config.tz
                )
                store.set_state("retention_health", {
                    "ok": True, "last_run_at": cleanup_time.isoformat(), **counts,
                })
                next_retention_cleanup = time.monotonic() + 24 * 60 * 60
            except Exception as exc:
                store.set_state("retention_health", {
                    "ok": False, "last_run_at": cleanup_time.isoformat(),
                    "error_type": type(exc).__name__,
                })
                next_retention_cleanup = time.monotonic() + 60 * 60
        try:
            events = await asyncio.to_thread(observer.sample)
            if agent_observers:
                names = list(agent_observers)
                sampled = await asyncio.gather(
                    *(asyncio.to_thread(agent_observers[name].poll) for name in names),
                    return_exceptions=True,
                )
                source_health = {}
                for name, result in zip(names, sampled):
                    source = agent_observers[name]
                    if isinstance(result, BaseException):
                        source_health[name] = {
                            "available": False,
                            "error_type": type(result).__name__,
                        }
                        continue
                    events.extend(result)
                    try:
                        health = source.health()
                        source_health[name] = dict(health) if isinstance(health, dict) else {"available": True}
                    except Exception as exc:
                        source_health[name] = {
                            "available": False,
                            "error_type": type(exc).__name__,
                        }
                observer_health = observer.health()
                observer_health["agent_sources"] = source_health
            else:
                observer_health = observer.health()
            if stop.is_set():
                break
            store.record_events(events)
            for event in events:
                if event.type == "window.foreground":
                    store.set_state("current_foreground", {"time": event.time.isoformat(), "event_id": event.id, "project": event.project, **event.data})
            store.set_state("observer_health", observer_health)
            if events or time.monotonic() - last_export >= 60:
                now = datetime.now(timezone.utc)
                recent = store.events_between(now - timedelta(hours=2), now, limit=2000)
                store.update_context(summarize_activity(recent))
            if time.monotonic() - last_export >= 60:
                day = datetime.now(config.tz).date()
                if prior_day is not None and prior_day != day:
                    store.export_day(prior_day, config.tz)
                store.export_day(day, config.tz)
                prior_day, last_export = day, time.monotonic()
        except Exception as exc:
            store.set_state("observer_health", {"available": False, "error_type": type(exc).__name__})
        try:
            await asyncio.wait_for(stop.wait(), config.observe_interval)
        except TimeoutError:
            pass


async def run_daemon(config) -> None:
    config.validate(live=True)
    lock = InstanceLock(config.data_dir)
    store = None
    brain = client = supervisor = observer = None
    browser_receiver = None
    tasks = []
    previous_handlers = {}
    stop = asyncio.Event()
    loop = asyncio.get_running_loop()
    try:
        store = MemoryStore(config.data_dir)
        brain = build_backend(config)
        client = TelegramClient(config.token, api_base=config.telegram_api_base)
        supervisor = TaskSupervisor(config, store, brain, client, stop)
        from .adapters.browser import BrowserContextObserver
        from .adapters.codex import CodexSessionObserver
        from .adapters.opencode import OpenCodeSessionObserver
        from .adapters.terminal import TerminalObserver
        from .adapters.vscode import VscodeCompanionObserver

        observer = WindowsActivityObserver(config.project_roots, config.observe_interval)
        agent_observers = {
            "codex": CodexSessionObserver(),
            "opencode": OpenCodeSessionObserver(),
            "terminal": TerminalObserver(),
            "vscode": VscodeCompanionObserver(),
        }
        if not observer.detect():
            raise RuntimeError("Live observation requires an interactive Windows desktop session")
        try:
            observer.start()
        except Exception:
            store.set_state("observer_health", observer.health())
            raise
        browser_receiver = BrowserContextReceiver(config.data_dir)
        browser_observer = BrowserContextObserver(browser_receiver.root, receiver=browser_receiver)
        agent_observers["browser"] = browser_observer
        browser_receiver.start()
        supervisor.browser_context_provider = browser_observer.current_contexts
        supervisor.browser_capture_provider = browser_receiver.request_capture
        initial_browser_events = await asyncio.to_thread(browser_observer.poll)
        if initial_browser_events:
            store.record_events(initial_browser_events)
        store.set_state("observer_health", observer.health())
        for signum in (signal.SIGINT, signal.SIGTERM):
            previous_handlers[signum] = signal.getsignal(signum)
            signal.signal(signum, lambda *_: loop.call_soon_threadsafe(stop.set))
        store.record_event(TimelineEvent("daemon", "DAEMON_STARTED", {"backend": config.backend, "actions_enabled": config.pc_access == "full", "pc_access": config.pc_access}))
        gateway = TelegramGateway(client, TrustGate(config.primary_id, config.secondary_id), store, supervisor.submit, supervisor.shutdown, stop)
        tasks = [
            asyncio.create_task(gateway.run()),
            asyncio.create_task(observe_loop(observer, store, config, stop, agent_observers)),
        ]
        stopped = asyncio.create_task(stop.wait())
        tasks.append(stopped)
        done, _ = await asyncio.wait(tasks, return_when=asyncio.FIRST_COMPLETED)
        for task in done:
            if task is not stopped:
                task.result()
    finally:
        stop.set()
        if supervisor:
            await supervisor.shutdown()
        for task in tasks:
            task.cancel()
        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)
        if browser_receiver:
            browser_receiver.stop()
        if observer:
            await asyncio.to_thread(observer.stop)
            if store:
                store.set_state("observer_health", observer.health())
        if brain:
            try:
                await asyncio.wait_for(brain.close(), 4)
            except (Exception, asyncio.CancelledError):
                pass
        if client:
            await client.close()
        if store:
            try:
                store.export_day(datetime.now(config.tz).date(), config.tz)
            finally:
                store.close()
        for signum, handler in previous_handlers.items():
            signal.signal(signum, handler)
        lock.close()
