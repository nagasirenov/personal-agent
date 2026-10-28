from __future__ import annotations

import asyncio
import tempfile
import unittest
from datetime import datetime, timedelta, timezone
from pathlib import Path
from types import SimpleNamespace

from personal_agent.daemon import observe_loop
from personal_agent.memory import MemoryStore
from personal_agent.models import TimelineEvent


class DaemonAgentObserverTests(unittest.IsolatedAsyncioTestCase):
    async def test_agent_events_and_health_are_collected(self) -> None:
        class WindowsObserver:
            def sample(self):
                return []

            def health(self):
                return {"available": True, "last_error": None}

        class AgentObserver:
            def __init__(self) -> None:
                self.calls = 0

            def poll(self):
                self.calls += 1
                if self.calls > 1:
                    return []
                return [TimelineEvent(
                    source="codex",
                    type="codex.task_started",
                    data={"session_id": "opaque", "task": "Update observer"},
                    project="agent",
                    time=datetime.now(timezone.utc),
                    id="codex-event-1",
                )]

            def health(self):
                return {"available": True, "events_emitted": 1}

        with tempfile.TemporaryDirectory() as temp:
            store = MemoryStore(Path(temp) / "state")
            source = AgentObserver()
            stop = asyncio.Event()
            config = SimpleNamespace(
                tz=timezone.utc,
                observe_interval=10,
            )
            task = asyncio.create_task(observe_loop(
                WindowsObserver(), store, config, stop, {"codex": source},
            ))
            try:
                await asyncio.sleep(0.1)
                stop.set()
                await task
                now = datetime.now(timezone.utc)
                events = store.events_between(now - timedelta(minutes=1), now + timedelta(seconds=1))
                self.assertEqual([event.id for event in events], ["codex-event-1"])
                health = store.get_state("observer_health", {})
                self.assertEqual(health["agent_sources"]["codex"]["available"], True)
                self.assertEqual(health["agent_sources"]["codex"]["events_emitted"], 1)
            finally:
                stop.set()
                if not task.done():
                    await task
                store.close()


if __name__ == "__main__":
    unittest.main()
