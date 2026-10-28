"""Synthetic TGS/WebM end-to-end check; Telegram always remains offline."""
from __future__ import annotations

import asyncio
from dataclasses import replace
from pathlib import Path
import shutil
import sys
import tempfile
from unittest.mock import AsyncMock, patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts.verify_conversation import EVALUATION_MODEL, EVALUATION_REASONING, OfflineTelegram, NoTranscriber, TracedBackend
from tests.test_animated_stickers import make_tgs, make_webm
from personal_agent.testing import ScriptedBrain, ScriptedTurn
from personal_agent.backends import resolve_codex_command
from personal_agent.config import load_config
from personal_agent.memory import MemoryStore
from personal_agent.models import IncomingMessage, Identity
from personal_agent.supervisor import TaskSupervisor


async def main():
    real = '--real' in sys.argv
    source = load_config(Path('config.local.toml'))
    with tempfile.TemporaryDirectory(prefix='agent-animated-') as directory, patch(
        'personal_agent.windows_uia.read_active_window', new=AsyncMock(return_value={
            'ok': False, 'kind': 'read_active_window', 'error': {'code': 'synthetic_test_no_desktop'}})
    ):
        root = Path(directory)
        fixtures = {'tgs': root / 'moving.tgs', 'webm': root / 'colors.webm'}
        make_tgs(fixtures['tgs'])
        make_webm(fixtures['webm'])

        class Telegram(OfflineTelegram):
            async def download(self, file_id, destination, max_bytes):
                shutil.copyfile(fixtures[file_id], destination)
                return destination.stat().st_size

        config = replace(source, data_dir=root, token='', primary_id=101, secondary_id=202,
                         project_roots={}, brain_timeout=150,
                         model=EVALUATION_MODEL, reasoning=EVALUATION_REASONING)
        if real:
            print(f'evaluation_model={config.model}; reasoning={config.reasoning}', flush=True)
        brain = TracedBackend(model=config.model, codex_command=resolve_codex_command(config.codex_executable),
                              reasoning_effort=config.reasoning, work_root=root) if real else ScriptedBrain([ScriptedTurn(reply='ok'), ScriptedTurn(reply='ok'), ScriptedTurn(
            calls=[('view_recent_media', lambda records: {'media_id': store.get_state('last_attachment_image:101')['attachment_id']})],
            reply='ok')], image_input_supported=True)
        store, telegram = MemoryStore(root), Telegram()
        supervisor = TaskSupervisor(config, store, brain, telegram, asyncio.Event(), transcriber=NoTranscriber())
        checks = {}
        try:
            for index, format in enumerate(('tgs', 'webm'), 1):
                text = ('Опиши направление движения красной фигуры в присланном стикере одним предложением.' if format == 'tgs'
                        else 'Какие цвета видны в начале и в конце анимации? Ответь одним предложением.')
                media = {'kind': 'sticker', 'file_id': format, 'sticker_format': format,
                         'is_animated': format == 'tgs', 'is_video': format == 'webm'}
                await supervisor.submit(IncomingMessage(index, 101, 101, text, index, media), Identity.PRIMARY)
                await asyncio.gather(*list(supervisor.tasks))
                count = brain.image_counts[-1] if real else len(brain.generate_calls[-1]['images'])
                checks[format + '_four_frames_reach_brain'] = count == 4
                if real:
                    answer = telegram.answers[-1].casefold()
                    checks[format + '_animation_understood'] = any(word in answer for word in ('вправо', 'направо', 'слева направо')) if format == 'tgs' else ('красн' in answer and 'син' in answer)
            await supervisor.submit(IncomingMessage(3, 101, 101, 'И ты его видишь? Назови цвет в конце стикера.', 3), Identity.PRIMARY)
            await asyncio.gather(*list(supervisor.tasks))
            checks['followup_views_four_frames_via_tool'] = (
                'view_recent_media' in brain.tool_kinds if real else len(brain.records[-1].outcome.images) == 4)
            checks['no_photo_delivery'] = telegram.photos == 0
            for key, ok in checks.items():
                print(f'{key}={"PASS" if ok else "FAIL"}', flush=True)
            print('telegram_network_calls=0', flush=True)
            if not all(checks.values()):
                raise SystemExit(1)
        finally:
            await supervisor.shutdown()
            if real:
                await brain.close()
            store.close()


if __name__ == '__main__':
    asyncio.run(main())
