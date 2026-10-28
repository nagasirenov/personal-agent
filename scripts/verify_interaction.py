"""Real model regression using synthetic inputs and an offline Telegram sink."""
from __future__ import annotations

import asyncio
from dataclasses import replace
from pathlib import Path
import shutil
import sys
import tempfile
from unittest.mock import AsyncMock, patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts.verify_conversation import EVALUATION_MODEL, EVALUATION_REASONING, OfflineTelegram, NoTranscriber, TracedBackend, red_png
from personal_agent.backends import resolve_codex_command
from personal_agent.config import load_config
from personal_agent.memory import MemoryStore
from personal_agent.models import Identity, IncomingMessage
from personal_agent.self_knowledge import preferences_for
from personal_agent.supervisor import TaskSupervisor


async def main():
    source = load_config(Path('config.local.toml'))
    profile_only = '--profile-only' in sys.argv
    tool_choice_only = '--tool-choice-only' in sys.argv
    visual_tool_choice_only = '--visual-tool-choice-only' in sys.argv
    profile_text = 'Мой проект называется Полярис-731. Я изучаю астрономию.'
    with tempfile.TemporaryDirectory(prefix='agent-interaction-') as folder, patch(
        'personal_agent.windows_uia.read_active_window', new=AsyncMock(return_value={
            'ok': False, 'kind': 'read_active_window', 'error': {'code': 'synthetic_test_no_desktop'}})
    ), patch('personal_agent.tools.capture_screenshot',
             side_effect=AssertionError('Real screen capture is disabled in this verification.')) as screen_capture:
        root = Path(folder)
        image = root / 'fixture-red.png'
        red_png(image)

        class Telegram(OfflineTelegram):
            async def download(self, file_id, destination, max_bytes):
                if file_id == 'synthetic-profile':
                    destination.write_text(profile_text, encoding='utf-8')
                else:
                    shutil.copyfile(image, destination)
                return destination.stat().st_size

        config = replace(source, data_dir=root, token='', primary_id=101, secondary_id=202,
                         project_roots={'fixture': root / 'synthetic-fixture'} if tool_choice_only else {},
                         pc_access='read' if profile_only else 'full', brain_timeout=150,
                         model=EVALUATION_MODEL, reasoning=EVALUATION_REASONING)
        print(f'evaluation_model={config.model}; reasoning={config.reasoning}', flush=True)
        brain = TracedBackend(model=config.model, codex_command=resolve_codex_command(config.codex_executable),
                              reasoning_effort=config.reasoning, work_root=root)
        telegram = Telegram()
        store = MemoryStore(root)
        supervisor = TaskSupervisor(config, store, brain, telegram, asyncio.Event(), transcriber=NoTranscriber())

        async def ask(text, attachment=None):
            index = len(telegram.answers) + 1
            previous_turn = store.get_state('last_completed_turn', None)
            await supervisor.submit(IncomingMessage(index, 101, 101, text, index, attachment), Identity.PRIMARY)
            await asyncio.gather(*list(supervisor.tasks))
            if store.get_state('last_completed_turn', None) == previous_turn:
                error = store.get_state('last_turn_error', {}) or {}
                raise RuntimeError('Real check failed: ' + str(error.get('code', 'no_completed_turn')))
            return telegram.answers[-1]

        try:
            if visual_tool_choice_only:
                screen = root / 'memory' / 'screenshots' / 'synthetic-red.png'
                red_png(screen)
                request = 'Оцени оформление открытого приложения по цвету главной панели. Хочу увидеть точную копию его текущего вида в этом чате.'
                with patch('personal_agent.tools.capture_screenshot', return_value=screen) as synthetic_capture:
                    reply = await ask(request)
                checks = {
                    'model_selected_screenshot': 'take_screenshot' in brain.tool_kinds,
                    'model_used_synthetic_image': 'take_screenshot' in brain.tool_kinds,
                    'model_described_synthetic_red_image': 'красн' in reply.casefold() or 'red' in reply.casefold(),
                    'model_requested_photo_delivery': telegram.photos == 1,
                    'only_synthetic_capture_used': synthetic_capture.called and not screen_capture.called,
                    'telegram_network_calls_zero': True,
                }
                for key, ok in checks.items():
                    print(f'{key}={"PASS" if ok else "FAIL"}', flush=True)
                print(f'model_calls={len(brain.image_counts)}; model_tool_kinds={brain.tool_kinds}; telegram_network_calls=0', flush=True)
                if not all(checks.values()):
                    print('synthetic_reply=' + repr(reply), flush=True)
                    print('image_counts=' + repr(brain.image_counts), flush=True)
                    raise SystemExit(1)
                return
            if tool_choice_only:
                fixture = root / 'synthetic-fixture'
                fixture.mkdir()
                note = fixture / 'maintenance-note.txt'
                note.write_text(
                    'Synthetic backup maintenance date: 2026-11-09. '
                    'Validation marker: BACKUP_NOTE_FIXTURE_731.', encoding='utf-8')
                read_request = (f'Какая дата указана в заметке {note} о следующем обслуживании '
                                'резервного питания? Ответь только датой.')
                read_reply = await ask(read_request)

                target = root / 'synthetic-actions' / 'result.txt'
                action_text = 'SYNTHETIC_ACTION_STATE_4921'
                action_request = (f'К концу синтетической проверки файл {target} должен содержать '
                                  f'ровно {action_text}. После достижения этого состояния сообщи результат.')
                executed_kinds = []

                async def synthetic_action(kind, arguments, *, data_dir):
                    executed_kinds.append(kind)
                    path = Path(arguments.get('path', '')).resolve() if isinstance(arguments, dict) else Path()
                    if (kind != 'write_file' or path != target.resolve()
                            or not path.is_relative_to(root.resolve())
                            or arguments.get('text') != action_text
                            or arguments.get('append', False) is not False):
                        return {'ok': False, 'kind': kind, 'error': {'code': 'synthetic_action_denied'}}
                    path.parent.mkdir(parents=True, exist_ok=True)
                    path.write_text(action_text, encoding='utf-8')
                    return {'ok': True, 'kind': kind, 'path': str(path),
                            'bytes_written': len(action_text.encode('utf-8'))}

                with patch('personal_agent.pc_tools.execute_pc_request', new=synthetic_action):
                    action_reply = await ask(action_request)
                checks = {
                    'model_selected_read_file': 'read_file' in brain.tool_kinds,
                    'model_read_synthetic_note': '2026-11-09' in read_reply,
                    'model_selected_write_file': 'write_file' in brain.tool_kinds,
                    'only_synthetic_write_executor_called': executed_kinds == ['write_file'],
                    'synthetic_action_result_matches': target.is_file() and target.read_text(encoding='utf-8') == action_text,
                    'real_screen_capture_not_called': not screen_capture.called,
                    'telegram_network_calls_zero': True,
                }
                for key, ok in checks.items():
                    print(f'{key}={"PASS" if ok else "FAIL"}', flush=True)
                print(f'model_calls={len(brain.image_counts)}; model_tool_kinds={brain.tool_kinds}; telegram_network_calls=0', flush=True)
                if not all(checks.values()):
                    print('synthetic_replies=' + repr([read_reply, action_reply]), flush=True)
                    raise SystemExit(1)
                return
            if profile_only:
                read_reply = await ask('', {'kind': 'document', 'file_id': 'synthetic-profile',
                                           'file_name': 'message.txt', 'mime_type': 'text/plain'})
                save_reply = await ask('Сохрани этот текст как мой постоянный профиль пользователя. Проверь чтением из хранилища.')
                saved = store.get_user_profile(101)
                receipt = store.get_state('profile_save_receipt:101', {})
                await supervisor.shutdown()
                store.close()
                store = MemoryStore(root)
                supervisor = TaskSupervisor(config, store, brain, telegram, asyncio.Event(), transcriber=NoTranscriber())
                recalled = await ask('Как называется мой проект? Ответь только названием.')
                checks = {
                    'captionless_text_understood': 'полярис-731' in read_reply.casefold(),
                    'profile_saved_and_read_back': bool(saved and saved['text'] == profile_text and receipt.get('read_back_verified')),
                    'profile_recalled_after_restart': 'полярис-731' in recalled.casefold(),
                    'other_sender_has_no_profile': store.get_user_profile(202) is None,
                }
                for key, ok in checks.items():
                    print(f'{key}={"PASS" if ok else "FAIL"}', flush=True)
                print('synthetic_save_reply=' + repr(save_reply), flush=True)
                print(f'model_calls={len(brain.image_counts)}; telegram_network_calls=0', flush=True)
                if not all(checks.values()):
                    print('synthetic_recall=' + repr(recalled), flush=True)
                    raise SystemExit(1)
                return
            await ask('Запомни: пиши без Markdown, скриншоты присылай только по прямой просьбе.')
            photo = await ask('Какой основной цвет? Ответь одним словом.', {'kind': 'photo', 'file_id': 'synthetic-photo'})
            screen = root / 'memory' / 'screenshots' / 'red.png'
            red_png(screen)
            with patch('personal_agent.tools.capture_screenshot', return_value=screen):
                analysis = await ask('Что сейчас у меня на экране? Назови основной цвет одним словом.')
                private_count = telegram.photos
                await ask('Пришли мне скриншот.')
            checks = {
                'preference_persisted': preferences_for(store, 101).get('response_format') == 'plain_text',
                'telegram_photo_seen': 'красн' in photo.casefold() or 'red' in photo.casefold(),
                'screen_seen_without_photo': private_count == 0 and ('красн' in analysis.casefold() or 'red' in analysis.casefold()),
                'explicit_delivery_once': telegram.photos == 1,
            }
            for key, ok in checks.items():
                print(f'{key}={"PASS" if ok else "FAIL"}', flush=True)
            print(f'model_calls={len(brain.image_counts)}; telegram_network_calls=0', flush=True)
            if not all(checks.values()):
                print('synthetic_screen_answer=' + repr(analysis), flush=True)
                print('image_counts=' + repr(brain.image_counts), flush=True)
                raise SystemExit(1)
        finally:
            await supervisor.shutdown()
            await brain.close()
            store.close()


if __name__ == '__main__':
    asyncio.run(main())
