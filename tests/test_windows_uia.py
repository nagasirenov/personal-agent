import asyncio
import json
import unittest
from unittest.mock import AsyncMock, patch

from personal_agent.models import Identity
from personal_agent.observation_broker import ObservationDenied, ObservationScope, validate_observation_plan
from personal_agent.observation_requests import ObservationRequest
from personal_agent.testing import ScriptedBrain, ScriptedTurn
from personal_agent.windows_uia import _normalize_response, read_active_window
from tests.test_service_observation_integration import ObservationHarness, _message


class WindowsUiaTests(unittest.IsolatedAsyncioTestCase):
    def test_unicode_bounded_content_and_title_redaction(self):
        token = '123456789:abcdefghijklmnopqrstuvwxyz_12345'
        raw = json.dumps({
            'ok': True, 'window_title': 'token=' + token, 'process_id': 123,
            'text': 'Отчёт готов. ' + token + ' ' + 'a' * 13000,
            'element_count': 4, 'truncated': False,
        }, ensure_ascii=False).encode('utf-8')
        result = _normalize_response(raw)
        self.assertTrue(result['ok'])
        self.assertTrue(result['text'].startswith('Отчёт готов.'))
        self.assertNotIn(token, json.dumps(result))
        self.assertLessEqual(len(result['text']), 12000)
        self.assertTrue(result['truncated'])
        self.assertTrue(result['sufficient_text'])

    def test_window_title_alone_is_not_content_and_errors_are_bounded(self):
        result = _normalize_response(b'{"ok":true,"window_title":"App","text":"","element_count":1}')
        self.assertFalse(result['sufficient_text'])
        self.assertEqual(_normalize_response(b'x' * 128001)['error']['code'], 'uia_response_too_large')
        self.assertEqual(_normalize_response(b'private stderr')['error']['code'], 'uia_invalid_response')

    def test_window_request_requires_authorized_scope_and_no_path(self):
        plan = [ObservationRequest('read_active_window')]
        with self.assertRaises(ObservationDenied):
            validate_observation_plan(plan, ObservationScope())
        self.assertEqual(validate_observation_plan(plan, ObservationScope(allow_active_window=True)), plan)
        with self.assertRaises(ObservationDenied):
            validate_observation_plan([ObservationRequest('read_active_window', 'C:/')],
                                      ObservationScope(allow_active_window=True))

    async def test_cancel_terminates_owned_uia_helper(self):
        entered = asyncio.Event()

        class Process:
            returncode = None
            async def communicate(self):
                entered.set()
                await asyncio.Future()

        process = Process()
        with patch('personal_agent.windows_uia.sys.platform', 'win32'), \
                patch('personal_agent.windows_uia.shutil.which', return_value='powershell.exe'), \
                patch('personal_agent.windows_uia.asyncio.create_subprocess_exec', new=AsyncMock(return_value=process)), \
                patch('personal_agent.windows_uia.terminate_tree', new=AsyncMock()) as terminate:
            task = asyncio.create_task(read_active_window())
            await asyncio.wait_for(entered.wait(), 1)
            task.cancel()
            with self.assertRaises(asyncio.CancelledError):
                await task
            terminate.assert_awaited_once_with(process)

    async def test_model_reads_window_without_photo_or_persisting_raw_text(self):
        brain = ScriptedBrain([ScriptedTurn(calls=[('read_active_window', {})], reply='Открыт редактор.')])
        harness = ObservationHarness(brain)
        observation = {'ok': True, 'kind': 'read_active_window', 'text': 'UIA_EPHEMERAL_MARKER', 'sufficient_text': True}
        try:
            with patch('personal_agent.windows_uia.read_active_window', new=AsyncMock(return_value=observation)) as read, \
                    patch('personal_agent.tools.capture_screenshot') as screenshot:
                await harness.supervisor.submit(_message('Что сейчас у меня на экране?'), Identity.PRIMARY)
                await harness.wait_idle()
            read.assert_awaited_once()
            screenshot.assert_not_called()
            self.assertIn('UIA_EPHEMERAL_MARKER', brain.records[0].outcome.output)
            self.assertNotIn('UIA_EPHEMERAL_MARKER', brain.generate_calls[0]['prompt'])
            self.assertEqual(harness.telegram.sent_photos, [])
            self.assertIsNone(harness.store.get_state('last_observation:101', None))
            self.assertEqual(harness.store.retrieve('UIA_EPHEMERAL_MARKER'), [])
        finally:
            await harness.close()

    async def test_secondary_has_no_active_window_tool(self):
        brain = ScriptedBrain([ScriptedTurn(calls=[('read_active_window', {})], reply='Видно отчёт.')])
        harness = ObservationHarness(brain)
        try:
            with patch('personal_agent.windows_uia.read_active_window', new=AsyncMock(return_value={
                'ok': True, 'kind': 'read_active_window', 'text': 'Текущий отчёт', 'sufficient_text': True,
            })) as read:
                message = _message('Разберись с открытым окном')
                message = type(message)(update_id=1, sender_id=202, chat_id=202, text=message.text, message_id=1)
                await harness.supervisor.submit(message, Identity.SECONDARY)
                await harness.wait_idle()
            read.assert_not_awaited()
            self.assertNotIn('read_active_window', brain.generate_calls[0]['tool_names'])
            self.assertEqual([record.error_code for record in brain.records], ['unknown_tool'])
            self.assertEqual(harness.telegram.sent_text[-1], (202, 'Видно отчёт.'))
        finally:
            await harness.close()
