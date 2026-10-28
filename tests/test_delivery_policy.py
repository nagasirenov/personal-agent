from unittest.mock import AsyncMock, patch
import unittest

from personal_agent.delivery import format_response
from personal_agent.models import Identity
from personal_agent.testing import ScriptedBrain, ScriptedTurn
from tests.test_service_observation_integration import ObservationHarness, _message


class DeliveryPolicyTests(unittest.IsolatedAsyncioTestCase):
    def test_plain_text_preserves_content_and_links(self):
        result = format_response('# Заголовок\n**Ответ**\n- `C:\\demo`\n[сайт](https://example.test)', {'markdown_allowed': False})
        self.assertEqual(result, 'Заголовок\nОтвет\nC:\\demo\nсайт (https://example.test)')

    def test_plain_text_keeps_code_contents_literal(self):
        source = 'Код:\n```python\ndef __init__():\n    return "**literal**"\n```\n`__name__`'
        self.assertEqual(format_response(source, {'response_format': 'plain_text'}),
                         'Код:\ndef __init__():\n    return "**literal**"\n__name__')

    async def test_screen_question_is_answered_from_uia_text_without_capture_or_photo(self):
        brain = ScriptedBrain([ScriptedTurn(calls=[("read_active_window", {})], reply='Открыт редактор.')])
        harness = ObservationHarness(brain)
        try:
            window = {'ok': True, 'kind': 'read_active_window', 'sufficient_text': True,
                      'data': {'title': 'Editor', 'text': 'UIA_TEXT_731'}}
            with patch('personal_agent.tools.capture_screenshot') as capture, \
                    patch('personal_agent.windows_uia.read_active_window', new=AsyncMock(return_value=window)):
                await harness.supervisor.submit(_message('Что сейчас у меня на экране?'), Identity.PRIMARY)
                await harness.wait_idle()
            capture.assert_not_called()
            self.assertIn('UIA_TEXT_731', brain.records[0].outcome.output)
            self.assertEqual(harness.telegram.sent_photos, [])
            self.assertEqual(harness.telegram.sent_text[-1][1], 'Открыт редактор.')
        finally:
            await harness.close()

    async def test_model_falls_back_to_capture_when_uia_text_is_unavailable(self):
        brain = ScriptedBrain([ScriptedTurn(
            calls=[("read_active_window", {}), ("take_screenshot", {})], reply='Текст анализа')],
            image_input_supported=True)
        harness = ObservationHarness(brain)
        image = harness.screenshot_file('screen.png')
        try:
            with patch('personal_agent.tools.capture_screenshot', return_value=image) as capture, \
                    patch('personal_agent.windows_uia.read_active_window', new=AsyncMock(return_value={
                        'ok': False, 'kind': 'read_active_window', 'error': {'code': 'uia_unavailable'}})):
                await harness.supervisor.submit(_message('Что сейчас у меня на экране?'), Identity.PRIMARY)
                await harness.wait_idle()
            capture.assert_called_once()
            self.assertEqual(brain.records[0].error_code, 'uia_unavailable')
            self.assertEqual(harness.telegram.sent_photos, [])
            self.assertEqual(brain.records[1].outcome.images, [image])
            self.assertEqual(harness.telegram.sent_text[-1][1], 'Текст анализа')
        finally:
            await harness.close()

    async def test_internal_model_capture_is_never_auto_sent(self):
        brain = ScriptedBrain([ScriptedTurn(calls=[("take_screenshot", {})], reply='Вижу окно')],
                              image_input_supported=True)
        harness = ObservationHarness(brain)
        harness.enable_full_access()
        image = harness.screenshot_file('screen.png')
        try:
            with patch('personal_agent.tools.capture_screenshot', return_value=image):
                await harness.supervisor.submit(_message('Помоги разобраться с открытым окном'), Identity.PRIMARY)
                await harness.wait_idle()
            self.assertEqual(harness.telegram.sent_photos, [])
            self.assertEqual(brain.records[0].outcome.images, [image])
            self.assertFalse(brain.records[0].data['delivered_to_user'])
        finally:
            await harness.close()

    async def test_photo_is_sent_only_with_send_to_user_or_exact_slash_command(self):
        brain = ScriptedBrain([
            ScriptedTurn(calls=[("take_screenshot", {"send_to_user": False})]),
            ScriptedTurn(calls=[("take_screenshot", {"send_to_user": True})]),
            ScriptedTurn(calls=[("take_screenshot", {"send_to_user": "true"})]),
        ])
        harness = ObservationHarness(brain)
        image = harness.screenshot_file()
        try:
            with patch('personal_agent.tools.capture_screenshot', return_value=image):
                await harness.supervisor.submit(_message('Пришли мне скрин', update_id=1), Identity.PRIMARY)
                await harness.wait_idle()
                self.assertEqual(harness.telegram.sent_photos, [])  # wording alone sends nothing
                await harness.supervisor.submit(_message('покажи экран', update_id=2), Identity.PRIMARY)
                await harness.wait_idle()
                self.assertEqual(len(harness.telegram.sent_photos), 1)
                await harness.supervisor.submit(_message('ещё', update_id=3), Identity.PRIMARY)
                await harness.wait_idle()
            self.assertEqual(len(harness.telegram.sent_photos), 1)
            self.assertEqual(brain.records[2].error_code, 'invalid_arguments')
        finally:
            await harness.close()


if __name__ == '__main__':
    unittest.main()
