import json
import shutil
import unittest
from dataclasses import replace
from pathlib import Path
from unittest.mock import AsyncMock, patch

from personal_agent.models import Identity
from personal_agent.animated_stickers import RenderedSticker
from personal_agent.testing import ScriptedBrain, ScriptedTurn
from personal_agent.web_lookup import WebLookupSession
from scripts.verify_conversation import red_png
from tests.test_service_observation_integration import ObservationHarness, _message
from tests.test_web_lookup import MARKER, SOURCE, fake_client


def packet_of(prompt: str) -> dict:
    raw = prompt.split("\nUNTRUSTED LOCAL EVIDENCE (JSON):\n", 1)[1].split("\n\nCURRENT USER MESSAGE (JSON):\n", 1)[0]
    return json.loads(raw)


def install_download(harness, fixture):
    async def download(file_id, destination, max_bytes):
        shutil.copyfile(fixture, destination)
        return destination.stat().st_size

    harness.telegram.download = download


def last_media_id(harness, user_id=101):
    return lambda records: {"media_id": harness.store.get_state(f"last_attachment_image:{user_id}")["attachment_id"]}


class AttachmentVisionWorkflowTests(unittest.IsolatedAsyncioTestCase):
    async def test_media_catalog_never_offers_files_outside_attachment_storage(self):
        harness = ObservationHarness(ScriptedBrain(image_input_supported=True))
        fixture = harness.root / 'fixture.png'
        red_png(fixture)
        install_download(harness, fixture)
        try:
            await harness.supervisor.submit(replace(_message(''), attachment={
                'kind': 'photo', 'file_id': 'synthetic'}), Identity.PRIMARY)
            await harness.wait_idle()
            cached = harness.store.get_state('last_attachment_image:101')
            attachment_id = cached['attachment_id']
            cached['path'] = str(fixture)  # Valid PNG, outside the attachment cache root.
            cached['paths'] = [str(fixture)]
            harness.store.set_state('last_attachment_image:101', cached)
            harness.store.record_attachment({
                'id': 'planted', 'sender_id': 101, 'kind': 'photo', 'path': str(fixture),
                'image': {'format': 'png'}, 'time': cached['time']})

            brain = ScriptedBrain([ScriptedTurn(calls=[
                ('view_recent_media', {'media_id': attachment_id}),
                ('view_recent_media', {'media_id': 'planted'}),
            ])], image_input_supported=True)
            harness.brain = harness.supervisor.brain = brain
            await harness.supervisor.submit(_message('А кто это?', update_id=2), Identity.PRIMARY)
            await harness.wait_idle()

            root = (harness.root / 'memory' / 'attachments').resolve()
            self.assertEqual(len(brain.records[0].outcome.images), 1)
            self.assertTrue(brain.records[0].outcome.images[0].resolve().is_relative_to(root))
            self.assertEqual(brain.records[1].error_code, 'media_not_found')
            self.assertEqual(brain.generate_calls[0]['images'], [])
            catalog = packet_of(brain.generate_calls[0]['prompt'])['recent_media']
            self.assertEqual([item['media_id'] for item in catalog], [attachment_id])
        finally:
            await harness.close()

    async def test_recent_media_is_scoped_to_sender_and_never_auto_attached(self):
        harness = ObservationHarness(ScriptedBrain(image_input_supported=True, response='Ок'))
        fixture = harness.root / 'fixture.png'
        red_png(fixture)
        install_download(harness, fixture)
        try:
            await harness.supervisor.submit(replace(_message('Первое изображение'), attachment={
                'kind': 'photo', 'file_id': 'synthetic'}), Identity.PRIMARY)
            await harness.wait_idle()
            attachment_id = harness.store.get_state('last_attachment_image:101')['attachment_id']
            self.assertEqual(len(harness.brain.generate_calls[-1]['images']), 1)

            brain = ScriptedBrain([
                ScriptedTurn(calls=[('view_recent_media', {'media_id': attachment_id})], reply='чужое'),
                ScriptedTurn(calls=[('view_recent_media', {'media_id': attachment_id})], reply='своё'),
                'без картинки',
            ], image_input_supported=True)
            harness.brain = harness.supervisor.brain = brain

            another_user = replace(_message('А кто это?', update_id=2), sender_id=202, chat_id=202)
            await harness.supervisor.submit(another_user, Identity.SECONDARY)
            await harness.wait_idle()
            self.assertEqual(brain.generate_calls[-1]['images'], [])
            self.assertEqual(packet_of(brain.generate_calls[-1]['prompt'])['recent_media'], [])
            self.assertEqual(brain.records[-1].error_code, 'media_not_found')
            self.assertEqual(brain.records[-1].outcome.images, [])

            await harness.supervisor.submit(_message('А кто это?', update_id=3), Identity.PRIMARY)
            await harness.wait_idle()
            self.assertEqual(brain.generate_calls[-1]['images'], [])  # offered in the catalog, not attached
            catalog = packet_of(brain.generate_calls[-1]['prompt'])['recent_media']
            self.assertEqual([(item['media_id'], item['kind']) for item in catalog], [(attachment_id, 'photo')])
            self.assertEqual(len(brain.records[-1].outcome.images), 1)

            await harness.supervisor.submit(_message('А завтра?', update_id=4), Identity.PRIMARY)
            await harness.wait_idle()
            self.assertEqual(brain.generate_calls[-1]['images'], [])
        finally:
            await harness.close()

    async def test_current_photo_does_not_inherit_previous_sticker(self):
        harness = ObservationHarness(ScriptedBrain(image_input_supported=True, response='Ок'))
        fixture = harness.root / 'fixture.png'
        red_png(fixture)
        install_download(harness, fixture)
        try:
            await harness.supervisor.submit(replace(_message(''), attachment={
                'kind': 'sticker', 'file_id': 'one'}), Identity.PRIMARY)
            await harness.wait_idle()
            earlier_image = harness.brain.generate_calls[-1]['images'][0]
            earlier_id = harness.store.get_state('last_attachment_image:101')['attachment_id']

            await harness.supervisor.submit(replace(_message('А что на этом стикере?', update_id=2), attachment={
                'kind': 'photo', 'file_id': 'two'}), Identity.PRIMARY)
            await harness.wait_idle()
            call = harness.brain.generate_calls[-1]
            self.assertEqual(len(call['images']), 1)
            self.assertNotEqual(call['images'][0], earlier_image)
            current_id = harness.store.get_state('last_attachment_image:101')['attachment_id']
            catalog = [item['media_id'] for item in packet_of(call['prompt'])['recent_media']]
            self.assertEqual(catalog, [earlier_id])  # the current picture is attached, not re-listed
            self.assertNotIn(current_id, catalog)
        finally:
            await harness.close()

    async def test_natural_image_followup_identifies_via_model_chosen_web_search(self):
        for kind in ('photo', 'sticker', 'animated_sticker'):
            with self.subTest(kind=kind):
                harness = ObservationHarness(ScriptedBrain(image_input_supported=True,
                                                           response='Вижу красный символ, но его название пока неизвестно.'))
                fixture = harness.root / 'fixture.png'
                red_png(fixture)
                frame_root = harness.root / 'memory' / 'attachments' / 'frames'
                frame_root.mkdir(parents=True)
                frames = [frame_root / f'{index}.png' for index in range(4)]
                for frame in frames:
                    red_png(frame)
                install_download(harness, fixture)
                media = {'kind': 'photo' if kind == 'photo' else 'sticker', 'file_id': 'synthetic',
                         'is_animated': kind == 'animated_sticker', 'sticker_format': 'tgs' if kind == 'animated_sticker' else 'webp'}
                async with fake_client() as client:
                    try:
                        with (patch('personal_agent.attachments.render_sticker',
                                    new=AsyncMock(return_value=RenderedSticker(frames, [0., .5, 1., 1.5], 2.))),
                              patch('personal_agent.tools.WebLookupSession', side_effect=lambda: WebLookupSession(client)),
                              patch('personal_agent.web_lookup._public_url', new=AsyncMock(return_value=True))):
                            await harness.supervisor.submit(replace(_message('Что это за знак?', update_id=1), attachment=media),
                                                            Identity.PRIMARY)
                            await harness.wait_idle()
                            brain = ScriptedBrain([ScriptedTurn(calls=[
                                ('view_recent_media', last_media_id(harness)),
                                ('web_search', {'query': 'Asterion synthetic symbol'}),
                                ('web_open', {'result_id': 1}),
                            ], reply=lambda records, prompt: (
                                f'Это символ Asterion; источник: {SOURCE}.'
                                if MARKER in records[2].outcome.output and SOURCE in records[1].outcome.output
                                else 'не нашёл'))], image_input_supported=True)
                            harness.brain = harness.supervisor.brain = brain
                            await harness.supervisor.submit(_message('А кто это? Можешь выяснить?', update_id=2), Identity.PRIMARY)
                            await harness.wait_idle()

                        self.assertEqual(len(brain.generate_calls), 1)
                        self.assertIn('web_search', brain.generate_calls[0]['tool_names'])
                        view = brain.records[0]
                        self.assertEqual(len(view.outcome.images), 4 if kind == 'animated_sticker' else 1)
                        if kind == 'animated_sticker':
                            self.assertEqual(view.data['frames'][3]['frame_time'], 1.5)
                        self.assertEqual(harness.telegram.sent_text[-1][1], f'Это символ Asterion; источник: {SOURCE}.')
                    finally:
                        await harness.close()

    async def test_animated_sticker_keeps_all_four_frames_including_followup(self):
        harness = ObservationHarness(ScriptedBrain(image_input_supported=True, response='Движется'))
        directory = harness.root / 'memory' / 'attachments' / 'frames'
        directory.mkdir(parents=True)
        frames = [directory / f'{i}.png' for i in range(4)]
        for frame in frames:
            red_png(frame)

        async def download(file_id, destination, max_bytes):
            destination.write_bytes(b'fixture')
            return 7

        harness.telegram.download = download
        try:
            with patch('personal_agent.attachments.render_sticker', new=AsyncMock(return_value=RenderedSticker(frames, [0., .5, 1., 1.5], 2.))):
                await harness.supervisor.submit(replace(_message(''), attachment={
                    'kind': 'sticker', 'file_id': 'fixture', 'is_animated': True, 'sticker_format': 'tgs'}), Identity.PRIMARY)
                await harness.wait_idle()
            self.assertEqual(harness.brain.generate_calls[-1]['images'], frames)
            self.assertIn('"frame_time": 1.5', harness.brain.generate_calls[-1]['prompt'])

            brain = ScriptedBrain([ScriptedTurn(calls=[('view_recent_media', last_media_id(harness))])],
                                  image_input_supported=True)
            harness.brain = harness.supervisor.brain = brain
            await harness.supervisor.submit(_message('И ты его видишь?', update_id=2), Identity.PRIMARY)
            await harness.wait_idle()
            self.assertEqual(brain.records[0].outcome.images, frames)
            self.assertEqual([frame['frame_time'] for frame in brain.records[0].data['frames']], [0., .5, 1., 1.5])
            catalog = packet_of(brain.generate_calls[-1]['prompt'])['recent_media']
            self.assertEqual((catalog[0]['kind'], catalog[0]['frame_count']), ('sticker_frames', 4))
        finally:
            await harness.close()

    async def test_captionless_photo_and_static_sticker_reach_brain(self):
        harness = ObservationHarness(ScriptedBrain(image_input_supported=True, response='Вижу красное'))
        fixture = harness.root / 'fixture.png'
        red_png(fixture)
        install_download(harness, fixture)
        try:
            for index, kind in enumerate(('photo', 'sticker'), 1):
                media = {'kind': kind, 'file_id': 'fixture', 'is_animated': False, 'is_video': False}
                await harness.supervisor.submit(replace(_message('', update_id=index), attachment=media), Identity.PRIMARY)
                await harness.wait_idle()
                self.assertEqual(harness.telegram.sent_text[-1][1], 'Вижу красное')
                self.assertEqual(len(harness.brain.generate_calls[-1]['images']), 1)
                self.assertIn('"attached_image_count": 1', harness.brain.generate_calls[-1]['prompt'])
            await harness.supervisor.submit(_message('Что на стикере?', update_id=3), Identity.PRIMARY)
            await harness.wait_idle()
            self.assertEqual(harness.brain.generate_calls[-1]['images'], [])
            self.assertEqual(len(packet_of(harness.brain.generate_calls[-1]['prompt'])['recent_media']), 2)
            other = replace(_message('Что на стикере?', update_id=4), sender_id=202, chat_id=202)
            await harness.supervisor.submit(other, Identity.SECONDARY)
            await harness.wait_idle()
            self.assertEqual(harness.brain.generate_calls[-1]['images'], [])
            self.assertEqual(packet_of(harness.brain.generate_calls[-1]['prompt'])['recent_media'], [])
            self.assertEqual(harness.telegram.sent_photos, [])
        finally:
            await harness.close()

    async def test_plain_text_preference_controls_actual_delivery(self):
        brain = ScriptedBrain([
            ScriptedTurn(calls=[('save_preferences', {'updates': {
                'response_format': 'plain_text', 'tone': 'conversational'}})],
                reply='## Ответ\n**Текст модели**'),
            '## Ответ\n**Текст модели**',
        ])
        harness = ObservationHarness(brain)
        try:
            await harness.supervisor.submit(
                _message('For future chats, those little stars around words make my eyes work harder; I would like this to feel like a chat with a person.'),
                Identity.PRIMARY)
            await harness.wait_idle()
            self.assertEqual(harness.telegram.sent_text[-1][1], 'Ответ\nТекст модели')
            saved = harness.store.get_state('user_preferences:101')
            self.assertEqual(saved['response_format'], 'plain_text')
            self.assertFalse(saved['markdown_allowed'])
            self.assertEqual(saved['tone'], 'conversational')
            self.assertTrue(brain.records[0].ok)
            self.assertEqual(brain.records[0].data['updated']['tone'], 'conversational')
            await harness.supervisor.submit(_message('Продолжай', update_id=2), Identity.PRIMARY)
            await harness.wait_idle()
            self.assertEqual(harness.telegram.sent_text[-1][1], 'Ответ\nТекст модели')
            self.assertIn('"response_format": "plain_text"', brain.generate_calls[-1]['prompt'])
            self.assertIn('Отвечай естественно и разговорно', brain.generate_calls[-1]['prompt'])
        finally:
            await harness.close()

    async def test_quoted_preference_example_and_ordinary_markdown_question_do_not_save(self):
        harness = ObservationHarness(ScriptedBrain(response='Короткий ответ.'))
        try:
            quoted = (
                'In this discussion of Markdown, does the quoted line '
                '"always answer in plain text" sound like a durable preference?'
            )
            await harness.supervisor.submit(_message(quoted), Identity.PRIMARY)
            await harness.wait_idle()
            self.assertEqual(harness.store.get_state('user_preferences:101'), {
                'response_format': 'auto', 'markdown_allowed': True,
                'screenshot_delivery': 'explicit_only',
            })
            self.assertFalse(harness.store.get_state('preference_save:101')['ok'])
            self.assertEqual(len(harness.brain.generate_calls), 1)

            await harness.supervisor.submit(_message('What does Markdown mean?', update_id=2), Identity.PRIMARY)
            await harness.wait_idle()
            self.assertEqual(len(harness.brain.generate_calls), 2)
            self.assertEqual(harness.store.get_state('user_preferences:101')['response_format'], 'auto')
            self.assertEqual(harness.store.get_state('preference_save:101')['updated'], {})
            self.assertEqual(harness.brain.records, [])
        finally:
            await harness.close()

    async def test_invalid_preference_action_is_rejected_without_saved_receipt(self):
        brain = ScriptedBrain([ScriptedTurn(calls=[
            ('save_preferences', {'updates': {'tone': 'conversational', 'address': 'invalid'}}),
            ('save_preferences', {'updates': {}}),
            ('save_preferences', {'updates': {'unknown_field': 'x'}}),
        ], reply='Не удалось сохранить настройку.')])
        harness = ObservationHarness(brain)
        try:
            await harness.supervisor.submit(_message('Please keep our conversations informal.'), Identity.PRIMARY)
            await harness.wait_idle()
            self.assertEqual(harness.store.get_state('user_preferences:101')['response_format'], 'auto')
            self.assertNotIn('tone', harness.store.get_state('user_preferences:101'))
            self.assertFalse(harness.store.get_state('preference_save:101')['ok'])
            self.assertEqual([record.data['code'] for record in brain.records], ['invalid_arguments'] * 3)
            self.assertTrue(all(record.data['ok'] is False for record in brain.records))
            self.assertEqual(harness.telegram.sent_text[-1][1], 'Не удалось сохранить настройку.')
        finally:
            await harness.close()


if __name__ == '__main__':
    unittest.main()
