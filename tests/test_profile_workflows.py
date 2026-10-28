import asyncio
import unittest
from dataclasses import replace

from personal_agent.memory import MemoryStore
from personal_agent.models import Identity
from personal_agent.supervisor import TaskSupervisor
from personal_agent.testing import ScriptedBrain, ScriptedTurn
from tests.test_service_observation_integration import ObservationHarness, FakeTranscriber, _message


SAVE_PROFILE = 'Сохрани этот текст как мой постоянный профиль пользователя. Проверь чтением из хранилища.'


def use_brain(harness, brain):
    harness.brain = harness.supervisor.brain = brain
    return brain


async def upload(harness, text, update_id, caption='', filename='message.txt'):
    async def download(file_id, destination, max_bytes):
        raw = text.encode('utf-8')
        destination.write_bytes(raw)
        return len(raw)
    harness.telegram.download = download
    message = replace(_message(caption, update_id=update_id), attachment={
        'kind': 'document', 'file_id': 'synthetic-profile', 'file_name': filename, 'mime_type': 'text/plain',
    })
    await harness.supervisor.submit(message, Identity.PRIMARY)
    await harness.wait_idle()


def latest_attachment_id(harness, user_id=101):
    return harness.store.recent_attachments(user_id, limit=1)[0]['id']


class ProfileWorkflowTests(unittest.IsolatedAsyncioTestCase):
    async def test_natural_profile_request_uses_model_selected_tool_and_persists(self):
        profile_text = 'Интересуется астрономией; работает над проектом Орбита.'
        brain = ScriptedBrain([
            ScriptedTurn(calls=[('save_profile', {'text': profile_text, 'append': True})],
                         reply='Записал это в профиль.'),
            'Ты интересуешься астрономией и работаешь над проектом Орбита.',
        ])
        harness = ObservationHarness(brain)
        try:
            await harness.supervisor.submit(
                _message('На будущее учитывай: я обожаю астрономию и работаю над проектом Орбита.', update_id=1),
                Identity.PRIMARY)
            await harness.wait_idle()

            saved = harness.store.get_user_profile(101)
            self.assertEqual(saved['text'], profile_text)
            self.assertTrue(harness.store.get_state('profile_save_receipt:101')['read_back_verified'])
            self.assertIn('save_profile', brain.generate_calls[0]['tool_names'])
            self.assertTrue(brain.records[0].data['read_back_verified'])
            self.assertEqual(harness.telegram.sent_text[-1][1], 'Записал это в профиль.')

            await harness.supervisor.submit(_message('Чем я интересуюсь?', update_id=2), Identity.PRIMARY)
            await harness.wait_idle()
            self.assertIn(profile_text, brain.generate_calls[-1]['prompt'])
        finally:
            await harness.close()

    async def test_profile_wording_alone_never_saves_anything(self):
        harness = ObservationHarness()
        try:
            await harness.supervisor.submit(_message(SAVE_PROFILE, update_id=1), Identity.PRIMARY)
            await harness.wait_idle()
            await upload(harness, 'Старый документ', 2)
            await harness.supervisor.submit(_message('Сохрани следующий файл как мой постоянный профиль', update_id=3), Identity.PRIMARY)
            await harness.wait_idle()
            await upload(harness, 'Новый профиль', 4)
            await upload(harness, 'Профиль из текущего вложения', 5, SAVE_PROFILE)
            self.assertIsNone(harness.store.get_user_profile(101))
            self.assertEqual(harness.brain.records, [])
            self.assertIsNone(harness.store.get_state('profile_save_receipt:101'))
        finally:
            await harness.close()

    async def test_current_attachment_can_be_saved_by_model_without_phrase_gate(self):
        source_text = 'Мне нравится астрономия, а сейчас я изучаю переменные звёзды.'
        profile_text = 'Интересуется астрономией и изучает переменные звёзды.'
        natural_request = 'Пусть факты из этой анкеты учитываются при следующих разговорах.'
        harness = ObservationHarness()
        brain = use_brain(harness, ScriptedBrain([ScriptedTurn(
            calls=[('save_profile', lambda records: {
                'text': profile_text, 'append': True,
                'source_attachment_id': harness.store.get_state('last_text_attachment:101')['attachment_id']})],
            reply='Сохранил факты из анкеты.')]))
        try:
            await upload(harness, source_text, 1, caption=natural_request, filename='survey.txt')

            self.assertIn(source_text, brain.generate_calls[0]['prompt'])
            attachment_id = harness.store.get_state('last_text_attachment:101')['attachment_id']
            saved = harness.store.get_user_profile(101)
            receipt = harness.store.get_state('profile_save_receipt:101')
            self.assertEqual(saved['text'], profile_text)
            self.assertEqual(saved['source_attachment_ids'], [attachment_id])
            self.assertEqual(receipt['source_attachment_id'], attachment_id)
            self.assertTrue(receipt['read_back_verified'])
        finally:
            await harness.close()

    async def test_recent_attachment_is_model_selected_loaded_and_saved_with_provenance(self):
        source_text = 'Люблю походы и планирую пройти маршрут по Алтаю.'
        profile_text = 'Любит походы; планирует пройти маршрут по Алтаю.'
        natural_request = 'Хочу, чтобы ты помнил в следующих чатах факты из profile.txt.'
        harness = ObservationHarness(ScriptedBrain(['Файл получил.']))
        try:
            await upload(harness, source_text, 1, filename='profile.txt')
            attachment_id = latest_attachment_id(harness)
            brain = use_brain(harness, ScriptedBrain([ScriptedTurn(calls=[
                ('read_attachment', {'attachment_id': attachment_id}),
                ('save_profile', {'text': profile_text, 'append': True, 'source_attachment_id': attachment_id}),
            ], reply='Добавил сведения из profile.txt в профиль.')]))
            await harness.supervisor.submit(_message(natural_request, update_id=2), Identity.PRIMARY)
            await harness.wait_idle()

            self.assertIn('"recent_documents"', brain.generate_calls[0]['prompt'])
            self.assertIn(attachment_id, brain.generate_calls[0]['prompt'])
            self.assertIn(source_text, brain.records[0].data['text'])
            saved = harness.store.get_user_profile(101)
            receipt = harness.store.get_state('profile_save_receipt:101')
            self.assertEqual(saved['text'], profile_text)
            self.assertEqual(saved['source_attachment_ids'], [attachment_id])
            self.assertEqual(receipt['source_attachment_id'], attachment_id)
            self.assertTrue(receipt['read_back_verified'])
        finally:
            await harness.close()

    async def test_save_before_read_is_refused_then_succeeds_after_read_in_the_same_turn(self):
        source_text = 'Люблю наблюдать кометы и изучаю проект Вега-842.'
        profile_text = 'Наблюдает кометы и изучает проект Вега-842.'
        harness = ObservationHarness(ScriptedBrain(['Файл получил.']))
        try:
            await upload(harness, source_text, 1, filename='profile.txt')
            attachment_id = latest_attachment_id(harness)
            save = {'text': profile_text, 'append': True, 'source_attachment_id': attachment_id}
            brain = use_brain(harness, ScriptedBrain([ScriptedTurn(calls=[
                ('save_profile', save),
                ('read_attachment', {'attachment_id': attachment_id}),
                ('save_profile', save),
            ], reply='Сохранил сведения.')]))
            await harness.supervisor.submit(_message('Запомни факты из profile.txt.', update_id=2), Identity.PRIMARY)
            await harness.wait_idle()

            self.assertEqual(brain.records[0].error_code, 'profile_source_not_loaded')
            self.assertTrue(brain.records[1].ok)
            self.assertTrue(brain.records[2].ok)
            saved = harness.store.get_user_profile(101)
            self.assertEqual(saved['text'], profile_text)
            self.assertEqual(saved['source_attachment_ids'], [attachment_id])
        finally:
            await harness.close()

    async def test_attachment_tools_cannot_read_or_attribute_another_users_file(self):
        harness = ObservationHarness()
        try:
            await upload(harness, 'Синтетический закрытый профиль.', 1, filename='private.txt')
            attachment_id = latest_attachment_id(harness)

            self.assertEqual(harness.supervisor.profile_store.attachment_candidates(202), [])
            result = harness.supervisor.profile_store.load_attachment(202, attachment_id)
            self.assertFalse(result['ok'])
            self.assertEqual(result['error']['code'], 'attachment_not_available')
            denied_save = harness.supervisor.profile_store.save_from_model(
                202, 'Чужие данные', source_attachment_id=attachment_id, authorized_source_attachment_ids=())
            self.assertEqual(denied_save['error']['code'], 'profile_source_not_loaded')

            brain = use_brain(harness, ScriptedBrain([ScriptedTurn(calls=[
                ('read_attachment', {'attachment_id': attachment_id}),
                ('save_profile', {'text': 'Чужие данные', 'source_attachment_id': attachment_id}),
            ], reply='Не удалось загрузить источник.')]))
            message = replace(_message('Запомни сведения из моего файла.', update_id=2),
                              sender_id=202, chat_id=202)
            await harness.supervisor.submit(message, Identity.SECONDARY)
            await harness.wait_idle()
            self.assertIsNone(harness.store.get_user_profile(202))
            self.assertEqual([record.error_code for record in brain.records],
                             ['attachment_not_available', 'profile_source_not_loaded'])
            self.assertTrue(all('Синтетический закрытый профиль.' not in call['prompt']
                                for call in brain.generate_calls))
            self.assertNotIn(attachment_id, brain.generate_calls[0]['prompt'])
        finally:
            await harness.close()

    async def test_profile_tool_validates_arguments_and_size(self):
        harness = ObservationHarness()
        try:
            await harness.supervisor.submit(_message('Запомни: я люблю чай', update_id=1), Identity.PRIMARY)
            await harness.wait_idle()
            brain = use_brain(harness, ScriptedBrain([ScriptedTurn(calls=[
                ('save_profile', {'text': 'Я люблю чай', 'append': 'yes'}),
                ('save_profile', {'text': 'я' * 24001}),
                ('save_profile', {'text': 'fact', 'source_attachment_id': ''}),
                ('save_profile', {'text': '   '}),
                ('save_profile', {'text': 'fact', 'extra': 1}),
                ('save_profile', {'text': 'Я люблю чай', 'append': True}),
            ])]))
            await harness.supervisor.submit(_message('Запомни: я люблю чай', update_id=2), Identity.PRIMARY)
            await harness.wait_idle()
            self.assertEqual([record.error_code for record in brain.records],
                             ['invalid_arguments', 'profile_too_large', 'invalid_arguments',
                              'invalid_arguments', 'invalid_arguments', None])
            self.assertEqual(harness.store.get_user_profile(101)['text'], 'Я люблю чай')
        finally:
            await harness.close()

    async def test_captionless_document_is_shown_to_model_and_survives_restart_for_read_attachment(self):
        harness = ObservationHarness()
        text = 'Мне нравится астрономия. Мой проект называется Космос-713.'
        try:
            await upload(harness, text, 1)
            self.assertIn(text, harness.brain.generate_calls[-1]['prompt'])
            self.assertIsNone(harness.store.get_user_profile(101))
            attachment_id = latest_attachment_id(harness)
            await harness.supervisor.shutdown()
            harness.store.close()
            harness.store = MemoryStore(harness.root)
            brain = ScriptedBrain([ScriptedTurn(calls=[
                ('read_attachment', {'attachment_id': attachment_id}),
                ('save_profile', {'text': text, 'source_attachment_id': attachment_id}),
            ], reply='Сохранил.'), 'Ты интересуешься астрономией.'])
            harness.brain = brain
            harness.supervisor = TaskSupervisor(harness.config, harness.store, brain,
                                               harness.telegram, asyncio.Event(), transcriber=FakeTranscriber())
            await harness.supervisor.submit(_message(SAVE_PROFILE, update_id=2), Identity.PRIMARY)
            await harness.wait_idle()
            self.assertEqual(harness.store.get_user_profile(101)['text'], text)
            self.assertIsNone(harness.store.get_user_profile(202))
            self.assertTrue(harness.store.get_state('profile_save_receipt:101')['read_back_verified'])
            await harness.supervisor.submit(_message('Чем я интересуюсь?', update_id=3), Identity.PRIMARY)
            await harness.wait_idle()
            self.assertIn(text, brain.generate_calls[-1]['prompt'])
        finally:
            await harness.close()

    async def test_large_document_preview_is_not_an_authorized_source_but_full_read_is(self):
        harness = ObservationHarness()
        text = 'Профиль ' * 2300
        try:
            await upload(harness, text, 1)
            self.assertTrue(harness.store.get_state('last_text_attachment:101')['truncated'])
            attachment_id = latest_attachment_id(harness)
            brain = use_brain(harness, ScriptedBrain([ScriptedTurn(calls=[
                ('save_profile', {'text': text, 'source_attachment_id': attachment_id}),
                ('read_attachment', {'attachment_id': attachment_id}),
                # One chunk is not the whole document: it cannot authorize the profile import yet.
                ('save_profile', {'text': text, 'source_attachment_id': attachment_id}),
                ('read_attachment', lambda records: {'attachment_id': attachment_id,
                                                     'offset': records[1].data['next_offset']}),
                ('save_profile', {'text': text, 'source_attachment_id': attachment_id}),
            ])]))
            await harness.supervisor.submit(_message('Сохрани файл в профиль', update_id=2), Identity.PRIMARY)
            await harness.wait_idle()
            self.assertEqual(brain.records[0].error_code, 'profile_source_not_loaded')
            self.assertTrue(brain.records[1].data['truncated'])
            self.assertFalse(brain.records[1].data['eof'])
            self.assertEqual(brain.records[2].error_code, 'profile_source_not_loaded')
            self.assertTrue(brain.records[3].data['eof'])
            self.assertTrue(brain.records[4].ok)
            self.assertEqual(harness.store.get_user_profile(101)['text'], text.strip())
        finally:
            await harness.close()

    async def test_failed_save_preserves_profile_and_is_not_a_turn_error(self):
        harness = ObservationHarness()
        try:
            harness.store.save_user_profile(101, 'Мой прежний профиль', source_attachment_id='', append=False)
            brain = use_brain(harness, ScriptedBrain([ScriptedTurn(
                calls=[('save_profile', {'text': 'я' * 24001, 'append': False})], reply='Слишком большой.')]))
            await harness.supervisor.submit(_message('Замени профиль', update_id=1), Identity.PRIMARY)
            await harness.wait_idle()
            self.assertEqual(harness.store.get_user_profile(101)['text'], 'Мой прежний профиль')
            self.assertEqual(brain.records[0].error_code, 'profile_too_large')
            self.assertFalse(brain.records[0].data['read_back_verified'])
            self.assertIsNone(harness.store.get_state('last_turn_error', None))
            self.assertEqual(harness.telegram.sent_text[-1][1], 'Слишком большой.')
        finally:
            await harness.close()

    async def test_read_attachment_rejects_non_text_and_unknown_ids(self):
        harness = ObservationHarness()
        try:
            brain = use_brain(harness, ScriptedBrain([ScriptedTurn(calls=[
                ('read_attachment', {'attachment_id': 'does-not-exist'}),
                ('read_attachment', {'attachment_id': 5}),
                ('read_attachment', {}),
            ])]))
            await harness.supervisor.submit(_message('прочитай файл', update_id=1), Identity.PRIMARY)
            await harness.wait_idle()
            self.assertEqual([record.error_code for record in brain.records],
                             ['attachment_not_available', 'invalid_arguments', 'invalid_arguments'])
        finally:
            await harness.close()
