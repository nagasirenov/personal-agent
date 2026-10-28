"""Opt-in Luna check with synthetic files, an isolated store and offline Telegram."""
from __future__ import annotations

import asyncio
import json
from pathlib import Path
import sys
import tempfile
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from personal_agent.backends import CodexSubscriptionBackend, resolve_codex_command
from personal_agent.config import Config
from personal_agent.memory import MemoryStore
from personal_agent.models import Identity, IncomingMessage
from personal_agent.supervisor import TaskSupervisor
from personal_agent.testing import block_host_desktop, traced_handler


SOURCE = 'Я увлекаюсь астрономией и работаю над вымышленным проектом «Полярис-742».'
REQUEST = 'Хочу, чтобы ты помнил в следующих чатах факты из profile.txt.'


class OfflineTelegram:
    def __init__(self):
        self.answers = []

    async def download(self, file_id, destination, max_bytes):
        assert file_id == 'synthetic-profile'
        raw = SOURCE.encode('utf-8')
        assert len(raw) <= max_bytes
        destination.write_bytes(raw)
        return len(raw)

    async def send_text(self, chat_id, text):
        assert chat_id == 101
        self.answers.append(text)

    async def send_photo(self, *args, **kwargs):
        raise AssertionError('Photo delivery is not part of this test')


class NoTranscriber:
    async def close(self):
        pass


class TracedLuna(CodexSubscriptionBackend):
    def __init__(self, root):
        super().__init__(model='gpt-6-luna', reasoning_effort='medium', work_root=root, codex_command=resolve_codex_command('codex'))
        self.tool_calls = []
        self.calls = 0
        self.payload_settings = []

    def _request_gateway(self, prompt, images, *native):
        gateway = super()._request_gateway(prompt, images, *native)
        payload = gateway._make_payload()
        self.payload_settings.append({'model': payload['model'], 'reasoning': payload['reasoning']})
        assert self.payload_settings[-1] == {'model': 'gpt-6-luna', 'reasoning': {'effort': 'medium'}}
        return gateway

    async def generate(self, prompt, request_id=None, *, images=None, tools=None, tool_handler=None):
        self.calls += 1
        return await super().generate(prompt, request_id, images=images, tools=tools,
                                      tool_handler=traced_handler(tool_handler, self.tool_calls))


async def main():
    block_host_desktop()
    with tempfile.TemporaryDirectory(prefix='profile-luna-fixture-') as folder:
        root = Path(folder)
        config = Config(data_dir=root, primary_id=101, secondary_id=202,
                        model='gpt-6-luna', reasoning='medium', timezone_name='UTC', brain_timeout=150)
        store, telegram, brain = MemoryStore(root), OfflineTelegram(), TracedLuna(root)
        aliases = {'memory': str(root / 'memory')}
        with patch('personal_agent.observation_requests.known_paths', return_value=aliases), \
             patch('personal_agent.tools.local_drive_roots', return_value=(root,)), \
             patch('personal_agent.tools.execute_read_request', side_effect=AssertionError('PC reads are disabled in the synthetic check')):
            supervisor = TaskSupervisor(config, store, brain, telegram, asyncio.Event(), transcriber=NoTranscriber())

            async def deny_pc(*args, **kwargs):
                raise AssertionError('PC tools are disabled in the synthetic check')

            supervisor.action_broker.execute = deny_pc
            source_reads = []
            load_source = supervisor.profile_store.load_attachment

            def trace_load(user_id, attachment_id, *rest):
                result = load_source(user_id, attachment_id, *rest)
                source_reads.append({'user_id': user_id, 'attachment_id': attachment_id,
                                     'ok': result.get('ok')})
                return result

            supervisor.profile_store.load_attachment = trace_load

            async def turn(message):
                previous = store.get_state('last_completed_turn')
                await supervisor.submit(message, Identity.PRIMARY)
                await asyncio.gather(*list(supervisor.tasks))
                if store.get_state('last_completed_turn') == previous:
                    raise RuntimeError('turn_failed: ' + str(store.get_state('last_turn_error', {})))

            try:
                await turn(IncomingMessage(1, 101, 101, 'Кратко перескажи содержимое вложения.', 1,
                           attachment={'kind': 'document', 'file_id': 'synthetic-profile',
                                       'file_name': 'profile.txt', 'mime_type': 'text/plain'}))
                attachment_id = store.get_state('last_text_attachment:101')['attachment_id']
                negative_trace = [call['name'] for call in brain.tool_calls]
                negative_ok = (not source_reads and store.get_user_profile(101) is None
                               and not set(negative_trace) & {'read_attachment', 'save_profile'})
                print('negative_request=' + 'Кратко перескажи содержимое вложения.', flush=True)
                print('negative_trace=' + json.dumps(negative_trace, ensure_ascii=False), flush=True)
                print('negative_no_profile_or_profile_tool=' + str(negative_ok), flush=True)
                if not negative_ok:
                    raise AssertionError('Unrequested profile action or save')

                await turn(IncomingMessage(2, 101, 101, REQUEST, 2))
                positive_trace = [call['name'] for call in brain.tool_calls[len(negative_trace):]]
                saved = store.get_user_profile(101)
                receipt = store.get_state('profile_save_receipt:101')
                positive_ok = bool(saved and receipt and receipt.get('ok') is True
                    and receipt.get('read_back_verified') is True
                    and receipt.get('source_attachment_id') == attachment_id
                    and receipt.get('sha256') == saved.get('sha256')
                    and saved.get('user_id') == 101
                    and attachment_id in saved.get('source_attachment_ids', [])
                    and (source_reads == [{'user_id': 101, 'attachment_id': attachment_id, 'ok': True}]
                         or (len(source_reads) >= 1 and all(sr.get('user_id') == 101 and sr.get('attachment_id') == attachment_id and sr.get('ok') is True for sr in source_reads)))
                    and 'save_profile' in positive_trace
                    and 'астрон' in saved['text'].casefold() and '742' in saved['text'])
                print('positive_request=' + REQUEST, flush=True)
                print('positive_trace=' + json.dumps(positive_trace, ensure_ascii=False), flush=True)
                print('source_reads=' + json.dumps(source_reads), flush=True)
                print('saved_profile=' + json.dumps(saved, ensure_ascii=False), flush=True)
                print('receipt=' + json.dumps(receipt, ensure_ascii=False), flush=True)
                print('positive_tool_selection_and_receipt=' + str(positive_ok), flush=True)
                print('gateway_payload_settings=' + json.dumps(brain.payload_settings), flush=True)
                print(f'model_calls={brain.calls}; telegram_network_calls=0; pc_tools=disabled', flush=True)
                if not positive_ok:
                    raise AssertionError('Profile tool selection or verification failed')
            finally:
                await supervisor.shutdown()
                await brain.close()
                store.close()


if __name__ == '__main__':
    asyncio.run(main())
