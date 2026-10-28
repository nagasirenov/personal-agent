"""Opt-in real Luna visual follow-up check; all application inputs and web are synthetic."""
import asyncio
import json
import shutil
import struct
import sys
import tempfile
import zlib
from pathlib import Path
from unittest.mock import AsyncMock, patch

import httpx

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from personal_agent.backends import resolve_codex_command
from personal_agent.config import Config
from personal_agent.memory import MemoryStore
from personal_agent.models import Identity, IncomingMessage
from personal_agent.supervisor import TaskSupervisor
from personal_agent.testing import block_host_desktop, traced_handler
from personal_agent.web_lookup import WebLookupSession
from scripts.verify_web_lookup import CheckedLuna, NoTranscriber

SOURCE = 'https://example.org/synthetic-mascot'
MARKER = 'Zorbi fixture mascot'


def make_image(path):
    # A red square face with two cyan eyes and a blue triangular hat.
    rows = []
    for y in range(128):
        row = bytearray()
        for x in range(128):
            color = (255, 255, 255)
            if 38 <= y <= 112 and 27 <= x <= 100:
                color = (235, 32, 32)
            if 55 <= y <= 67 and (40 <= x <= 51 or 76 <= x <= 87):
                color = (0, 230, 230)
            if 8 <= y < 38 and abs(x - 64) <= y - 8:
                color = (20, 40, 235)
            row.extend(color)
        rows.append(b'\0' + row)
    def chunk(kind, data):
        return struct.pack('>I', len(data)) + kind + data + struct.pack('>I', zlib.crc32(kind + data))
    path.write_bytes(b'\x89PNG\r\n\x1a\n' + chunk(b'IHDR', struct.pack('>IIBBBBB', 128, 128, 8, 2, 0, 0, 0))
                     + chunk(b'IDAT', zlib.compress(b''.join(rows))) + chunk(b'IEND', b''))


class Telegram:
    def __init__(self, image):
        self.image, self.sent = image, []

    async def download(self, file_id, destination, max_bytes):
        assert file_id == 'synthetic-image'
        shutil.copyfile(self.image, destination)
        return destination.stat().st_size

    async def send_text(self, chat_id, text):
        self.sent.append(text)


class TraceLuna(CheckedLuna):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.trace = []

    async def generate(self, prompt, request_id=None, *, images=None, tools=None, tool_handler=None):
        entry = {'images': len(images or []), 'tools': [], 'tool_images': 0}
        self.trace.append(entry)
        calls = []
        answer = await super(CheckedLuna, self).generate(prompt, request_id=request_id, images=images, tools=tools,
                                                         tool_handler=traced_handler(tool_handler, calls))
        entry['tools'] = [call['name'] for call in calls]
        entry['tool_images'] = sum(call['images'] for call in calls)
        entry['source_evidence'] = any(MARKER in call['output'] for call in calls)
        entry['response'] = answer
        return answer


async def main():
    if '--real' not in sys.argv:
        raise SystemExit('Requires --real; uses gpt-6-luna medium only')
    block_host_desktop()
    requests = []
    def handler(request):
        requests.append({'path': request.url.path, 'query': request.url.params.get('q', '')})
        if request.url.path == '/search':
            return httpx.Response(200, content=("<rss version='2.0'><channel><item>"
                "<title>Zorbi: red square mascot with cyan eyes and blue hat</title>"
                f"<link>{SOURCE}</link><description>Fictional character reference</description>"
                "</item></channel></rss>").encode())
        if str(request.url) == SOURCE:
            return httpx.Response(200, headers={'content-type': 'text/html'}, content=(
                f'<html><body><h1>{MARKER}</h1><p>Zorbi is a fictional red square character '
                'with cyan eyes and a blue triangular hat.</p></body></html>').encode())
        raise AssertionError('Unexpected synthetic web request')

    with tempfile.TemporaryDirectory(prefix='visual-lookup-') as temp:
        root = Path(temp)
        image = root / 'fixture.png'
        make_image(image)
        config = Config(data_dir=root, primary_id=101, secondary_id=202, backend='codex_subscription',
                        model='gpt-6-luna', reasoning='medium', brain_timeout=150, timezone_name='UTC')
        brain = TraceLuna(model=config.model, reasoning_effort=config.reasoning,
                          codex_command=resolve_codex_command(config.codex_executable), work_root=root)
        telegram = Telegram(image)
        store = MemoryStore(root)
        supervisor = TaskSupervisor(config, store, brain, telegram, asyncio.Event(), transcriber=NoTranscriber())
        supervisor.context.knowledge = None
        async with httpx.AsyncClient(transport=httpx.MockTransport(handler), trust_env=False) as client:
            try:
                with (patch('personal_agent.tools.WebLookupSession', side_effect=lambda: WebLookupSession(client)),
                      patch('personal_agent.web_lookup._public_url', new=AsyncMock(return_value=True))):
                    await supervisor.submit(IncomingMessage(1, 101, 101, 'Опиши, что видишь.', 1,
                        attachment={'kind': 'sticker', 'file_id': 'synthetic-image', 'is_animated': False}), Identity.PRIMARY)
                    await asyncio.gather(*list(supervisor.tasks))
                    boundary = len(brain.trace)
                    await supervisor.submit(IncomingMessage(2, 101, 101, 'А кто это? Можешь выяснить?', 2), Identity.PRIMARY)
                    await asyncio.gather(*list(supervisor.tasks))
                print('model=gpt-6-luna reasoning=medium', flush=True)
                print('telegram=' + json.dumps(telegram.sent, ensure_ascii=False), flush=True)
                print('model_calls=' + json.dumps(brain.trace, ensure_ascii=False), flush=True)
                print('synthetic_web=' + json.dumps(requests, ensure_ascii=False), flush=True)
                initial = telegram.sent[0].casefold() if telegram.sent else ''
                color = any(word in initial for word in ('красн', 'red'))
                face = any(word in initial for word in ('глаз', 'eyes', 'лиц', 'face'))
                follow = brain.trace[boundary:]
                plans = [name for call in follow for name in call['tools']]
                ok = (boundary >= 1 and brain.trace[0]['images'] == 1 and color and face
                      and len(follow) == 1 and follow[0]['tool_images'] == 1
                      and 'view_recent_media' in plans and 'web_search' in plans and 'web_open' in plans
                      and any(r['path'] == '/search' for r in requests)
                      and any(r['path'] == '/synthetic-mascot' for r in requests)
                      and follow[-1]['source_evidence'] and SOURCE in telegram.sent[-1])
                print('result=' + ('PASS' if ok else 'FAIL'), flush=True)
                if not ok:
                    raise SystemExit(1)
            finally:
                await supervisor.shutdown()
                store.close()


if __name__ == '__main__':
    asyncio.run(main())
