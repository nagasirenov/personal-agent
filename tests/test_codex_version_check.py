"""Codex version trust: built-in set + versions recorded by the local wire self-check, probe caching."""
from __future__ import annotations

import asyncio
import json
import tempfile
import unittest
from pathlib import Path

import httpx

from personal_agent import codex_wire_check as wire
from personal_agent.backends import BackendError, CodexSubscriptionBackend
from tests.test_backends import FakeProcess, FakeSubscriptionBackend

PONG = b'{"type":"item.completed","item":{"type":"agent_message","text":"PONG"}}\n'
NEW_VERSION = "0.200.1"


class VersionTrustTests(unittest.IsolatedAsyncioTestCase):
    def make(self, temp: str, processes=None, **kwargs) -> FakeSubscriptionBackend:
        backend = FakeSubscriptionBackend(processes if processes is not None else [FakeProcess(stdout=PONG)],
                                          work_root=Path(temp), **kwargs)
        backend.probe_version = NEW_VERSION
        return backend

    async def test_unknown_version_passing_the_wire_check_is_recorded_and_not_rechecked(self):
        with tempfile.TemporaryDirectory() as temp:
            backend = self.make(temp, [FakeProcess(stdout=PONG), FakeProcess(stdout=PONG)])
            backend.wire_result = (True, "all wire checks passed")
            self.assertEqual(await backend.generate("ping", "r1"), "PONG")
            self.assertEqual(await backend.generate("ping", "r2"), "PONG")
            self.assertEqual(backend.wire_calls, [NEW_VERSION])
            recorded = json.loads((Path(temp) / "verified_codex_versions.json").read_text(encoding="utf-8"))
            self.assertEqual(list(recorded["versions"]), [NEW_VERSION])
            self.assertEqual(recorded["versions"][NEW_VERSION]["method"], "local_wire_self_check")

            # A fresh process (new backend, same data dir) trusts the recorded version without the check.
            again = self.make(temp)
            self.assertEqual(await again.generate("ping", "r3"), "PONG")
            self.assertEqual(again.wire_calls, [])

    async def test_unknown_version_failing_the_wire_check_is_refused_with_detail_and_cached(self):
        with tempfile.TemporaryDirectory() as temp:
            backend = self.make(temp)
            backend.wire_result = (False, "failed checks: cli_parsed_final_text")
            for request_id in ("r1", "r2"):
                with self.assertRaises(BackendError) as caught:
                    await backend.generate("ping", request_id)
                self.assertEqual(caught.exception.code, "unverified_codex_version")
                self.assertIn(NEW_VERSION, caught.exception.public_message)
                self.assertIn("cli_parsed_final_text", caught.exception.public_message)
            self.assertEqual(backend.wire_calls, [NEW_VERSION])  # one self-check per process
            self.assertEqual(backend.spawned, [])
            self.assertFalse((Path(temp) / "verified_codex_versions.json").exists())
            health = await backend.health()
            self.assertFalse(health["version_verified"])
            self.assertEqual(health["version_check"], "wire_self_check_failed")

    async def test_builtin_and_recorded_versions_need_no_wire_check_and_bad_files_are_ignored(self):
        with tempfile.TemporaryDirectory() as temp:
            builtin = self.make(temp)
            builtin.probe_version = builtin.VERIFIED_CODEX_VERSION
            self.assertEqual(await builtin.generate("ping", "r1"), "PONG")
            self.assertEqual(builtin.wire_calls, [])

            path = Path(temp) / "verified_codex_versions.json"
            for content in ("{not json", "[]", json.dumps({"versions": "oops"}),
                            json.dumps({"versions": {"not a version": {}}})):
                path.write_text(content, encoding="utf-8")
                backend = self.make(temp)
                self.assertEqual(backend._recorded_versions(), set(), content)
                with self.assertRaises(BackendError):
                    await backend.generate("ping", "r2")
            path.write_text(json.dumps({"versions": {NEW_VERSION: {}}}), encoding="utf-8")
            trusted = self.make(temp)
            self.assertEqual(await trusted.generate("ping", "r3"), "PONG")
            self.assertEqual(trusted.wire_calls, [])

    async def test_concurrent_requests_share_one_self_check(self):
        with tempfile.TemporaryDirectory() as temp:
            backend = self.make(temp, [FakeProcess(stdout=PONG), FakeProcess(stdout=PONG)])
            release = asyncio.Event()

            async def slow_check(version):
                backend.wire_calls.append(version)
                await release.wait()
                return True, "ok"

            backend._wire_self_check = slow_check
            first = asyncio.create_task(backend.generate("a", "r1"))
            second = asyncio.create_task(backend.generate("b", "r2"))
            await asyncio.sleep(0.05)
            release.set()
            self.assertEqual(await asyncio.gather(first, second), ["PONG", "PONG"])
            self.assertEqual(backend.wire_calls, [NEW_VERSION])

    async def test_cancelling_a_turn_does_not_abort_the_shared_self_check(self):
        with tempfile.TemporaryDirectory() as temp:
            backend = self.make(temp, [FakeProcess(stdout=PONG)])
            release = asyncio.Event()

            async def slow_check(version):
                await release.wait()
                return True, "ok"

            backend._wire_self_check = slow_check
            turn = asyncio.create_task(backend.generate("a", "r1"))
            await asyncio.sleep(0.05)
            turn.cancel()
            with self.assertRaises(asyncio.CancelledError):
                await turn
            release.set()
            self.assertEqual(await backend.generate("b", "r2"), "PONG")
            self.assertTrue(backend.is_version_verified(NEW_VERSION))

    async def test_probe_results_are_cached_but_failures_are_not(self):
        with tempfile.TemporaryDirectory() as temp:
            backend = self.make(temp, [FakeProcess(stdout=PONG) for _ in range(3)])
            backend.probe_version = backend.VERIFIED_CODEX_VERSION
            for index in range(3):
                await backend.generate("ping", f"r{index}")
            self.assertEqual(backend.probe_calls.count(("--version",)), 1)
            self.assertEqual(backend.probe_calls.count(("login", "status")), 1)

            logged_out = self.make(temp, [FakeProcess(stdout=PONG)])
            logged_out.probe_version = logged_out.VERIFIED_CODEX_VERSION
            logged_out.login_ok = False
            with self.assertRaises(BackendError) as caught:
                await logged_out.generate("ping", "r1")
            self.assertEqual(caught.exception.code, "chatgpt_login_required")
            logged_out.login_ok = True  # recovery is noticed at once: the failure was not cached
            self.assertEqual(await logged_out.generate("ping", "r2"), "PONG")

    async def test_cache_expires(self):
        with tempfile.TemporaryDirectory() as temp:
            backend = self.make(temp, [FakeProcess(stdout=PONG) for _ in range(2)])
            backend.probe_version = backend.VERIFIED_CODEX_VERSION
            backend.PROBE_TTL_SECONDS = 0.0
            await backend.generate("ping", "r1")
            await backend.generate("ping", "r2")
            self.assertEqual(backend.probe_calls.count(("--version",)), 2)

    async def test_unparseable_version_output_fails_closed(self):
        with tempfile.TemporaryDirectory() as temp:
            backend = self.make(temp)
            backend.probe_version = "garbage"
            with self.assertRaises(BackendError) as caught:
                await backend.generate("ping", "r1")
            self.assertEqual(caught.exception.code, "unverified_codex_version")
            self.assertEqual(backend.wire_calls, [])


GOOD = dict(
    stub_requests=[{"method": "POST", "path": "/v1/responses", "bearer_auth": True, "header_names": [],
                    "raw": json.dumps({"instructions": "PERSONA", "stream": True, "store": False,
                                       "input": [{"role": "user", "content": [{"type": "input_text", "text": "PROMPT"}]}],
                                       "tools": [{"type": "function", "name": "probe"}]}).encode()}],
    proxy_attempts=[], gateway_requests=1, gateway_error=None, result_text="MARK", marker="MARK",
    instructions="PERSONA", prompt="PROMPT", tool_names=["probe"], generate_error=None)


class WireCheckJudgementTests(unittest.TestCase):
    def failing(self, **changes) -> list[str]:
        checks = wire.evaluate(**{**GOOD, **changes})
        return [name for name, ok in checks.items() if not ok]

    def test_a_clean_run_passes_every_check(self):
        self.assertEqual(self.failing(), [])

    def test_known_housekeeping_and_gateway_route_probes_are_tolerated(self):
        self.assertEqual(self.failing(proxy_attempts=["CONNECT chatgpt.com:443", "CONNECT ab.chatgpt.com:443"],
                                      gateway_error="gateway_route_rejected"), [])
        self.assertEqual(wire.unexpected_hosts(["CONNECT chatgpt.com.evil.net:443", "CONNECT [::1]:443"]),
                         ["CONNECT chatgpt.com.evil.net:443", "CONNECT [::1]:443"])

    def test_each_deviation_is_caught(self):
        def body(**changes):
            payload = json.loads(GOOD["stub_requests"][0]["raw"])
            payload.update(changes)
            return [{**GOOD["stub_requests"][0], "raw": json.dumps(payload).encode()}]

        extra_message = body(input=[{"role": "user", "content": [{"type": "input_text", "text": "PROMPT"}]},
                                    {"role": "developer", "content": [{"type": "input_text", "text": "HOST"}]}])
        cases = {
            "no_request_at_all": (dict(stub_requests=[], gateway_requests=0), "cli_reached_gateway_route"),
            "gateway_upstream_error": (dict(gateway_error="gateway_upstream_http_500"), "gateway_served_exactly_one_request"),
            "two_upstream_requests": (dict(stub_requests=GOOD["stub_requests"] * 2), "exactly_one_upstream_request"),
            "wrong_route": (dict(stub_requests=[{**GOOD["stub_requests"][0], "path": "/other"}]),
                            "upstream_route_and_method"),
            "no_bearer": (dict(stub_requests=[{**GOOD["stub_requests"][0], "bearer_auth": False}]),
                          "cli_sent_bearer_auth"),
            "foreign_instructions": (dict(stub_requests=body(instructions="HOST")), "persona_instructions_exact"),
            "extra_input_message": (dict(stub_requests=extra_message), "only_our_prompt_as_input"),
            "extra_tool": (dict(stub_requests=body(tools=[{"name": "probe"}, {"name": "shell"}])),
                           "tools_exactly_those_offered"),
            "stored_request": (dict(stub_requests=body(store=True)), "request_is_stateless_stream"),
            "sse_not_parsed": (dict(result_text=None, generate_error="empty_response"), "cli_parsed_final_text"),
            "other_host": (dict(proxy_attempts=["CONNECT api.example.net:443"]), "no_connection_to_unexpected_hosts"),
        }
        for name, (changes, expected) in cases.items():
            with self.subTest(case=name):
                self.assertIn(expected, self.failing(**changes))

    def test_failure_description_names_checks_and_never_a_secret(self):
        checks = wire.evaluate(**{**GOOD, "proxy_attempts": ["CONNECT api.example.net:443"], "result_text": None,
                                  "generate_error": "empty_response"})
        detail = wire.describe_failures(checks, "empty_response", ["CONNECT api.example.net:443"])
        self.assertIn("cli_parsed_final_text", detail)
        self.assertIn("empty_response", detail)
        self.assertIn("CONNECT api.example.net:443", detail)


class WireStubTests(unittest.IsolatedAsyncioTestCase):
    async def test_stub_answers_with_a_parseable_final_sse_and_keeps_only_credential_shape(self):
        stub = wire.CaptureStub("HELLO_MARKER").start()
        try:
            async with httpx.AsyncClient(trust_env=False) as client:
                response = await client.post(stub.base_url + "/responses", json={"x": 1},
                                             headers={"Authorization": "Bearer super-secret-token-value"})
            self.assertEqual(response.headers["content-type"], "text/event-stream")
            self.assertIn("response.completed", response.text)
            self.assertIn("HELLO_MARKER", response.text)
            self.assertEqual(len(stub.requests), 1)
            self.assertTrue(stub.requests[0]["bearer_auth"])
            self.assertNotIn("super-secret", repr(stub.requests))
        finally:
            stub.stop()

    def test_proxy_sink_records_attempts_without_forwarding(self):
        import socket
        sink = wire.ProxySink().start()
        try:
            with socket.create_connection(("127.0.0.1", sink.port), timeout=2) as connection:
                connection.sendall(b"CONNECT chatgpt.com:443 HTTP/1.1\r\nHost: chatgpt.com\r\n\r\n")
                connection.shutdown(socket.SHUT_WR)
                connection.recv(10)
            for _ in range(20):
                if sink.attempts:
                    break
                import time
                time.sleep(0.05)
            self.assertEqual(sink.attempts, ["CONNECT chatgpt.com:443"])
            self.assertIn("127.0.0.1", sink.env["NO_PROXY"])
        finally:
            sink.stop()


if __name__ == "__main__":
    unittest.main()
