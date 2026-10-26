from __future__ import annotations

import json
import os
import re
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from personal_agent.observation_broker import (
    ObservationDenied,
    ObservationScope,
    observation_protocol,
    parse_observation_plan,
    validate_observation_plan,
)
from personal_agent.observation_requests import ObservationRequest


class ObservationBrokerTests(unittest.TestCase):
    def assertDenied(self, code: str, fn) -> None:  # noqa: N802 - unittest style
        with self.assertRaises(ObservationDenied) as caught:
            fn()
        self.assertEqual(caught.exception.code, code)
        self.assertTrue(caught.exception.public_message)

    def test_parse_plain_fenced_and_normal_answer(self) -> None:
        raw = r'{"observations":[{"kind":"read_file","path":"C:\\Temp\\a.txt"}]}'
        expected = [ObservationRequest("read_file", r"C:\Temp\a.txt")]
        self.assertEqual(parse_observation_plan(raw), expected)
        self.assertEqual(parse_observation_plan(f"```json\n{raw}\n```"), expected)
        self.assertIsNone(parse_observation_plan("I can answer this from the data already provided."))

    def test_rejects_malformed_unknown_keys_kinds_grants_and_oversize(self) -> None:
        bad = [
            ('{"observations":[{"kind":"read_file"}]}', "malformed_plan"),
            ('{"observations":[{"kind":"read_file","path":"C:\\\\x","write":true}]}', "malformed_plan"),
            ('{"observations":[{"kind":"shell","path":"C:\\\\x"}]}', "unsupported_kind"),
            ('{"observations":[{"kind":"browser","path":"C:\\\\x"}]}', "unsupported_kind"),
            ('{"observations":[{"kind":"worker","path":"C:\\\\x"}]}', "unsupported_kind"),
            ('{"observations":[{"kind":"write_file","path":"C:\\\\x"}]}', "unsupported_kind"),
            ('{"observations":[{"kind":"screenshot","path":"C:\\\\x"}]}', "malformed_plan"),
        ]
        for text, code in bad:
            with self.subTest(text=text):
                self.assertDenied(code, lambda text=text: parse_observation_plan(text))
        four = {"observations": [{"kind": "screenshot"}] * 4}
        self.assertDenied("request_budget", lambda: parse_observation_plan(json.dumps(four)))
        self.assertDenied("plan_too_large", lambda: parse_observation_plan("{" + "x" * 20000))

    def test_budget_duplicates_and_screenshot_scope(self) -> None:
        scope = ObservationScope(allow_screenshot=False, max_requests=2)
        self.assertDenied(
            "request_budget",
            lambda: validate_observation_plan([ObservationRequest("screenshot")] * 3, scope),
        )
        self.assertDenied(
            "duplicate_request",
            lambda: validate_observation_plan([ObservationRequest("screenshot")] * 2, scope),
        )
        self.assertDenied(
            "screenshot_denied",
            lambda: validate_observation_plan([ObservationRequest("screenshot")], scope),
        )
        self.assertEqual(
            validate_observation_plan([ObservationRequest("screenshot")], ObservationScope(allow_screenshot=True)),
            [ObservationRequest("screenshot")],
        )

    def test_authorized_project_paths_and_out_of_root_denial(self) -> None:
        with tempfile.TemporaryDirectory() as temp, tempfile.TemporaryDirectory() as outside:
            root = Path(temp)
            file = root / "notes.txt"
            file.write_text("ok", encoding="utf-8")
            scope = ObservationScope((root,))
            requests = [ObservationRequest("list_directory", str(root)), ObservationRequest("read_file", str(file))]
            validated = validate_observation_plan(requests, scope)
            self.assertEqual([r.kind for r in validated], ["list_directory", "read_file"])
            self.assertDenied(
                "path_outside_scope",
                lambda: validate_observation_plan([ObservationRequest("read_file", str(Path(outside) / "x.txt"))], scope),
            )

    def test_rejects_relative_unc_device_shell_and_credentials(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            scope = ObservationScope((Path(temp),))
            cases = [
                (r"relative\x.txt", "relative_path"),
                (r"\\server\share\x.txt", "network_or_device_path"),
                (r"\\?\C:\Temp\x.txt", "network_or_device_path"),
                (str(Path(temp) / "x.txt") + "; whoami", "unsafe_path"),
                (str(Path(temp) / ".env"), "credential_path_denied"),
                (str(Path(temp) / ".ssh" / "config"), "credential_path_denied"),
            ]
            for raw, code in cases:
                with self.subTest(raw=raw):
                    self.assertDenied(
                        code,
                        lambda raw=raw: validate_observation_plan([ObservationRequest("read_file", raw)], scope),
                    )

    def test_reparse_component_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            child = root / "linked"
            child.mkdir()
            junction_name = "personal_agent.observation_broker.os.path.isjunction"
            with patch(junction_name, side_effect=lambda value: Path(value) == child):
                self.assertDenied(
                    "path_outside_scope",
                    lambda: validate_observation_plan(
                        [ObservationRequest("read_file", str(child / "x.txt"))], ObservationScope((root,))
                    ),
                )

    def test_protocol_describes_real_one_round_capability(self) -> None:
        protocol = observation_protocol(ObservationScope((Path(r"C:\Project"),), allow_screenshot=False))
        example = re.search(r'Example: (\{.*\})\. Allowed kinds:', protocol).group(1)
        self.assertEqual(parse_observation_plan(example), [ObservationRequest("list_directory", r"C:\project")])
        self.assertIn("one read-only observation round", protocol)
        self.assertIn("screenshot=not allowed", protocol)
        self.assertIn("After observation results, answer normally", protocol)
        self.assertNotIn("if the application sends", protocol.casefold())


if __name__ == "__main__":
    unittest.main()
