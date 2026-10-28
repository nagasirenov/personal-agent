from __future__ import annotations

import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from personal_agent.observation_broker import (
    ObservationDenied,
    ObservationScope,
    validate_observation_plan,
)
from personal_agent.observation_requests import ObservationRequest


class ObservationBrokerTests(unittest.TestCase):
    def assertDenied(self, code: str, fn) -> None:  # noqa: N802 - unittest style
        with self.assertRaises(ObservationDenied) as caught:
            fn()
        self.assertEqual(caught.exception.code, code)
        self.assertTrue(caught.exception.public_message)

    def test_rejects_unknown_kinds_and_malformed_requests(self) -> None:
        for request in (ObservationRequest("read_file"), ObservationRequest("shell", r"C:\x"),
                        ObservationRequest("write_file", r"C:\x"), ObservationRequest("screenshot", r"C:\x")):
            with self.subTest(request=request):
                self.assertRaises(ObservationDenied, validate_observation_plan, [request],
                                  ObservationScope(allow_screenshot=True))
        self.assertDenied("request_budget", lambda: validate_observation_plan(
            [ObservationRequest("screenshot")] * 4, ObservationScope(allow_screenshot=True)))

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


if __name__ == "__main__":
    unittest.main()
