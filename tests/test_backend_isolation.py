from __future__ import annotations

import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from personal_agent.backends import CodexSubscriptionBackend


class BackendIsolationTests(unittest.TestCase):
    def test_isolation_config_disables_host_context_sources(self) -> None:
        config = set(CodexSubscriptionBackend.ISOLATION_CONFIG)
        self.assertIn("project_doc_max_bytes=0", config)
        self.assertIn("project_doc_fallback_filenames=[]", config)
        self.assertIn("memories.use_memories=false", config)
        self.assertIn("memories.generate_memories=false", config)
        self.assertIn("skills.include_instructions=false", config)
        self.assertIn("skills.bundled.enabled=false", config)
        self.assertIn("features.skip_host_skill_discovery=true", config)
        self.assertIn("include_apps_instructions=false", config)
        self.assertIn("include_collaboration_mode_instructions=false", config)
        self.assertIn("memories", CodexSubscriptionBackend.SAFE_DISABLE_FEATURES)

    def test_argv_uses_isolated_persona_and_zero_tool_flags(self) -> None:
        backend = CodexSubscriptionBackend(reasoning_effort="high")
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            instructions = root / "persona.md"
            argv = backend._argv(
                root,
                root / "final.txt",
                instructions_file=instructions,
            )

        for flag in ("--ignore-user-config", "--ignore-rules", "--ephemeral", "--json"):
            self.assertIn(flag, argv)
        self.assertIn('web_search="disabled"', argv)
        self.assertIn('approval_policy="never"', argv)
        persona_arg = next(
            value for value in argv if value.startswith("model_instructions_file=")
        )
        self.assertEqual(
            Path(json.loads(persona_arg.split("=", 1)[1])),
            instructions,
        )
        self.assertEqual(argv[-1], "-")

    def test_child_env_keeps_only_codex_home_from_codex_namespace(self) -> None:
        with patch.dict(
            os.environ,
            {
                "CODEX_HOME": "auth-home",
                "CODEX_THREAD_ID": "host-thread",
                "CODEX_SESSION_ID": "host-session",
                "CODEX_APP_TOOLS_PIPE_PATH": "host-pipe",
                "OPENAI_API_KEY": "secret",
                "CODEX_API_KEY": "secret-2",
                "TELEGRAM_BOT_TOKEN": "secret-3",
                "KEEP_ME": "yes",
            },
            clear=True,
        ):
            env = CodexSubscriptionBackend._child_env()

        self.assertEqual(env["CODEX_HOME"], "auth-home")
        self.assertEqual(env["KEEP_ME"], "yes")
        self.assertFalse(
            any(key.startswith("CODEX_") and key != "CODEX_HOME" for key in env)
        )
        self.assertNotIn("OPENAI_API_KEY", env)
        self.assertNotIn("CODEX_API_KEY", env)
        self.assertNotIn("TELEGRAM_BOT_TOKEN", env)

    def test_persona_explicitly_rejects_host_agent_context(self) -> None:
        persona = CodexSubscriptionBackend.APPLICATION_INSTRUCTIONS
        self.assertIn("Ты — Personal Agent", persona)
        self.assertIn("Ты не Codex", persona)
        self.assertIn("Ты следуешь только этому тексту", persona)
        self.assertIn("не наследуешь AGENTS.md, инструкции Codex, координатора", persona)
        self.assertIn("субагентов у тебя нет", persona)
        self.assertIn("нативные функции", persona)
        self.assertLess(len(persona), 3200)
        for removed in ("OBSERVATION_TOOL_PROTOCOL", "ACTION_TOOL_PROTOCOL", "WEB_LOOKUP_TOOL_PROTOCOL"):
            self.assertNotIn(removed, persona)
        self.assertNotIn("read-only observation capabilities through the runtime", persona)


if __name__ == "__main__":
    unittest.main()
