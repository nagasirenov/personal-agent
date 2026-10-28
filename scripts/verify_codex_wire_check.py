"""Run the local Codex wire self-check once against the installed CLI. No model call, no Telegram.

Usage: python scripts/verify_codex_wire_check.py [--record DIR]

Without --record nothing is persisted. With --record DIR the passing version is written to
DIR/verified_codex_versions.json exactly as the daemon would do.
"""
from __future__ import annotations

import asyncio
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from personal_agent.backends import CodexSubscriptionBackend, resolve_codex_command  # noqa: E402


async def main() -> int:
    record = Path(sys.argv[sys.argv.index("--record") + 1]) if "--record" in sys.argv else None
    command = resolve_codex_command("codex")
    with tempfile.TemporaryDirectory(prefix="personal-agent-wirecheck-") as temp:
        backend = CodexSubscriptionBackend(model="gpt-6-luna", reasoning_effort="medium", codex_command=command,
                                           work_root=record if record is not None else temp)
        returncode, raw = await backend._probe("--version")
        version = backend._parse_version(raw)
        print(f"codex={command}\nversion={version} (probe rc={returncode})")
        print(f"builtin_verified={version in backend.VERIFIED_CODEX_VERSIONS}")
        ok, detail = await backend._wire_self_check(version or "unknown")
        print(f"wire_self_check={'PASS' if ok else 'FAIL'}: {detail}")
        if ok and record is not None and version:
            backend._record_version(version)
            print(f"recorded in {backend._verified_versions_path()}")
        await backend.close()
        return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
