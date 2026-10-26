# Backend isolation verifier

## Enforced request boundary

The raw pinned CLI still injected global guidance despite the discovery flags:
11 substantial lines from the real global file, the subagent-permission rule and
one AGENTS instruction frame were observed. The global file and settings were
left unchanged. Disabling project docs alone is insufficient on this client.

Production now uses a request-local `BrainRequestGateway`. The CLI supplies its
existing subscription credential to loopback; the gateway discards the entire
CLI input/instructions/tools/metadata and constructs a fresh request only from
the explicit application persona, model, reasoning, prompt and image arguments.
It forwards authentication only to the selected official upstream (or the
explicitly enabled loopback bridge). Cookies, thread IDs and coordinator headers
are not forwarded. Auth files and cookies are never extracted or copied.

The egress body has `store: false`, `stream: true`, `tools: []`, one user message
and exactly the Personal Agent instructions. Cancellation closes the stream and
local listener. This uses the documented [ChatGPT subscription gateway route](https://learn.chatgpt.com/docs/enterprise/sign-in-with-chatgpt-through-a-gateway).

Verified egress: zero real/global/synthetic AGENTS matches, zero coordinator
markers, one user prompt and exactly one app persona. Real text and synthetic
red-image requests returned `ISOLATED_OK` and `Red`. No private screenshot was
uploaded. The full project suite passed 90 tests.

Local app context uses the persistent `personal-agent-v2` namespace. Pre-update
assistant replies stay on disk but are excluded from conversation/FTS model
context; user messages, project observations and legitimate per-account style
preferences remain available.

The Personal Agent Codex backend is intended to send only the Personal Agent model
instructions and the current user prompt. Host Codex project rules, global
`AGENTS.md`, user instructions, coordinator chat context, skills, memories, and
tools must stay outside that model request.

`scripts/verify_backend_isolation.py` is a local wire-level regression check. It
starts a fake upstream Responses endpoint, creates a unique synthetic `AGENTS.md`
in the parent directory of the production request working directory, and invokes
the pinned Codex CLI through `CodexSubscriptionBackend`. It captures the egress
of the production `BrainRequestGateway`: exactly the body the model would see.
The test subclass overrides only the upstream destination and sets fixture mode,
which strips credentials before forwarding to the fake upstream. Production CLI
argument construction, request rebuilding, persona and input path remain in use.

Run it manually from the repository root:

```powershell
.\.venv\Scripts\python.exe scripts\verify_backend_isolation.py
```

The verifier captures the actual Responses JSON body and requires:

- no `Authorization` header at the fake endpoint;
- zero tool definitions (the CLI may encode this as `tools: []` or omit the field);
- the synthetic and real global `AGENTS.md` content to be absent;
- no `# AGENTS.md` instruction frame;
- no known coordinator transport/chat markers;
- no configured global `user_instructions` value;
- the Personal Agent persona to appear exactly once as an instruction/developer
  message;
- the supplied synthetic user prompt to appear exactly once, with no other
  instruction-bearing chat messages.

Private host guidance and credentials are never printed or written to disk by the
verifier. It reads the global `AGENTS.md` and `config.toml` only to derive
in-memory comparison strings and reports only counts/booleans. The fake endpoint
intentionally returns an error after capturing the request, so a backend error
after capture is expected and does not cause network fallback.

`tests/test_backend_isolation.py` contains offline unit checks for the production
isolation flags, zero-tool invocation shape, persona boundary, and child
environment scrubbing. It does not make a model or cloud request.

The wire check is intentionally separate from the automatic unit suite because it
requires the locally pinned Codex CLI and an existing ChatGPT subscription login.
If the CLI changes its request encoding or payload schema, the verifier fails
closed and reports the observed encoding/check that could not be verified.
