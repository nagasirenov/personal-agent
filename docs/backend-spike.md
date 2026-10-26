# Phase 0 Brain backend feasibility spike

Date: 2026-10-26. Scope was limited to Brain transport feasibility and the no-PC-tools boundary. No Telegram live test was performed.

## Result

`CodexSubscriptionBackend` is the supported real safe Brain backend for this spike. It uses the installed Codex CLI with the user's existing official ChatGPT login, sends the prompt through stdin, creates a fresh isolated working directory per request, and fails closed if the verified safety boundary cannot be reproduced.

`codex_subscription` is the selected application default after integration and a real successful `PONG` response from the final adapter. `MockBackend` is explicitly offline and used by `demo`. `CodexChatGPTWebBackend` is experimental and requires `enabled=True`; it uses the same verified Codex CLI boundary with an explicit loopback Responses provider rather than reading or copying bridge credentials.

Constructor surface:

```python
MockBackend()
CodexSubscriptionBackend(
    model="gpt-6-sol",
    codex_command="codex",
    reasoning_effort="medium",
    work_root=None,
    data_dir=None,
)
CodexChatGPTWebBackend(
    base_url="http://127.0.0.1:17841/v1",
    model="gpt-6-sol",
    enabled=False,
    codex_command="codex",
    reasoning_effort="medium",
    work_root=None,
    data_dir=None,
)
```

Pass only one of `work_root` or `data_dir`. Main can pass `config.data_dir` so request temp directories stay under agent-owned storage.

`BrainBackend.generate` now accepts direct image inputs without changing existing callers:

```python
await backend.generate(
    prompt,
    request_id,
    images=[Path("agent-owned-capture.png")],  # optional, keyword-only
)
```

`images` is `Sequence[Path] | None`, defaults to `None`, is limited to 8 files, and each path must resolve to an existing regular local file. The subscription backend passes validated absolute paths through Codex CLI `-i/--image`; `view_image` remains disabled and no image tool is enabled.

## Verified subscription path

Current isolation boundary: CLI discovery flags do not prevent this client from
injecting global AGENTS guidance. Every production generation now passes through
`BrainRequestGateway`, which reconstructs the upstream body from application
arguments and drops all native CLI context. See [backend-isolation.md](backend-isolation.md)
for actual egress verification. `CODEX_HOME` is retained only for official auth;
other inherited Codex app/session/tool variables are removed. Authentication is
forwarded in memory by the documented subscription gateway flow, not copied from
auth files. Tools remain disabled both in the CLI and the reconstructed body.

Installed Codex was `0.164.0-alpha.11.1`, and `codex login status` reported `Logged in using ChatGPT`. A wire-level fake Responses server captured the outgoing request from the safe CLI shape and observed `model: gpt-6-sol`, `stream: true`, and `tools: []`. The captured request had authorization present from Codex itself. No model tools were present, covering shell, MCP, browser, workers/subagents, apps, computer use, and native patch-style tools at the transport boundary rather than by prompt instruction.

A real safe CLI turn returned exactly `PONG`. The final integrated adapter was then also smoke-tested with a request directory under `.state/backend-smoke`, stdin transport, read-only sandbox, explicit disabled web search and disabled shell snapshot; it returned exactly `PONG`. No user activity or Telegram context was included in these probes.

The production adapter therefore pins the exact verified Codex version and refuses generation on any other version until that version is re-probed. It passes `--ignore-user-config`, `--ignore-rules`, `--ephemeral`, `--json`, and `--skip-git-repo-check`, and explicitly disables `shell_tool`, `multi_agent`, `apps`, `plugins`, `browser_use`, `browser_use_external`, `computer_use`, `view_image`, `image_generation`, `shell_snapshot`, and `hooks`. Web search is disabled using `-c web_search="disabled"` (not a feature flag). It also sets `model_reasoning_effort`, a read-only sandbox and `approval_policy="never"` explicitly.

Prompts are not placed in argv. The CLI receives `-` and the prompt bytes on stdin. `OPENAI_API_KEY` and `CODEX_API_KEY` are removed from the child environment, so this backend relies on the existing ChatGPT subscription login rather than API-key billing. If the safe CLI invocation, version check, or ChatGPT login check fails, there is no less-restricted fallback.

The installed `0.164.0-alpha.11.1` CLI was locally checked to support `exec -i/--image`. Its `model_instructions_file` setting was also verified with `codex debug prompt-input`: the referenced file is inserted as developer-role `model.base_instructions`. Each Brain request therefore writes a small application-instructions file inside its isolated request directory and passes it with `-c model_instructions_file=...`. Those instructions define the model as the Personal Agent Brain, state that Codex CLI is only transport, and forbid inferring PC/filesystem abilities from the Codex scratch directory, sandbox, repository view, or tool inventory.

Expected operational failures are surfaced as `BackendError(code, public_message)`. Main can persist `code` plus the bounded, scrubbed `public_message` in `last_turn_error` or Telegram diagnostics. A compatibility `.message` alias currently mirrors `public_message`. Provider stderr is used only for local classification (for example rate limit, oversized input, image rejection, transport outage, or model unavailability) and is never returned verbatim. Unknown nonzero exits become `safe_invocation_failed` with only the numeric exit code. The historical 08:55:54Z failure was recorded only as `RuntimeError`; with no preserved stderr its original cause cannot be recovered and is not inferred here.

## ChatGPT-web bridge path

The installed bridge root was `C:\Users\nagas\.codex-chatgpt-web`, version `6.1.13-win32-x64`, with loopback API `http://127.0.0.1:17841/v1`. Existing bridge/global Codex configuration was not modified or restarted, and no auth files, cookies, or bearer credentials were copied or extracted.

Direct bridge probing established that the bridge expects Codex-native turn metadata/environment context and an existing Bearer path. Because the adapter must not extract or copy that credential, production `CodexChatGPTWebBackend` does not send raw HTTP itself. Instead it invokes the same pinned, zero-tool Codex CLI and supplies an explicit local model provider:

- `base_url = http://127.0.0.1:17841/v1` (loopback only)
- `wire_api = responses`
- `requires_openai_auth = true`
- `supports_websockets = false`

The CLI therefore supplies its own existing official auth while `--ignore-user-config` prevents inherited user MCP/apps/hooks/tool configuration. The same explicit tool disables, stdin prompt transport, API-key scrubbing, version pin, and isolated working directory apply to this backend. The `chatgpt-web/` model prefix is preserved because the installed bridge uses it to distinguish Web requests from ordinary Codex pass-through. Web generation was not live-verified; health reports CLI readiness and explicitly marks bridge reachability as unchecked.

## Cancellation and remaining limits

Cancellation is request-scoped. The subscription and web adapters track only their own Codex subprocesses; cancellation terminates the owned process tree. Tests cover targeted cancellation and cancel-all with fake processes. Real remote bridge cancellation semantics were not live-tested, so the adapter does not claim that cancelling the local CLI process guarantees immediate server-side turn cancellation.

CLI streaming was verified at the wire level (`stream: true`). The backend API currently returns a completed string rather than exposing incremental chunks. Direct bridge streaming was not live-tested. Direct subscription CLI image input is wired through `generate(..., *, images=...)`; `model_info()` exposes `image_input_supported`, `image_input_verified`, and `image_input_limit`. The experimental ChatGPT-web bridge keeps both image capability flags false until that provider path is independently verified.

A real direct-image smoke test used a generated nonprivate 64x64 solid-red PNG. `codex debug prompt-input` showed the attachment as `input_image` / `user.image`, and `CodexSubscriptionBackend.generate(..., images=[fixture])` returned `Red` to a one-word dominant-color prompt. No current-screen image or private user content was sent.

Capability research was timeboxed. The user deferred live Telegram verification.
