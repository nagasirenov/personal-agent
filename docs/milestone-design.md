# Milestone 1: passive observer and local memory

The input specification is preserved in `docs/requirements.ru.md`. Implementation proceeds in vertical slices, not all twelve phases in one change.

## Boundary

OBSERVE produces timestamped, untrusted facts. THINK receives bounded, selected evidence and returns text. Explicit current user requests can invoke bounded directory/text reads and a single desktop capture through deterministic OBSERVE handlers. ACT is unavailable in this release: the gate always denies environment changes, including requests from PRIMARY. There are no file mutation, browser control, shell, installation, Git write or external worker launch endpoints. Internal writes are restricted to the configured agent-owned root. Brain transports must not expose PC tools; direct image inputs are data, not tools.

Telegram authorizes numeric user IDs, accepts private messages only and silently ignores other IDs. The exact trimmed emergency phrase from PRIMARY is handled before attachment downloads, queues or Brain. The gateway keeps polling while generation runs. New PRIMARY messages cancel that user's old reasoning turn and are persisted as steering events. SECONDARY turns have independent cancellation and cannot stop PRIMARY.

## Storage and cost

SQLite is canonical state (WAL, FTS5, events, messages, attachment provenance, polling checkpoint, telemetry). Markdown provides local daily/project/context views. Observations remain local; only selective context for an explicit message goes to the selected Brain. An explicitly requested screenshot is sent to the requesting Telegram account and, with a supported backend, attached to one vision turn. No background cloud requests, screenshot timers, audio streaming or vector DB. Observations and previous assistant messages never trigger capabilities or grant permissions.

## Operation

The daemon is an explicitly started foreground Python process, suitable for a later user-managed Scheduled Task. No autostart registration or application settings changes happen during setup. Missing Telegram credentials or unavailable optional STT are reported directly. A second daemon cannot share the same state root. Polling offsets and completed/in-flight state survive restarts; interrupted replies are marked rather than silently replayed.

## Verification

Offline integration tests use a fake Telegram transport and Brain. They cover identities, injection, emergency precedence and cancellation, nonblocking polling, attachment path confinement, read-only requests, local memory retrieval and telemetry. Windows sensors receive a separate live smoke check. Live Telegram and STT recognition require user-supplied credentials and a local speech model and are not inferred from mocked tests.
