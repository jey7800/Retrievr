# 0012 — Custom LLM transport as a supported extension point

Date: 2026-11-23 · Status: accepted · Extends 0007

## Context

Hosts often reach their LLM through a gateway that is OpenAI-compatible in its request
and response bodies but not in its access: OAuth client-credentials tokens that expire,
one URL per model deployment, an `api-version` query parameter, `max_completion_tokens`
instead of `max_tokens`. Configuration (`base_url` + a static key) cannot describe this,
and the library must stay free of any gateway-specific code.

`Retrievr(llm_transport=...)` already existed for tests, but was undocumented, still
required the API key variable, and closed the transport at the end of each run (one
`LLMClient` per run), which broke a host transport on the second run.

## Decision

`llm_transport` (an `httpx.BaseTransport`) is part of the public API:

- it receives every LLM request: `POST {base_url}/chat/completions` with the standard
  chat completions body, and returns the gateway's standard JSON reply;
- with a transport, the API key environment variable is optional; when it is unset no
  `Authorization` header is sent (the transport authenticates); when set, it is sent;
- the library never closes a transport it was given: the host owns it.

The command line has no transport option; hosts with a gateway use the Python API.

## Consequences

Gateway adapters live in the host application (a few dozen lines), and updating the
library does not touch them. Covered by `tests/unit/test_llm_transport.py` (no key,
static key, transport reused across runs, price table keeps the budget cap effective).
