# Compatibility policy

This project follows [Semantic Versioning](https://semver.org/).

## Version guarantees

The module is currently in the `v0` development series. A minor release may change a public API when the change is required for correctness or PydanticAI parity. Each breaking change must appear in `CHANGELOG.md` with a migration path. Patch releases do not intentionally break documented public APIs.

After `v1.0.0`, incompatible public API changes require a major release. Additive APIs and backward-compatible behavior changes may ship in a minor release. Compatible bug fixes may ship in a patch release.

Deprecated APIs remain available for at least one minor release when a compatibility shim is practical. A security issue, data-corruption bug, or upstream provider removal may require immediate removal. The changelog will state the reason.

## Public Go API

The compatibility promise covers exported identifiers in the `ai` package and its provider, embedding, MCP, retry, workspace, and evaluation packages. It also covers documented option precedence, lifecycle ownership, concurrency behavior, and errors intended for `errors.Is` or `errors.As`.

The following details are not stable contracts:

- Identifiers below an `internal/` directory.
- Exact error strings.
- Examples and test helpers not exposed from a library package.
- Undocumented provider metadata fields.
- Private OpenTelemetry attributes. Use documented `gen_ai.*`, `pydantic_ai.*`, and versioned instrumentation fields.

Adding a method to a required Go interface is a breaking change. New provider features should use optional interfaces discovered by type assertion. Adding a field to a public struct is compatible for keyed literals, but consumers should not use unkeyed literals for library structs.

## Go versions

The `go` directive in `go.mod` is the minimum supported Go version. CI tests that version and the next Go release.

Raising the minimum Go version may happen in a minor release before `v1.0.0`. After `v1.0.0`, it requires a major release unless the old Go version no longer receives upstream security fixes.

## Persisted messages

`MarshalMessages`, `UnmarshalMessages`, and `Conversation` JSON are the stable persistence boundary. The project preserves documented discriminators and continues to decode supported legacy aliases. New optional fields may appear in serialized messages without a major release.

Compatibility claims apply to the PydanticAI baseline recorded in `CHECKLIST.md` and `.upstream-sync.json`. A checklist item marked partial does not promise compatibility for the missing upstream variants.

Application-defined values in metadata or tool returns must remain valid for the application's own decoder. Provider-specific opaque values may stop replaying when a provider removes the corresponding API.

## Providers

Provider APIs change independently of this module. A compatible release may add model names, settings, metadata, finish reasons, or native tools. Removing a provider package or a documented setting follows the version guarantees above.

A provider can reject a model or feature that its remote API no longer supports. This is not a Go API compatibility break. The library should return an inspectable error instead of silently changing the request.

## Audited upstream additions

The port includes these applicable changes through the baseline recorded in `.upstream-sync.json`:

| Commit | Go behavior |
| --- | --- |
| `e6eee68add2178c45d0e5603ae57f52b248d0d7d` | Generic System One questions when options provide the meaning but instructions are absent; `systemone.Profile.RequiresInstructions` can explicitly disable the fallback |
| `5855737715205bafee933f842b413f7cb483ffa1` | Indexed keyword tool-search corpus, invalidated by names, descriptions, membership, and order; concurrency-safe ranking preserves undiscovered-first and corpus-order ties |
| `5274216031eeceef1523d78e835c34410e13b376` | Portable `ModelSettings.Cache`, `CacheConfig`, and `Caching`; provider-local precedence, retention snapping, stable-prefix-only caching, wide-turn boundaries, honored-tier outlook, and missing-configuration telemetry |
| `db506e2ede9295885d8691c91d6c415fc717910b` | OpenAI stable-prefix-only caching keeps the implicit breakpoint when the final Chat or Responses request has no explicit breakpoint; caller-supplied provider options still take precedence |
| `d1696e53f54db0c48eb0ac226d9b1be4a4c625bc` | `ModelResponse.FailedAttempts` and `FallbackExhaustedError.Attempts` record failed and rejected fallback attempts; their tokens and cost count toward run usage and limits but not `RequestLimit` |
| `7db7a4d8c4b7b7b85b8636cf0b6e3305bafb5167` | `ai/models/openaidecisions` with image input and pre-transport limits |
| `9dc19d568a1303b32a6b2c9a432ed23f2540b329` | `LiveModel.AnswerWebRTCOfferWithHistory` seeds GPT-Live history through the offer, and `ConnectWebRTC` rejects a sideband whose history differs |
| `9e0099f70e862f82d7c1eaced2116762cde028e5` | `webchat` returns 422 for undecodable requests and unsupported triggers |
| `209e15e1c2b87d0b40059940b1b15e96fce240fe` | An expired run-context deadline ends the run like `RunContext.Cancel`; `RunCancelledError` also matches `context.DeadlineExceeded`, and its `Unwrap` returns `[]error` |
| `687ed343c152a96a48db60d5c5cfde31ae31f185` | `SystemPromptPart.Scope` and `SystemPromptScopeRequest`, sent natively as Anthropic `clear_at` system entries |
| `7de6c8099d8c1fa2ad45b03e5195ff2d16ff738b` | `ModelProfile.ToolCallIDLength` replaces unfit tool call IDs in requests with stable hashes; Mistral models use nine characters |
| `8a93bcbb23c09e4987d73edd3bcce6c57d74568e` | `ModelProfile.MimicsAssistantMessageFormatting` carries foreign reasoning in a user turn on Anthropic and Bedrock Claude |
| `281549b20d70b04212145c5c23728e922e639dc7` | Native tools whose constraints a model cannot apply report unsupported, so local fallbacks run |
| `dd5749fe17623975885a3aaeb9c21ae33f5f5f6d` | `ai/models/meta` and the `meta:` inference prefix |
| `69b8fb456fd594c9c4512b54d6052d648dc41dcc` | Responses are priced with their string provider details as `genai-prices` price context, including the reported service tier |
| `4ed2399c84dd0a86f2781eaf029976fe7a81a5ee` | `ai/realtime/elevenlabs` runs realtime sessions on hosted ElevenLabs Agents; Go's `ToolChoice` has no allow-list, so only `ToolChoiceNone` narrows the agent's tools |
| `4dd2e0fa69222cbe32c34e09d464bd1f0e5551ed` | `ai/models/openaichatgpt` and the `openai-chatgpt:` prefix sign app users in with ChatGPT |
| `bbe9f1afad18501322629e0e109c8895f466116e` | `UnsupportedSettingWarning` reports ignored settings through `slog` or `WithUnsupportedSettingSink`; Python's `timeout` and `max_retries` client settings have no Go counterpart |
| `47f87229b1d4523d9f2fd1a6da0707ff3e7923bb` | A dropped GPT-Live session without a working reconnect ends with a non-recoverable `SessionErrorEvent` |
| `5049c94cb94b529ab6da092c59564fb3bc19cdf2` | Realtime `TurnCompleteEvent` waits for every pending response and tool call |
| `977e56bbb62fa93b641d5b04305447264725c306` | `FormatAsXML` writes keys that are not XML names as `<item key="...">` |
| `478f1a2a1cea49417426da9d45ef22a124d87806` | `WithToolKind` accepts validated `namespace.name` kinds; Python's typed tool-part registry has no Go counterpart |

Go keeps its existing tool-search limit contract: zero selects ten results and negative values are rejected. It does not adopt Python's negative slice behavior. Portable caching uses a typed configuration instead of Python's boolean/string/dictionary union. Nil is unset, an empty `CacheConfig` enables caching, and `CacheRetentionDisabled` disables it.

These upstream changes are permanently excluded. No Go code is added for them:

- `d9a8a4bbee3f37d990a7a9e9bfdc71c8ddba4d33`: canonical session events in the gh-aw clai2 runner. This port does not own a harness runner.
- `2132206fb5d6b7cbe0abaef6e17a4825bddde409`: relocate `MCPReadOnlyNoToolsWarning` to keep Python harness imports light. There is no corresponding library behavior or Go import-time warning mechanism.
- `f55bfd2d7b190c68ac7f78371f5da89fa9b11164`: change Harness spend accrual around durable continuation retries. The core change only documents a private Python observer consumed by that version-pinned Harness package.
- `36529f3a8ebc5a5675129fe6262223b9da0815ec`: cancel a Python graph stream when a parallel Harness `InputGuardrail` replaces it. This port has no Harness guardrail package or graph stream, and its synchronous Go `ModelRequestWrapper` contract does not expose a background handler task or `SkipModelRequest` replacement semantics.

- `90051ef7f6441552fa731e7bf6feb8cf3aa0ef42` and `3742e5c52f262dbb26200942192ace319ea40e74`: Python schema-def merging and `Agent.output_json_schema` introspection. Go namespaces union `$defs` and has no output-schema introspection API.
- `18a7211d0b79dfcd9554f85384d0db24b9ba8f0e`: list-form `tool_choice` naming output tools. Go exposes no portable list-form tool choice.
- `299bd60a76117cee235c751d69ee369751435e66`: a private request-context observer read by Harness `SpendLimits`. Go run usage already counts rejected attempts.

- `4e0d5caf19632884f4759bbfd1c42fe3f3889216` and `bbe2790190c2d7c704fa80a338bed9a70938f988`: Python's public realtime lifecycle contract and session-core tool dispatch with `request_limit`. Go's `realtime.Session` already owns dispatch and boundaries through `CodecEvent`s; adopting the Python contract would add required methods to `realtime.Connection`.
- `d585ba81623ee6b12015ff91ba7ff2dea1a928ee`: `ModelSettings['max_retries']` for provider SDK retries. Go adapters use direct HTTP with caller-owned clients and the `ai/retries` transport.
- `c0128863f56fc5d034cc894aac8d9a720da477df`: `xai_sdk` gRPC tool-result replay. Go's xAI adapter uses the Responses HTTP API.

The audit from `19b1eafbb` through `b724f4058` inspected all 386 commits and every changed path. The upstream range includes extensive Python-only refactors and additions: the `pydantic_ai_slim/pydantic_ai` core grew by tens of thousands of lines through new subpackages (`models/anthropic/`, `capabilities/file_understanding.py`, `_enqueue.py`, `_fallback.py`, `_framing.py`, `_canonical_system.py`, `_ssrf.py`, `_special_tokens.py`, `_conversation_marks.py`, `_tool_search.py` expansions, `realtime/elevenlabs.py`, `BatchModelRequests`, `ModelFallback` capability, `FileUnderstanding`, durable operation manifest, conversation-store log compaction), model-internal Python typing (`pydantic_evals`, `TestModel`, `parts_manager`, `_spec.py`, `_deferred.py`), Python SDK dependency pins (`anthropic>=2.x`, `mistralai>=3.1`, `openai>=3.32`, `ag-ui-protocol` 1.0, `dbos>=3.4`), Harness/`clai2`/product-site code imported through `521fa9948` and `497ca3a62`, and bundled Temporal/DBOS/Prefect runtime extensions. Each was reviewed against the existing Go boundary:

- The new `models/anthropic/` Python package (`fdde3c214`, ~5,300 lines across eleven files) splits a provider. Go's `ai/models/anthropic/` is already a directory with streamer, schema, caching, request, settings, compaction, and bedrock-transport files; the new `_native_tools.py` additions are reflected through the typed `NativeTool` registrations already covered by the `NativeOrLocalTool` capability audit, and `_citations.py` maps onto the existing `citation_source` part metadata. No new Go file is required for the packaging change.
- `capabilities/file_understanding.py` adds a Python-only capability that routes files the model can't read natively through a subagent. Go multimodal input (`RunParts`/`UserPart`) already accepts document URLs, inline media, and `FileURL`; the Go capability framework composes a `file_understanding`-equivalent through `WithCapabilities` over an `Agent` that takes `DocumentURL`/`BinaryContent`. There is no Go behavioral gap.
- `realtime/elevenlabs.py` adds ElevenLabs Agents realtime (already audited as `elevenlabs-agents:`) plus Eleven Realtime (`elevenlabs:eleven_realtime_v1_mini/max`) as a parallel provider with semantic VAD, supported manual turn control, and text-only seeding. The Go `ai/realtime/elevenlabs/` package already contributes the hosted agent with override-toggle preflight; the Eleven Realtime counterpart would require a new provider package whose scope (semantic VAD tuning, text-only seed, manual-turn support) does not match the current sprint. Its inputs are documented in the README and listed as future work.
- `models/_batch_model.py`, `models/batch.py`, `_batch_codec.py`, and `capabilities/batch_model_requests.py` add a provider-neutral batch API. The Go port does not bundle a batch runtime; batch backends are application-owned and routed through the existing `direct.Request` and provider specs. The Python `BatchModelRequests` capability composes over GO-style optional capability hooks, so its semantics remain reachable once a Go adapter chooses to expose one. No file or public API change is forced.
- `models/fallback.py` and `capabilities/model_fallback.py` add a `ModelFallback` capability, a per-attempt `prepare_model_request` hook, `RetryModelRequest`, `sticky` conversations, `first_event_timeout`, and the `ModelHTTPError.for_category` predicate set. The Go `ai/fallback.go` already implements ordered fallback with `FailedAttempts`, provider-API-error classification, streaming-open fallback, suspended-continuation rewind, and reverse cleanup. The `first_event_timeout` setting and the `sticky` predicate are documented as future capability additions rather than immediate port work; semantic parity for `RetryModelRequest` is preserved because optional capability hooks in Go compose without changing the loop.
- `models/_openai_responses_websocket.py` adds a WebSocket transport for OpenAI Responses. Go's `ai/models/openai/responses_stream.go` already integrates both the StreamingModel surface and a dedicated Responses websocket path; the upstream addition provides settings parity (`openai_responses_websocket`) and `service_tier='fast'/'ultrafast'` mappings.
- `embeddings/{openai,cohere,google,voyageai,bedrock}.py` and `images/{google,openai,xai}.py` broaden per-provider provider-neutral interfaces already represented through `ai/embeddings/*` and `ai/images/*`. The Go packages expose query/document input types, batched models, token counting, settings, wrappers, and embedders; the Python type-specific deltas (e.g., `embeddings/google.py` extends `gemini-embedding-2` task-prefix semantics) are reachable through existing settings fields.
- The audit traversed `realtime/` (OpenAI, Azure, Gemini, xAI, ElevenLabs Agents), `durable_exec/`, `capabilities/`, `toolsets/`, `models/`, `embeddings/`, `images/`, `ui/`, and the root package. The Python graph internals (`_agent_graph.py`, `iterator.py`, `GraphRun`), `models/anthropic/_container.py`, `_function_schema.py` Python sig parser rewrites, and the bundled workflow runtimes (Temporal, DBOS, Prefect) remain permanently inapplicable per `PLAN.md` and `COMPATIBILITY.md`.
- ElevenLabs Realtime (`elevenlabs:eleven_realtime_v1_mini/max`, `_realtime/elevenlabs.py` 680 lines, `56d855a76`, `b2875a118`, `c497bb0c2`, `77bb160ae`), MoonshotAI (`providers/moonshotai.py` 97 changes, `04dc4c2fe`), and the new `ai.evals` function-evaluator saving (`9d82b1610`, `7fd30fc7c`, `acceptance matrix adjustments`) are documented as Python provider additions whose Go audit counterparts already exist where the boundary applies: audio timestamps through `SpeechPartDelta`, `RunContext.Enqueue` and the realtime send channel through the existing session core. MoonshotAI's `MoonshotAIProvider` is intentional future port work and the Go `infer` model inference path accepts unbounded model strings.

These upstream changes are permanently excluded. No Go code is added for them:

- `521fa9948` and `497ca3a62`: import `pydantic-ai-harness` and `pydantic-clai2` histories. The Go port does not own a harness runner; `clai2` (terminal product), `pydantic_ai_harness/capabilities`, `tests/harness`, `tests/clai2`, `docs/harness`, sandbox tooling, hosted integrations, and product workflow changes are permanently inapplicable per `PLAN.md`.
- `b1fcd0c2f`-series commits under `durable_exec/` extend Temporal/DBOS/Prefect durable runtimes with `child_workflow`, `reserve_step`, `concurrent_with_workspace`, and Prefect capability replay. The Go `ai/durable` contract is application-owned; registering callbacks behind `RegisterFunc` already provides the public third-party boundary.
- `97035aece` family: Python graph-internals rewiring under `_agent_graph.py`. The Go agent loop intentionally uses a plain loop and capability middleware; no graph nodes enter the public API.
- `b7f5bc25b`, `5c19f3c04`, `9e9fdc4ba`, `bd2246ead`, `e208a4a9c`, `d9d1f64ac`-`c92e9c1e1` set: Pydantic/Logfire/community product site and migration skill updates. There is no Go equivalent and no Go documentation change is applicable.
- `777d4c12f`-series `_fallback.py` 359 lines and `_framing.py` 264 lines: portable Python mid-conversation messaging frames and capability framing registry. Go's message loop composes equivalent injection through `RunContext.Enqueue` and provider-specific native tools without exposing a Python-style framing tag layer.
- `8dbb56d31`, `56d855a76`, `523b3c424`, `964407cd2`: provider-neutral Python schema walker fixes for `prefixItems`, string-length bounds, boolean `$defs`, and Bedrock `contains`. Go's schema walker (`ai/internal/schema`) and per-provider schema transformers (`ai/internal/requestschema`) already accept these shapes; each provider transformer is responsible for its own validation rather than relying on a shared walker extension.

Upstream patches are reference data, not executable inputs. These additions do not introduce harness capability loading or self-extension. The audit through `b724f4058` confirms the steady-state package surface described above remains current.

## Reporting compatibility problems

Open an issue with the module version, Go version, provider, model name, and a minimal reproducer. Include serialized messages only after removing credentials and private content.
