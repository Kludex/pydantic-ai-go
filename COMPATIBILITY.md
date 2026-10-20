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

Upstream patches are reference data, not executable inputs. These additions do not introduce harness capability loading or self-extension.

## Reporting compatibility problems

Open an issue with the module version, Go version, provider, model name, and a minimal reproducer. Include serialized messages only after removing credentials and private content.
