# Changelog

All notable changes to this project are documented here.

The format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/). This project follows [Semantic Versioning](https://semver.org/).

## [Unreleased]

### Added

- Portable prompt caching with `ModelSettings.Cache`, `CacheConfig`, and the composable `Caching` capability. Provider-local cache settings take precedence. Retention snaps to supported tiers, and stable-prefix-only caching avoids writing one-off conversations.
- Once-per-conversation cache-configuration telemetry for long requests to models that need caching enabled explicitly. Harness-only upstream commits `d9a8a4bb` and `2132206f` remain permanently excluded with no Go code.
- Serializable `Conversation` bundles that preserve history, cumulative usage, identity, and deferred requests across runs.
- Core workspaces with local command and filesystem backends, read-only policy, trusted persisted references, and run-scoped selection.
- System One decision models through `ai/models/systemone`, including custom and Ollama-compatible `/v1/systemone` endpoints.
- TypeSafe Jev generation through `ai/models/typesafe` and provider-prefixed model inference.
- Runtime described output choices through `Choice`, `NewChoices`, `NewStringChoices`, and `NewChoicesAgent`.
- A once-per-process interactive first-run banner with resolved agent, model, output, tool, and capability details; coding-agent detection; CI, test, environment, API, and instrumentation suppression; and terminal CLI placement.

### Changed

- Send a generic System One question for option-described scalar outputs with no instructions. Set `systemone.Profile.RequiresInstructions` to false for compatible servers that permit omitted instructions.
- Index local keyword tool-search terms and refresh the index when the corpus changes. Preserve undiscovered-first ranking and current corpus-order ties across concurrent runs.
- Respect provider-honored cache tiers in cache outlook, preserve previous request boundaries after wide Bedrock and OpenRouter tool turns, and prefer explicit final Anthropic cache markers over automatic caching.
- Add provider cache-outlook defaults, current OpenRouter routing values, current Anthropic output limits, and GPT-6.1 Sol behavior.
- Reject forced tool choice in persistent OpenAI-protocol realtime sessions, support Vertex Gemini Live text output, expose realtime context-window usage, and classify dated Gemini 3.8 Live models correctly.
- Accept TOML as text input, reject duplicate model tool-call IDs before execution, expose detached history repair, and hide MCP Apps tools that are not visible to the model.
- Align TypeSafe Jev routing with decision-model behavior: named routes, speculative field questions, opt-in route thresholds, route handoffs, reasoning-aware history, route premises, and a 32,000-token context window. `typesafe.Settings.ToolCallThreshold` is deprecated and ignored; use `RouteThreshold` to hand low-confidence routes to a fallback model instead of silently selecting the output route.
- Expose the current prompt and routed request to model selectors.
- Price one-hour Anthropic cache writes, realtime image input, and duration-billed voice usage with dedicated usage counters.
- Add Gemini 3.8 Live behavior, five-minute audio-view buffering, refused-input rollback, stalled-exchange tracking, and recoverable OpenAI realtime response failures.
- Add GitHub Copilot device authorization, realtime `WaitForReply`, and terminal error delivery through every realtime consumer path.
- Forward Vertex AI `gs://` image references, require Gemini Developer API `v1alpha` for proactive audio, and default future Gemini models to thinking-level behavior.
- Extend TypeSafe Jev with described boolean criteria, whole-number choices, described null choices, and omission of selected optional defaults.
- Omit unsupported sampling settings for Bedrock Converse GPT-5.6 Sol/Luna/Terra and GPT-6 Sol/Luna/Astra models.
- Add GPT-6 Sol, GPT-6 Luna, and Claude Opus 5.5 model behavior, and upgrade `genai-prices` to v0.1.8.
- Extend TypeSafe Jev with bounded-number scaling, boolean option maps, ten-level rubric limits, recursive and fixed-array rejection, route descriptions and attribution, stable tool handoff history, and argument-bearing output functions.
- Require JSON request media types by default in AG-UI and Vercel handlers, redact telemetry errors and instruction content when content capture is disabled, and harden local web fetching against equivalent domain spellings and invalid charsets.
- Add stable `RunResult` JSON, realtime enqueue delivery events, playback draining and automatic response pricing, and Bedrock adaptive and `xhigh` thinking behavior.

### Deprecated

- `ResolvePromptCacheRetention` and `PromptCacheRetentionModel` are deprecated in favor of `ResolveCacheRetention` and `CacheRetentionModel`.

### Fixed

- Make standard and realtime message enqueueing safe against concurrent run or session completion, and reject retained contexts after shutdown.
- Forward blocked domains to OpenAI Responses web search and pair multiple anonymous hosted tool-search calls and results in provider order.

## [0.4.0] - 2026-09-14

### Added

- Context-aware background pricing updates with immediate and hourly downloads, atomic last-good snapshots, shared workers, bounded responses, and caller-provided clients and error handling.
- A typed agent loop with structured output, tools, retries, usage limits, cancellation, deferred execution, message enqueueing, manual runs, and synchronous or streamed results.
- OpenAI Chat Completions and Responses, OpenAI Codex subscription authentication with OAuth PKCE and safe credential rotation, Amazon Bedrock Converse with streaming, native output, prompt caching, token counting, Nova code interpreter, inference profiles, guardrails, performance options, request metadata, and prompt variables, Anthropic Messages with legacy Bedrock InvokeModel generation, streaming, and token counting, Google Gemini and Vertex AI, Azure OpenAI, GitHub Copilot, vLLM, Groq, OpenRouter, and Z.AI model providers.
- Provider-neutral native tools for web search, web fetch, code execution, image generation, file search, MCP servers, advisors, memory, and X search where providers support them.
- Capability middleware for runs, model requests, tool validation and execution, output validation and processing, instructions, history processing, event streams, and deferred calls.
- MCP clients and toolsets for Streamable HTTP, SSE, stdio, shared sessions, OAuth, sampling, elicitation, prompts, resources, and configuration files.
- MCP server guidance and a client-sampling model with basic, multimodal, tool-enabled, and structured-output requests.
- AG-UI and Vercel AI adapters with secure client-history sanitization, multimodal input, text, reasoning, and tool streaming, standalone transformation, and SSE HTTP serving. AG-UI emits version-gated thinking or reasoning events, timestamps every event, preserves compaction and tool-availability activities, registers frontend tools for external execution, emits resumable external-work interrupts, optionally round-trips generated and provider-hosted files through reserved activities, and completes canceled runs without an unsupported outcome. Both protocols also support approval requests, strict resume decisions, and replacement arguments. Vercel AI accepts file input, streams generated files, emits custom data and source chunks from tool metadata, round-trips compaction, tool-availability, provider, and message metadata, supports dynamic tool history, resumes external deferred tools, emits version-correct invalid-input and denied-output lifecycles, closes open tool inputs before errors, and emits cancellation chunks.
- An official A2A Go SDK executor with sanitized task history, streamed artifacts, dependency resolution, and task lifecycle states.
- Terminal and browser chat entry points with provider-prefixed model inference, MCP configuration, streamed tool status, and session history.
- Durable operation backend contracts with stable naming, codecs, cache identity, explicit model ownership, and Temporal, DBOS, and Prefect registration policies.
- Typed embedding clients for OpenAI-compatible APIs, Google Gemini and Vertex AI, Cohere, VoyageAI, Amazon Bedrock, and local Ollama models.
- Direct image generation and editing through OpenAI, Google Gemini and Vertex AI, and xAI, with native-or-local capability fallback, portable geometry, provider settings, batches, wrappers, deterministic fakes, and OpenTelemetry instrumentation. xAI uses its official gRPC protocol.
- Typed prompt templates and deterministic XML formatting for structured prompt data.
- Agent delegation through typed tools with nested usage and limit propagation.
- OpenTelemetry tracing and metrics for agents, model requests, tools, output functions, compaction, and embeddings.
- Bidirectional realtime sessions with portable speech history, audio and transcript taps, concurrent tools, interruption, reconnection, OpenAI Realtime, Azure OpenAI and Voice Live, Gemini Live, and xAI Grok Voice transports.
- A runnable Logfire OTLP export example with privacy-safe defaults.
- PydanticAI-compatible message serialization, multimodal content, speech history, compaction boundaries, cache points, sanitization, and history repair.
- Evaluation task adapters for `pydantic-evals-go`.
- Recorded generation and Google Gemini/AWS Bedrock embedding provider tests, deterministic fake models, race tests, and per-package 100% statement coverage.

### Changed

- `images.NewImageGenerationCapability` now returns `*images.ImageGenerationCapability[Deps]` so repeated declarations merge direct settings and native configuration. Pass it through `ai.WithCapabilities` as before. If you passed its result to `Agent.AddNativeOrLocal`, use `ai.WithCapabilities` instead.
- Preserve separate OpenAI Chat text parts when streamed tool calls appear between text chunks, including tagged thinking streams.
- Capabilities with stable IDs now combine within one registration layer and are replaced as complete wrapper subtrees by matching run capabilities. Invalid collisions and unsafe default merges fail before execution.
- Dynamic image-generation and X-search fallback subagents now resolve the same native configuration as the outer path. Models without native tool-availability deltas receive provider-neutral synthetic search exchanges without splitting parallel result batches.
- All library packages now live under `ai/`. The core package uses `github.com/Kludex/pydantic-ai-go/ai`, and subpackages use paths such as `github.com/Kludex/pydantic-ai-go/ai/models/openai`.
- Browser chat now serves PydanticAI's official `@pydantic/ai-chat-ui` with cached remote or local HTML, model and native-tool selectors, approval continuation, and configuration and health endpoints. `webchat.Config.AllowedHosts` now permits only IP addresses and localhost by default; set it to `[]string{"*"}` only to restore unrestricted host access behind an authentication boundary.
- Forward visible typed custom events through AG-UI and Vercel AI, and add capability-owned attribution, immediate decision dispatch, ordered listeners, listener timeouts, and durable event envelopes.
- Preserve kind-colliding application tool-return maps, restore uploaded-file serialization defaults, and normalize Vercel URL and JavaScript binary tool outputs.
- Resolve bundled standard and realtime model context windows from `genai-prices` v0.1.6 metadata, while preserving explicit and unknown profile values.
- Add Logfire messages and JSON schemas to model and tool spans, including model-visible retry and terminal-failure results.
- Widen Vercel AI `Chunk.Data` and `UIMessagePart.Data` from object-only maps to arbitrary JSON values. Existing map values remain valid.
- Emit Vercel AI response metadata through the protocol's final `message-metadata` chunk instead of attaching it to `finish`.
- Widen AG-UI `Message.Content` from text to arbitrary protocol content so typed multimodal input and structured tool results can round-trip.
- Default AG-UI streams to protocol version `0.1.19`; set `Config.Version` or `StreamConfig.Version` for an older frontend.
- Widen AG-UI `Event.Content` to carry either tool-result text or structured activity snapshots.
- Add opt-in AG-UI file preservation without weakening the separate uploaded-file trust setting.
- Forward detached AG-UI state, context, custom properties, and resolved run IDs to dependencies implementing `RunInputReceiver`.
- Add a run-scoped AG-UI queue for state snapshots, JSON Patch deltas, and custom events.
- Widen AG-UI `Event.Delta` to carry state patch arrays as well as streamed text.
- Preserve AG-UI developer messages, typed-tool kinds, failed outcomes, and provider-native identities through encrypted metadata and echoed history.
- Infer embedding models for the audited OpenAI-compatible provider aliases, including environment-configured LiteLLM, Snowflake, vLLM, and GitHub Copilot endpoints.
- Preserve Anthropic stale-thinking recovery and active-turn container uploads, normalize OpenAI-compatible finish behavior, and map Anthropic web-search usage for pricing.
- Send native JSON Schema output through Anthropic `output_config.format` on supported Claude model families.
- Extend reflected JSON Schema with embedded structs, nullable pointers, JSON/time/text/base64 representations, and common validation annotations.
- Add typed Gemini and Vertex cached-content references with API-required system and tool omission.
- Add typed Vertex Model Armor prompt and response screening configuration for non-streaming requests.
- Add typed Anthropic container reuse, managed skills, fresh-container requests, and model-gated code execution versions.
- Add profile-aware Anthropic adaptive thinking, effort mapping, and model-specific reasoning restrictions.
- Add OpenAI Chat Completions input for provider-hosted document file IDs.
- Add names, descriptions, extension URIs, and detached metadata to streamed A2A artifacts.
- Expand reflected JSON Schema annotations with required fields, constants, compositions, conditions, containment, content metadata, and comma-safe JSON values.
- Add sanitized A2A related-task histories and artifacts to agent context.
- Resume A2A deferred calls and approvals from authoritative stored task state without repeating completed tool side effects.
- Adapt official A2A clients into blocking static and streamed models with multimodal requests, task continuity, extension and push configuration, artifact metadata, and inspectable failures.
- Configure implicit native web search on Groq compound models with validated domain filters.
- Translate portable reasoning by Groq model family and report Qwen 3 effort overrides.
- Normalize Groq compound search executions into provider-native call and return parts.
- Restrict Groq multimodal prompts to supported URL and inline images before transport.
- Recover Groq `tool_use_failed` payloads as retryable static and streamed model output.
- Read terminal Groq streaming usage from the `x_groq` envelope.
- Split Groq `<think>` content across static and streamed chunk boundaries.
- Align reflected schemas with custom JSON marshalers, `json:",string"`, fixed arrays, JSON-compatible map keys, numeric JSON values, and Go's embedded-field selection rules.
- Complete Draft 2020-12 reflected-schema annotations with structured merge and replacement escapes.
- Normalize tagged Chat Completions reasoning, moderation metadata, and `o1-mini` instruction roles.
- Add local and Ollama Cloud Chat Completions models with image, reasoning, and structured-output support.
- Add Cerebras Chat Completions models with GLM and GPT-OSS reasoning behavior.
- Add Hugging Face Inference Providers with routed endpoints, tagged reasoning, images, and function tools.
- Add Cohere v2 Chat models with thinking, function tools, structured output, and billed usage.
- Add Crusoe Serverless Inference models with guided output and family-aware reasoning.
- Infer uploaded-file media types and stable identifiers when callers omit them.
- Complete the audited OpenAI Responses, native-tool, compaction, and serialized-content surfaces.
- Preserve Bedrock static and streamed request IDs, traces, performance settings, and additional response fields.
- Preserve Gemini static and streamed traffic-type metadata and complete its native-tool audit.
- Preserve Anthropic static and streamed text citations as detached part metadata.
- Complete OpenRouter downstream profile handling with safe non-leading system-prompt fallback.
- Add native Mistral Chat Completions with streaming, thinking, tools, caching, and multimodal input.
- Add Snowflake Cortex models with account-scoped endpoints and Claude reasoning.
- Add Amazon Bedrock Mantle models with endpoint routing, bearer or SigV4 authentication, and response-scoped tool-call IDs.
- Add xAI generation models with reasoning, uploaded files, native tools, and X and collections search.

### Deprecated

- Nothing is deprecated.

### Removed

- Nothing has been removed.

### Fixed

- Accept OpenAI Responses web-search progress events without aborting streamed runs.
- Keep realtime sessions open when automatic barge-in is enabled for a model without interruption support.
- Let a realtime tool close its session without racing the connection pump into waiting on that tool.

### Security

- No security fixes have been released yet.

[Unreleased]: https://github.com/Kludex/pydantic-ai-go/compare/v0.4.0...HEAD
[0.4.0]: https://github.com/Kludex/pydantic-ai-go/commits/v0.4.0
