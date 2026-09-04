# Changelog

All notable changes to this project are documented here.

The format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/). This project follows [Semantic Versioning](https://semver.org/).

## [Unreleased]

### Added

- A typed agent loop with structured output, tools, retries, usage limits, cancellation, deferred execution, message enqueueing, manual runs, and synchronous or streamed results.
- OpenAI Chat Completions and Responses, Amazon Bedrock Converse with streaming, native output, prompt caching, token counting, Nova code interpreter, inference profiles, guardrails, performance options, request metadata, and prompt variables, Anthropic Messages with legacy Bedrock InvokeModel generation, streaming, and token counting, Google Gemini and Vertex AI, Azure OpenAI, Groq, OpenRouter, and Z.AI model providers.
- Provider-neutral native tools for web search, web fetch, code execution, image generation, file search, MCP servers, advisors, memory, and X search where providers support them.
- Capability middleware for runs, model requests, tool validation and execution, output validation and processing, instructions, history processing, event streams, and deferred calls.
- MCP clients and toolsets for Streamable HTTP, SSE, stdio, shared sessions, OAuth, sampling, elicitation, prompts, resources, and configuration files.
- MCP server guidance and a client-sampling model with basic, multimodal, tool-enabled, and structured-output requests.
- AG-UI and Vercel AI adapters with secure client-history sanitization, multimodal input, text, reasoning, and tool streaming, standalone transformation, and SSE HTTP serving. AG-UI emits version-gated thinking or reasoning events, timestamps every event, preserves compaction and tool-availability activities, registers frontend tools for external execution, emits resumable external-work interrupts, optionally round-trips generated and provider-hosted files through reserved activities, and completes canceled runs without an unsupported outcome. Both protocols also support approval requests, strict resume decisions, and replacement arguments. Vercel AI accepts file input, streams generated files, emits custom data and source chunks from tool metadata, round-trips compaction, tool-availability, provider, and message metadata, supports dynamic tool history, resumes external deferred tools, emits version-correct invalid-input and denied-output lifecycles, closes open tool inputs before errors, and emits cancellation chunks.
- An official A2A Go SDK executor with sanitized task history, streamed artifacts, dependency resolution, and task lifecycle states.
- Terminal and browser chat entry points with provider-prefixed model inference, MCP configuration, streamed tool status, and session history.
- Durable operation backend contracts with stable naming, codecs, cache identity, explicit model ownership, and Temporal, DBOS, and Prefect registration policies.
- Typed embedding clients for OpenAI-compatible APIs, Google Gemini and Vertex AI, Cohere, VoyageAI, Amazon Bedrock, and local Ollama models.
- Typed prompt templates and deterministic XML formatting for structured prompt data.
- Agent delegation through typed tools with nested usage and limit propagation.
- OpenTelemetry tracing and metrics for agents, model requests, tools, output functions, compaction, and embeddings.
- A runnable Logfire OTLP export example with privacy-safe defaults.
- PydanticAI-compatible message serialization, multimodal content, speech history, compaction boundaries, cache points, sanitization, and history repair.
- Evaluation task adapters for `pydantic-evals-go`.
- Recorded generation and Google Gemini/AWS Bedrock embedding provider tests, deterministic fake models, race tests, and per-package 100% statement coverage.

### Changed

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
- Infer embedding models for the audited OpenAI-compatible provider aliases, including environment-configured LiteLLM and Snowflake endpoints.
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

- No fixes have been released yet.

### Security

- No security fixes have been released yet.
