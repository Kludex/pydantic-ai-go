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
- AG-UI and Vercel AI adapters with secure client-history sanitization, text, reasoning, and tool streaming, standalone transformation, and SSE HTTP serving.
- An official A2A Go SDK executor with sanitized task history, streamed artifacts, dependency resolution, and task lifecycle states.
- Terminal and browser chat entry points with provider-prefixed model inference, MCP configuration, streamed tool status, and session history.
- Typed embedding clients for OpenAI-compatible APIs, Google Gemini and Vertex AI, Cohere, VoyageAI, Amazon Bedrock, and local Ollama models.
- Typed prompt templates and deterministic XML formatting for structured prompt data.
- Agent delegation through typed tools with nested usage and limit propagation.
- OpenTelemetry tracing and metrics for agents, model requests, tools, output functions, compaction, and embeddings.
- A runnable Logfire OTLP export example with privacy-safe defaults.
- PydanticAI-compatible message serialization, multimodal content, speech history, compaction boundaries, cache points, sanitization, and history repair.
- Evaluation task adapters for `pydantic-evals-go`.
- Recorded provider tests, deterministic fake models, race tests, and per-package 100% statement coverage.

### Changed

- No released behavior has changed yet.

### Deprecated

- Nothing is deprecated.

### Removed

- Nothing has been removed.

### Fixed

- No fixes have been released yet.

### Security

- No security fixes have been released yet.
