# PydanticAI Feature Parity Checklist

This is the living source of truth for parity work. Update it whenever a feature lands, a gap is discovered, or an API decision changes.

Upstream baseline: `pydantic/pydantic-ai@bf2fb0555cedeb86ef4376629b1317b0ea1a9b2f` (`v2.35.3-17`).

Status:

- [x] Implemented and covered by public-API tests.
- [~] Partially implemented or behavior differs from upstream.
- [ ] Not implemented.
- [-] Intentionally excluded. The reason must be recorded.

## P0 - Core correctness and stable public API

### Agent loop

- [x] Typed `Agent[Deps, Output]` and `Run`.
- [x] Plain-text and tool-based structured output.
- [x] Native structured output for OpenAI Chat Completions and Google Gemini.
- [x] Output validators with model retries via `AddOutputValidator`.
- [x] Dynamic instructions via `AddInstructionsFunc`.
- [x] Message history input and complete/new message results.
- [x] Request, token, output, and total usage limits.
- [x] Run, model-request, and tool-call capability middleware.
- [x] Execute independent tool calls concurrently by default while preserving model order.
- [x] Per-tool sequential execution barrier via `WithSequential()`.
- [x] Agent-wide sequential tool execution via `WithSequentialToolExecution()`.
- [x] Output-tool end strategies via `WithEndStrategy`: `graceful` default, `early`, and `exhaustive`, including retry-wins.
- [ ] Run cancellation initiated from `RunContext`, including draining concurrent tools.
- [ ] Per-run overrides for model, settings, instructions, output type/mode, limits, and tools.
- [ ] Model selection and model-ID resolution per request step.
- [ ] Usage/cost details beyond basic token counts, including cached, audio, and reasoning tokens.

### Messages and persisted history

- [x] Implemented message subset uses upstream-compatible discriminators and validates with upstream `ModelMessagesTypeAdapter`.
- [x] Text, thinking, tool call, tool return with success/failed outcome, retry prompt, and system/user prompt parts.
- [x] Text, image URL, and inline binary user content.
- [~] Pinned upstream fixtures cover basic and multimodal histories; Go output validates with `ModelMessagesTypeAdapter`. Add fixtures as each remaining part type lands.
- [ ] Instruction parts and stable instruction IDs.
- [ ] Native tool call/return parts.
- [ ] File, document, audio, video, speech, uploaded-file, and cache-point content.
- [ ] Compaction and builtin-tool return parts.
- [ ] Tool availability delta parts.
- [ ] Provider details, metadata, run ID, conversation ID, finish reason, and response IDs.
- [ ] Retry prompt structured validation errors.
- [ ] Rich `ToolReturn`: separate return value, extra content, metadata, and revealed tools.
- [ ] Stream event parity: part start/delta/end, final result, and enqueued-message events.

### Tools and toolsets

- [x] Typed function tools with reflected JSON Schema.
- [x] Dependency-aware and simple tool signatures.
- [x] Raw-schema dynamic tool registration.
- [x] Tool and argument-unmarshal retries use independent per-tool counters; output retries use a separate counter.
- [~] JSON Schema supports common structs, arrays, maps, descriptions, and enums; it is not full JSON Schema parity.
- [~] Explicit strict tool mode via `WithStrict()` / `WithoutStrict()` on OpenAI, Anthropic, and Gemini; automatic provider/model defaults and schema compatibility checks remain.
- [x] Per-tool and agent-wide per-step preparation and omission via `AddPreparedTool` and `AddToolsPrepareFunc`, applied in upstream order.
- [ ] Argument validators before approval/execution.
- [x] Agent-wide and per-run function/output retry budgets, plus per-function-tool overrides and `RunContext` retry metadata.
- [ ] Per-toolset retry defaults and output-tool-specific overrides once those abstractions land.
- [x] `ToolFailedf` terminal failure results with persisted `failed` outcome and no retry-budget cost.
- [x] Unknown or prepared-out tool calls produce corrective prompts with currently available tool names and per-name retry budgets.
- [x] Per-tool deadlines via `WithToolTimeout`; cooperating cancellation becomes a retry and consumes only that tool's budget.
- [ ] Tool metadata and provider-specific options.
- [ ] Toolsets: function, combined, filtered, prefixed, renamed, prepared, and approval-required.
- [ ] Deferred/lazy tool loading and tool search.
- [ ] Native/builtin tools distinct from function tools.

### Deferred execution and approval

- [ ] Approval-required tools and dynamic approval requests.
- [ ] External/deferred tool calls.
- [ ] Deferred request/result message types and pause/resume flow.
- [ ] Inline deferred-call handling capability.
- [ ] Approved/denied results, argument overrides, and approval metadata.
- [ ] Repair incomplete tool-call histories when resumed.

### Streaming

- [x] Provider-optional `StreamingModel` and `Agent.RunStream`.
- [x] Text, thinking, and partial tool-argument events.
- [x] Non-streaming fallback replay.
- [x] OpenAI Chat Completions SSE streaming.
- [ ] Anthropic streaming.
- [ ] Google Gemini streaming.
- [ ] OpenAI Responses streaming.
- [ ] Output validation during streaming and partial structured output.
- [ ] Streaming final-output commitment: upstream `run_stream` locks the first matching output and behaves like `early`; Go currently accumulates the response and applies the configured strategy.
- [ ] Stream event processors and capability wrapper.
- [ ] Streamed tool execution events and deferred results.

## P1 - Providers and model behavior

### Provider implementations

- [x] OpenAI Chat Completions.
- [~] OpenAI Responses: non-streaming text, reasoning summaries, and function calls; native output, multimodal content, builtin tools, and streaming remain.
- [~] Anthropic Messages: non-streaming text, thinking, function tools, and multimodal input; streaming and advanced thinking remain.
- [~] Google Gemini: non-streaming text, thought parts, function tools, native output, and multimodal input; streaming remains.
- [ ] OpenAI-compatible provider configuration without provider-specific forks.
- [ ] Azure OpenAI.
- [ ] AWS Bedrock.
- [ ] Groq, Mistral, Cohere, Cerebras, xAI, OpenRouter, Ollama, Hugging Face, and other upstream providers.
- [ ] Provider profiles/capability detection instead of model-name conditionals.
- [ ] Provider HTTP retries and configurable retry policy.
- [ ] Model fallback chains and instrumentation wrappers.

### Model settings

- [x] Max tokens, temperature, top-p, seed, and stop sequences in the common settings type.
- [~] Providers only forward settings they support; compatibility is not validated by profiles.
- [ ] Timeout and request-level deadline settings.
- [x] `ParallelToolCalls` generation setting for OpenAI Chat/Responses and Anthropic; Gemini exposes no equivalent request setting.
- [ ] Thinking/reasoning effort and token budgets.
- [ ] Logprobs, penalties, service tier, response metadata, and provider-specific settings.
- [ ] Settings merge semantics across model, agent, capability, and run levels.

### Outputs

- [x] Text output.
- [x] Tool output.
- [~] Native output on supported providers.
- [x] Native structured output alongside function calls obeys end strategies; plain text remains non-preemptive.
- [ ] Prompted JSON output fallback, including end-strategy handling.
- [ ] Multiple output alternatives / union outputs.
- [ ] Image and binary outputs.
- [ ] Output tool name/description customization and sequential flag.
- [ ] Output preparation per step.

## P1 - Capabilities and ecosystem

### Capability framework

- [x] Setup contributions for static instructions and raw tools.
- [x] Run, model request, tool call, and dynamic instruction hooks.
- [x] Ordered middleware composition; first capability is outermost.
- [x] History processing can be expressed as model-request middleware.
- [ ] Before/after/error hooks in addition to wrappers.
- [ ] Output validation/processing hooks.
- [ ] Tool validation hook separate from tool execution.
- [ ] Event-stream wrapper and per-event processor.
- [ ] Capability ordering constraints and outermost/innermost tiers.
- [ ] Combined and wrapper capabilities.
- [ ] Capability-provided model settings and adaptive model selection.
- [ ] Deferred-call handler hook.

### Built-in capabilities

- [ ] MCP client capability and MCP toolset.
- [ ] Web search, web fetch, X search, and provider-native tools.
- [ ] Thinking configuration.
- [ ] Compaction and history processing helpers.
- [ ] Tool search and deferred capability loading.
- [ ] Prefix/prepare tools and set-tool-metadata helpers.
- [ ] Reinjected system prompts and content-filter error handling.
- [ ] Thread/concurrency executor configuration.

### Integrations

- [ ] First-class `pydantic-evals-go` task adapter.
- [ ] OpenTelemetry parity with PydanticAI span names, attributes, events, and privacy controls.
- [ ] Logfire guidance and examples.
- [ ] AG-UI adapter.
- [ ] Vercel AI protocol adapter.
- [ ] A2A integration.

## P2 - Broader PydanticAI surface

- [ ] Direct model API without an agent loop.
- [ ] Embeddings API and provider implementations.
- [ ] Realtime voice/audio API and providers.
- [ ] MCP server support.
- [ ] Agent-to-agent delegation examples and usage propagation.
- [ ] CLI and web chat entry points.
- [ ] Prompt templates and format helpers.
- [ ] Durable execution integrations.
- [-] Public graph API and graph-backed loop - excluded because this project intentionally uses a plain loop and capability middleware.

## Quality, documentation, and maintenance

- [x] `gofmt`, `go vet`, `golangci-lint`, tests, and per-package 100% coverage in pre-commit.
- [x] Recorded OpenAI and Anthropic traffic with credentials filtered.
- [ ] Record Google Gemini cassettes when credentials are available.
- [ ] Race-detector CI and concurrency stress tests.
- [ ] CI workflow for supported Go versions and replay-only cassettes.
- [ ] API examples for tools, streaming, multimodal input, capabilities, and each provider.
- [ ] Go package documentation for every public contract.
- [ ] Compatibility policy, semantic versioning policy, and changelog.
- [ ] Benchmark loop overhead, streaming, schema reflection, and parallel tools.
- [ ] Pin `.upstream-sync.json` to the audited upstream commit.
- [ ] After parity, add the daily `gh-aw` upstream-sync workflow described in `PLAN.md`.

## Next work

1. Extend upstream message fixtures as remaining persisted part types land.
2. Match streaming final-output commitment and validation semantics.
3. Add provider-profile defaults and schema compatibility checks for strict mode.
4. Add Anthropic, Gemini, and OpenAI Responses streaming.
5. Design deferred tools and approvals around explicit pause/resume values rather than exceptions.
6. Add MCP once raw/dynamic tool lifecycle and deferred calls are stable.
