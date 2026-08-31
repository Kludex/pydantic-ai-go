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
- [x] Dynamic instructions reevaluated before every model request via `AddInstructionsFunc` and `WithRunInstructionsFunc`.
- [x] Message history input and complete/new message results.
- [x] Request, token, output, and total usage limits.
- [x] Run, model-request, and tool-call capability middleware.
- [x] Execute independent tool calls concurrently by default while preserving model order.
- [x] Per-tool sequential execution barrier via `WithSequential()`.
- [x] Agent-wide sequential tool execution via `WithSequentialToolExecution()`.
- [x] Output-tool end strategies via `WithEndStrategy`: `graceful` default, `early`, and `exhaustive`, including retry-wins.
- [x] Run cancellation through idempotent `RunContext.Cancel`, terminal `RunCancelledError`, retained usage/history, drained concurrent tools, completed sibling results, and resumable interrupted history.
- [~] Per-run overrides cover fixed/adaptive model selection, static and per-step settings/instructions, output mode, usage limits, retry limits, history, reusable tools, and additive capabilities with setup contributions. Typed output specialization and toolsets remain.
- [x] Typed agent/run and untyped capability model selectors run before every logical request with step, completed-history, prior-model, model-ID, deps, and usage context.
- [x] Application model IDs resolve through ordered agent/capability resolvers, cache once per run, preserve the selection token across steps, and fail with inspectable `UnknownModelIDError`.
- [x] Model-less agents can bootstrap through agent/capability selectors or model IDs, return `ErrNoModel` when unresolved, and attribute selected models on request and outer run spans.
- [ ] Lifecycle entry/exit hooks for models selected during a run.
- [x] Fresh run and conversation IDs populate all generated requests/responses; explicit run IDs reject history collisions, while conversation IDs inherit from history or reset through `WithConversationID("new")`.
- [~] Usage includes requests, successful function-tool calls, inclusive input/output totals, cache read/write, audio, reasoning, prediction, arbitrary integer detail keys, and optional USD cost, with projected tool-call and known-cost limits. Automatic pricing remains.

### Messages and persisted history

- [x] Implemented message subset uses upstream-compatible discriminators and validates with upstream `ModelMessagesTypeAdapter`.
- [x] Text, thinking, tool call, tool return with success/failed outcome, retry prompt, and system/user prompt parts.
- [x] Text, image URL, and inline binary user content.
- [~] Pinned upstream fixtures cover basic, multimodal, interrupted, synthesized-return, request-instruction, dynamic-system-prompt, structured-retry, response/part-metadata, and arbitrary usage-detail histories; Go output validates with `ModelMessagesTypeAdapter`. Add fixtures as each remaining part type lands.
- [x] Provider parameters preserve static/dynamic instruction boundaries, and each sent `ModelRequest` persists its effective joined instructions across serialization and compatible history merging.
- [x] Legacy system prompts support static values, one-time functions, and explicit stable dynamic IDs; dynamic parts retain timestamps/IDs in serialized history and reevaluate after model selection when history resumes.
- [x] System, user, tool-return, and retry request parts preserve upstream-compatible timestamps; generated parts receive UTC timestamps without rewriting history values.
- [ ] Native tool call/return parts.
- [ ] File, document, audio, video, speech, uploaded-file, and cache-point content, including provider prompt-cache placement from static instruction boundaries.
- [ ] Compaction and builtin-tool return parts.
- [ ] Tool availability delta parts.
- [x] Implemented requests/responses preserve timestamps, local metadata, run/conversation IDs, normalized finish reasons, provider details/URLs/names, response IDs, and lifecycle state, including deprecated `vendor_*` aliases.
- [x] Text, thinking, and function-tool-call parts preserve IDs, signatures, provider names/details, and typed tool kinds through serialization, fallback replay, keyed streaming accumulation, and consumer-safe copies.
- [x] Retry prompts preserve structured validation errors and timestamps, format provider feedback consistently, and retain JSON Schema keyword, location, message, and offending input details.
- [x] Interrupted tool-return outcomes, request state, and synthesized history repair after run cancellation.
- [x] Synthesized-return metadata markers and deterministic, idempotent repair of trailing, interior, shadowed-ID, malformed-order, and empty-ID dangling calls.
- [x] Orphaned tool results are removed while plain validation feedback is preserved; consecutive requests and synthetic responses are merged with tool results hoisted before user-facing content.
- [~] `ToolReturn` metadata is preserved; separate return value, extra content, and revealed tools remain.
- [x] Stable stream part IDs and keyed/interleaved text, thinking, signature, provider-metadata, and tool-argument deltas across bundled providers and fallback replay.
- [x] Explicit `PartStartEvent`, `PartDeltaEvent`, `PartEndEvent`, `FinalResultEvent`, and metadata-bearing `FinishEvent` with typed, applicable deltas.
- [ ] Enqueued-message events.

### Tools and toolsets

- [x] Typed function tools with reflected JSON Schema, including reusable `Tool[Deps]` values that can be registered on agents or scoped to one run.
- [x] Dependency-aware and simple tool signatures.
- [x] Raw-schema dynamic tool registration.
- [x] Tool and argument-unmarshal retries use independent per-tool counters; output retries use a separate counter.
- [~] Reflected JSON Schema supports common structs, arrays, maps, descriptions, and enums; schema generation is not yet full Pydantic parity.
- [x] Provider schema transforms for implemented providers: Gemini full JSON Schema wire fields, OpenAI compatibility inference/forced rewrites including recursive roots, and opt-in Anthropic strict-subset conversion.
- [x] Provider-aware strict tool mode via `WithStrict()` / `WithoutStrict()`: OpenAI infers schema compatibility, Anthropic is explicit and model-gated, and Gemini 2.5+ defaults to request-wide `VALIDATED`; each provider supports alias/proxy overrides.
- [x] Surface Anthropic's lossy strict transformation of dynamic-map schemas through `anthropic.WithSchemaWarningHandler`.
- [x] Per-tool and agent-wide per-step preparation and omission via `AddPreparedTool` and `AddToolsPrepareFunc`, applied in upstream order.
- [x] Complete Draft 2020-12 tool schemas, including prepared raw schemas and asserted formats, compile before each request and validate arguments before Go decoding or execution; failures produce persisted structured retry details.
- [x] Typed semantic argument validators run after JSON Schema validation and Go decoding but before execution for dependency-aware, simple, prepared, and raw-schema tools, with retry and terminal-failure semantics.
- [x] Agent-wide and per-run function/output retry budgets, plus per-function-tool overrides and `RunContext` retry metadata.
- [ ] Per-toolset retry defaults and output-tool-specific overrides once those abstractions land.
- [x] `ToolFailedf` terminal failure results with persisted `failed` outcome and no retry-budget cost.
- [x] Failed and interrupted tool returns use Anthropic error results and Gemini error responses.
- [x] Unknown or prepared-out tool calls produce corrective prompts with currently available tool names and per-name retry budgets.
- [x] Per-tool deadlines via `WithToolTimeout`; cooperating cancellation becomes a retry and consumes only that tool's budget.
- [~] Tool metadata is cloned for per-step preparation and excluded from provider payloads; provider-specific options remain.
- [~] Reusable function tools and per-run additive tools are implemented. Function, combined, filtered, prefixed, renamed, prepared, and approval-required toolset composition remains.
- [ ] Deferred/lazy tool loading and tool search.
- [ ] Native/builtin tools distinct from function tools.

### Deferred execution and approval

- [ ] Approval-required tools and dynamic approval requests.
- [ ] External/deferred tool calls.
- [ ] Deferred request/result message types and pause/resume flow.
- [ ] Automatically continue provider responses in `suspended` state, including Anthropic `pause_turn` and OpenAI background responses.
- [ ] Inline deferred-call handling capability.
- [ ] Approved/denied results, argument overrides, and approval metadata.
- [ ] Repair incomplete tool-call histories when resumed.

### Streaming

- [x] Provider-optional `StreamingModel` and `Agent.RunStream`.
- [x] Normalized part lifecycle and final-result events over text, thinking, and partial tool arguments, with stable IDs and interleaved-delta routing.
- [x] Non-streaming fallback replay.
- [x] OpenAI Chat Completions SSE streaming.
- [x] Anthropic SSE streaming for text, thinking, function calls, usage, errors, and cancellation.
- [x] Google Gemini SSE streaming for text, thinking, function calls with IDs, usage, errors, and cancellation.
- [x] OpenAI Responses SSE streaming for text, reasoning summaries, function arguments, usage, errors, and cancellation.
- [x] Final streamed output validation; retry requests fail clearly because `RunStream` cannot start another model round after committing output.
- [x] Typed partial text and structured output snapshots through `StreamedRun.Outputs`, with partial-validator context, retry suppression, required-field checks, and a final fully validated value.
- [x] Configurable soft-maximum partial-output debouncing through `OutputsDebounced`, including text, structured output, validator grouping, final flush, and consumer-break cleanup.
- [x] Complete Draft 2020-12 constraint and format validation for partial and final tool/native outputs, followed by custom Go decoding and semantic validators.
- [x] Streaming final-output commitment: `RunStream` locks the first matching text, native, or output-tool result. Configured end strategies still govern co-emitted tools, but retries cannot revoke the committed result.
- [x] Consumer-only stream transformation through `RunEventStreamWrapper` and `StreamEventProcessor`, including automatic streaming for `Run`.
- [x] `FunctionToolCallEvent`, `FunctionToolResultEvent`, `OutputToolCallEvent`, and `OutputToolResultEvent`, with concurrent results emitted in completion order.
- [ ] Streamed deferred request and result events.

## P1 - Providers and model behavior

### Provider implementations

- [x] OpenAI Chat Completions.
- [~] OpenAI Responses: text/reasoning/function-call streaming plus response/item IDs, encrypted reasoning, function namespaces, status, timestamps, and background metadata; native output, multimodal content, builtin tools, and background continuation remain.
- [~] Anthropic Messages: text/thinking/function-tool streaming, signed-thinking round trips, multimodal input, response IDs, stop reasons, and suspended `pause_turn` state; automatic pause continuation, citations, and native tools remain.
- [~] Google Gemini: text/thinking/function-tool streaming, thought-signature round trips, native output, multimodal input, function-call/response IDs, normalized finish reasons, full JSON Schema wire fields, and Gemini 2.5+ strict defaults; native tools and advanced metadata remain.
- [ ] OpenAI-compatible provider configuration without provider-specific forks.
- [ ] Azure OpenAI.
- [ ] AWS Bedrock.
- [ ] Groq, Mistral, Cohere, Cerebras, xAI, OpenRouter, Ollama, Hugging Face, and other upstream providers.
- [ ] Provider profiles/capability detection instead of model-name conditionals.
- [ ] Provider HTTP retries and configurable retry policy.
- [ ] Model fallback chains and instrumentation wrappers.

### Model settings

- [x] Max tokens, temperature, top-p, seed, stop sequences, and a cooperative per-request timeout in the common settings type.
- [~] Providers only forward settings they support; compatibility is not validated by profiles. OpenAI, Anthropic, and Gemini normalize their available cache, audio, reasoning, and prediction usage details.
- [x] Static or per-step `RequestTimeout` bounds synchronous requests and full stream consumption while preserving earlier parent cancellation.
- [x] `ParallelToolCalls` generation setting for OpenAI Chat/Responses and Anthropic; Gemini exposes no equivalent request setting.
- [ ] Thinking/reasoning effort and token budgets.
- [ ] Logprobs, penalties, service tier, response metadata, and provider-specific settings.
- [x] Fieldwise settings resolve per selected model in model-default, agent, capability, and run order, with prior layers visible to each callback; bundled providers expose detached defaults through `WithDefaultSettings`.

### Outputs

- [x] Text output.
- [x] Tool output.
- [~] Native output on supported providers.
- [x] Native structured output alongside function calls obeys end strategies; plain text remains non-preemptive.
- [ ] Prompted JSON output fallback, including end-strategy handling.
- [ ] Multiple output alternatives / union outputs.
- [ ] Image and binary outputs.
- [x] Output tool name, description, strict mode, and sequential execution-barrier configuration through `OutputToolConfig`.
- [x] Output-tool definitions can be modified, renamed, or omitted from fresh copies before each request through `AddOutputToolPrepareFunc`.

## P1 - Capabilities and ecosystem

### Capability framework

- [x] Setup contributions for static instructions, model settings, and raw tools.
- [x] Run, model request, tool call, and dynamic instruction hooks.
- [x] Ordered middleware composition; first capability is outermost.
- [x] History processing can be expressed as model-request middleware.
- [ ] Before/after/error hooks in addition to wrappers.
- [ ] Output validation/processing hooks.
- [ ] Tool validation hook separate from tool execution.
- [x] Event-stream wrapper and per-event processor with standard capability middleware ordering.
- [ ] Capability ordering constraints and outermost/innermost tiers.
- [ ] Combined and wrapper capabilities.
- [x] Capability-provided static/per-step model settings and adaptive model selection.
- [x] Per-run capabilities are set up once per run, contribute static instructions/settings/raw tools, participate in every middleware hook, enable event processing for `Run`, and leave agent configuration unchanged.
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
- [ ] OpenTelemetry parity with PydanticAI span names, attributes, events, arbitrary usage-detail attributes, and privacy controls.
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
2. Add per-run typed output specialization and composable toolsets.
3. Add rich tool return values, extra content, and revealed tools.
4. Design deferred tools and approvals around explicit pause/resume values rather than exceptions, building on pre-execution argument validation.
5. Add streamed deferred request and result events with that lifecycle.
6. Add MCP once raw/dynamic tool lifecycle and deferred calls are stable.
