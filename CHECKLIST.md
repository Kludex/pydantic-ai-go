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
- [x] Per-run overrides cover fixed/adaptive model selection, static and per-step settings/instructions, type-safe output specialization, output mode/tool configuration, usage limits, retry limits, history, reusable tools/toolsets, and additive capabilities with setup contributions.
- [x] Typed agent/run and untyped capability model selectors run before every logical request with step, completed-history, prior-model, model-ID, deps, and usage context.
- [x] Application model IDs resolve through ordered agent/capability resolvers, cache once per run, preserve the selection token across steps, and fail with inspectable `UnknownModelIDError`.
- [x] Model-less agents can bootstrap through agent/capability selectors or model IDs, return `ErrNoModel` when unresolved, and attribute selected models on request and outer run spans.
- [x] Optional `ModelOpener` lifecycle runs once per distinct selected model, closes in reverse selection order before toolsets, uses a non-canceled cleanup context, and propagates open/close failures across ordinary and streamed runs.
- [x] Fresh run and conversation IDs populate all generated requests/responses; explicit run IDs reject history collisions, while conversation IDs inherit from history or reset through `WithConversationID("new")`.
- [~] Usage includes requests, successful function-tool calls, inclusive input/output totals, cache read/write, audio, reasoning, prediction, arbitrary integer detail keys, and optional USD cost, with projected tool-call and known-cost limits. Automatic pricing remains.

### Messages and persisted history

- [x] Implemented message subset uses upstream-compatible discriminators and validates with upstream `ModelMessagesTypeAdapter`.
- [x] Text, thinking, tool call, tool return with success/failed outcome, retry prompt, and system/user prompt parts.
- [x] Text, image URL, and inline binary user content.
- [~] Pinned upstream fixtures cover basic, multimodal, interrupted, synthesized-return, request-instruction, dynamic-system-prompt, structured-retry, response/part-metadata, compaction, tool-availability, typed tool-search, and arbitrary usage-detail histories; Go output validates with `ModelMessagesTypeAdapter`. Add fixtures as each remaining part type lands.
- [x] Provider parameters preserve static/dynamic instruction boundaries, and each sent `ModelRequest` persists its effective joined instructions across serialization and compatible history merging.
- [x] Legacy system prompts support static values, one-time functions, and explicit stable dynamic IDs; dynamic parts retain timestamps/IDs in serialized history and reevaluate after model selection when history resumes.
- [x] System, user, tool-return, and retry request parts preserve upstream-compatible timestamps; generated parts receive UTC timestamps without rewriting history values.
- [ ] Native tool call/return parts.
- [ ] File, document, audio, video, speech, uploaded-file, and cache-point content, including provider prompt-cache placement from static instruction boundaries.
- [~] Compaction parts preserve readable summaries, opaque IDs/details, serialization, stream lifecycle, deferred-tool visibility boundaries, and same-provider Anthropic/OpenAI Responses round trips with latest-boundary history trimming. Explicit compaction capabilities/settings and builtin-tool return parts remain.
- [x] Tool availability delta parts, including the legacy `added` decode alias and upstream-compatible serialization.
- [x] Implemented requests/responses preserve timestamps, local metadata, run/conversation IDs, normalized finish reasons, provider details/URLs/names, response IDs, and lifecycle state, including deprecated `vendor_*` aliases.
- [x] Text, thinking, and function-tool-call parts preserve IDs, signatures, provider names/details, and typed tool kinds through serialization, fallback replay, keyed streaming accumulation, and consumer-safe copies.
- [x] Retry prompts preserve structured validation errors and timestamps, format provider feedback consistently, and retain JSON Schema keyword, location, message, and offending input details.
- [x] Failed, denied, and interrupted tool-return outcome values, typed return tool kinds, request state, and synthesized history repair after run cancellation.
- [x] Synthesized-return metadata markers and deterministic, idempotent repair of trailing, interior, shadowed-ID, malformed-order, and empty-ID dangling calls.
- [x] Orphaned tool results are removed while plain validation feedback is preserved; consecutive requests and synthetic responses are merged with tool results hoisted before user-facing content.
- [x] Rich `ToolReturn` values separate the provider-facing return value, trailing multimodal user content, local metadata, and deferred-tool reveals while preserving provider-valid concurrent ordering.
- [x] Stable stream part IDs and keyed/interleaved text, thinking, signature, provider-metadata, and tool-argument deltas across bundled providers and fallback replay.
- [x] Explicit `PartStartEvent`, `PartDeltaEvent`, `PartEndEvent`, `FinalResultEvent`, and metadata-bearing `FinishEvent` with typed, applicable deltas.
- [x] Concurrent-safe `RunContext.Enqueue`, `EnqueueWhenIdle`, and explicit priorities inject grouped user content, request parts, or complete messages; `asap` drains before the next request or redirects final output, `when_idle` redirects only at termination, and `EnqueuedMessagesEvent` carries stamped detached messages. Undelivered groups persist through serialized and repeated deferred pauses and resumable cancellation histories, then drain only after deferred results resolve.

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
- [x] Per-toolset retry and timeout defaults preserve explicit per-tool overrides.
- [x] Output-tool-specific retry overrides through `OutputToolConfig.MaxRetries`, including per-run output-tool configuration.
- [x] `ToolFailedf` terminal failure results with persisted `failed` outcome and no retry-budget cost.
- [x] Reflected scalar/collection/structured tool return schemas, explicit rich-return schema overrides, detached inspection, and rejection of nested rich `ToolReturn` values.
- [x] Failed and interrupted tool returns use Anthropic error results and Gemini error responses.
- [x] Unknown or prepared-out tool calls produce corrective prompts with currently available tool names and per-name retry budgets; a deferred-but-hidden tool receives one free availability correction before later refusals charge its budget.
- [x] Per-tool deadlines via `WithToolTimeout`; cooperating cancellation becomes a retry and consumes only that tool's budget.
- [~] Tool metadata is cloned for per-step preparation and excluded from provider payloads; provider-specific options remain.
- [x] Function toolsets compose through combined, filtered, prefixed, renamed, prepared, metadata, retry-default, and timeout-default wrappers; listing and instructions reevaluate per step, wrapped calls retain original names, and toolsets can be agent-wide or per-run.
- [x] `RequireApprovalToolset` wraps all or selected original names, and `RequireApprovalToolsetWhen` evaluates validated calls dynamically; both forward instructions and run/step/open/close lifecycle.
- [x] Stateful remote toolsets support local `ToolsetID` propagation, per-run isolation, per-step replacement, open/close lifecycle, reverse-order rollback, and lifecycle forwarding through built-in wrappers.
- [x] Deferred tools can be marked individually or through `DeferLoadingToolset`, remain unavailable until revealed, deduplicate concurrent reveals in model order, and retain visibility through serialized/resumed history.
- [x] Local `search_tools` discovery through `WithToolSearch`, with typed results, configurable detached search callbacks, word-bounded relevance, undiscovered-first ranking, result limits, and independent retries.
- [~] Anthropic 4.5+ models render deferred definitions, local search results as `tool_reference` blocks, and other reveals as `tool_addition` blocks with the required beta header. OpenAI Responses maps local search to client-executed `tool_search` in streaming and non-streaming requests, preserves final streamed call IDs, replays `tool_search_output`, and sends other reveals through `additional_tools`. Provider-managed search remains.
- [x] Reset derived deferred-tool discovery visibility at compaction part boundaries while allowing calls generated in the compacting response to use the request-time visibility snapshot; post-boundary reveals remain visible.
- [ ] Native/builtin tools distinct from function tools.

### Deferred execution and approval

- [x] Static and dynamic approval through `WithApprovalRequired`, `WithDynamicApproval`, and explicit `RequestToolApproval` return values, including validated pending calls and local execution after resume.
- [x] Static and dynamic external execution through typed/raw constructors, `WithExternalExecution`, `WithDynamicExternalExecution`, and explicit `RequestExternalToolExecution` values, including rich, failed, or retrying results.
- [x] Explicit `DeferredToolRequests` / `DeferredToolResults` values, detached pending results, preserved partial history, all-result validation, `WithDeferredToolResults` pause/resume, and ordered inline partial resolution.
- [x] Suspended provider responses continue automatically as one logical turn in ordinary and streamed runs, with accumulate/same-ID/fresh-replacement merge modes, early usage checks, separate generation/poll ceilings, context-aware delays, and best-effort cancellation. `Resume` and `ResumeStream` continue persisted suspended history without injecting a prompt or retaining the stale suspended snapshot. Anthropic `pause_turn` reissues immediately; OpenAI Responses background mode supports create, retrieve, streamed cursor resume, static retrieval fallback, polling configuration, and cancel.
- [x] Ordered `DeferredToolCallHandler` capability hook plus `DeferredToolHandlerFunc`, with partial resolution and unresolved-call bubbling.
- [x] Approved/denied results, validated argument overrides, and detached approval metadata through `RunContext.ToolCallApproved` and `ToolCallMetadata`.
- [x] Preserve intended deferred gaps while repairing unrelated incomplete tool-call histories on ordinary resume; completed siblings are not re-executed.
- [x] Resolved calls can defer again with a new approval/external kind and detached metadata; completed sibling results remain in history, unsent resume prompts are omitted on another pause, and the next inline handler receives the replacement request without a duplicate request event. Pending kinds survive serialization through `DeferredToolKindsMetadataKey`, preventing cross-kind result substitution.

### Streaming

- [x] Provider-optional `StreamingModel` and `Agent.RunStream`.
- [x] Normalized part lifecycle and final-result events over text, thinking, compaction, and partial tool arguments, with stable IDs and interleaved-delta routing.
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
- [x] `DeferredToolRequestsEvent` after individual call events and `DeferredToolResultsEvent` for each inline handler result batch, without duplicate call events.
- [x] `EnqueuedMessagesEvent` is emitted once per delivered enqueue group and participates in ordinary event capability ordering and consumer cancellation.
- [x] Stream consumer detachment preserves a detached `SuspendedRun` response/history/usage snapshot without canceling the server-side job; explicit context cancellation still performs best-effort provider cancellation. OpenAI Responses tracks response IDs and the latest consumed sequence cursor before terminal events, so detached jobs resume without replaying consumed events.
- [x] Continuation-segment deltas stream live with append indexes for fresh generations and reused indexes for provider-marked background snapshots; mid-segment metadata preserves resumable detachment snapshots and the latest OpenAI sequence cursor.

## P1 - Providers and model behavior

### Provider implementations

- [x] OpenAI Chat Completions.
- [~] OpenAI Responses: text/reasoning/function-call/compaction streaming plus authoritative terminal snapshots, response/item IDs, encrypted reasoning and compaction round trips, latest-compaction history trimming with standing-prompt provenance, stateful context management and explicit stateless `/responses/compact` with message/custom triggers, durable history replacement, combined usage accounting, function namespaces, portable reasoning effort, logprob requests/static text metadata, service tiers, status, timestamps, configurable background create/poll/retrieve/cancel/detach continuation, and streaming/non-streaming client-executed deferred-tool search/reveal rendering; streamed logprob metadata, provider-managed search, native output, multimodal content, and other builtin tools remain.
- [~] Anthropic Messages: text/thinking/function-tool/compaction streaming, portable effort-to-budget and explicit-budget thinking configuration, service-tier mapping and response metadata, signed-thinking and readable/encrypted compaction round trips, latest-compaction trimming, explicit token-triggered compaction capabilities with summary/pause options, required beta/default context management with extension overrides, multimodal input, response IDs, stop reasons, automatic ordinary/streamed `pause_turn` continuation, and provider-native deferred-definition/reveal rendering; adaptive thinking profiles, native server search, citations, and other native tools remain.
- [~] Google Gemini: text/thinking/function-tool streaming, generation-aware thinking levels/budgets and thought inclusion, portable penalties/logprobs/service tiers, returned static/streamed logprob and tier metadata, thought-signature round trips, native output, multimodal input, function-call/response IDs, normalized finish reasons, full JSON Schema wire fields, and Gemini 2.5+ strict defaults; native tools and advanced metadata remain.
- [ ] OpenAI-compatible provider configuration without provider-specific forks.
- [ ] Azure OpenAI.
- [ ] AWS Bedrock.
- [ ] Groq, Mistral, Cohere, Cerebras, xAI, OpenRouter, Ollama, Hugging Face, and other upstream providers.
- [ ] Provider profiles/capability detection instead of model-name conditionals.
- [ ] Provider HTTP retries and configurable retry policy.
- [ ] Model fallback chains and instrumentation wrappers.

### Model settings

- [x] Max tokens, temperature, top-p, seed, stop sequences, and a cooperative per-request timeout in the common settings type.
- [~] Providers only forward settings they support; compatibility is not validated by profiles. OpenAI, Anthropic, and Gemini normalize their available cache, audio, reasoning, prediction, log-probability, service-tier, and finish details.
- [x] Static or per-step `RequestTimeout` bounds synchronous requests and full stream consumption while preserving earlier parent cancellation.
- [x] `ParallelToolCalls` generation setting for OpenAI Chat/Responses and Anthropic; Gemini exposes no equivalent request setting.
- [x] Detached `ThinkingSettings` with enabled/disabled and minimal through xhigh effort levels, explicit token budgets, and thought-inclusion control. OpenAI maps levels to reasoning effort, Anthropic maps levels or explicit budgets to extended thinking, and Gemini maps by generation to thinking levels or budgets.
- [~] Portable presence/frequency penalties, logit bias, logprobs/top-logprobs, auto/default/flex/priority service tiers, and detached per-request extra headers are fieldwise merged. OpenAI Chat maps all supported fields; Responses maps logprobs and tiers; Gemini maps penalties, logprobs, and tier values; Anthropic maps compatible tiers. Static and streamed OpenAI Chat/Gemini responses preserve logprobs and service metadata, and static Responses preserves per-text logprobs. OpenAI and Anthropic support detached extra request-body fields with typed-field conflict protection. Provider-specific typed overrides and remaining streaming metadata remain.
- [x] Fieldwise settings resolve per selected model in model-default, agent, capability, and run order, with prior layers visible to each callback; bundled providers expose detached defaults through `WithDefaultSettings`.

### Outputs

- [x] Text output.
- [x] Tool output.
- [~] Native output on supported providers.
- [x] Native structured output alongside function calls obeys end strategies; plain text remains non-preemptive.
- [x] Prompted JSON output fallback for reflected structured outputs, including default/custom schema instructions, validation retries, streaming, per-run overrides, and end-strategy handling.
- [ ] Multiple output alternatives / union outputs.
- [ ] Provider-profile default output modes and provider-specific prompted-output templates.
- [ ] Image and binary outputs.
- [x] Output tool name, description, strict mode, sequential execution barrier, and independent retry configuration through `OutputToolConfig`, with per-run replacement.
- [x] Output-tool definitions can be modified, renamed, or omitted from fresh copies before each request through `AddOutputToolPrepareFunc`.
- [x] Per-run output specialization through type-safe `RunAs`, `RunPartsAs`, `RunStreamAs`, and `RunStreamPartsAs`, with validator incompatibility rejected explicitly.

## P1 - Capabilities and ecosystem

### Capability framework

- [x] Setup contributions for static instructions, model settings, and raw tools.
- [x] Run, model request, tool call, and dynamic instruction hooks.
- [x] Ordered middleware composition; first capability is outermost.
- [x] `HistoryProcessor` provides composable request-only history middleware with detached input/output snapshots and `RunInfo`; durable history replacement remains an explicit model-request hook action.
- [x] Runs, prepared model requests, function-tool validation/execution, and output validation/processing have ordered before hooks plus reverse-ordered after/error hooks in addition to middleware wrappers. Model hooks can detach snapshots, modify requests, switch models, recover errors, and request budgeted retries that preserve rejected responses. Tool and output hooks transform raw or typed values, recover ordinary errors, and preserve retry/failure control flow. Run wrappers can short-circuit, transform, or recover type-checked outcomes while cancellation remains terminal.
- [x] Output validation/processing hooks cover raw structured repair, schema/decoding/semantic validation, final typed processing, wrapper and recovery composition, normal and early outputs, and streaming partial/final values.
- [x] Audited upstream user-prompt/model-request/call-tools/end node lifecycles against the plain loop. Focused request, tool validation/execution, output, run-outcome, retry, deferred, and enqueue hooks cover semantic interception; public internal-node replacement remains intentionally excluded with the graph API.
- [x] Function-tool schema validation, typed decoding, and semantic argument validation are separate from local execution, with dedicated wrappers and before/after/error hooks. Static approvals and external calls defer only after validation, wrapper-modified raw arguments are revalidated, and validated values retain their registered concrete Go type. After-validation and before/after-execution hooks can request durable approval or external execution without changing tool registration; validation-error hooks cannot defer invalid arguments.
- [x] Event-stream wrapper and per-event processor with standard capability middleware ordering.
- [x] Stable capability ordering through outermost/default/innermost tiers, type/interface or pointer-instance `Wraps`/`WrappedBy` edges, dependency requirements, cycle detection, nested-group flattening, sorted setup contributions, and independently scoped run capabilities whose requirements can use agent capabilities.
- [~] `CombineCapabilities` packages and recursively flattens ordered groups for agent-wide or per-run registration without changing setup or middleware order. Transparent wrapper-capability helpers remain.
- [x] Capability-provided static/per-step model settings and adaptive model selection.
- [x] Per-run capabilities are set up once per run, contribute static instructions/settings/raw tools, participate in every middleware hook, enable event processing for `Run`, and leave agent configuration unchanged.
- [x] Deferred-call handler hook with detached requests/results, ordered composition, partial handling, and streamed lifecycle events.

### Built-in capabilities

- [ ] MCP client capability and MCP toolset.
- [ ] Web search, web fetch, X search, and provider-native tools.
- [x] Portable thinking configuration through common model settings, including per-run and dynamic setting layers.
- [~] Provider-neutral compaction boundaries, composable message history processors, stateful OpenAI Responses/Anthropic compaction, OpenAI stateless message/custom triggers, direct compaction requests, durable history replacement, and usage-limit accounting are complete. Token-based trimming, model-wrapper unwrapping, dedicated compaction tracing, and provider-neutral summarization helpers remain.
- [~] Local tool search is available as a composable toolset; deferred capability loading remains.
- [x] Prefix, rename, filter, prepare, combine, and set-tool-metadata helpers through composable toolsets.
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

- [ ] Iterative/manual agent run driver, including external `AgentRun.Enqueue`; run-context enqueue is complete.
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
- [x] CI runs the race detector plus repeated concurrent/parallel/enqueue stress tests.
- [x] CI covers Go 1.25 and 1.26, vet, lint, tests, 100% per-package coverage, replay-only cassettes, and a clean post-test worktree.
- [ ] API examples for tools, streaming, multimodal input, capabilities, and each provider.
- [ ] Go package documentation for every public contract.
- [ ] Compatibility policy, semantic versioning policy, and changelog.
- [ ] Benchmark loop overhead, streaming, schema reflection, and parallel tools.
- [ ] Pin `.upstream-sync.json` to the audited upstream commit.
- [ ] After parity, add the daily `gh-aw` upstream-sync workflow described in `PLAN.md`.

## Next work

1. Expose external enqueue through an iterative/manual run driver; run-context queues now survive deferred pauses.
2. Add transparent wrapper-capability composition without exposing graph internals; grouped composition and ordering constraints are complete.
3. Add provider-managed search for Anthropic and OpenAI Responses; OpenAI Responses client search already uses native wire items in streaming and non-streaming requests.
4. Add token-aware trimming and provider-neutral summarization helpers; request-only message processors and provider-native compaction are complete.
5. Add MCP now that raw/dynamic toolsets have run/step lifecycle and deferred calls have an explicit pause/resume boundary.
6. Extend upstream message fixtures as remaining persisted part types land.
