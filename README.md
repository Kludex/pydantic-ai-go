# pydantic-ai-go

An idiomatic Go library for the LLM agent loop.

You create an `Agent` with a `Model`, register tools on it, and call `Run`. The agent loops: it sends the conversation to the model, executes any tool calls in the response, and repeats until the model produces a final output, which is unmarshalled into your `Output` type.

## Install

```sh
go get github.com/Kludex/pydantic-ai-go
```

## Example

```go
package main

import (
	"context"
	"fmt"

	ai "github.com/Kludex/pydantic-ai-go"
	"github.com/Kludex/pydantic-ai-go/models/openai"
)

type Deps struct {
	DefaultUnit string
}

type WeatherArgs struct {
	City string `json:"city" jsonschema:"description=City name"`
}

func main() {
	model := openai.NewModel("gpt-4o-mini") // uses OPENAI_API_KEY
	agent := ai.NewAgent[Deps, string](model,
		ai.WithInstructions("You are a weather assistant."),
	)
	ai.AddTool(agent, "get_weather",
		func(ctx context.Context, rc *ai.RunContext[Deps], args WeatherArgs) (string, error) {
			return fmt.Sprintf("sunny, 21 %s in %s", rc.Deps.DefaultUnit, args.City), nil
		},
		ai.WithDescription("Get current weather for a city"),
	)

	result, err := agent.Run(context.Background(), "What's the weather in Berlin?", Deps{DefaultUnit: "C"})
	if err != nil {
		panic(err)
	}
	fmt.Println(result.Output)
}
```

The tool's argument schema is reflected from `WeatherArgs` - the model sees the `json` names and `jsonschema` constraints. The agent enforces the complete Draft 2020-12 schema before Go decoding and tool execution. Invalid arguments go back to the model as a retry prompt instead of failing the run.

## Per-run configuration

Run options change one invocation without mutating the agent. You can safely use different options in concurrent runs. Every run creates a run ID and a conversation ID. Use `WithRunID` for an application ID and `WithConversationID` to join related runs. A conversation ID is inherited from message history; pass `WithConversationID("new")` to start another conversation:

```go
result, err := agent.Run(
	ctx,
	"What's the weather in Oslo?",
	deps,
	ai.WithRunID("weather-run-42"),
	ai.WithConversationID("customer-session-7"),
	ai.WithRunModel(fasterModel),
	ai.WithRunInstructions("Prefer concise answers."),
	ai.WithRunModelSettings(ai.ModelSettings{MaxTokens: 200}),
	ai.WithRunModelSettingsFunc(func(
		ctx context.Context, rc *ai.RunContext[Deps],
	) (ai.ModelSettings, error) {
		return ai.ModelSettings{MaxTokens: 200 + rc.Usage().Requests*50}, nil
	}),
	ai.WithRunInstructionsFunc(func(
		ctx context.Context, rc *ai.RunContext[Deps],
	) (string, error) {
		return fmt.Sprintf("This is model step %d.", rc.Usage().Requests+1), nil
	}),
	ai.WithRunUsageLimits(ai.UsageLimits{TotalTokenLimit: 1_000}),
)
```

Run settings merge field by field. Bundled models accept provider-level defaults through `openai.WithDefaultSettings`, `anthropic.WithDefaultSettings`, and `google.WithDefaultSettings`. After model selection, settings resolve in model, agent, capability, then run order. Dynamic callbacks run before every model request, so they can adapt after tool calls and retries. Each callback sees prior layers through `RunContext.ModelSettings`. The effective joined instructions are persisted on every sent `ModelRequest`, including middleware changes, so serialized histories retain request context. A zero `UsageLimits` value disables agent-level limits for that run.

Prefer instructions for new applications. If you resume histories that use legacy system prompts, register a stable application ID so the prompt can be reevaluated with the new dependencies:

```go
agent.AddDynamicSystemPromptFunc("tenant-policy", func(
	_ context.Context,
	rc *ai.RunContext[Deps],
) (string, error) {
	return "Use " + rc.Deps.DefaultUnit + " for temperatures.", nil
})
```

The ID is serialized as `SystemPromptPart.DynamicRef`. Keep it stable across deployments. A resumed run updates every matching part after selecting the model, without duplicating system prompts in the new request.

## Model selection

Use `AddModelSelector` when later steps need a different model:

```go
agent.AddModelSelector(func(
	ctx context.Context,
	selection ai.ModelSelectionContext[Deps],
) (ai.ModelSelection, error) {
	if selection.Step == 1 {
		return ai.ModelSelection{Model: fastModel}, nil
	}
	return ai.ModelSelection{Model: reasoningModel}, nil
})
```

`Step` starts at 1. `Messages` is a detached snapshot of completed turns and excludes the pending request. `Usage` contains work completed before the step. Model selection runs before dynamic settings, instructions, and tool preparation. You can pass `nil` to `NewAgent` when a selector or model ID always supplies the model. A request without any selected model returns `ErrNoModel`.

Application IDs keep tenant or deployment lookup outside the selector:

```go
agent.AddModelIDResolver(func(
	ctx context.Context,
	resolution ai.ModelResolutionContext[Deps],
	modelID string,
) (ai.Model, error) {
	return resolution.Deps.Models.Lookup(ctx, modelID)
})

result, err := agent.Run(ctx, "hello", deps, ai.WithRunModelID("tenant-primary"))
```

Resolvers run in registration order, and each ID is resolved once per run. An unresolved ID returns `UnknownModelIDError` and matches `ErrUnknownModelID`. `WithRunModel` skips selectors. `WithRunModelSelector` replaces agent and capability selectors for one run.

A model can implement `ModelOpener` when it owns run-scoped resources. `OpenModel` runs once for each distinct selected model. Its `ModelCloseFunc` runs with a non-canceled context in reverse selection order, including failed, canceled, deferred, and partially consumed streamed runs. Models close before toolsets because model selection happens after toolset acquisition.

## Suspended responses

Use OpenAI background mode for model requests that may take a long time:

```go
package main

import (
	"context"
	"fmt"
	"log"

	ai "github.com/Kludex/pydantic-ai-go"
	"github.com/Kludex/pydantic-ai-go/models/openai"
)

func main() {
	model := openai.NewResponsesModel(
		"gpt-5",
		openai.WithBackgroundMode(true),
	)
	agent := ai.NewAgent[struct{}, string](model)
	result, err := agent.Run(context.Background(), "Solve the problem.", struct{}{})
	if err != nil {
		log.Fatal(err)
	}
	fmt.Println(result.Output)
}
```

The agent polls an OpenAI background response until it completes. It also resumes Anthropic `pause_turn` responses immediately. Continuation segments form one response in history and contribute usage once. Fresh generations stop after `MaxGenerationContinuations`; polls of one response ID stop after `MaxBackgroundPolls`. A model can implement `ModelContinuationDelayer` and `SuspendedResponseCanceler` to control polling and best-effort cleanup.

Use `agent.Resume(ctx, history, deps)` or `agent.ResumeStream(ctx, history, deps)` when persisted history ends with a suspended response. These methods do not add a user prompt. They replace the suspended history entry with the final merged response. Invalid history returns `ErrNoSuspendedResponse`.

`RunStream` emits part events from every segment. Accumulated segments receive continuous part indexes. Repeated snapshots for one response ID reuse their index space. Each segment emits its own `FinishEvent`, while `RunResult.Usage()` contains the merged usage.

Breaking out of `StreamedRun.Events()` detaches from a provider-managed suspended response without canceling its server-side job. `StreamedRun.Suspended()` then returns a `SuspendedRun` with detached `Response()`, `Messages()`, and `Usage()` snapshots. Persist `Messages()` and pass them to `Resume` later. Canceling the run context instead invokes `SuspendedResponseCanceler` so the provider can stop the job.

## Concurrent tools

Independent tool calls from one model response run concurrently. Results still go back to the model in the order it requested them.

Use `ai.WithSequential()` when a tool changes shared state or must run alone:

```go
ai.AddTool(agent, "update_database", updateDatabase, ai.WithSequential())
```

The sequential tool is a barrier. Earlier calls finish before it starts. Later calls wait until it finishes. Use `ai.WithSequentialToolExecution()` on the agent when every tool must run serially.

Local execution is separate from model generation. Set `ModelSettings.RequestTimeout` to limit each model request, including consumption of a streaming response. It does not limit tools or the whole run.

Set `ModelSettings.ParallelToolCalls` to tell OpenAI or Anthropic whether the model may emit parallel calls:

```go
parallel := false
agent := ai.NewAgent[Deps, string](
	model,
	ai.WithModelSettings(ai.ModelSettings{ParallelToolCalls: &parallel}),
)
```

Configure portable reasoning with `ThinkingSettings`:

```go
budget := 8_192
includeThoughts := true
agent := ai.NewAgent[Deps, string](
	model,
	ai.WithModelSettings(ai.ModelSettings{Thinking: &ai.ThinkingSettings{
		Level:           ai.ThinkingLevelHigh,
		TokenBudget:     &budget,
		IncludeThoughts: &includeThoughts,
	}}),
)
```

`Level` maps to OpenAI reasoning effort, an Anthropic token budget, and Gemini thinking levels or budgets. `ThinkingLevelEnabled` uses the provider default. `ThinkingLevelDisabled` requests no reasoning where the provider supports it. An explicit `TokenBudget` overrides the portable effort mapping on Anthropic and Gemini. Gemini also forwards `IncludeThoughts`.

Configure portable sampling controls and service tiers on the same settings value:

```go
logprobs := true
topLogprobs := 5
presencePenalty := 0.2
agent := ai.NewAgent[Deps, string](
	model,
	ai.WithModelSettings(ai.ModelSettings{
		PresencePenalty: &presencePenalty,
		Logprobs:         &logprobs,
		TopLogprobs:      &topLogprobs,
		ServiceTier:      ai.ServiceTierPriority,
	}),
)
```

OpenAI Chat and Gemini accept presence and frequency penalties. OpenAI Chat also accepts `LogitBias`. OpenAI Chat, OpenAI Responses, and Gemini return available log probabilities in provider details. Unified service tiers map to each provider's wire values, including Anthropic's `standard_only` and Gemini's `standard` values.

Use `ExtraHeaders` for provider preview headers or gateway routing. OpenAI and Anthropic also accept `ExtraBody` for new provider fields that do not have a typed setting yet. Both maps are detached per run. Extra body fields cannot replace typed request fields. This prevents an extension value from silently changing the model, tools, output schema, or another validated setting.

A tool can stop its run through `RunContext.Cancel`. Cancellation reaches sibling tools through `context.Context`, waits for their cleanup, and returns an error matching `ai.ErrRunCancelled`:

```go
ai.AddTool(agent, "stop", func(
	ctx context.Context,
	rc *ai.RunContext[Deps],
	args StopArgs,
) (string, error) {
	rc.Cancel()
	return "", nil
})
```

Use `errors.As` with `*ai.RunCancelledError` to inspect the history and usage retained before cancellation. Pass `cancelled.Messages()` to `ai.WithMessageHistory` to resume. Dangling calls receive synthesized `interrupted` returns beside the matching turn, before user-facing content. Orphaned results whose calls are missing are removed. Consecutive same-role history is merged into provider-valid turns. Inspect `ToolReturnPart.Metadata[ai.SynthesizedToolReturnMetadataKey]` to distinguish a repaired return from an executed tool result.

Use `ai.WithStrict()` to ask the provider to constrain generated arguments to the tool schema:

```go
ai.AddTool(agent, "book_table", bookTable, ai.WithStrict())
```

Strict mode prevents malformed arguments before they reach your code. OpenAI enables it automatically when a schema is compatible, and rewrites incompatible constraints when you use `ai.WithStrict()`. Use `openai.WithStrictToolSupport(false)` for compatible endpoints that reject strict definitions. Anthropic uses explicit strict mode on supported Claude models; use `anthropic.WithStrictToolSupport` for aliases or newly released models. Set `anthropic.WithSchemaWarningHandler` to inspect lossy conversions, such as dynamic map schemas that Anthropic closes with `additionalProperties: false`. Gemini 2.5 and newer use request-wide `VALIDATED` mode by default. Use `ai.WithoutStrict()` to keep a Gemini request on `AUTO`, or `google.WithStrictToolSupport` for model aliases and compatible proxies.

## Reusable and per-run tools

Create a `Tool` when the same implementation belongs to several agents or only one run:

```go
weatherTool := ai.NewTool("get_weather", func(
	_ context.Context,
	rc *ai.RunContext[Deps],
	args WeatherArgs,
) (string, error) {
	return fmt.Sprintf("sunny, 21 %s in %s", rc.Deps.DefaultUnit, args.City), nil
})

result, err := agent.Run(
	ctx,
	"What's the weather in Oslo?",
	deps,
	ai.WithRunTools(weatherTool),
)
```

`WithRunTools` does not mutate the agent or leak tools into concurrent runs. A tool name cannot duplicate an agent tool or another per-run tool. Use `agent.AddTool(weatherTool)` to register the same value permanently. `Tool.Definition()` returns a detached copy for inspection, including the reflected `ReturnSchema`. Use `WithReturnSchema` when a rich `ToolReturn` needs an explicit schema for its inner value.

Return `ToolReturn` when a tool needs to keep application metadata or send additional user content outside the provider's tool-result message:

```go
ai.AddSimpleTool(agent, "inspect_receipt", func(
	_ context.Context,
	args ReceiptArgs,
) (ai.ToolReturn, error) {
	return ai.ToolReturn{
		ReturnValue: Receipt{Total: args.Total},
		Content: []ai.UserContent{
			ai.TextContent{Text: "The original receipt follows."},
			ai.BinaryContent{Data: args.Image, MediaType: "image/png"},
		},
		Metadata: map[string]any{"audit_id": args.AuditID},
		Tools:    []string{"archive_receipt"},
	}, nil
})
```

`ReturnValue` becomes the tool result. `Content` becomes a trailing user prompt after every result in the same concurrent batch, which preserves provider-valid tool-call ordering. `Metadata` remains in local history and is not sent to the model. `Tools` reveals matching deferred tools by their model-facing names. Unknown, already-visible, and already-revealed names are ignored.

Compose larger collections with toolsets:

```go
weatherTools := ai.NewFunctionToolset(weatherTool)
publicWeatherTools := ai.FilterToolset(weatherTools, func(
	_ context.Context,
	_ *ai.RunContext[Deps],
	tool ai.ToolDefinition,
) (bool, error) {
	return tool.Metadata["internal"] != true, nil
})
agent.AddToolset(ai.PrefixToolset(publicWeatherTools, "weather"))
```

The model sees `weather_get_weather`, while the function receives `get_weather` through `RunContext.ToolName`. You can also use `CombineToolsets`, `RenameToolset`, `PrepareToolset`, and `SetToolsetMetadata`. `RequireApprovalToolset` requires approval for every wrapped tool or a selected set of original names. `RequireApprovalToolsetWhen` evaluates a policy for each validated call. `WithToolsetMaxRetries` and `WithToolsetTimeout` provide defaults without replacing explicit tool options. Toolsets list tools and contribute optional instructions before each model step. Pass them through `WithRunToolsets` to scope them to one run.

Use `WithDeferredLoading()` on one tool or wrap a collection with `DeferLoadingToolset`. Deferred definitions remain hidden until `ToolReturn.Tools` reveals them. The first premature call to each hidden tool receives a free availability correction, so it does not consume the budget needed by a later valid call. Reveals are deduplicated in model-call order and persisted as `ToolAvailabilityDeltaPart`, so resumed histories retain the same visibility.

Add local discovery when the model should search a large deferred catalog:

```go
catalog := ai.WithToolSearch(
	ai.DeferLoadingToolset(ai.NewFunctionToolset(githubTools...)),
	ai.ToolSearchConfig[Deps]{MaxResults: 5},
)
agent.AddToolset(catalog)
```

The wrapper exposes `search_tools`. Default search uses case-insensitive word overlap across tool names and descriptions, prioritizes undiscovered matches, and returns typed `ToolSearchResult` values. Set `ToolSearchConfig.Search` to use an external index. Custom search receives detached definitions and its unknown or duplicate names are ignored.

Anthropic 4.5+ models advertise the hidden schemas with `defer_loading`, replay local search results as `tool_reference` blocks, and render other reveals as `tool_addition` blocks. Streaming and non-streaming OpenAI Responses requests use client-executed `tool_search`, `tool_search_output`, and `additional_tools` items, including final streamed call IDs. This keeps the stable visible-tool prefix small without changing local execution safety. Use each provider's `WithDeferredToolSupport(false)` option for compatible endpoints that do not implement its native wire protocol.

### Pause for approval or external execution

```go
func runWithApproval(ctx context.Context, model ai.Model) (string, error) {
	type DeleteArgs struct {
		Path string `json:"path"`
	}

	agent := ai.NewAgent[struct{}, string](model)
	ai.AddSimpleTool(agent, "delete_file", func(
		_ context.Context,
		args DeleteArgs,
	) (string, error) {
		return "deleted " + args.Path, nil
	}, ai.WithApprovalRequired())
	ai.AddExternalTool[struct{}, string, DeleteArgs, string](agent, "archive_file")

	paused, err := agent.Run(ctx, "Archive and delete old.log", struct{}{})
	if err != nil {
		return "", err
	}
	requests := paused.Deferred()
	if requests == nil {
		return paused.Output, nil
	}

	results := ai.DeferredToolResults{
		Approvals: map[string]ai.ToolApproval{},
		Calls:     map[string]any{},
	}
	for _, call := range requests.Approvals {
		results.Approvals[call.ToolCallID] = ai.ApproveTool()
	}
	for _, call := range requests.Calls {
		results.Calls[call.ToolCallID] = "archived by worker"
	}

	completed, err := agent.Run(
		ctx,
		"Continue after the approved work.",
		struct{}{},
		ai.WithMessageHistory(paused.Messages()),
		ai.WithDeferredToolResults(results),
	)
	if err != nil {
		return "", err
	}
	return completed.Output, nil
}
```

`WithApprovalRequired` validates arguments and pauses before local execution. Add `WithApprovalMetadata` when the caller needs context such as a policy reason. For a decision based on arguments or dependencies, register the function with `WithDynamicApproval` and return `RequestToolApproval` before performing the protected operation. The resumed call receives `ToolCallApproved == true`. It can complete, request approval again, or request external execution when the corresponding dynamic option is enabled. The pending kind is persisted under `DeferredToolKindsMetadataKey`, so serialized history cannot bypass an approval by submitting an external result. `NewExternalTool` and `NewRawExternalTool` return every validated call for another process to execute. Use `WithDynamicExternalExecution` and return `RequestExternalToolExecution` when only selected calls need an external worker. A paused `RunResult` has a non-nil `Deferred()` value and a zero `Output`. Resume with its message history and results for every pending call. Use `ApproveToolWithArgs` to replace arguments, or `DenyTool` to return a denial without execution. Per-call result metadata is available through `RunContext.ToolCallMetadata`, and approved tools receive `ToolCallApproved == true`.

Use `DeferredToolHandlerFunc` as a capability when the resolver runs in the same process. Handlers run in capability order. Each handler receives a detached copy of the unresolved requests and may return results for any subset. Resolved calls continue inline. A resolved call can defer again with a new kind and metadata, which the next handler receives. Calls omitted by every handler remain in `RunResult.Deferred()`. Streams emit one `DeferredToolRequestsEvent` before each handler chain and `DeferredToolResultsEvent` for each returned result batch.

Approval protects against the model acting without confirmation. It does not replace authentication or authorization for clients that can submit message history and approval results.

Stateful toolsets can implement three small optional interfaces:

```go
func (t *RemoteTools) ToolsetID() string {
	return "inventory"
}

func (t *RemoteTools) ForRun(
	ctx context.Context,
	rc *ai.RunContext[Deps],
) (ai.Toolset[Deps], error) {
	return &RemoteTools{endpoint: t.endpoint}, nil
}

func (t *RemoteTools) OpenToolset(
	ctx context.Context,
	rc *ai.RunContext[Deps],
) (ai.Toolset[Deps], ai.ToolsetCloseFunc, error) {
	client, err := connectInventory(ctx, t.endpoint)
	if err != nil {
		return nil, nil, err
	}
	opened := &RemoteTools{endpoint: t.endpoint, client: client}
	return opened, func(ctx context.Context) error {
		return client.Close(ctx)
	}, nil
}

func (t *RemoteTools) ForRunStep(
	ctx context.Context,
	rc *ai.RunContext[Deps],
) (ai.Toolset[Deps], error) {
	return t.withRegion(rc.Deps.Region), nil
}
```

`ForRun` creates isolated state once per run. `OpenToolset` acquires resources once and cleanup runs exactly once in reverse registration order. `ForRunStep` can replace the active view before each model request. Built-in combined and wrapper toolsets propagate all three hooks. `ToolsetID` is copied to `ToolDefinition.ToolsetID` as local lifecycle metadata.

## Dynamic tools

Use `ai.AddPreparedTool` when one tool's availability or schema depends on the run:

```go
ai.AddPreparedTool(
	agent,
	"get_weather",
	func(_ context.Context, rc *ai.RunContext[Deps], args WeatherArgs) (string, error) {
		return fmt.Sprintf("sunny, 21 %s in %s", rc.Deps.DefaultUnit, args.City), nil
	},
	func(_ context.Context, rc *ai.RunContext[Deps], tool ai.ToolDefinition) (*ai.ToolDefinition, error) {
		if rc.Deps.DefaultUnit == "" {
			return nil, nil
		}
		tool.Description = "Get current weather in " + rc.Deps.DefaultUnit
		return &tool, nil
	},
)
```

The callback receives a fresh definition before every model request. Return `nil` to omit that tool for the step.

Use `AddToolsPrepareFunc` to filter or modify all function tools together:

```go
agent.AddToolsPrepareFunc(func(
	_ context.Context,
	rc *ai.RunContext[Deps],
	tools []ai.ToolDefinition,
) ([]ai.ToolDefinition, error) {
	if rc.Deps.DefaultUnit == "" {
		return nil, nil
	}
	return tools, nil
})
```

The hook receives fresh copies, so you can safely change descriptions, nested schemas, and metadata. Return an empty or nil slice to expose no function tools for that step. Output tools are prepared separately by the agent and are not included.

Use `ai.WithToolMetadata(map[string]any{"owner": "billing"})` to attach local data for preparation and filtering. Metadata is copied with the definition and is never sent to the model provider.

## Argument validation

Use `AddToolWithArgsValidator` when valid JSON is not enough. The validator receives typed arguments and the same `RunContext` as the tool:

```go
ai.AddToolWithArgsValidator(
	agent,
	"get_weather",
	func(_ context.Context, rc *ai.RunContext[Deps], args WeatherArgs) (string, error) {
		return fmt.Sprintf("sunny, 21 %s in %s", rc.Deps.DefaultUnit, args.City), nil
	},
	func(_ context.Context, _ *ai.RunContext[Deps], args WeatherArgs) error {
		if args.City == "" {
			return ai.Retryf("city must not be empty")
		}
		return nil
	},
)
```

The validator runs after JSON decoding and before the tool. `Retryf` asks the model for corrected arguments and consumes that tool's retry budget. `ToolFailedf` records a terminal failed result without running the tool. Other errors abort the run. Use `AddPreparedToolWithArgsValidator`, `AddSimpleToolWithArgsValidator`, or `AddRawToolWithArgsValidator` for the corresponding registration style.

## Retry budgets

Function tools track retries independently. Output validation has a separate budget. Both default to one retry:

```go
agent := ai.NewAgent[Deps, Weather](
	model,
	ai.WithRetryLimits(ai.RetryLimits{
		Tools:  2,
		Output: 1,
	}),
)
```

Use `ai.WithToolMaxRetries(4)` when registering one tool to override the function-tool budget. Use `ai.WithRunRetryLimits(...)` to override both agent defaults for one run. Explicit per-tool limits still win.

Inside tools and output validators, `rc.Retry` is the current counter for that tool or output path. `rc.MaxRetries` is the limit that applies to it.

Return `ai.Retryf(...)` when the model should correct the call. Return `ai.ToolFailedf(...)` when the call completed unsuccessfully and the model should adapt instead of retrying. A terminal failure does not consume the tool's retry budget.

Unknown tool names also go back to the model as retry prompts. The prompt lists only tools exposed for that request, so a tool omitted by preparation cannot be executed from a stale call. JSON Schema failures are retained as structured `ValidationError` values on `RetryPromptPart.Errors`, including the failing location and input. Providers receive the same formatted feedback from `RetryPromptPart.ModelResponse()`.

Use `ai.WithToolTimeout(5 * time.Second)` to give one tool call a deadline. The tool must honor `ctx.Done()`. A tool-specific timeout becomes a retry, while cancellation of the parent run remains `context.Canceled`.

## Usage

`result.Usage()` includes requests, successful function-tool calls, token totals, cache reads and writes, audio tokens, reasoning tokens, and prediction tokens when the provider reports them. Provider-specific integer counters are preserved and accumulated in `Usage.Details`. Values returned by `Usage()` and `Usage.Clone()` are detached, so callers can safely modify detail maps. `Usage.CacheHitRatio()` returns the fraction of input tokens read from cache.

Providers do not calculate prices. A model or capability can set `Usage.CostUSD`. `UsageLimits.CostLimitUSD` enforces known costs and does not reject a run when cost is unavailable. `UsageLimits.ToolCallLimit` rejects a batch before any function tool runs when its projected successful-call count exceeds the limit.

## Structured output

Use any struct as the `Output` type and the agent asks the model for it via a final output tool. The result arrives typed and validated:

```go
type Weather struct {
	City  string  `json:"city"`
	TempC float64 `json:"temp_c"`
}

agent := ai.NewAgent[Deps, Weather](model)
result, err := agent.Run(ctx, "Weather in SF?", deps)
// result.Output is a Weather
```

`Output = string` means plain text - no output tool is involved.

Specialize the output type for one run without changing the agent:

```go
agent := ai.NewAgent[Deps, string](model)
result, err := ai.RunAs[Weather](ctx, agent, "Weather in SF?", deps)
// result.Output is a Weather
```

Use `RunPartsAs`, `RunStreamAs`, or `RunStreamPartsAs` for multimodal and streaming runs. Go methods cannot introduce a new type parameter, so these are package functions instead of `Agent` methods. A specialized run returns `ErrOutputTypeOverrideWithValidators` when the agent has output validators, because those validators accept the agent's declared output type.

Customize the tool contract without changing the output type:

```go
strict := true
outputRetries := 2
agent := ai.NewAgent[Deps, Weather](model, ai.WithOutputTool(ai.OutputToolConfig{
	Name:        "weather_result",
	Description: "Return the validated weather.",
	Strict:      &strict,
	Sequential:  true,
	MaxRetries:  &outputRetries,
}))
```

`MaxRetries` overrides the general output retry budget while leaving function-tool budgets unchanged. Use `WithRunOutputTool` to replace this configuration for one run. Use `AddOutputToolPrepareFunc` to rename, modify, or omit a fresh output-tool definition before each model request. Preparation runs after model selection and dynamic settings.

Use prompted output when a provider does not implement native JSON Schema output and you do not want an output tool:

```go
agent := ai.NewAgent[Deps, Weather](
	model,
	ai.WithOutputMode(ai.OutputModePrompted),
)
result, err := agent.Run(ctx, "Weather in SF?", deps)
```

The agent appends the schema to its instructions, validates the returned JSON, and sends validation failures back for correction. Use `WithPromptedOutputTemplate` or `WithRunPromptedOutputTemplate` to replace the instructions. If a custom template omits `{schema}`, the schema is appended automatically.

### Tool calls alongside output

The default `ai.EndStrategyGraceful` runs function tools emitted alongside an output tool. The first successful output wins. A function-tool retry suppresses that output so the model can correct the call.

Use `ai.EndStrategyEarly` when function tools should be skipped after an output succeeds:

```go
agent := ai.NewAgent[Deps, Weather](
	model,
	ai.WithEndStrategy(ai.EndStrategyEarly),
)
```

Use `ai.EndStrategyExhaustive` when every output and function tool must run. Independent calls run concurrently, and the first successful output in emission order wins.

With native or prompted structured output, `ai.EndStrategyEarly` also lets valid JSON preempt function tools. Invalid JSON falls through to the tools without consuming a retry. Plain text never preempts a tool call because it may only describe the work the model is about to perform.

## Streaming

`RunStream` yields normalized part lifecycle events, and the typed result is available once the stream completes:

```go
stream := agent.RunStream(ctx, "tell me a story", deps)
for event, err := range stream.Events() {
	if err != nil {
		panic(err)
	}
	switch event := event.(type) {
	case ai.PartStartEvent:
		if text, ok := event.Part.(ai.TextPart); ok {
			fmt.Print(text.Content)
		}
	case ai.PartDeltaEvent:
		if text, ok := event.Delta.(ai.TextPartDelta); ok {
			fmt.Print(text.ContentDelta)
		}
	}
}
result := stream.Result()
```

OpenAI Chat Completions, OpenAI Responses, Anthropic Messages, and Google Gemini stream text, thinking, tool arguments, and usage from their SSE APIs. Models that do not implement `ai.StreamingModel` still work: each response is replayed as events.

`PartStartEvent` contains the first content for a part. Later content, thinking signatures, and provider names arrive through typed `PartDeltaEvent` values. `PartEndEvent` contains the complete part and merged provider details. Each event carries a stable `PartID` and response index, so you can route interleaved deltas without relying on arrival order. `FinalResultEvent` follows the first part matching the configured output.

Bundled providers populate part IDs. A custom `StreamingModel` emits provider-facing `ModelStreamEvent` values and can leave `PartID` empty only when its parts are strictly sequential.

Function and output tools emit call events before execution and result events when each call settles. Concurrent result events use completion order, while the request parts stored in history keep model order.

Use `Outputs` instead of `Events` when you want typed snapshots. Output validators receive `RunContext.PartialOutput == true` for partial values and `false` for the final value:

```go
stream := agent.RunStream(ctx, "weather in Berlin", deps)
for output, err := range stream.Outputs() {
	if err != nil {
		panic(err)
	}
	fmt.Printf("%+v\n", output)
}
```

Structured snapshots satisfy the output's complete Draft 2020-12 schema. Incomplete prefixes and values that fail constraints, formats, or custom Go decoding are withheld until they become valid.

Use `stream.OutputsDebounced(100 * time.Millisecond)` to group bursty partial snapshots. The interval is a soft maximum: the next snapshot after the interval flushes the latest value from the preceding group. Stream completion flushes the final partial. Pass zero to disable grouping.

The last value is always the fully validated output, even when it equals the preceding partial value. `Events`, `Outputs`, and `OutputsDebounced` are alternative views; consume only one for each run.

`RunStream` commits the first matching text, native, or output-tool result. The configured end strategy still controls co-emitted tools, but a tool retry cannot revoke that result. If an output validator requests a retry, the streamed run returns `UnexpectedModelBehaviorError` because output has already been committed. Use `Run` when validation should start another model round.

## Inject messages during a run

```go
func newIncidentAgent(model ai.Model) *ai.Agent[struct{}, string] {
	agent := ai.NewAgent[struct{}, string](model)
	ai.AddTool(agent, "raise_alert", func(
		_ context.Context,
		rc *ai.RunContext[struct{}],
		_ struct{},
	) (string, error) {
		_, err := rc.Enqueue(
			ai.SystemPromptPart{Content: "Use concise incident language."},
			ai.TextContent{Text: "Production is degraded. Prioritize triage."},
		)
		return "alert raised", err
	})
	return agent
}
```

`RunContext.Enqueue` injects a group before the next model request. Use `EnqueueWhenIdle` for follow-up work that should wait until the run would otherwise finish. `EnqueueWithPriority` accepts an explicit `PendingMessagePriority`. Calls are safe from concurrently executing tools.

You can enqueue user content, request parts, complete requests, and complete responses. Adjacent user content becomes one `UserPromptPart`. Complete messages preserve their boundaries. Every group must end in a `ModelRequest` or content that forms one.

Each non-empty call returns an ID. Streams emit `EnqueuedMessagesEvent` with that ID and detached copies of the messages after run, conversation, and timestamp fields are filled. An `asap` message arriving during final validation redirects the run into another request instead of being dropped. A `when_idle` message redirects only after the current output candidate is complete.

## Multimodal input

`RunParts` sends images and files alongside text:

```go
result, err := agent.RunParts(ctx, []ai.UserContent{
	ai.TextContent{Text: "What is in this image?"},
	ai.ImageURL{URL: "https://example.com/cat.png"},
}, deps)
```

## Capabilities

A capability is a reusable, composable unit of agent behavior: it can contribute tools and instructions at setup, and intercept the run, every model request, and every tool call. One capability works with any agent, regardless of its `Deps` and `Output` types.

```go
type Redactor struct{}

func (Redactor) Setup(*ai.CapabilityRegistry) error { return nil }

func (Redactor) WrapToolCall(ctx context.Context, ri *ai.RunInfo, call ai.ToolCallPart, next ai.ToolCallFunc) (any, error) {
	if call.ToolName == "delete_everything" {
		return nil, ai.Retryf("that tool is not allowed")
	}
	return next(ctx, call)
}

agent := ai.NewAgent[Deps, string](model, ai.WithCapabilities(Redactor{}))
```

Implement any of `BeforeModelRequestHook`, `AfterModelRequestHook`, `ModelRequestErrorHook`, `RunWrapper`, `ModelRequestWrapper`, `ToolCallWrapper`, `RunEventStreamWrapper`, `StreamEventProcessor`, `InstructionsProvider`, `ModelSettingsProvider`, `ModelSelectionProvider`, or `ModelIDResolver` - the agent discovers them by type assertion, the same pattern as `http.Flusher`. Function adapters are available as `BeforeModelRequestFunc`, `AfterModelRequestFunc`, and `ModelRequestErrorFunc`.

Before hooks run in capability order. After and error hooks run in reverse order. This matches middleware nesting. An error hook may return a replacement response. Return `ai.Retryf(...)` from a before or after hook to consume the output retry budget and ask the model to respond again. A response rejected by an after hook remains in history. `ModelRequestContext.Clone` detaches mutable request data when a hook needs to retain or inspect a snapshot.

Tool hooks split argument validation from local execution. Use `BeforeToolValidationHook`, `AfterToolValidationHook`, `ToolValidationErrorHook`, and `ToolValidationWrapper` around raw JSON, schema validation, typed decoding, and semantic validators. Use the corresponding execution hooks and `ToolExecutionWrapper` only after arguments are valid and approval or external-execution deferral has been resolved. Validated hook arguments use the concrete Go `Args` type registered for the tool. Static approval and external calls are validated before they are returned to the caller. An after-validation or before-execution hook can return `RequestToolApproval(...)` or `RequestExternalToolExecution(...)` to defer one call without changing its registration.

Slice order is middleware order: the first capability is outermost. Usage limits are implemented on this same surface internally. Pass `ai.WithRunCapabilities(...)` to scope setup contributions and middleware to one run. Agent capabilities remain outermost. A run capability is set up once for that run and may contribute instructions, settings, and raw tools without modifying the shared agent.

Stream wrappers only change events seen by the consumer. They do not change accumulated history, tool execution, or final output. Adding one also enables provider streaming for `Run`, so processors run whether you call `Run` or `RunStream`.

## Why no graph?

The agent run is a plain loop: call model, execute tool calls, repeat. PydanticAI's graph layer exists for history and durability reasons that do not apply here. Fewer layers means the whole loop fits in one file you can read.

## Testing your agents

`models/fakes` ships two `ai.Model` implementations so agent tests never touch the network:

- `fakes.NewTestModel()` calls every registered tool once with schema-conformant arguments, then produces a final output.
- `fakes.NewFunctionModel(fn)` delegates every request to your function - script any conversation.

```go
agent := ai.NewAgent[Deps, string](fakes.NewTestModel())
```

## Interoperability

Message history serializes to PydanticAI's JSON format via `ai.MarshalMessages` / `ai.UnmarshalMessages`, so histories exchange cleanly with [PydanticAI](https://ai.pydantic.dev), [pydantic-evals-go](https://github.com/Kludex/pydantic-evals-go), and Logfire. Requests and responses retain timestamps, run and conversation IDs, metadata, normalized finish reasons, provider details, response IDs, and lifecycle state. Text, thinking, and tool-call parts also retain provider IDs, signatures, details, and typed tool kinds. OpenTelemetry spans follow the GenAI semantic conventions and are free unless you set a global tracer provider.

## Status

v0.3 - capabilities (hook interfaces, `WithCapabilities`) and the OpenAI Responses API model (`openai.NewResponsesModel`), on top of v0.2 streaming, three providers, multimodal input, and native JSON output mode, and the v0.1 loop, tools, structured output, usage limits, fakes, and tracing. See [PLAN.md](PLAN.md) for the roadmap: MCP and provider-native tools (v0.4).
