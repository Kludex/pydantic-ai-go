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

The tool's argument schema is reflected from `WeatherArgs` - the model sees the `json` names and the `jsonschema` descriptions. If the model sends arguments that fail to unmarshal, the error goes back to the model as a retry prompt instead of failing the run.

## Concurrent tools

Independent tool calls from one model response run concurrently. Results still go back to the model in the order it requested them.

Use `ai.WithSequential()` when a tool changes shared state or must run alone:

```go
ai.AddTool(agent, "update_database", updateDatabase, ai.WithSequential())
```

The sequential tool is a barrier. Earlier calls finish before it starts. Later calls wait until it finishes. Use `ai.WithSequentialToolExecution()` on the agent when every tool must run serially.

Local execution is separate from model generation. Set `ModelSettings.ParallelToolCalls` to tell OpenAI or Anthropic whether the model may emit parallel calls:

```go
parallel := false
agent := ai.NewAgent[Deps, string](
	model,
	ai.WithModelSettings(ai.ModelSettings{ParallelToolCalls: &parallel}),
)
```

Use `ai.WithStrict()` to ask the provider to constrain generated arguments to the tool schema:

```go
ai.AddTool(agent, "book_table", bookTable, ai.WithStrict())
```

Strict mode prevents malformed arguments before they reach your code. OpenAI enables it automatically when a schema is compatible, and rewrites incompatible constraints when you use `ai.WithStrict()`. Use `openai.WithStrictToolSupport(false)` for compatible endpoints that reject strict definitions. Anthropic uses explicit strict mode on supported Claude models; use `anthropic.WithStrictToolSupport` for aliases or newly released models. Set `anthropic.WithSchemaWarningHandler` to inspect lossy conversions, such as dynamic map schemas that Anthropic closes with `additionalProperties: false`. Gemini 2.5 and newer use request-wide `VALIDATED` mode by default. Use `ai.WithoutStrict()` to keep a Gemini request on `AUTO`, or `google.WithStrictToolSupport` for model aliases and compatible proxies.

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

The hook receives fresh copies, so you can safely change descriptions and nested schemas. Return an empty or nil slice to expose no function tools for that step. Output tools are prepared separately by the agent and are not included.

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

Unknown tool names also go back to the model as retry prompts. The prompt lists only tools exposed for that request, so a tool omitted by preparation cannot be executed from a stale call.

Use `ai.WithToolTimeout(5 * time.Second)` to give one tool call a deadline. The tool must honor `ctx.Done()`. A tool-specific timeout becomes a retry, while cancellation of the parent run remains `context.Canceled`.

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

With native structured output, `ai.EndStrategyEarly` also lets valid JSON preempt function tools. Invalid JSON falls through to the tools without consuming a retry. Plain text never preempts a tool call because it may only describe the work the model is about to perform.

## Streaming

`RunStream` yields events as the model produces them - text deltas, tool call starts, argument fragments - and the typed result is available once the stream completes:

```go
stream := agent.RunStream(ctx, "tell me a story", deps)
for event, err := range stream.Events() {
	if err != nil { /* handle */ }
	if delta, ok := event.(ai.TextDeltaEvent); ok {
		fmt.Print(delta.Delta)
	}
}
result := stream.Result()
```

OpenAI Chat Completions, OpenAI Responses, Anthropic Messages, and Google Gemini stream text, thinking, tool arguments, and usage from their SSE APIs. Models that do not implement `ai.StreamingModel` still work: each response is replayed as events.

Every text, thinking, tool-start, and tool-argument event carries a stable `PartID`. Use it to route interleaved deltas without relying on arrival order. Bundled providers populate it, and custom streaming models can leave it empty only for strictly sequential parts.

`RunStream` commits the first matching text, native, or output-tool result. The configured end strategy still controls co-emitted tools, but a tool retry cannot revoke that result. If an output validator requests a retry, the streamed run returns `UnexpectedModelBehaviorError` because output has already been committed. Use `Run` when validation should start another model round.

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

Implement any of `RunWrapper`, `ModelRequestWrapper`, `ToolCallWrapper`, or `InstructionsProvider` - the agent discovers them by type assertion, the same pattern as `http.Flusher`. Slice order is middleware order: the first capability is outermost. Usage limits are implemented on this same surface internally.

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

Message history serializes to PydanticAI's JSON format via `ai.MarshalMessages` / `ai.UnmarshalMessages`, so histories exchange cleanly with [PydanticAI](https://ai.pydantic.dev), [pydantic-evals-go](https://github.com/Kludex/pydantic-evals-go), and Logfire. OpenTelemetry spans follow the GenAI semantic conventions and are free unless you set a global tracer provider.

## Status

v0.3 - capabilities (hook interfaces, `WithCapabilities`) and the OpenAI Responses API model (`openai.NewResponsesModel`), on top of v0.2 streaming, three providers, multimodal input, and native JSON output mode, and the v0.1 loop, tools, structured output, usage limits, fakes, and tracing. See [PLAN.md](PLAN.md) for the roadmap: MCP and provider-native tools (v0.4).
