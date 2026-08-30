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

Use `ai.WithStrict()` to ask the provider to constrain generated arguments to the tool schema:

```go
ai.AddTool(agent, "book_table", bookTable, ai.WithStrict())
```

Strict mode prevents malformed arguments before they reach your code. OpenAI and Anthropic configure it per tool. Gemini applies its request-wide `VALIDATED` mode. Use `ai.WithoutStrict()` when you need to disable a provider default explicitly.

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

Models that do not implement `ai.StreamingModel` still work: each response is replayed as events.

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
