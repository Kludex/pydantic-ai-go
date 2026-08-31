# pydantic-ai-go

Build typed LLM agents in Go.

`pydantic-ai-go` provides the agent loop, tools, structured output, streaming, usage limits, model middleware, and provider integrations. It follows PydanticAI's behavior while keeping the public API idiomatic for Go.

## Install

You need Go 1.25 or newer.

```console
go get github.com/Kludex/pydantic-ai-go
```

Set the API key for your provider:

```console
export OPENAI_API_KEY="..."
```

## Create an agent with a tool

Create an `Agent`, register a typed function, then call `Run`:

```go
package main

import (
	"context"
	"fmt"
	"log"

	ai "github.com/Kludex/pydantic-ai-go"
	"github.com/Kludex/pydantic-ai-go/models/openai"
)

type WeatherArgs struct {
	City string `json:"city" jsonschema:"description=City name"`
}

func main() {
	model := openai.NewModel("gpt-5-mini")
	agent := ai.NewAgent[struct{}, string](
		model,
		ai.WithInstructions("You are a concise weather assistant."),
	)

	ai.AddSimpleTool(agent, "get_weather", func(
		_ context.Context,
		args WeatherArgs,
	) (string, error) {
		return fmt.Sprintf("It is 21 C and sunny in %s.", args.City), nil
	}, ai.WithDescription("Get the current weather for a city"))

	result, err := agent.Run(
		context.Background(),
		"What is the weather in Berlin?",
		struct{}{},
	)
	if err != nil {
		log.Fatal(err)
	}

	fmt.Println(result.Output)
}
```

The type parameters are `Agent[Deps, Output]`.

- `Deps` is application state available to tools and callbacks.
- `Output` is the final result type. Use `string` for text.
- `WeatherArgs` becomes the tool's JSON Schema.

The agent validates tool arguments before your function runs. Invalid arguments go back to the model as a retry prompt. Independent tool calls run concurrently and their results keep model order.

Use `AddTool` instead of `AddSimpleTool` when your function needs `RunContext`. The context contains typed dependencies, usage, history, retry state, and cancellation for the current run. Pass dependencies as the third argument to `Run`. One agent can serve concurrent runs safely.

## Return structured data

Use a struct as the output type:

```go
package main

import (
	"context"
	"fmt"
	"log"

	ai "github.com/Kludex/pydantic-ai-go"
	"github.com/Kludex/pydantic-ai-go/models/openai"
)

type City struct {
	Name       string `json:"name"`
	Country    string `json:"country"`
	Population int    `json:"population"`
}

func main() {
	agent := ai.NewAgent[struct{}, City](openai.NewModel("gpt-5-mini"))

	result, err := agent.Run(
		context.Background(),
		"Return information about Paris.",
		struct{}{},
	)
	if err != nil {
		log.Fatal(err)
	}

	fmt.Printf("%s, %s: %d\n", result.Output.Name, result.Output.Country, result.Output.Population)
}
```

The agent reflects a Draft 2020-12 JSON Schema from `City`. It validates the model response against that schema, decodes it into `City`, and retries invalid output.

`OutputModeAuto` is the default. It reads the selected model's `ModelProfile`; the standard profile chooses tool output because it works across providers. Use `WithOutputMode(ai.OutputModeNative)` to require native JSON Schema output, or `OutputModePrompted` to require JSON in text.

## Stream a response

`RunStream` returns normalized events across all providers:

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
	agent := ai.NewAgent[struct{}, string](openai.NewModel("gpt-5-mini"))
	stream := agent.RunStream(context.Background(), "Write a short poem.", struct{}{})

	for event, err := range stream.Events() {
		if err != nil {
			log.Fatal(err)
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
	if result == nil {
		log.Fatal("stream ended without a result")
	}
	fmt.Printf("\nTokens: %d\n", result.Usage().TotalTokens())
}
```

Consume one stream view only. Use `Events` for lifecycle events or `Outputs` for typed partial output snapshots. Cancel the context to stop the request.

## Continue a conversation

Pass the previous result's messages into the next run:

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
	ctx := context.Background()
	agent := ai.NewAgent[struct{}, string](openai.NewModel("gpt-5-mini"))

	first, err := agent.Run(ctx, "My name is Ada.", struct{}{})
	if err != nil {
		log.Fatal(err)
	}
	second, err := agent.Run(
		ctx,
		"What is my name?",
		struct{}{},
		ai.WithMessageHistory(first.Messages()),
	)
	if err != nil {
		log.Fatal(err)
	}

	fmt.Println(second.Output)
}
```

Use `MarshalMessages` and `UnmarshalMessages` to persist history. Their JSON format is compatible with PydanticAI.

## Choose a provider

The bundled providers use the same `ai.Model` interface.

| Provider | Constructor | Environment variable |
| --- | --- | --- |
| OpenAI Chat Completions | `openai.NewModel("gpt-5-mini")` | `OPENAI_API_KEY` |
| OpenAI Responses | `openai.NewResponsesModel("gpt-5-mini")` | `OPENAI_API_KEY` |
| Azure OpenAI | `azure.NewModel("deployment", azure.Config{})` | `AZURE_OPENAI_ENDPOINT`, `AZURE_OPENAI_API_KEY` |
| Anthropic | `anthropic.NewModel("claude-sonnet-4-5")` | `ANTHROPIC_API_KEY` |
| Google Gemini | `google.NewModel("gemini-2.5-flash")` | `GEMINI_API_KEY` |

Each constructor supports custom HTTP clients and default model settings. See [Provider configuration](docs/providers.md) for OpenAI-compatible endpoints, Azure API versions, and provider-specific options.

## Test an agent without network calls

Use `models/fakes` in tests:

```go
package weather_test

import (
	"context"
	"testing"

	ai "github.com/Kludex/pydantic-ai-go"
	"github.com/Kludex/pydantic-ai-go/models/fakes"
)

func TestAgent(t *testing.T) {
	agent := ai.NewAgent[struct{}, string](fakes.NewFunctionModel(func(
		_ context.Context,
		_ []ai.ModelMessage,
		_ ai.ModelRequestParams,
	) (*ai.ModelResponse, error) {
		return &ai.ModelResponse{
			Parts: []ai.ResponsePart{ai.TextPart{Content: "hello"}},
		}, nil
	}))

	result, err := agent.Run(t.Context(), "Say hello.", struct{}{})
	if err != nil {
		t.Fatal(err)
	}
	if result.Output != "hello" {
		t.Fatalf("got %q", result.Output)
	}
}
```

`fakes.NewFunctionModel` gives you complete control over each model response. `fakes.NewTestModel` can generate schema-valid calls for every registered tool.

## API guide

| You need to | Use |
| --- | --- |
| Change one run without mutating the agent | `WithRunModelSettings`, `WithRunInstructions`, and other `RunOption` values |
| Reuse or combine tools | `NewTool`, `NewFunctionToolset`, and the toolset wrappers |
| Require approval | `WithApprovalRequired` and `DeferredToolResults` |
| Connect an MCP server | [`mcp.NewStreamableHTTPToolset`, `Connect`, or `LoadToolsets`](docs/mcp.md) |
| Limit usage or cost | `UsageLimits` |
| Configure a provider or compatible endpoint | [Provider configuration](docs/providers.md) |
| Add fallback models | `NewFallbackModel` |
| Limit concurrency | `NewConcurrencyLimiter` |
| Add middleware | `Capability` and its focused hook interfaces |
| Add OpenTelemetry | [`NewInstrumentation` or `NewInstrumentedModel`](docs/observability.md) |
| Call a model without an agent | `RequestModel` or `StreamModel` |
| Drive a run one event at a time | `AgentRun` |

Run options are detached from agent configuration. They are safe to use in concurrent runs. See the [Go package documentation](https://pkg.go.dev/github.com/Kludex/pydantic-ai-go) for each public contract and [`CHECKLIST.md`](CHECKLIST.md) for remaining parity work.

## Design

The agent is a plain loop:

1. Build a model request.
2. Ask the model for a response.
3. Validate and execute tool calls.
4. Repeat until a final output is valid.

Capabilities intercept these semantic boundaries. The package does not expose an internal graph API. This keeps the control flow readable and lets Go interfaces provide small extension points.

Per-run settings, tools, capabilities, output types, histories, and provider details are copied before use. One run cannot mutate another run's configuration.

## Development

Run the complete local checks:

```console
gofmt -w $(git ls-files '*.go')
go vet ./...
golangci-lint run ./...
go test ./...
scripts/coverage.sh
go test -race ./...
```

The project requires 100% statement coverage for every tested package.

## Status

The API is under active development while it approaches feature parity with [PydanticAI](https://ai.pydantic.dev/). Review [`CHECKLIST.md`](CHECKLIST.md) before depending on a provider-specific or advanced feature.

## License

This project is licensed under the MIT License. See [`LICENSE`](LICENSE).
