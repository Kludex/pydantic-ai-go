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

v0.1 - the core loop, tools, structured output, usage limits, OpenAI provider, fakes, and tracing. See [PLAN.md](PLAN.md) for the roadmap: streaming and more providers (v0.2), capabilities (v0.3), MCP (v0.4).
