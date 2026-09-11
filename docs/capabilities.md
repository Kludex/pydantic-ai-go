# Capabilities

A capability adds behavior at a semantic boundary in the agent loop. You can reuse it across agents without changing the model or tool implementation.

## Compose built-in capabilities

```go
package main

import (
	"context"
	"errors"
	"fmt"
	"log"

	ai "github.com/Kludex/pydantic-ai-go/ai"
	"github.com/Kludex/pydantic-ai-go/ai/models/openai"
)

type WeatherArgs struct {
	City string `json:"city"`
}

type Weather struct {
	TemperatureC int `json:"temperature_c"`
	Conditions   string `json:"conditions"`
}

func main() {
	agent := ai.NewAgent[struct{}, string](
		openai.NewModel("gpt-5"),
		ai.WithCapabilities(
			ai.IncludeToolReturnSchemas{},
			ai.RaiseContentFilterError{},
		),
	)
	ai.AddSimpleTool(agent, "weather", func(_ context.Context, args WeatherArgs) (Weather, error) {
		return Weather{TemperatureC: 18, Conditions: "cloudy in " + args.City}, nil
	})

	result, err := agent.Run(context.Background(), "What is the weather in London?", struct{}{})
	if err != nil {
		var filtered *ai.ContentFilterError
		if errors.As(err, &filtered) {
			fmt.Println(string(filtered.Body()))
			return
		}
		log.Fatal(err)
	}
	fmt.Println(result.Output)
}
```

`IncludeToolReturnSchemas` sends reflected return types for selected tools. Gemini receives a native `responseJsonSchema` field. OpenAI-compatible and Anthropic models receive the schema in the tool description.

`RaiseContentFilterError` makes partial filtered responses fail instead of returning their partial text. Use `errors.As` to inspect its response body.

The agent already raises `ContentFilterError` when a filtered response has no parts. Add the capability when partial text or refusal text must also end the run.

## Select return schemas

```go
package main

import (
	"context"
	"fmt"

	ai "github.com/Kludex/pydantic-ai-go/ai"
	"github.com/Kludex/pydantic-ai-go/ai/models/fakes"
)

type Result struct {
	Value string `json:"value"`
}

func main() {
	capability := ai.IncludeToolReturnSchemas{
		Select: func(
			_ context.Context,
			_ *ai.RunInfo,
			definition ai.ToolDefinition,
		) (bool, error) {
			return definition.Metadata["include_return"] == true, nil
		},
	}
	agent := ai.NewAgent[struct{}, string](
		fakes.NewTestModel(),
		ai.WithCapabilities(capability),
	)
	agent.AddTool(ai.NewSimpleTool[struct{}](
		"lookup",
		func(context.Context, struct{}) (Result, error) {
			return Result{Value: "found"}, nil
		},
		ai.WithToolMetadata(map[string]any{"include_return": true}),
	))

	result, err := agent.Run(context.Background(), "Look up the value.", struct{}{})
	if err != nil {
		panic(err)
	}
	fmt.Println(result.Output)
}
```

A nil selector includes every tool. `WithReturnSchemaIncluded(false)` opts one tool out. An explicit per-tool setting takes precedence over the capability.

Use `WithToolReturnSchemas` when only one toolset should advertise returns.

## Reinject a trusted legacy system prompt

```go
package main

import (
	"context"
	"fmt"
	"log"

	ai "github.com/Kludex/pydantic-ai-go/ai"
	"github.com/Kludex/pydantic-ai-go/ai/models/openai"
)

func main() {
	history := []ai.ModelMessage{
		ai.ModelRequest{Parts: []ai.RequestPart{
			ai.UserPromptPart{Content: "My name is Ada."},
		}},
		ai.ModelResponse{Parts: []ai.ResponsePart{
			ai.TextPart{Content: "Hello, Ada."},
		}},
	}
	agent := ai.NewAgent[struct{}, string](
		openai.NewModel("gpt-5"),
		ai.WithSystemPrompt("You are a concise assistant."),
		ai.WithCapabilities(ai.ReinjectSystemPrompt{ReplaceExisting: true}),
	)
	result, err := agent.Run(
		context.Background(),
		"What is my name?",
		struct{}{},
		ai.WithMessageHistory(history),
	)
	if err != nil {
		log.Fatal(err)
	}
	fmt.Println(result.Output)
}
```

Use `ReplaceExisting: true` when history came from an untrusted client. The capability strips history-provided system parts before it injects the server configuration.

Reinjection changes only the request snapshot sent to the model. It does not rewrite durable history. Every later request applies the same filter again.

Prefer `WithInstructions` for new applications. Instructions are prepared for every request and do not need reinjection.

## Combine repeated capabilities

```go
package main

import (
	"context"
	"fmt"

	ai "github.com/Kludex/pydantic-ai-go/ai"
	"github.com/Kludex/pydantic-ai-go/ai/models/fakes"
)

type SearchPolicy struct {
	Domains []string
}

func (SearchPolicy) CapabilityID() string {
	return "search_policy"
}

func (SearchPolicy) CombineCapabilities(capabilities []ai.Capability) (ai.Capability, error) {
	return ai.MergeCapabilities(capabilities...)
}

func (policy SearchPolicy) Setup(registry *ai.CapabilityRegistry) error {
	registry.AddInstructions(fmt.Sprintf("Search only these domains: %v", policy.Domains))
	return nil
}

func main() {
	agent := ai.NewAgent[struct{}, string](
		fakes.NewTestModel(),
		ai.WithCapabilities(
			SearchPolicy{Domains: []string{"go.dev"}},
			SearchPolicy{Domains: []string{"pkg.go.dev"}},
		),
	)
	result, err := agent.Run(context.Background(), "Find the context package.", struct{}{})
	if err != nil {
		panic(err)
	}
	fmt.Println(result.Output)
}
```

`CapabilityIDProvider` identifies one logical capability. Repeated IDs in the same agent or run layer require `CapabilityCombiner`. The default `MergeCapabilities` helper preserves a value stated by only one declaration. It unions maps and slices. Later scalar values win.

The default merge accepts exported struct fields. It rejects unexported fields and incompatible collection types. Implement `CombineCapabilities` yourself when your capability caches derived state. Rebuild that state from the merged fields before you return.

A run capability replaces an agent capability with the same ID. It does not merge with the agent configuration. The complete run wrapper subtree replaces the agent subtree, then each layer keeps its declared middleware order. A shared ID with a different capability type is always rejected.

Leave `CapabilityID` empty when several independent instances are valid. Distinct non-empty IDs also keep instances separate.

## Implement one hook

```go
package main

import (
	"context"
	"fmt"

	ai "github.com/Kludex/pydantic-ai-go/ai"
	"github.com/Kludex/pydantic-ai-go/ai/models/fakes"
)

func main() {
	capability := ai.BeforeModelRequestFunc(func(
		_ context.Context,
		_ *ai.RunInfo,
		request ai.ModelRequestContext,
	) (ai.ModelRequestContext, error) {
		request.Params.Settings.ExtraHeaders = map[string]string{
			"X-Request-Source": "example",
		}
		return request, nil
	})
	agent := ai.NewAgent[struct{}, string](
		fakes.NewTestModel(),
		ai.WithCapabilities(capability),
	)
	result, err := agent.Run(context.Background(), "Say hello.", struct{}{})
	if err != nil {
		panic(err)
	}
	fmt.Println(result.Output)
}
```

Use the narrowest hook that matches the behavior. Hooks receive detached values. Set `ModelRequestContext.ReplaceHistory` only when a request transformation must also replace durable history.

Capability order is middleware order. Before hooks run from first to last. After and error hooks run in reverse order.

## Emit typed events

```go
package main

import (
	"context"
	"fmt"
	"time"

	ai "github.com/Kludex/pydantic-ai-go/ai"
	"github.com/Kludex/pydantic-ai-go/ai/models/fakes"
)

type Progress struct {
	Completed int `json:"completed"`
	Total     int `json:"total"`
}

func main() {
	agent := ai.NewAgent[struct{}, string](fakes.NewTestModel())
	ai.OnEvent(agent, func(
		_ context.Context,
		_ *ai.RunContext[struct{}],
		event *ai.CustomEvent[Progress],
	) error {
		fmt.Printf("%d/%d\n", event.Data.Completed, event.Data.Total)
		return nil
	}, ai.WithEventListenerTimeout(time.Second))

	ai.AddTool(agent, "work", func(
		_ context.Context,
		run *ai.RunContext[struct{}],
		_ struct{},
	) (string, error) {
		if err := run.Emit(ai.NewCustomEvent("progress", Progress{Completed: 1, Total: 1})); err != nil {
			return "", err
		}
		return "done", nil
	})

	if _, err := agent.Run(context.Background(), "Run the work tool.", struct{}{}); err != nil {
		panic(err)
	}
}
```

`CustomEvent` carries application data. It is visible to UI adapters by default. Use `SetUIVisible(false)` for server-only events. Use `ProjectForUI` when the frontend needs a smaller payload.

A capability emits `CapabilityEvent` instead. `CapabilityRegistry.AddContextTool` gives a capability-owned tool a `RunInfo`, so emitted events include the capability ID, tool name, and tool call ID. Capability events stay server-side unless an application listener republishes them as custom events.

Listeners normally run when an event reaches its stream position. Use `SetDispatch(ai.EventDispatchImmediate)` for a mutable capability decision that must settle before `RunInfo.Emit` returns. Agent listeners run after ordinary capability listeners and before capabilities in the `CapabilityInnermost` tier.
