# Model wrappers

Wrap a model when behavior must follow that model everywhere it is used. Embed `ModelWrapper` so optional streaming, token-counting, compaction, lifecycle, provider identity, and native-tool capabilities keep working.

```go
package main

import (
	"context"
	"fmt"
	"log"

	ai "github.com/Kludex/pydantic-ai-go/ai"
	"github.com/Kludex/pydantic-ai-go/ai/models/fakes"
)

type HeaderModel struct {
	*ai.ModelWrapper
	Tenant string
}

func (model *HeaderModel) Request(
	ctx context.Context,
	messages []ai.ModelMessage,
	params ai.ModelRequestParams,
) (*ai.ModelResponse, error) {
	request := (ai.ModelRequestContext{Messages: messages, Params: params}).Clone()
	if request.Params.Settings.ExtraHeaders == nil {
		request.Params.Settings.ExtraHeaders = map[string]string{}
	}
	request.Params.Settings.ExtraHeaders["X-Tenant"] = model.Tenant
	return model.ModelWrapper.Request(ctx, request.Messages, request.Params)
}

func main() {
	base := fakes.NewFunctionModel(func(
		_ context.Context,
		_ []ai.ModelMessage,
		params ai.ModelRequestParams,
	) (*ai.ModelResponse, error) {
		if params.Settings.ExtraHeaders["X-Tenant"] != "acme" {
			return nil, fmt.Errorf("missing tenant header")
		}
		return &ai.ModelResponse{Parts: []ai.ResponsePart{
			ai.TextPart{Content: "tenant selected"},
		}}, nil
	})
	model := &HeaderModel{ModelWrapper: ai.WrapModel(base), Tenant: "acme"}
	agent := ai.NewAgent[struct{}, string](model)
	result, err := agent.Run(context.Background(), "Check the tenant.", struct{}{})
	if err != nil {
		log.Fatal(err)
	}
	fmt.Println(result.Output)
}
```

Clone messages, request parameters, settings, and metadata before you mutate them. The example uses `ModelRequestContext.Clone` so a reusable wrapper cannot modify another concurrent run.

Override `StreamRequest` separately when behavior must inspect individual stream events. The embedded method otherwise delegates native streaming and replays non-streaming responses for models without it.

## Built-in wrappers

| Constructor | Behavior |
| --- | --- |
| `NewInstrumentedModel` | Records model request, stream, token, cost, and compaction telemetry |
| `NewConcurrencyLimitedModel` | Shares a context-aware request and token-counting concurrency gate |
| `NewFallbackModel` | Tries ordered models after eligible provider failures or rejected responses |
| `NewProfiledModel` | Overrides model feature and structured-output profile detection |

Wrapper order is explicit. The outer constructor sees the operation first. For example, `NewInstrumentedModel(NewConcurrencyLimitedModel(model, gate))` includes queue wait time in the request span. Reverse the order when you want telemetry to begin only after admission.

`ModelWrapper` delegates all optional interfaces implemented by the wrapped model. A custom wrapper should override only behavior it changes. Use `UnwrapModel` when profile detection or diagnostics need the innermost concrete model. Cyclic or nil wrapper chains return the outer model instead of looping.

## Choose a capability or a wrapper

Use a capability when behavior belongs to an agent run and must follow adaptive model selection. Model-request capabilities receive the model selected for the current step and compose with run, tool, output, and event hooks.

Use a wrapper when behavior belongs to one model instance. Transport instrumentation, concurrency limits, provider profiles, and fallback selection need to remain attached when the same model is passed to direct requests or several agents.

Do not instrument both the agent capability and its wrapped model only to create duplicate spans. Duplicate model-request instrumentation is suppressed, but one instrumentation boundary is easier to reason about.

## Preserve ownership

Wrapping does not transfer ownership. `ModelWrapper.OpenModel` delegates `ModelOpener` and returns the wrapped cleanup function. An agent opens each distinct selected model once per run and closes it in reverse selection order.

Do not store `context.Context` in a wrapper. Pass the request context to the wrapped operation so cancellation, deadlines, tracing, and admission all share one lifecycle.
