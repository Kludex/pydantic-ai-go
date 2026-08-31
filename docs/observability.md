# OpenTelemetry

Instrument an agent with one capability:

```go
package main

import (
	"context"
	"fmt"
	"log"

	ai "github.com/Kludex/pydantic-ai-go"
	"github.com/Kludex/pydantic-ai-go/models/openai"
	"go.opentelemetry.io/otel"
	"go.opentelemetry.io/otel/exporters/stdout/stdouttrace"
	sdktrace "go.opentelemetry.io/otel/sdk/trace"
)

func main() {
	exporter, err := stdouttrace.New(stdouttrace.WithPrettyPrint())
	if err != nil {
		log.Fatal(err)
	}
	provider := sdktrace.NewTracerProvider(sdktrace.WithBatcher(exporter))
	defer func() {
		if err := provider.Shutdown(context.Background()); err != nil {
			log.Printf("flush traces: %v", err)
		}
	}()
	otel.SetTracerProvider(provider)

	agent := ai.NewAgent[struct{}, string](
		openai.NewModel("gpt-5-mini"),
		ai.WithAgentName("support"),
		ai.WithAgentDescription("Answers support questions"),
		ai.WithMetadata(map[string]any{"service": "support"}),
		ai.WithCapabilities(ai.NewInstrumentation()),
	)
	result, err := agent.Run(context.Background(), "What is 2 + 2?", struct{}{})
	if err != nil {
		log.Fatal(err)
	}
	fmt.Println(result.Output)
}
```

Install the stdout exporter for this example:

```console
go get go.opentelemetry.io/otel/exporters/stdout/stdouttrace
```

Use an OTLP exporter instead when you send traces to an observability service. The instrumentation uses the standard OpenTelemetry tracer and meter providers.

## Recorded spans

`NewInstrumentation` records the complete agent hierarchy:

- One `invoke_agent <name>` span for the run.
- One `chat <model>` client span for each model request.
- One `execute_tool <name>` span for each local tool execution.
- One `execute_tool <name>` span for each user output function.
- One failed `execute_tool <name>` span for a tool call rejected during argument validation.

Set the application identity with `WithAgentName` and `WithAgentDescription`. Use `WithAgentDescriptionFunc` when the description depends on typed run dependencies. `WithInstrumentationAgentName` remains available when one telemetry pipeline needs to override the application name.

The run span records the agent description, cumulative usage, cost, application metadata from `WithMetadata` or `WithRunMetadata`, and whether formatted instructions changed between requests. Its `logfire.json_schema` describes the recorded run fields. Request spans record provider, model, request settings, response details, tool definitions, messages, usage, cost, and streaming time to first chunk.

Output-function spans include the validated model value as their arguments and the converted final value as their result. They use the output-tool name in tool mode and the registered function name in native or prompted mode. Plain validation and output validators do not create output-function spans.

Tool deferrals are control flow in the default format. Their spans remain successful and use `pydantic_ai.tool.deferral.name` and `pydantic_ai.tool.deferral.metadata` attributes.

## Protect content

Disable content before you send telemetry outside your trust boundary. Pass `WithInstrumentationContent(false)`, `WithInstrumentationBinaryContent(false)`, and `WithInstrumentationModelRequestParameters(false)` to `NewInstrumentation`.

`WithInstrumentationContent(false)` keeps message roles and part types but removes prompts, completions, tool arguments, tool results, and final output. This preserves trace structure without exporting user content.

Application run metadata is still exported in the `metadata` attribute because it is intended for trace filtering and evaluation. Do not put secrets in metadata. `WithInstrumentationBinaryContent(false)` redacts `BinaryContent` values nested inside metadata.

`WithInstrumentationBinaryContent(false)` keeps media types but removes inline bytes. It follows maps, slices, `ToolReturn`, and deferred metadata. It does not inspect fields inside your own struct types.

`WithInstrumentationModelRequestParameters(false)` removes the complete request-parameter snapshot. Tool names and public JSON Schemas remain available through `gen_ai.tool.definitions`.

> [!WARNING]
> Tool schemas and names can contain sensitive business information. The model-request-parameter switch does not hide `gen_ai.tool.definitions`.

## Instrument direct model requests

Use `NewInstrumentedModel` when you call `RequestModel`, `StreamModel`, or the model interface directly:

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
	model := ai.NewInstrumentedModel(openai.NewModel("gpt-5-mini"))
	response, err := ai.RequestModel(
		context.Background(),
		model,
		[]ai.ModelMessage{ai.ModelRequest{Parts: []ai.RequestPart{
			ai.UserPromptPart{Content: "What is 2 + 2?"},
		}}},
		ai.ModelRequestParams{},
	)
	if err != nil {
		log.Fatal(err)
	}
	fmt.Println(response.Text())
}
```

Do not wrap an agent's model and add `NewInstrumentation` only to get duplicate spans. The agent capability already records model requests. Duplicate instrumentation is suppressed when both are present.

## Select a telemetry format

Version 5 is the default. It matches the current upstream default. Pass `WithInstrumentationVersion(6)` to emit tool call responses with the OpenTelemetry `tool` role instead of the legacy `user` role.

Versions 2 through 6 are supported for compatibility. Versions 2 and 3 retain the older multimodal message shape. Version 2 also uses the legacy `agent run`, `running tool`, `running output function`, `tool_arguments`, and `tool_response` names.

Run spans use `gen_ai.aggregated_usage.*` by default. This prevents observability backends from adding cumulative run usage to child request usage. Pass `WithInstrumentationAggregatedUsageAttributeNames(false)` when an existing dashboard requires `gen_ai.usage.*` names.
