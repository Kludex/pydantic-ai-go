# OpenTelemetry

Instrument an agent with one capability:

```go
package main

import (
	"context"
	"fmt"
	"log"

	ai "github.com/Kludex/pydantic-ai-go/ai"
	"github.com/Kludex/pydantic-ai-go/ai/models/openai"
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

## Control the first-run banner

```go
package main

import ai "github.com/Kludex/pydantic-ai-go/ai"

func main() {
	ai.SetBannerEnabled(false)
}
```

Before you configure instrumentation, the first agent run in a process writes a short banner to `stderr`.
The banner appears only when `stderr` is a terminal or a recognized coding agent reads the process output.
It is not shown for instrumented agents, in CI, or under `go test`.

Set `PYDANTIC_AI_NO_BANNER` to any value to disable the banner without changing code.
You can also call `SetBannerEnabled(false)` before the first run.
The setting and the once-per-process claim are safe for concurrent runs.

`cli.Run` writes the banner before its first prompt.
If you build another terminal interface, call `Agent.WriteBanner` with the interface output before you start reading prompts.
The method omits the tool count when run-scoped capabilities or toolsets cannot be inspected safely before the run.

## Send telemetry to Logfire

```go
package main

import (
	"context"
	"fmt"
	"log"

	ai "github.com/Kludex/pydantic-ai-go/ai"
	"github.com/Kludex/pydantic-ai-go/ai/models/openai"
	"go.opentelemetry.io/otel"
	"go.opentelemetry.io/otel/exporters/otlp/otlpmetric/otlpmetrichttp"
	"go.opentelemetry.io/otel/exporters/otlp/otlptrace/otlptracehttp"
	sdkmetric "go.opentelemetry.io/otel/sdk/metric"
	sdktrace "go.opentelemetry.io/otel/sdk/trace"
)

func main() {
	ctx := context.Background()
	traceExporter, err := otlptracehttp.New(ctx)
	if err != nil {
		log.Printf("create trace exporter: %v", err)
		return
	}
	tracerProvider := sdktrace.NewTracerProvider(sdktrace.WithBatcher(traceExporter))
	defer func() {
		if err := tracerProvider.Shutdown(context.Background()); err != nil {
			log.Printf("flush traces: %v", err)
		}
	}()

	metricExporter, err := otlpmetrichttp.New(ctx)
	if err != nil {
		log.Printf("create metric exporter: %v", err)
		return
	}
	meterProvider := sdkmetric.NewMeterProvider(
		sdkmetric.WithReader(sdkmetric.NewPeriodicReader(metricExporter)),
	)
	defer func() {
		if err := meterProvider.Shutdown(context.Background()); err != nil {
			log.Printf("flush metrics: %v", err)
		}
	}()
	otel.SetTracerProvider(tracerProvider)
	otel.SetMeterProvider(meterProvider)

	agent := ai.NewAgent[struct{}, string](
		openai.NewModel("gpt-5-mini"),
		ai.WithAgentName("logfire-example"),
		ai.WithCapabilities(ai.NewInstrumentation(
			ai.WithInstrumentationContent(false),
			ai.WithInstrumentationBinaryContent(false),
			ai.WithInstrumentationModelRequestParameters(false),
		)),
	)
	result, err := agent.Run(ctx, "What is 2 + 2?", struct{}{})
	if err != nil {
		log.Printf("run agent: %v", err)
		return
	}
	fmt.Println(result.Output)
}
```

Install the OTLP exporters:

```console
go get go.opentelemetry.io/otel/exporters/otlp/otlptrace/otlptracehttp
go get go.opentelemetry.io/otel/exporters/otlp/otlpmetric/otlpmetrichttp
```

Create a Logfire write token. Then configure the standard OTLP environment variables:

```console
export OTEL_SERVICE_NAME=my-agent
export OTEL_EXPORTER_OTLP_ENDPOINT=https://logfire-us.pydantic.dev
export OTEL_EXPORTER_OTLP_PROTOCOL=http/protobuf
export OTEL_EXPORTER_OTLP_HEADERS='Authorization=your-write-token'
export OPENAI_API_KEY=your-openai-api-key
go run ./examples/logfire
```

Use the endpoint for your Logfire region. An endpoint and token from different regions fail authentication. You can create and revoke write tokens in [Logfire project settings](https://pydantic.dev/docs/logfire/manage/create-write-tokens/).

The example exports traces and metrics through OTLP. It disables prompts, outputs, binary values, and full request parameters because telemetry often leaves your trust boundary. Enable only the content your Logfire project is allowed to retain.

## Recorded spans

`NewInstrumentation` records the complete agent hierarchy:

- One `invoke_agent <name>` span for the run.
- One `chat <model>` client span for each model request.
- One `compact <model>` client span for each explicit provider compaction request.
- One `execute_tool <name>` span for each local tool execution.
- One `execute_tool <name>` span for each user output function.
- One failed `execute_tool <name>` span for a tool call rejected during argument validation.

Set the application identity with `WithAgentName` and `WithAgentDescription`. Use `WithAgentDescriptionFunc` when the description depends on typed run dependencies. `WithInstrumentationAgentName` remains available when one telemetry pipeline needs to override the application name.

The run span records the agent description, cumulative usage, cost, application metadata from `WithMetadata` or `WithRunMetadata`, and whether formatted instructions changed between requests. Its `logfire.json_schema` describes the recorded run fields. Request spans record provider, model, request settings, response details, tool definitions, messages, usage, cost, and streaming time to first chunk.

Explicit compaction uses `gen_ai.operation.name=compact`. Automatic stateless compaction is a separate child of the run span before the primary request. See [Compact model history](compaction.md).

Output-function spans include the validated model value as their arguments and the converted final value as their result. They use the output-tool name in tool mode and the registered function name in native or prompted mode. Plain validation and output validators do not create output-function spans.

Tool deferrals are control flow in the default format. Their spans remain successful and use `pydantic_ai.tool.deferral.name` and `pydantic_ai.tool.deferral.metadata` attributes.

## Prompt-cache health

You get cache-health telemetry without another option. This example uses a fake model to reproduce a collapsed cache without contacting a provider:

```go
package main

import (
	"context"
	"log"
	"time"

	ai "github.com/Kludex/pydantic-ai-go/ai"
	"github.com/Kludex/pydantic-ai-go/ai/models/fakes"
	"go.opentelemetry.io/otel/exporters/stdout/stdouttrace"
	sdktrace "go.opentelemetry.io/otel/sdk/trace"
)

func main() {
	ctx := context.Background()
	exporter, err := stdouttrace.New(stdouttrace.WithPrettyPrint())
	if err != nil {
		log.Fatal(err)
	}
	provider := sdktrace.NewTracerProvider(sdktrace.WithSyncer(exporter))
	defer func() { _ = provider.Shutdown(ctx) }()

	calls := 0
	model := fakes.NewFunctionModel(func(context.Context, []ai.ModelMessage, ai.ModelRequestParams) (*ai.ModelResponse, error) {
		calls++
		usage := ai.Usage{InputTokens: 20000, CacheWriteTokens: 14000}
		if calls > 1 {
			usage = ai.Usage{InputTokens: 20000, CacheReadTokens: 1000}
		}
		return &ai.ModelResponse{
			Parts: []ai.ResponsePart{ai.TextPart{Content: "done"}}, Usage: usage,
			ProviderName: "example", ModelName: "cache-model",
		}, nil
	})
	agent := ai.NewAgent[struct{}, string](ai.NewProfiledModel(model, ai.ModelProfile{
		DefaultOutputMode: ai.OutputModeTool, DefaultCacheRetention: time.Hour,
	}), ai.WithCapabilities(ai.NewInstrumentation(ai.WithInstrumentationTracerProvider(provider))))
	first, err := agent.Run(ctx, "first turn", struct{}{})
	if err != nil {
		log.Fatal(err)
	}
	if _, err := agent.Run(ctx, "second turn", struct{}{}, ai.WithConversation(first.Conversation())); err != nil {
		log.Fatal(err)
	}
}
```

The second `chat` span reports an `unexpected` collapse and emits a `pydantic_ai.cache.collapse` event. You can use the event for alerts. The detector reads normalized provider usage; it does not inspect or hash your prompts.

| Request-span attribute | Meaning |
|---|---|
| `pydantic_ai.cache.hit_ratio` | `CacheReadTokens / InputTokens`. Input tokens include cache reads and writes. Zero input tokens give a zero ratio. |
| `pydantic_ai.cache.established_tokens` | The cached prefix later requests are judged against. It grows with reads plus writes and re-baselines after a collapse. |
| `pydantic_ai.cache.collapsed` | `true` when reads fall short by more than 5% and at least 2,000 tokens. Small misses and provider rounding do not count. |
| `pydantic_ai.cache.missed_tokens` | Previously established tokens not read back on a collapse. |
| `pydantic_ai.cache.collapse_reason` | The classification below. More values may be added. |
| `pydantic_ai.cache.not_enabled` | `true` when a model needs request-side configuration but a request of at least 4,096 input tokens has none, no explicit marker, and no cache usage. |

The collapse attributes are absent on healthy requests. Requests that have never established a cache and report neither reads nor writes normally have no cache attributes. A missing configuration on a cacheable model instead emits `pydantic_ai.cache.not_enabled` once per conversation and provider/endpoint/model key. The event carries `input_tokens`, plus provider and model names when available. Explicitly disabled caching, provider-local cache settings, and implicit provider caching suppress it. Use [`Caching`](caching.md) to enable portable caching. The run span has no cache ratio because an average across different models is misleading. Use its `gen_ai.aggregated_usage.*` counts if you need a run-level figure.

| Collapse reason | Meaning | Emits the event |
|---|---|---|
| `unexpected` | The prefix shrank while the known retention window should still have been active. | Yes |
| `ttl_expired` | The same provider, endpoint, and model have been idle beyond their retention window. | No |
| `compacted` | A new `CompactionPart` in the served history or response replaced the prefix by design. | No |
| `unknown` | No retention window is known, so expiry cannot be ruled out. | No |
| `unreported` | Neither reads nor writes were reported. This can mean caching was disabled or a full miss on a read-only-reporting provider. | No |

The event carries `established_tokens`, `cache_read_tokens`, and `missed_tokens`, plus `provider_name` and `model_name` when available. A sustained collapse records attributes on each request but emits the event once. A healthy ordinary read re-arms it. Unreported usage preserves the previous prefix, idle clock, and alert latch because it says nothing about the provider's cached copy.

Retention comes from `ResolveCacheRetention` with the served request's settings, then `ModelProfile.DefaultCacheRetention`. Known retention is extended by explicit `CachePoint` lifetimes, using the same rule as `PromptCacheOutlook`. Cache points do not make an unknown provider lifetime known. A `FallbackModel` without an explicit profile leaves retention unknown rather than guessing which child's profile applies.

Marks are shared within your process and keyed by conversation ID, provider, endpoint, and response model. Serialized history and `Conversation` snapshots keep the ID, so the first request of a continued turn is checked against the previous turn. New conversations and model or endpoint switches start clean. Switching back checks that cache's own mark and idle clock. Marks are forgotten after 24 idle hours or when 4,096 other conversations have reported cache usage more recently. An active run keeps its marks if the process store evicts them. Nothing is persisted or shared between workers.

Suspended continuations are one logical request. Cache health uses the final served segment, not the sum of segment usage; billing and usage telemetry still include every segment. Native-tool responses can sum cache reads across internal calls. They can prove a collapse but cannot raise the prefix or re-arm alerts. A reported single model pass without compaction, or a separate `tool_use_prompt_tokens` count, uses ordinary accounting. Responses rejected by a later hook and sampled-out requests still advance the marks.

`NewInstrumentedModel`, `RequestModel`, `StreamModel`, and the default agent request tracing use the same detector. Direct calls share marks only when their messages carry a conversation ID. Explicit `compact` spans do not record cache health. Only provider-native `CompactionPart` markers explain compaction; an application history rewrite can still produce an unexpected collapse.

## Protect content

Disable content before you send telemetry outside your trust boundary. Pass `WithInstrumentationContent(false)`, `WithInstrumentationBinaryContent(false)`, and `WithInstrumentationModelRequestParameters(false)` to `NewInstrumentation`.

`WithInstrumentationContent(false)` keeps message roles and part types but removes prompts, completions, tool arguments, tool results, and final output. This preserves trace structure without exporting user content.

Application run metadata is still exported in the `metadata` attribute because it is intended for trace filtering and evaluation. Do not put secrets in metadata. `WithInstrumentationBinaryContent(false)` redacts `BinaryContent` values nested inside metadata.

`WithInstrumentationBinaryContent(false)` keeps media types but removes inline bytes. It applies to files, retained `SpeechPart` audio, maps, slices, `ToolReturn`, and deferred metadata. It does not inspect fields inside your own struct types.

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

	ai "github.com/Kludex/pydantic-ai-go/ai"
	"github.com/Kludex/pydantic-ai-go/ai/models/openai"
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
