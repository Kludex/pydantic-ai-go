# pydantic-ai-go

Build typed LLM agents in Go.

`pydantic-ai-go` provides the agent loop, tools, structured output, streaming, usage limits, model middleware, and provider integrations. It follows PydanticAI's behavior while keeping the public API idiomatic for Go.

## Install

You need Go 1.25 or newer.

```console
go get github.com/Kludex/pydantic-ai-go/ai
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

	ai "github.com/Kludex/pydantic-ai-go/ai"
	"github.com/Kludex/pydantic-ai-go/ai/models/openai"
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

	ai "github.com/Kludex/pydantic-ai-go/ai"
	"github.com/Kludex/pydantic-ai-go/ai/models/openai"
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

`OutputModeAuto` is the default. It reads the selected model's `ModelProfile`; the standard profile chooses tool output because it works across providers. Use `WithOutputMode(ai.OutputModeNative)` to require native JSON Schema output, or `OutputModePrompted` to require JSON in text. See [Structured output](docs/outputs.md) for validation and multiple output alternatives.

## Stream a response

`RunStream` returns normalized events across all providers:

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

	ai "github.com/Kludex/pydantic-ai-go/ai"
	"github.com/Kludex/pydantic-ai-go/ai/models/openai"
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
| OpenAI Codex subscription | `openaicodex.NewModel("gpt-5.6-luna")` | Codex CLI or caller-owned credential source |
| Azure OpenAI | `azure.NewModel("deployment", azure.Config{})` | `AZURE_OPENAI_ENDPOINT`, `AZURE_OPENAI_API_KEY` |
| Amazon Bedrock | `bedrock.NewModel("us.amazon.nova-lite-v1:0")` | Standard AWS SDK configuration |
| Amazon Bedrock Mantle | `bedrockmantle.NewModel("openai.gpt-5.6-luna")` | `AWS_BEARER_TOKEN_BEDROCK` or standard AWS SDK configuration |
| Anthropic | `anthropic.NewModel("claude-sonnet-4-5")` | `ANTHROPIC_API_KEY` |
| Google Gemini | `google.NewModel("gemini-2.5-flash")` | `GOOGLE_API_KEY` |
| GitHub Copilot | `githubcopilot.NewModel("claude-haiku-4.5")` | `GITHUB_COPILOT_API_KEY` |
| Cohere | `cohere.NewModel("command-r7b-12-2024")` | `CO_API_KEY` |
| Crusoe | `crusoe.NewModel("openai/gpt-oss-120b")` | `CRUSOE_API_KEY` |
| Cerebras | `cerebras.NewModel("gpt-oss-120b")` | `CEREBRAS_API_KEY` |
| Groq | `groq.NewModel("openai/gpt-oss-20b")` | `GROQ_API_KEY` |
| Hugging Face | `huggingface.NewModel("Qwen/Qwen3-32B")` | `HF_TOKEN` |
| Mistral | `mistral.NewModel("mistral-large-latest")` | `MISTRAL_API_KEY` |
| Ollama | `ollama.NewModel("qwen3")` | `OLLAMA_BASE_URL`, `OLLAMA_API_KEY` |
| OpenRouter | `openrouter.NewModel("anthropic/claude-sonnet-4.6")` | `OPENROUTER_API_KEY` |
| Snowflake Cortex | `snowflake.NewModel("claude-sonnet-4-6")` | `SNOWFLAKE_ACCOUNT`, `SNOWFLAKE_TOKEN` |
| vLLM | `vllm.NewModel("Qwen/Qwen3-32B")` | `VLLM_BASE_URL`, `VLLM_API_KEY` |
| xAI | `xai.NewModel("grok-4.3")` | `XAI_API_KEY` |
| Z.AI | `zai.NewModel("glm-5.3-flash")` | `ZAI_API_KEY` |

Provider options configure caller-owned clients, endpoints, credentials, and default model settings. See [Provider configuration](docs/providers.md) for OpenAI-compatible endpoints and Azure API versions. See [OpenAI Codex subscription](docs/openai-codex.md) for read-only CLI credentials, OAuth PKCE, and rotated credential storage. See [Amazon Bedrock](docs/bedrock.md) for AWS SDK configuration and Converse behavior.

## Test an agent without network calls

Use `ai/models/fakes` in tests:

```go
package weather_test

import (
	"context"
	"testing"

	ai "github.com/Kludex/pydantic-ai-go/ai"
	"github.com/Kludex/pydantic-ai-go/ai/models/fakes"
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
| Name and describe an agent for integrations | `WithAgentName`, `WithAgentDescription`, and `WithAgentDescriptionFunc` |
| Attach application metadata to a run | `WithMetadata`, `AddMetadataFunc`, and `WithRunMetadata` |
| Transform structured data or plain text | [`NewOutputFunction` or `NewTextOutputFunction`](docs/outputs.md) |
| Return one of several typed outputs | [`NewUnionOutput` and `NewUnionAgent`](docs/outputs.md) |
| Render instructions from typed dependencies | [`PromptTemplate` and `FormatAsXML`](docs/prompt-templates.md) |
| Address and rewrite instruction blocks | [Stable instruction IDs](docs/instructions.md) |
| Reuse or combine tools | `NewTool`, `NewFunctionToolset`, and the toolset wrappers |
| Delegate work to another agent | [`ToolReturn.Usage` and shared dependencies](docs/delegation.md) |
| Require approval or external execution | [`WithApprovalRequired` and `DeferredToolResults`](docs/deferred-execution.md) |
| Connect an MCP server | [`mcp.NewStreamableHTTPToolset`, `Connect`, or `LoadToolsets`](docs/mcp.md) |
| Expose an agent through MCP | [An official SDK server with a typed agent tool](docs/mcp-server.md) |
| Stream an agent to an AG-UI frontend | [`agui.Adapter`](docs/ag-ui.md) |
| Stream an agent to Vercel AI UI | [`vercel.Adapter`](docs/vercel-ai.md) |
| Serve an agent over A2A | [`a2a.Executor`](docs/a2a.md) |
| Chat in a terminal or browser | [`cli.Run` and `webchat.NewHandler`](docs/chat.md) |
| Run a speech-to-speech session | [`realtime.Open`](docs/realtime.md) |
| Build a durable operation backend | [`durable.Backend` and `durable.Operation`](docs/durable.md) |
| Limit usage or cost | [`UsageLimits` and pre-request token counting](docs/usage.md) |
| Sanitize, trim, or summarize conversation history | [`SanitizeMessages` and history capabilities](docs/history.md) |
| Compact old provider history | [`ModelCompactor` and compaction capabilities](docs/compaction.md) |
| Send images, documents, audio, video, or uploaded files | [`RunParts`](docs/multimodal.md) |
| Use native-first web tools with local fallbacks | [`NewWebSearchCapability` and `NewWebFetchCapability`](docs/native-tools.md) |
| Configure a provider or compatible endpoint | [Provider configuration](docs/providers.md) |
| Use Amazon Bedrock Converse | [`bedrock.NewModel`](docs/bedrock.md) |
| Retry transient provider HTTP failures | [`retries.Transport`](docs/retries.md) |
| Add fallback models | `NewFallbackModel` |
| Limit concurrency | `NewConcurrencyLimiter` |
| Add agent middleware | [`Capability` and its focused hook interfaces](docs/capabilities.md) |
| Wrap one model | [`ModelWrapper` and built-in decorators](docs/model-wrappers.md) |
| Add OpenTelemetry or send telemetry to Logfire | [`NewInstrumentation` or `NewInstrumentedModel`](docs/observability.md) |
| Evaluate an agent | [`evals.NewTextTask` or `evals.NewTask`](docs/evals.md) |
| Generate query or document vectors | [`embeddings.Embedder`](docs/embeddings.md) |
| Generate or edit images directly | [`images.Generator`](docs/images.md) |
| Call a model without an agent | `RequestModel` or `StreamModel` |
| Drive a run one event at a time | `AgentRun` |

Run options are detached from agent configuration. They are safe to use in concurrent runs. See the [Go package documentation](https://pkg.go.dev/github.com/Kludex/pydantic-ai-go/ai) for each public contract and [`CHECKLIST.md`](CHECKLIST.md) for remaining parity work.

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
go test -run '^$' -bench . -benchmem ./ai
```

The project requires 100% statement coverage for every tested package. See the [benchmark guide](docs/benchmarks.md) to compare performance changes.

## Status

The API is under active development while it approaches feature parity with [PydanticAI](https://ai.pydantic.dev/). Review [`CHECKLIST.md`](CHECKLIST.md) before depending on a provider-specific or advanced feature.

See the [`COMPATIBILITY.md`](COMPATIBILITY.md) version guarantees and [`CHANGELOG.md`](CHANGELOG.md) before you upgrade.

## License

This project is licensed under the MIT License. See [`LICENSE`](LICENSE).
