# Provider configuration

Choose a model package and pass the model to your agent. Each package reads its standard API key environment variable by default.

## OpenAI

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
	agent := ai.NewAgent[struct{}, string](openai.NewModel("gpt-5"))
	result, err := agent.Run(context.Background(), "Explain why the sky is blue in one sentence.", struct{}{})
	if err != nil {
		log.Fatal(err)
	}
	fmt.Println(result.Output)
}
```

Set `OPENAI_API_KEY`. You can also set `OPENAI_BASE_URL` when a proxy preserves OpenAI's provider identity and behavior.

Use `openai.NewResponsesModel` instead of `openai.NewModel` when you need the Responses API. Both models accept image, document, and supported audio input. See [Multimodal input](multimodal.md) for provider-specific URL and inline-data behavior.

Responses assistant phases are retained in `TextPart.ProviderDetails["phase"]`. Same-provider history replays `commentary` and `final_answer` phases for `gpt-5.3-codex`, `gpt-5.4`, `gpt-5.5`, and `gpt-5.6` model families. Use `openai.WithResponsesPhaseSupport(true)` for a compatible gateway or future model. Use `false` when an endpoint rejects the field.

### Predicted output

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
	settings, err := (openai.Settings{
		Prediction: &openai.Prediction{Content: "package main\n\nfunc main() {\n}\n"},
	}).Build()
	if err != nil {
		log.Fatal(err)
	}

	agent := ai.NewAgent[struct{}, string](
		openai.NewModel("gpt-4.1"),
		ai.WithModelSettings(settings),
	)
	result, err := agent.Run(context.Background(), "Add a greeting to this Go program.", struct{}{})
	if err != nil {
		log.Fatal(err)
	}
	fmt.Println(result.Output)
}
```

`Prediction` sends expected output to a compatible Chat Completions model. The provider can generate matching content faster and reports accepted and rejected prediction tokens in usage. Use `ContentParts` when the expected output has separately cacheable text boundaries. Predicted output is not supported by the Responses API.

### Prompt caching

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
	settings, err := (openai.Settings{
		PromptCacheKey:       "product-reference",
		PromptCacheRetention: openai.PromptCacheRetention24Hours,
		PromptCacheOptions: &openai.PromptCacheOptions{
			Mode: openai.PromptCacheModeExplicit,
			TTL:  openai.PromptCacheTTL30Minutes,
		},
	}).Build()
	if err != nil {
		log.Fatal(err)
	}

	agent := ai.NewAgent[struct{}, string](
		openai.NewResponsesModel("gpt-5.6"),
		ai.WithModelSettings(settings),
	)
	result, err := agent.RunParts(context.Background(), []ai.UserContent{
		ai.TextContent{Text: "Stable product reference."},
		ai.CachePoint{},
		ai.TextContent{Text: "Summarize the reference."},
	}, struct{}{})
	if err != nil {
		log.Fatal(err)
	}
	fmt.Println(result.Output)
}
```

`PromptCacheOptions` controls request-wide GPT-5.6 caching for Chat Completions and Responses. OpenAI applies its 30-minute TTL to every explicit `CachePoint` and ignores each marker's portable TTL. `PromptCacheRetention24Hours` requests the legacy maximum retention independently. `ai.ResolvePromptCacheRetention` reports the longest requested lifetime for durable backends without treating in-memory caching as durable.

## OpenAI-compatible endpoints

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
	model := openai.NewModel("local-model", openai.WithProvider(openai.ProviderConfig{
		Name:    "local",
		BaseURL: "http://localhost:8000/v1",
	}))
	agent := ai.NewAgent[struct{}, string](model)
	result, err := agent.Run(context.Background(), "Say hello.", struct{}{})
	if err != nil {
		log.Fatal(err)
	}
	fmt.Println(result.Output)
}
```

`ProviderConfig.Name` is stored in message history and telemetry. Set it to a stable provider identifier. An empty `APIKey` omits the `Authorization` header, which supports local servers without authentication.

Use `ProviderConfig.Headers` and `ProviderConfig.Query` for provider-wide values. Use `ProviderConfig.PrepareRequest` for credentials that must be refreshed before each request. Per-request `ModelSettings.ExtraHeaders` are applied last.

Disable unsupported behavior explicitly:

```go
package main

import (
	"fmt"

	"github.com/Kludex/pydantic-ai-go/models/openai"
)

func main() {
	model := openai.NewResponsesModel(
		"compatible-model",
		openai.WithProvider(openai.ProviderConfig{
			Name:    "compatible",
			BaseURL: "https://example.com/v1",
			APIKey:  "your-api-key",
		}),
		openai.WithStrictToolSupport(false),
		openai.WithDeferredToolSupport(false),
	)
	fmt.Println(model.Name())
}
```

The compatibility layer sends OpenAI wire formats. It cannot make an endpoint support OpenAI features that the endpoint does not implement.

## OpenRouter

```go
package main

import (
	"context"
	"fmt"
	"log"

	ai "github.com/Kludex/pydantic-ai-go"
	"github.com/Kludex/pydantic-ai-go/models/openrouter"
)

func main() {
	settings, err := (openrouter.Settings{
		Provider: &openrouter.ProviderRouting{
			Only:           []string{"anthropic"},
			DataCollection: openrouter.DataCollectionDeny,
		},
		Usage:               &openrouter.UsageConfig{Include: true},
		CacheInstructions:   openrouter.CacheTTL1Hour,
		CacheToolDefinitions: openrouter.CacheTTL5Minutes,
	}).Build()
	if err != nil {
		log.Fatal(err)
	}

	agent := ai.NewAgent[struct{}, string](
		openrouter.NewModel("anthropic/claude-sonnet-4.6"),
		ai.WithModelSettings(settings),
		ai.WithNativeTools(ai.WebSearchTool{}),
	)
	result, err := agent.Run(context.Background(), "Find the latest PydanticAI release.", struct{}{})
	if err != nil {
		log.Fatal(err)
	}
	fmt.Println(result.Output)
}
```

Set `OPENROUTER_API_KEY`. Model names must use OpenRouter's `provider/model` form. The model uses `max_tokens`, sends portable thinking settings through OpenRouter's `reasoning` extension, and retains the routed provider, native finish reason, annotations, server-tool usage, and provider-reported cost.

OpenRouter reasoning details become separate `ThinkingPart` values. Text, summaries, encrypted signatures, stable IDs, formats, and indexes survive same-provider history replay and streaming. Embedded provider errors, transient responses without a completion, and nested provider responses remain inspectable model API failures or normalized responses instead of decoding errors.

The OpenRouter model supports native web search and advisor declarations. OpenRouter ignores `AdvisorTool.MaxUses` and `AdvisorTool.Caching`; it maps `MaxTokens` to `max_completion_tokens`. Use `WithAppAttribution` or `OPENROUTER_APP_URL` and `OPENROUTER_APP_TITLE` to identify your application.

`VideoURL` and inline `video/*` binary content use OpenRouter's `video_url` extension. `DocumentURL` uses OpenRouter's remote file support. Audio URLs are downloaded before they are sent. See [Multimodal input](multimodal.md) for safe forced downloads.

Use `openrouter.Settings` for fallback models, provider routing, presets, context transforms, reasoning, extended usage, and prompt caching. `CacheInstructions`, `CacheMessages`, and `CacheToolDefinitions` add explicit cache boundaries only for supported downstream providers. Anthropic receives the selected TTL and keeps a static instruction boundary before dynamic instructions. Gemini receives message or stable-instruction boundaries without an unsupported TTL. Other routed providers ignore these settings.

`Settings.Build` validates conflicts with `ExtraBody` and returns a detached `ModelSettings` snapshot.

OpenRouter adjusts JSON Schema for the routed provider. Google routes inline definitions, simplify nullable unions, stringify enum values, convert `oneOf`, move string formats into descriptions, and remove unsupported fields. Qwen, Amazon, and Meta routes inline non-recursive definitions. Recursive definitions remain referenced instead of expanding forever.

Anthropic routes cannot combine reasoning with forced tool choice. When structured output infers a required output tool, the model sends `tool_choice: "auto"` and keeps reasoning enabled. An explicit required or list choice fails before transport instead of letting OpenRouter silently remove reasoning.

## Z.AI

```go
package main

import (
	"context"
	"fmt"
	"log"

	ai "github.com/Kludex/pydantic-ai-go"
	"github.com/Kludex/pydantic-ai-go/models/zai"
)

func main() {
	clearThinking := false
	settings, err := (zai.Settings{
		Common: ai.ModelSettings{
			Thinking: &ai.ThinkingSettings{Level: ai.ThinkingLevelMedium},
		},
		ClearThinking: &clearThinking,
	}).Build()
	if err != nil {
		log.Fatal(err)
	}

	agent := ai.NewAgent[struct{}, string](
		zai.NewModel("glm-5.3-flash"),
		ai.WithModelSettings(settings),
	)
	result, err := agent.Run(context.Background(), "Explain why the sky is blue.", struct{}{})
	if err != nil {
		log.Fatal(err)
	}
	fmt.Println(result.Output)
}
```

Set `ZAI_API_KEY`.

Z.AI returns reasoning in `reasoning_content`. The model stores it as `ThinkingPart`. Later turns send it back unchanged to the same provider.

Thinking-capable GLM models use `clear_thinking: false` by default. This preserves earlier reasoning for multi-turn consistency. Set `Settings.ClearThinking` to `true` to clear it instead.

GLM 5.2 accepts the portable effort names. GLM 5.3 only accepts `low`, `high`, and `max`. The model maps `minimal`, `medium`, and `xhigh` to the nearest supported value.

The model normalizes the `sensitive`, `model_context_window_exceeded`, and `network_error` finish reasons.

Use `zai.WithProvider` for a gateway. The configured name becomes the persisted provider identity. Z.AI reasoning and finish-reason behavior stays enabled.

## Azure OpenAI and Azure AI Foundry

```go
package main

import (
	"context"
	"fmt"
	"log"

	ai "github.com/Kludex/pydantic-ai-go"
	"github.com/Kludex/pydantic-ai-go/models/azure"
)

func main() {
	model, err := azure.NewResponsesModel("my-deployment", azure.Config{})
	if err != nil {
		log.Fatal(err)
	}
	agent := ai.NewAgent[struct{}, string](model)
	result, err := agent.Run(context.Background(), "Say hello.", struct{}{})
	if err != nil {
		log.Fatal(err)
	}
	fmt.Println(result.Output)
}
```

Set `AZURE_OPENAI_ENDPOINT` and `AZURE_OPENAI_API_KEY`. Use an endpoint ending in `/openai/v1` for Azure's current OpenAI-compatible API. Azure AI Foundry serverless hosts ending in `.models.ai.azure.com` are normalized to `/v1` automatically.

Legacy Azure deployment endpoints also require `OPENAI_API_VERSION`:

```go
package main

import (
	"fmt"
	"log"

	"github.com/Kludex/pydantic-ai-go/models/azure"
)

func main() {
	model, err := azure.NewModel("my-deployment", azure.Config{
		Endpoint:   "https://my-resource.openai.azure.com",
		APIKey:     "your-api-key",
		APIVersion: "2025-04-01-preview",
	})
	if err != nil {
		log.Fatal(err)
	}
	fmt.Println(model.Name())
}
```

Use `Config.TokenProvider` instead of `Config.APIKey` for Microsoft Entra ID. The callback runs for every request and receives that request's context.

Azure Chat Completions rejects document input before transport. Use `azure.NewResponsesModel` when you need document URLs, uploaded documents, or inline document data.

## Anthropic

```go
package main

import (
	"context"
	"fmt"
	"log"

	ai "github.com/Kludex/pydantic-ai-go"
	"github.com/Kludex/pydantic-ai-go/models/anthropic"
)

func main() {
	agent := ai.NewAgent[struct{}, string](anthropic.NewModel("claude-sonnet-4-5"))
	result, err := agent.Run(context.Background(), "Say hello.", struct{}{})
	if err != nil {
		log.Fatal(err)
	}
	fmt.Println(result.Output)
}
```

Set `ANTHROPIC_API_KEY`. Use `anthropic.WithBaseURL` and `anthropic.WithHTTPClient` for a compatible gateway.

Anthropic accepts image URLs, PDF URLs, inline images, inline PDFs, and plain-text documents. Forced URL downloads use the shared SSRF protections. Anthropic does not accept audio or video input.

Supported Claude 4.1, 4.5, 4.6, 4.7, 4.8, and 5 families accept `OutputModeNative`. The provider sends your output schema through `output_config.format`. Unsupported models fail before transport instead of silently ignoring the schema.

### Prompt caching

```go
package main

import (
	"context"
	"fmt"
	"log"

	ai "github.com/Kludex/pydantic-ai-go"
	"github.com/Kludex/pydantic-ai-go/models/anthropic"
)

func main() {
	settings, err := (anthropic.Settings{
		Cache:             anthropic.CacheTTL1Hour,
		CacheInstructions: anthropic.CacheTTL5Minutes,
	}).Build()
	if err != nil {
		log.Fatal(err)
	}

	agent := ai.NewAgent[struct{}, string](
		anthropic.NewModel("claude-sonnet-4-6"),
		ai.WithInstructions("Use the product reference exactly."),
		ai.WithModelSettings(settings),
	)
	result, err := agent.Run(context.Background(), "Summarize the product.", struct{}{})
	if err != nil {
		log.Fatal(err)
	}
	fmt.Println(result.Output)
}
```

`Cache` lets Anthropic move one automatic breakpoint forward as the conversation grows. `CacheInstructions`, `CacheMessages`, and `CacheToolDefinitions` place explicit boundaries. Static instructions are cached before dynamic instructions. Automatic and explicit message caching are mutually exclusive. Anthropic keeps at most four cache points and removes the oldest message boundaries after reserving instruction, tool, and automatic slots.

## Google Gemini

```go
package main

import (
	"context"
	"fmt"
	"log"

	ai "github.com/Kludex/pydantic-ai-go"
	"github.com/Kludex/pydantic-ai-go/models/google"
)

func main() {
	agent := ai.NewAgent[struct{}, string](google.NewModel("gemini-2.5-flash"))
	result, err := agent.Run(context.Background(), "Say hello.", struct{}{})
	if err != nil {
		log.Fatal(err)
	}
	fmt.Println(result.Output)
}
```

Set `GOOGLE_API_KEY` or the legacy `GEMINI_API_KEY`. `GOOGLE_API_KEY` takes precedence. Use `google.WithBaseURL` and `google.WithHTTPClient` for a compatible Gemini Developer API gateway.

Gemini accepts image, document, audio, and video content. Ordinary URLs are downloaded with SSRF protection. Gemini Files API URLs and YouTube videos are sent directly.

## Google Cloud Vertex AI

```go
package main

import (
	"context"
	"fmt"
	"log"

	ai "github.com/Kludex/pydantic-ai-go"
	"github.com/Kludex/pydantic-ai-go/models/google"
)

func main() {
	model, err := google.NewVertexModel("gemini-2.5-flash", google.VertexConfig{})
	if err != nil {
		log.Fatal(err)
	}
	agent := ai.NewAgent[struct{}, string](model)
	result, err := agent.Run(context.Background(), "Say hello.", struct{}{})
	if err != nil {
		log.Fatal(err)
	}
	fmt.Println(result.Output)
}
```

Set `GOOGLE_CLOUD_PROJECT`. The location defaults to `GOOGLE_CLOUD_LOCATION` or `global`. The model uses Application Default Credentials and the `cloud-platform` scope.

Pass `VertexConfig.APIKey` for Vertex AI Express Mode. Pass `VertexConfig.TokenProvider` for workload identity or another application-owned credential flow. API keys and token providers are mutually exclusive.

Routing follows the configured transport. `global` uses `aiplatform.googleapis.com`. The `us` and `eu` multi-regions use their data-residency endpoints. Regional locations use `<location>-aiplatform.googleapis.com`. Vertex responses keep the `google-cloud` identity in histories, pricing, and telemetry.

Use `google.WithProvider` for a gateway or preconfigured transport. `ProviderConfig.Transport` controls Vertex-specific behavior. `ProviderConfig.Name` is independent persisted identity. This separation keeps history replay correct when a gateway's identity does not match its underlying transport.

Portable `ServiceTierFlex` and `ServiceTierPriority` values become Vertex spillover headers. `ServiceTierDefault` explicitly selects shared on-demand capacity. Per-request `ExtraHeaders` are applied last.

Vertex sends file URLs directly unless you set a forced download mode. YouTube and `gs://` video references remain provider-hosted.

## Request settings

Set portable defaults on the model when every agent should share them:

```go
package main

import (
	"fmt"

	ai "github.com/Kludex/pydantic-ai-go"
	"github.com/Kludex/pydantic-ai-go/models/openai"
)

func main() {
	temperature := 0.2
	model := openai.NewModel("gpt-5", openai.WithDefaultSettings(ai.ModelSettings{
		MaxTokens:   500,
		Temperature: &temperature,
	}))
	fmt.Println(model.Name())
}
```

Agent settings override model defaults. Capability and run settings override both. Unsupported portable fields are omitted by each provider.

## Structured-output profiles

Wrap a model when an OpenAI-compatible endpoint needs different structured-output defaults:

```go
package main

import (
	"fmt"

	ai "github.com/Kludex/pydantic-ai-go"
	"github.com/Kludex/pydantic-ai-go/models/openai"
)

func main() {
	model := ai.NewProfiledModel(
		openai.NewModel("compatible-model", openai.WithBaseURL("https://example.com/v1")),
		ai.ModelProfile{
			DefaultOutputMode:      ai.OutputModePrompted,
			PromptedOutputTemplate: "Return JSON matching this schema:\n{schema}",
		},
	)
	fmt.Println(model.Name())
}
```

`OutputModeAuto` uses this profile. It is the default for reflected structured output. An explicit agent or run output mode always wins. Adaptive model selection resolves the profile after selecting each model, and fallback chains resolve it separately for every attempted model.

Set `NativeOutputRequiresPrompt` when an endpoint supports native JSON Schema output but also requires the schema in its instructions. `NewProfiledModel` preserves streaming, lifecycle, tool-search, continuation, and other optional model behavior.

## Groq

```go
package main

import (
	"context"
	"fmt"
	"log"

	ai "github.com/Kludex/pydantic-ai-go"
	"github.com/Kludex/pydantic-ai-go/models/groq"
)

func main() {
	settings, err := (groq.Settings{
		ReasoningFormat: groq.ReasoningFormatParsed,
		ReasoningEffort: groq.ReasoningEffortHigh,
	}).Build()
	if err != nil {
		log.Fatal(err)
	}

	agent := ai.NewAgent[struct{}, string](
		groq.NewModel("openai/gpt-oss-20b"),
		ai.WithModelSettings(settings),
	)
	result, err := agent.Run(context.Background(), "Explain speculative decoding.", struct{}{})
	if err != nil {
		log.Fatal(err)
	}
	fmt.Println(result.Output)
}
```

Set `GROQ_API_KEY`. `GROQ_BASE_URL` overrides the default `https://api.groq.com/openai/v1` endpoint.

`ReasoningFormatParsed` returns reasoning as separate `ThinkingPart` values. Reasoning effort support depends on the selected Groq model family. `WithProvider` keeps Groq response parsing when you route requests through a gateway.
