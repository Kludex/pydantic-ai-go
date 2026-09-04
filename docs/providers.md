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

Chat Completions accepts provider-hosted documents as `ai.UploadedFile` values with `ProviderName: "openai"`. Uploaded image IDs are not valid Chat image inputs. Use an image URL, inline image data, or the Responses API instead.

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

## Cohere

```go
package main

import (
	"context"
	"fmt"
	"log"

	ai "github.com/Kludex/pydantic-ai-go"
	"github.com/Kludex/pydantic-ai-go/models/cohere"
)

func main() {
	topK := 20
	settings, err := (cohere.Settings{TopK: &topK}).Build()
	if err != nil {
		log.Fatal(err)
	}
	agent := ai.NewAgent[struct{}, string](
		cohere.NewModel("command-r7b-12-2024"),
		ai.WithModelSettings(settings),
	)
	result, err := agent.Run(context.Background(), "Explain why the sky is blue.", struct{}{})
	if err != nil {
		log.Fatal(err)
	}
	fmt.Println(result.Output)
}
```

Set `CO_API_KEY`. Set `CO_BASE_URL` for a compatible endpoint.

The model uses Cohere's v2 Chat API. It supports text, thinking, function tools, tool output, structured output through a function tool, Cohere finish reasons, billed-unit details, and cached-token usage. Cohere does not expose streaming or multimodal input through this adapter. Unsupported native tools, native JSON Schema output, and multimodal content fail before transport.

Use `cohere.Settings.TopK` for Cohere's `k` sampling setting. Portable max tokens, stop sequences, temperature, top-p, seed, presence penalty, and frequency penalty map directly.

## Crusoe

```go
package main

import (
	"context"
	"fmt"
	"log"

	ai "github.com/Kludex/pydantic-ai-go"
	"github.com/Kludex/pydantic-ai-go/models/crusoe"
)

func main() {
	agent := ai.NewAgent[struct{}, string](crusoe.NewModel("openai/gpt-oss-120b"))
	result, err := agent.Run(context.Background(), "Explain why the sky is blue.", struct{}{})
	if err != nil {
		log.Fatal(err)
	}
	fmt.Println(result.Output)
}
```

Set `CRUSOE_API_KEY`. Model names include the vendor prefix.

Crusoe uses the OpenAI-compatible Chat Completions wire format. Every served model supports guided JSON Schema output. The adapter normalizes the `reasoning` field used by most families and the `reasoning_content` field used by DeepSeek. Static and streamed text, reasoning, function tools, settings, and usage use the shared OpenAI-compatible lifecycle.

## Cerebras

```go
package main

import (
	"context"
	"fmt"
	"log"

	ai "github.com/Kludex/pydantic-ai-go"
	"github.com/Kludex/pydantic-ai-go/models/cerebras"
)

func main() {
	agent := ai.NewAgent[struct{}, string](cerebras.NewModel("gpt-oss-120b"))
	result, err := agent.Run(context.Background(), "Explain why the sky is blue.", struct{}{})
	if err != nil {
		log.Fatal(err)
	}
	fmt.Println(result.Output)
}
```

Set `CEREBRAS_API_KEY`.

Cerebras returns reasoning either through the OpenAI-compatible `reasoning` field or inside `<think>` tags. Both forms become `ThinkingPart`. GLM history replays reasoning with the tags Cerebras requires and defaults `clear_thinking` to `false`. Portable disabled thinking maps to `reasoning_effort: "none"` for GLM. GPT-OSS always reasons, so disabled thinking is omitted instead of sending an invalid setting. Cerebras documents `logit_bias` but does not apply it, so the adapter omits the portable field.

Use `cerebras.Settings` for the typed `ClearThinking` and legacy `DisableReasoning` controls. `Settings.Build` returns detached `ai.ModelSettings` and rejects conflicts with `ExtraBody`.

## Hugging Face Inference Providers

```go
package main

import (
	"context"
	"fmt"
	"log"

	ai "github.com/Kludex/pydantic-ai-go"
	"github.com/Kludex/pydantic-ai-go/models/huggingface"
)

func main() {
	model := huggingface.NewModel(
		"Qwen/Qwen3-32B",
		huggingface.WithInferenceProvider("together"),
	)
	agent := ai.NewAgent[struct{}, string](model)
	result, err := agent.Run(context.Background(), "Explain why the sky is blue.", struct{}{})
	if err != nil {
		log.Fatal(err)
	}
	fmt.Println(result.Output)
}
```

Set `HF_TOKEN`. The default endpoint lets Hugging Face select a provider automatically. Use `WithInferenceProvider` to route through a named provider or `WithBaseURL` for an explicit OpenAI-compatible endpoint.

The adapter supports static and streamed text, function tools, URL or inline image input, usage, finish reasons, and `<think>` reasoning. Reasoning becomes `ThinkingPart` and replays with tags on later same-provider turns. Hugging Face does not expose native JSON Schema output through this API. Use the default tool output or prompted output instead. Audio, video, documents, uploaded files, and non-image binary input fail before transport.

## Mistral

```go
package main

import (
	"context"
	"fmt"
	"log"

	ai "github.com/Kludex/pydantic-ai-go"
	"github.com/Kludex/pydantic-ai-go/models/mistral"
)

func main() {
	model := mistral.NewModel("mistral-large-latest")
	agent := ai.NewAgent[struct{}, string](model)
	result, err := agent.Run(context.Background(), "Explain why the sky is blue.", struct{}{})
	if err != nil {
		log.Fatal(err)
	}
	fmt.Println(result.Output)
}
```

Set `MISTRAL_API_KEY`. Set `MISTRAL_BASE_URL` only when you use a compatible gateway.

The adapter uses Mistral's native Chat Completions contract. It supports static and streamed text, thinking chunks, function tools, tool-based structured output, prompt cache keys, tool selection, parallel calls, usage, and portable generation settings.

Mistral accepts direct or downloaded images, inline images, direct or downloaded PDFs, and downloaded text documents. Audio, video, uploaded-file references, and other binary media fail before transport. Models in the Mistral Small 4 and Medium 3.5 families map enabled reasoning to `high` and disabled reasoning to `none`. Magistral reasoning is always enabled, so the adapter omits `reasoning_effort`.

## Ollama

```go
package main

import (
	"context"
	"fmt"
	"log"

	ai "github.com/Kludex/pydantic-ai-go"
	"github.com/Kludex/pydantic-ai-go/models/ollama"
)

func main() {
	agent := ai.NewAgent[struct{}, string](ollama.NewModel("qwen3"))
	result, err := agent.Run(context.Background(), "Explain why the sky is blue.", struct{}{})
	if err != nil {
		log.Fatal(err)
	}
	fmt.Println(result.Output)
}
```

The model connects to `http://localhost:11434/v1` by default. Set `OLLAMA_BASE_URL` or `OLLAMA_HOST` to use another server. Set `OLLAMA_API_KEY` when the endpoint requires authentication.

Ollama exposes Chat Completions through its OpenAI-compatible endpoint. The adapter preserves `reasoning` as `ThinkingPart`, sends `max_tokens`, rejects unsupported documents, and disables strict function-tool extensions. Local Ollama supports native JSON Schema output. Ollama Cloud does not enforce supplied schemas, so native output fails before transport there. Use tool or prompted output instead.

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

## Snowflake Cortex

```go
package main

import (
	"context"
	"fmt"
	"log"

	ai "github.com/Kludex/pydantic-ai-go"
	"github.com/Kludex/pydantic-ai-go/models/snowflake"
)

func main() {
	model := snowflake.NewModel("claude-sonnet-4-6")
	agent := ai.NewAgent[struct{}, string](model)
	result, err := agent.Run(context.Background(), "Explain why the sky is blue.", struct{}{})
	if err != nil {
		log.Fatal(err)
	}
	fmt.Println(result.Output)
}
```

Set `SNOWFLAKE_ACCOUNT` and `SNOWFLAKE_TOKEN`. `SNOWFLAKE_ACCOUNT` accepts a bare account identifier or its `snowflakecomputing.com` hostname. Use `SNOWFLAKE_BASE_URL` for private connectivity.

Snowflake Cortex exposes an OpenAI-compatible Chat Completions endpoint inside your Snowflake account. The adapter supports static and streamed text, reasoning details, function tools, JSON Schema output, generation settings, and usage for Claude and OpenAI model families. Claude models use Cortex's `reasoning` object and default to temperature `1` when reasoning is enabled.

Cortex does not support function tools or native JSON Schema output for its Llama, Mistral, Mixtral, DeepSeek, and Snowflake model families. The adapter rejects those requests before transport. Use prompted output for those models.

## xAI

```go
package main

import (
	"context"
	"fmt"
	"log"

	ai "github.com/Kludex/pydantic-ai-go"
	"github.com/Kludex/pydantic-ai-go/models/xai"
)

func main() {
	agent := ai.NewAgent[struct{}, string](
		xai.NewModel("grok-4.3"),
		ai.WithNativeTools(ai.XSearchTool{IncludeOutput: true}),
	)
	result, err := agent.Run(context.Background(), "What are people saying about Go on X?", struct{}{})
	if err != nil {
		log.Fatal(err)
	}
	fmt.Println(result.Output)
}
```

Set `XAI_API_KEY`.

The adapter uses xAI's Responses-compatible HTTP endpoint because xAI does not publish an official Go SDK. It supports static and streamed text, encrypted reasoning, function tools, native JSON Schema output, uploaded files, log probabilities, usage, and xAI conversation settings. Direct binary documents and document URLs are safely uploaded through the xAI Files API before generation.

Grok 4, code, and build models support hosted web search, X search, code execution, MCP servers, and managed collections search. `xai.Settings` configures output inclusion, stored-response continuity, reasoning effort, server-side turn limits, and multi-agent counts. Unsupported required native tools fail before transport.

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

### Adaptive thinking

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
		Common: ai.ModelSettings{Thinking: &ai.ThinkingSettings{Level: ai.ThinkingLevelHigh}},
		Effort: anthropic.EffortHigh,
	}).Build()
	if err != nil {
		log.Fatal(err)
	}
	agent := ai.NewAgent[struct{}, string](
		anthropic.NewModel("claude-sonnet-4-6"),
		ai.WithModelSettings(settings),
	)
	result, err := agent.Run(context.Background(), "Solve the problem carefully.", struct{}{})
	if err != nil {
		log.Fatal(err)
	}
	fmt.Println(result.Output)
}
```

Claude Sonnet 4.6+, Opus 4.6+, Fable 5, and Mythos 5 use adaptive thinking. Older models receive a token budget. Portable thinking levels select provider effort automatically, while `anthropic.Settings.Effort` provides an explicit override. Unsupported budget, sampling, effort, and forced-tool combinations fail or are omitted according to the model profile.

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

### Code execution containers

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
		Container: &anthropic.Container{Skills: []anthropic.ContainerSkill{{
			Type: "anthropic", SkillID: "xlsx", Version: "latest",
		}}},
		CodeExecutionToolVersion: anthropic.CodeExecutionToolVersionAuto,
	}).Build()
	if err != nil {
		log.Fatal(err)
	}
	agent := ai.NewAgent[struct{}, string](
		anthropic.NewModel("claude-sonnet-4-6"),
		ai.WithModelSettings(settings),
		ai.WithNativeTools(ai.CodeExecutionTool{}),
	)
	result, err := agent.Run(context.Background(), "Create a spreadsheet summary.", struct{}{})
	if err != nil {
		log.Fatal(err)
	}
	fmt.Println(result.Output)
}
```

Use `Container.ID` to select an existing container. Set `FreshContainer` to ignore container IDs in message history. The automatic code execution version chooses `20260120` only for supported models and otherwise uses `20250825`; forcing an unsupported version fails before transport.

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

### Cached content

```go
package main

import (
	"context"
	"log"

	ai "github.com/Kludex/pydantic-ai-go"
	"github.com/Kludex/pydantic-ai-go/models/google"
)

func main() {
	settings, err := (google.Settings{CachedContent: "cachedContents/example"}).Build()
	if err != nil {
		log.Fatal(err)
	}
	agent := ai.NewAgent[struct{}, string](
		google.NewModel("gemini-2.5-flash"),
		ai.WithModelSettings(settings),
	)
	if _, err := agent.Run(context.Background(), "Use the cached context.", struct{}{}); err != nil {
		log.Fatal(err)
	}
}
```

Create the cached-content resource with Gemini or Vertex before the run. The resource owns system instructions and tools. The provider omits local system and tool declarations because Google rejects requests that combine them with `cachedContent`.

### Model Armor

```go
package main

import (
	"context"
	"log"

	ai "github.com/Kludex/pydantic-ai-go"
	"github.com/Kludex/pydantic-ai-go/models/google"
)

func main() {
	settings, err := (google.Settings{ModelArmor: &google.ModelArmorConfig{
		PromptTemplateName:   "projects/project/locations/global/templates/prompt",
		ResponseTemplateName: "projects/project/locations/global/templates/response",
	}}).Build()
	if err != nil {
		log.Fatal(err)
	}
	model, err := google.NewVertexModel("gemini-2.5-flash", google.VertexConfig{})
	if err != nil {
		log.Fatal(err)
	}
	agent := ai.NewAgent[struct{}, string](model, ai.WithModelSettings(settings))
	if _, err := agent.Run(context.Background(), "Screen this request.", struct{}{}); err != nil {
		log.Fatal(err)
	}
}
```

Model Armor is available only with Vertex AI. Google applies it only to non-streaming generation, so `StreamRequest` omits the configuration. The Gemini Developer API rejects the setting before transport.

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

Groq accepts `ImageURL` and image `BinaryContent` prompt items. Unsupported audio, video, document, and uploaded-file items fail before transport. A `tool_use_failed` provider response becomes ordinary model output so the agent can validate the recovered call and send a retry instead of aborting the run.

`ReasoningFormatParsed` returns reasoning as separate `ThinkingPart` values. Raw `<think>` output is also normalized across static and streamed responses. `ModelSettings.Thinking` follows Groq's model families: GPT-OSS maps portable effort to `low`, `medium`, or `high`; Qwen 3 can disable reasoning with `none`; and legacy reasoning models map visibility without sending unsupported effort values. `WithReasoningWarningHandler` reports when Qwen 3 disabled thinking overrides an explicit effort setting. `WithProvider` keeps Groq response parsing when you route requests through a gateway.

### Compound web search

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
	agent := ai.NewAgent[struct{}, string](
		groq.NewModel("groq/compound"),
		ai.WithNativeTools(ai.WebSearchTool{
			AllowedDomains: []string{"go.dev"},
		}),
	)
	result, err := agent.Run(context.Background(), "What changed in the latest Go release?", struct{}{})
	if err != nil {
		log.Fatal(err)
	}
	fmt.Println(result.Output)
}
```

Groq compound models include web search automatically. `WebSearchTool` forwards allowed and blocked domains through `search_settings` without emitting a duplicate tool declaration. Static and streamed executions produce typed `NativeToolCallPart` and `NativeToolReturnPart` history. Other portable search constraints fail before transport because Groq does not support them on this API.
