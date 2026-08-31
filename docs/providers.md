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

Use `openai.NewResponsesModel` instead of `openai.NewModel` when you need the Responses API.

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
