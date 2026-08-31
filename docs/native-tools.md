# Use provider-native tools

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
	externalWebAccess := true
	agent := ai.NewAgent[struct{}, string](
		openai.NewResponsesModel("gpt-5"),
		ai.WithNativeTools(ai.WebSearchTool{
			SearchContextSize: ai.WebSearchContextHigh,
			AllowedDomains:    []string{"go.dev"},
			ExternalWebAccess: &externalWebAccess,
		}),
	)

	result, err := agent.Run(context.Background(), "What changed in the latest Go release?", struct{}{})
	if err != nil {
		log.Fatal(err)
	}
	fmt.Println(result.Output)
}
```

A native tool runs inside the model provider. The agent records its calls and returns in history, but it does not execute them as local Go functions.

`WebSearchTool` is provider-neutral. OpenAI Responses renders it as the hosted `web_search` tool. Gemini renders it as `googleSearch`. Anthropic renders the model-appropriate version of `web_search`. Static and streamed responses become `NativeToolCallPart` and `NativeToolReturnPart` values with `ToolPartKindWebSearch`.

## Configure search

| Field | Purpose |
| --- | --- |
| `SearchContextSize` | Select `low`, `medium`, or `high` retrieval context. The zero value means `medium`. |
| `UserLocation` | Localize results by city, country, region, or timezone. |
| `AllowedDomains` | Restrict results to selected domains where supported. |
| `BlockedDomains` | Exclude selected domains where supported. |
| `MaxUses` | Limit searches where supported. Zero uses the provider default. |
| `ExternalWebAccess` | Allow or forbid live web access where supported. `nil` uses the provider default. |
| `Optional` | Omit the tool instead of failing when the selected provider does not support it. |

Providers support different subsets of these fields. OpenAI Responses sends context size, location, allowed domains, and external web access. Anthropic sends location, allowed and blocked domains, and maximum uses. It selects `web_search_20260209` for models with dynamic filtering and `web_search_20250305` otherwise. Gemini currently sends only the native `googleSearch` declaration. Unsupported portable fields are left out of each request.

Gemini 3 can combine provider-native tools with function tools. Earlier Gemini models reject that combination before the HTTP request. Grounded Gemini responses retain complete `grounding_metadata` provider details while exposing search queries and returned web sources through normalized native-tool parts.

A required native tool fails before the provider request when the selected model adapter cannot render it. This prevents silent behavior changes. Set `Optional` only when your application has another valid path.

## Scope a tool to one run

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
	agent := ai.NewAgent[struct{}, string](openai.NewResponsesModel("gpt-5"))
	result, err := agent.Run(
		context.Background(),
		"Find today's Go security announcements.",
		struct{}{},
		ai.WithRunNativeTools(ai.WebSearchTool{AllowedDomains: []string{"go.dev"}}),
	)
	if err != nil {
		log.Fatal(err)
	}
	fmt.Println(result.Output)
}
```

`WithRunNativeTools` does not modify the shared agent. You can also call `Agent.AddNativeTool` before the first run. A capability can contribute a static native tool with `CapabilityRegistry.AddNativeTool`.

Native tools with the same `UniqueID` cannot appear twice in one request. Native tool definitions, domain lists, locations, and pointer settings are cloned before providers and hooks receive them.

## Resolve a tool from dependencies

```go
package main

import (
	"context"
	"fmt"
	"log"

	ai "github.com/Kludex/pydantic-ai-go"
	"github.com/Kludex/pydantic-ai-go/models/openai"
)

type Deps struct {
	DocumentationDomain string
}

func main() {
	agent := ai.NewAgent[Deps, string](openai.NewResponsesModel("gpt-5"))
	agent.AddNativeToolFunc(func(
		_ context.Context,
		runContext *ai.RunContext[Deps],
	) (ai.NativeTool, error) {
		return ai.WebSearchTool{
			AllowedDomains: []string{runContext.Deps.DocumentationDomain},
		}, nil
	})

	result, err := agent.Run(context.Background(), "Find the latest release notes.", Deps{
		DocumentationDomain: "go.dev",
	})
	if err != nil {
		log.Fatal(err)
	}
	fmt.Println(result.Output)
}
```

`AddNativeToolFunc` runs before every model request. This lets a tool follow dependencies, selected-model state, usage, or retry state. The callback can run concurrently across agent runs and must return detached state.

Use `WithRunNativeToolFunc` for a callback scoped to one run. Static tools remain in registration order with dynamic tools. Every resolved request is cloned and revalidated, including tools changed by model-request hooks.

## Fetch URLs

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
	agent := ai.NewAgent[struct{}, string](
		anthropic.NewModel("claude-sonnet-4-6"),
		ai.WithNativeTools(ai.WebFetchTool{
			AllowedDomains:   []string{"go.dev"},
			MaxUses:          3,
			MaxContentTokens: 4096,
			EnableCitations:  true,
		}),
	)

	result, err := agent.Run(context.Background(), "Summarize https://go.dev/doc/go1.25", struct{}{})
	if err != nil {
		log.Fatal(err)
	}
	fmt.Println(result.Output)
}
```

`WebFetchTool` lets Anthropic or Gemini retrieve URL content. Anthropic sends domain filters, maximum uses, content limits, and citation configuration. Gemini renders the portable tool as `urlContext` and leaves unsupported settings out.

Provider responses use `ToolPartKindWebFetch`. Gemini reconstructs calls and returns from `urlContextMetadata` while retaining the complete metadata in `ModelResponse.ProviderDetails`. Anthropic preserves native result payloads and caller metadata.

## Execute code

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
	agent := ai.NewAgent[struct{}, string](
		google.NewModel("gemini-3-flash"),
		ai.WithNativeTools(ai.CodeExecutionTool{}),
	)

	result, err := agent.Run(context.Background(), "Calculate the first 20 Fibonacci numbers with Python.", struct{}{})
	if err != nil {
		log.Fatal(err)
	}
	fmt.Println(result.Output)
}
```

`CodeExecutionTool` lets Gemini run model-generated code. Executable code and its result become normalized call and return parts with `ToolPartKindCodeExecution`. The provider language, source, outcome, and output remain available in those parts.

Uploaded execution files and Anthropic, OpenAI Responses, Bedrock, and xAI rendering remain provider-parity work.
