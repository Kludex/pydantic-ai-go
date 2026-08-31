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

`WebSearchTool` is provider-neutral. OpenAI Responses currently renders it as the hosted `web_search` tool. Its static and streamed responses become `NativeToolCallPart` and `NativeToolReturnPart` values with `ToolPartKindWebSearch`.

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

Providers support different subsets of these fields. OpenAI Responses currently sends context size, location, allowed domains, and external web access. Unsupported portable fields are left out of its request.

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
