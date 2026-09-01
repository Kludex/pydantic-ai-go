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
	includeRawAnnotations := true
	settings, err := (openai.Settings{IncludeRawAnnotations: &includeRawAnnotations}).Build()
	if err != nil {
		log.Fatal(err)
	}

	agent := ai.NewAgent[struct{}, string](
		openai.NewResponsesModel("gpt-5"),
		ai.WithModelSettings(settings),
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

OpenAI Responses omits raw provider annotations by default. Set `Settings.IncludeRawAnnotations` as shown above to retain citation annotations in `TextPart.ProviderDetails["annotations"]`. The setting applies to static responses, streams, and resumed background responses without entering the provider request body.

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

## Provide a local fallback

```go
package main

import (
	"context"
	"fmt"
	"log"

	ai "github.com/Kludex/pydantic-ai-go"
	"github.com/Kludex/pydantic-ai-go/models/openai"
)

type SearchArgs struct {
	Query string `json:"query"`
}

func main() {
	localSearch := ai.NewSimpleTool[struct{}](
		"local_search",
		func(_ context.Context, args SearchArgs) (string, error) {
			return "Local search result for: " + args.Query, nil
		},
	)

	agent := ai.NewAgent[struct{}, string](
		openai.NewModel("gpt-5"),
		ai.WithCapabilities(ai.NewWebSearchCapability(ai.WebSearchCapabilityConfig[struct{}]{
			Native: ai.WebSearchTool{},
			Local:  ai.NewFunctionToolset(localSearch),
		})),
	)
	result, err := agent.Run(context.Background(), "Search for the Go release notes", struct{}{})
	if err != nil {
		log.Fatal(err)
	}
	fmt.Println(result.Output)
}
```

`NewWebSearchCapability` registers both paths atomically. Built-in provider models receive exactly one implementation. A nil `Local` field requires native support.

The focused capability classifies `AllowedDomains`, `BlockedDomains`, `MaxUses`, and a false `ExternalWebAccess` value as native-only constraints. It suppresses the local implementation rather than silently ignoring one of those fields. Context size and user location may fall back locally because they only tune native results.

Use `NewNativeOrLocalTool` for other native tools. Use `NewNativeOrLocalToolset` when the fallback has multiple functions or a run-scoped lifecycle. You can register either pair with `WithCapabilities`, `WithRunCapabilities`, or `Agent.AddNativeOrLocal`. Fallback model chains resolve each pair separately for every candidate.

Use `WithNativeRequired("constraint name")` with the generic constructors when a native-only constraint makes the local implementation unsafe. The local tool is then suppressed, and an unsupported provider fails before its request. The native definition cannot be optional.

`WithNativeFallback` and `NativeFallbackToolset` remain available when you intentionally register both paths separately. Use `WithNativeCompanion` or `NativeCompanionToolset` when function definitions belong to a native tool's managed corpus. The marker remains only while that native tool is supported.

Custom models implement `NativeToolSupportModel` to participate in selection. A model without that interface receives both paths because the library cannot safely guess its native support.

## Use OpenAI Chat search models

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
	agent := ai.NewAgent[struct{}, string](
		openai.NewModel("gpt-4o-search-preview"),
		ai.WithNativeTools(ai.WebSearchTool{
			SearchContextSize: ai.WebSearchContextLow,
			UserLocation: &ai.WebSearchUserLocation{
				City: "Utrecht", Country: "NL",
			},
		}),
	)
	result, err := agent.Run(context.Background(), "What is the weather today?", struct{}{})
	if err != nil {
		log.Fatal(err)
	}
	fmt.Println(result.Output)
}
```

OpenAI search-preview Chat models receive `WebSearchTool` as `web_search_options`, not as a function tool. Chat supports context size and approximate user location. It ignores the portable domain, usage-limit, and live-access fields. Other Chat models fail before transport and direct you to `NewResponsesModel`. Use `WithChatWebSearchSupport` only when a future model or compatible gateway implements the same wire field.

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
	Country string
}

type SearchArgs struct {
	Query string `json:"query"`
}

func main() {
	localSearch := ai.NewSimpleTool[Deps](
		"local_search",
		func(_ context.Context, args SearchArgs) (string, error) {
			return "Local result for: " + args.Query, nil
		},
	)
	search := ai.NewDynamicNativeOrLocalTool(
		"web_search",
		func(_ context.Context, runContext *ai.RunContext[Deps]) (ai.NativeTool, error) {
			return ai.WebSearchTool{
				UserLocation: &ai.WebSearchUserLocation{
					Country: runContext.Deps.Country,
				},
			}, nil
		},
		localSearch,
	)
	agent := ai.NewAgent[Deps, string](
		openai.NewResponsesModel("gpt-5"),
		ai.WithCapabilities(search),
	)

	result, err := agent.Run(context.Background(), "Find the latest release notes.", Deps{
		Country: "GB",
	})
	if err != nil {
		log.Fatal(err)
	}
	fmt.Println(result.Output)
}
```

`NewDynamicNativeOrLocalTool` runs its resolver once before every model request. The declared ID is durable and must match every returned native tool. This lets configuration follow dependencies without allowing the local marker to drift. The callback can run concurrently across agent runs and must return detached state.

Use `AddNativeToolFunc` or `WithRunNativeToolFunc` when there is no local fallback. Static tools remain in registration order with dynamic tools. Every resolved request is cloned and revalidated, including tools changed by model-request hooks.

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

Use `NewWebFetchCapability` to pair this declaration with a local toolset. `AllowedDomains` and `BlockedDomains` can be enforced by the local implementation. `MaxUses` requires native support, so the focused capability suppresses the local path when it is set. A nil local toolset also requires native support.

Provider responses use `ToolPartKindWebFetch`. Gemini reconstructs calls and returns from `urlContextMetadata` while retaining the complete metadata in `ModelResponse.ProviderDetails`. Anthropic preserves native result payloads and caller metadata.

## Search managed files

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
	model := openai.NewResponsesModel(
		"gpt-5.4",
		openai.WithResponsesFileSearchResults(true),
	)
	agent := ai.NewAgent[struct{}, string](
		model,
		ai.WithNativeTools(ai.FileSearchTool{
			FileStoreIDs: []string{"vs_abc123"},
		}),
	)

	result, err := agent.Run(context.Background(), "Summarize the latest quarterly report.", struct{}{})
	if err != nil {
		log.Fatal(err)
	}
	fmt.Println(result.Output)
}
```

`FileSearchTool` references managed vector stores. OpenAI Responses sends `FileStoreIDs` as vector-store IDs. `WithResponsesFileSearchResults` requests the matched chunks in each normalized `NativeToolReturnPart`; leave it disabled when you only need the model's answer.

Google sends the same IDs as Gemini file-search store names. Older Gemini responses are reconstructed from executable queries and grounding contexts. Gemini 3 responses preserve explicit provider call IDs and fill empty tool responses from later grounding metadata. Same-provider histories replay through Google `toolCall` and `toolResponse` parts.

`MaxNumResults`, `Instructions`, and `RetrievalMode` are portable fields reserved for providers that expose those controls. OpenAI and Google ignore them. xAI collections-search rendering remains provider-parity work.

## Consult an advisor model

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
	maxUses := 2
	maxTokens := 2048
	agent := ai.NewAgent[struct{}, string](
		anthropic.NewModel("claude-sonnet-4-6"),
		ai.WithNativeTools(ai.AdvisorTool{
			Model:     "claude-opus-4-8",
			MaxUses:   &maxUses,
			MaxTokens: &maxTokens,
			Caching:   ai.AdvisorCaching1Hour,
		}),
	)

	result, err := agent.Run(context.Background(), "Review this migration plan for hidden risks.", struct{}{})
	if err != nil {
		log.Fatal(err)
	}
	fmt.Println(result.Output)
}
```

`AdvisorTool` lets an eligible Anthropic executor consult a stronger model during generation. `MaxUses` resets for every model request. `MaxTokens` must be at least 1024. Advisor calls and plaintext, encrypted, or error results remain in normalized history while the advisor stays enabled. They are removed from requests that disable the advisor and from token-count requests because Anthropic rejects inactive advisor blocks.

Advisor iteration tokens remain separate from executor totals. You can inspect `advisor_iterations`, `advisor_input_tokens`, `advisor_output_tokens`, and advisor cache-token fields in `Usage.Details`.

## Connect a provider-hosted MCP server

```go
package main

import (
	"context"
	"fmt"
	"log"
	"os"

	ai "github.com/Kludex/pydantic-ai-go"
	"github.com/Kludex/pydantic-ai-go/models/openai"
)

func main() {
	agent := ai.NewAgent[struct{}, string](
		openai.NewResponsesModel("gpt-5"),
		ai.WithNativeTools(ai.MCPServerTool{
			ID:                 "docs",
			URL:                "https://example.com/mcp",
			AuthorizationToken: os.Getenv("MCP_AUTH_TOKEN"),
			Description:        "Search the product documentation.",
			AllowedTools:       []string{"search"},
		}),
	)

	result, err := agent.Run(context.Background(), "How do I rotate an API key?", struct{}{})
	if err != nil {
		log.Fatal(err)
	}
	fmt.Println(result.Output)
}
```

`MCPServerTool` delegates the connection and tool execution to the model provider. OpenAI Responses accepts remote URLs and `x-openai-connector:<connector-id>` references. Anthropic accepts remote URLs and advertises its required MCP beta automatically. Both normalize provider-owned calls as `NativeToolCallPart` and `NativeToolReturnPart` values and replay their provider IDs on later requests. OpenAI also exposes server discovery as a native lifecycle.

OpenAI receives `AuthorizationToken` and `Headers`. Anthropic receives `AuthorizationToken` but does not support custom MCP headers or descriptions. Treat all credentials as secrets. Restrict `AllowedTools` to operations the model may execute without local approval. Use the [`mcp`](mcp.md) package instead when your application must own the session, inspect every call, or request approval locally.

## Add client-managed memory

```go
package main

import (
	"context"
	"fmt"
	"log"

	ai "github.com/Kludex/pydantic-ai-go"
	"github.com/Kludex/pydantic-ai-go/models/anthropic"
)

type MemoryCommand struct {
	Command string `json:"command"`
	Path    string `json:"path"`
}

func main() {
	agent := ai.NewAgent[struct{}, string](
		anthropic.NewModel("claude-sonnet-4-6"),
		ai.WithNativeTools(ai.MemoryTool{}),
	)
	ai.AddSimpleTool(agent, "memory", func(_ context.Context, command MemoryCommand) (string, error) {
		return fmt.Sprintf("execute %s on %s", command.Command, command.Path), nil
	})

	result, err := agent.Run(context.Background(), "Remember that I live in Mexico City.", struct{}{})
	if err != nil {
		log.Fatal(err)
	}
	fmt.Println(result.Output)
}
```

Anthropic's memory declaration controls the command schema shown to the model. Your local function tool named `memory` owns storage and executes each command. This keeps persistence, authorization, tenancy, and deletion policy inside your application. A missing or deferred `memory` function fails before transport instead of advertising a tool that cannot run.

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

`CodeExecutionTool` lets Gemini and Anthropic run model-generated code. Executable code and its result become normalized call and return parts with `ToolPartKindCodeExecution`. The provider language, source, outcome, and output remain available in those parts.

You can attach provider-hosted files through `CodeExecutionTool.Files`:

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
		ai.WithNativeTools(ai.CodeExecutionTool{Files: []ai.UploadedFile{
			{FileID: "file_abc123", ProviderName: "anthropic", MediaType: "text/csv"},
		}}),
	)

	result, err := agent.Run(context.Background(), "Calculate totals from the uploaded CSV.", struct{}{})
	if err != nil {
		log.Fatal(err)
	}
	fmt.Println(result.Output)
}
```

Only files whose `ProviderName` matches the selected provider are attached. Anthropic places uploads on the first user message so the cacheable prefix stays stable. It also retains and reuses the response container ID through subsequent tool turns.

Use `WithResponsesCodeExecutionOutputs` when you need OpenAI code-interpreter logs and generated images:

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
	model := openai.NewResponsesModel(
		"gpt-5.4",
		openai.WithResponsesCodeExecutionOutputs(true),
	)
	agent := ai.NewAgent[struct{}, string](
		model,
		ai.WithNativeTools(ai.CodeExecutionTool{}),
	)

	result, err := agent.Run(context.Background(), "Plot y = x squared from -5 to 5.", struct{}{})
	if err != nil {
		log.Fatal(err)
	}
	fmt.Println(result.Output)
	for _, message := range result.Messages() {
		response, ok := message.(ai.ModelResponse)
		if !ok {
			continue
		}
		for _, part := range response.Parts {
			if file, ok := part.(ai.FilePart); ok {
				fmt.Printf("generated %s: %d bytes\n", file.Content.MediaType, len(file.Content.Data))
			}
		}
	}
}
```

The option adds `code_interpreter_call.outputs` to the Responses `include` list. Image outputs become detached `FilePart` values. Logs remain on the matching `NativeToolReturnPart`.

Bedrock and xAI code-execution rendering remain provider-parity work.

## Generate images

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
	agent := ai.NewAgent[struct{}, string](
		openai.NewResponsesModel("gpt-5.4"),
		ai.WithNativeTools(ai.ImageGenerationTool{
			Quality:       ai.ImageGenerationQualityHigh,
			AspectRatio:   ai.ImageAspectRatio3x2,
			PartialImages: 2,
		}),
	)

	result, err := agent.Run(context.Background(), "Create a watercolor painting of a Go gopher.", struct{}{})
	if err != nil {
		log.Fatal(err)
	}
	fmt.Println(result.Output)
	for _, message := range result.Messages() {
		response, ok := message.(ai.ModelResponse)
		if !ok {
			continue
		}
		for _, part := range response.Parts {
			if file, ok := part.(ai.FilePart); ok {
				fmt.Printf("generated %s: %d bytes\n", file.Content.MediaType, len(file.Content.Data))
			}
		}
	}
}
```

`ImageGenerationTool` exposes portable action, background, input-fidelity, moderation, model, compression, format, partial-image, quality, size, and aspect-ratio settings. OpenAI Responses maps `1:1`, `2:3`, and `3:2` aspect ratios to supported pixel sizes. Unsupported or conflicting OpenAI dimensions fail before transport.

OpenAI Responses records static and streamed calls with `ToolPartKindImageGeneration`. Generated bytes become detached `FilePart` values. Each streamed partial image replaces the earlier file with the same part ID, so completed history contains only the latest image.

Google image models render aspect ratio and image size through `generationConfig.imageConfig`. Vertex AI also renders output format and JPEG compression. The Gemini Developer API leaves those two settings out. Google-generated images are ordinary `FilePart` values because Google does not expose image generation as a tool-call lifecycle. Action, background, input fidelity, moderation, model selection, and partial-image settings are OpenAI-specific.
